// 教程 14 阶段 2:激光 SLAM 前端(C++17 单文件,零外部依赖)= 阶段 1 Python 的位对齐移植。
//
// 与 autodrivedata/slam.py + bin/slam_odometry.py 逐位对齐的口径(改动必须同步两边):
//   * 下采样:体素键 ((kx+400)*801 + (ky+400))*201 + (kz+100),**按原始点序累加**,
//     输出按体素键升序(与 np.unique 的排序口径一致);
//   * 网格哈希最近邻:半径递增壳扫描,壳内偏移序 = `_radius_offsets`(a 外层 → c2 内层),
//     格内按扫描序、**严格小于才替换**(等距先到者胜);早停 = 扫完层 rad 后
//     `best_d < ((rad-1)·cell)²` 且 rad≥1 —— **rad=0 不允许早停**(自身格内有点
//     不等于全局最近,层 1 可能更近);
//   * 法向:27 邻域(GRID_OFFSETS 序)候选按 (d², 偏移序, 点序) 稳定排序取前 k,
//     PCA 协方差 → 最小特征向量(3×3 对称 Jacobi,升序);
//   * ICP 一步:λ 正则化法方程 (AᵀA+λI)x = Aᵀb,A[i,:3]=p×n、A[i,3:]=n、b=−n·(p−ref),
//     Cholesky 求解(替代 numpy 的 LU——唯一允许的舍入偏差 ~1e-12);
//   * 链式位姿 T_0→k = Δ_k · T_0→k−1,恒速先验 seed 只作 ICP 迭代起点(cur = seed·src),
//     **不得再乘进 init**(乘两次 = 恒速先验叠加两次 → 轨迹按 k² 发散)。
//
// 用法:slam_cpp <velodyne_dir> <out_json> [start] [end]
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

// ---- 常量(与 autodrivedata/slam.py 逐条对照)----
constexpr double kGridCell = 0.5;
constexpr double kVoxel = 0.5;
constexpr int kIcpMaxIter = 15;
constexpr double kIcpTolDelta = 1e-5;
constexpr int kIcpNormalK = 8;
// 法向可信门(**位对齐必需,不是调参**):λ1 ≤ kPlanarity·λ2 的邻域退化,最小特征
// 向量方向不定(Jacobi 与 LAPACK 会挑到不同向量,差 O(1))→ 置零向量,GN 侧剔除。
constexpr double kPlanarity = 1e-9;
constexpr double kIcpRegLam = 1e-4;
constexpr int kGridMaxRad = 16;

struct Vec3 {
    double x = 0.0, y = 0.0, z = 0.0;
};
inline Vec3 operator-(const Vec3& a, const Vec3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 operator+(const Vec3& a, const Vec3& b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline double dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

struct Mat4 {
    double m[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
};

inline void mat_identity(Mat4& m) {
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) m.m[i][j] = (i == j) ? 1.0 : 0.0;
}

inline void mat_mul(const Mat4& a, const Mat4& b, Mat4& out) {
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            double s = 0.0;
            for (int k = 0; k < 4; ++k) s += a.m[i][k] * b.m[k][j];
            out.m[i][j] = s;
        }
}

// 刚体变换的逆(Rᵀ, −Rᵀt;末行 [0,0,0,1])
inline void mat_inverse_rigid(const Mat4& m, Mat4& out) {
    mat_identity(out);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) out.m[i][j] = m.m[j][i];
    for (int i = 0; i < 3; ++i) {
        double s = 0.0;
        for (int j = 0; j < 3; ++j) s += m.m[j][i] * m.m[j][3];
        out.m[i][3] = -s;
    }
}

// ---- 体素下采样(accum.voxel_downsample 同款)----
std::vector<Vec3> voxel_downsample(const std::vector<Vec3>& pts) {
    const size_t n = pts.size();
    if (n == 0) return {};
    std::vector<int64_t> key(n);
    for (size_t i = 0; i < n; ++i) {
        int64_t kx = (int64_t)std::floor(pts[i].x / kVoxel);
        int64_t ky = (int64_t)std::floor(pts[i].y / kVoxel);
        int64_t kz = (int64_t)std::floor(pts[i].z / kVoxel);
        key[i] = ((kx + 400) * 801 + (ky + 400)) * 201 + (kz + 100);
    }
    std::vector<size_t> ord(n);
    for (size_t i = 0; i < n; ++i) ord[i] = i;
    std::stable_sort(ord.begin(), ord.end(), [&](size_t a, size_t b) { return key[a] < key[b]; });
    std::vector<Vec3> out;
    std::vector<double> cnt;
    size_t i = 0;
    while (i < n) {
        const int64_t k = key[ord[i]];
        Vec3 s{0.0, 0.0, 0.0};
        double c = 0.0;
        // 同键点按**原始索引序**累加(与 np.add.at 的累加序一致,浮点不满足结合律)
        while (i < n && key[ord[i]] == k) {
            const Vec3& p = pts[ord[i]];
            s = s + p;
            c += 1.0;
            ++i;
        }
        out.push_back({s.x / c, s.y / c, s.z / c});
        cnt.push_back(c);
    }
    return out;
}

// ---- 壳偏移(与 Python _radius_offsets 同序:a 外层 → c2 内层)----
const std::vector<std::vector<std::array<int, 3>>>& radius_offsets_by_layer() {
    static const std::vector<std::vector<std::array<int, 3>>> layers = [] {
        std::vector<std::vector<std::array<int, 3>>> v;
        for (int r = 0; r <= kGridMaxRad; ++r) {
            std::vector<std::array<int, 3>> offs;
            for (int a = -r; a <= r; ++a)
                for (int b = -r; b <= r; ++b)
                    for (int c = -r; c <= r; ++c) {
                        const int m = std::max(std::abs(a), std::max(std::abs(b), std::abs(c)));
                        if (m == r) offs.push_back({a, b, c});
                    }
            v.push_back(std::move(offs));
        }
        return v;
    }();
    return layers;
}

// 27 邻域(dz 外层 → dy → dx 内层;与 Python GRID_OFFSETS 同序,法向候选 tie-break 用)
const std::vector<std::array<int, 3>>& grid_offsets() {
    static const std::vector<std::array<int, 3>> v = [] {
        std::vector<std::array<int, 3>> o;
        for (int dz = -1; dz <= 1; ++dz)
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) o.push_back({dx, dy, dz});
        return o;
    }();
    return v;
}

// 体素整数坐标(不折线性键:键解码在负数/大坐标下易错,直接用三元组做哈希键)
struct Cell3 {
    int x = 0, y = 0, z = 0;
    bool operator==(const Cell3& o) const { return x == o.x && y == o.y && z == o.z; }
};
struct Cell3Hash {
    size_t operator()(const Cell3& c) const {
        size_t h = (size_t)(uint32_t)c.x * 73856093u;
        h ^= (size_t)(uint32_t)c.y * 19349663u;
        h ^= (size_t)(uint32_t)c.z * 83492791u;
        return h;
    }
};
inline Cell3 cell_of(const Vec3& p, double cell) {
    return {(int)std::floor(p.x / cell), (int)std::floor(p.y / cell), (int)std::floor(p.z / cell)};
}

// ---- 网格哈希(GridHash.nearest:半径递增壳扫描 + 安全早停)----
struct GridHash {
    const std::vector<Vec3>* pts = nullptr;
    double cell = kGridCell;
    std::unordered_map<Cell3, std::vector<int>, Cell3Hash> buckets;

    GridHash(const std::vector<Vec3>& p, double c) : pts(&p), cell(c) {
        buckets.reserve(p.size() * 2);
        for (size_t i = 0; i < p.size(); ++i) buckets[cell_of(p[i], cell)].push_back((int)i);
    }

    // 返回 (最近邻索引, 距离²);空云 → (-1, inf)
    std::pair<int, double> nearest(const Vec3& q) const {
        if (pts->empty()) return {-1, std::numeric_limits<double>::infinity()};
        const Cell3 cq = cell_of(q, cell);
        const int cx = cq.x, cy = cq.y, cz = cq.z;
        int best_i = -1;
        double best_d = std::numeric_limits<double>::infinity();
        const auto& layers = radius_offsets_by_layer();
        int rad = 0, empty_layers = 0;
        const size_t n = pts->size();
        while (true) {
            if (empty_layers >= 2 && rad > 4) {  // 稀疏云兜底:全云扫描
                for (size_t i = 0; i < n; ++i) {
                    const Vec3 d = (*pts)[i] - q;
                    const double dd = dot(d, d);
                    if (dd < best_d) { best_d = dd; best_i = (int)i; }
                }
                break;
            }
            size_t layer_points = 0;
            for (const auto& off : layers[rad]) {
                auto it = buckets.find(Cell3{cx + off[0], cy + off[1], cz + off[2]});
                if (it == buckets.end()) continue;
                layer_points += it->second.size();
                for (int i : it->second) {
                    const Vec3 d = (*pts)[i] - q;
                    const double dd = dot(d, d);
                    if (dd < best_d) { best_d = dd; best_i = i; }  // 严格小于:等距保留先到者
                }
            }
            // 安全早停(rad≥1):扫完层 rad 后未扫格必在某轴偏 ≥ rad 格 ⇒ 距离 ≥ (rad−1)·cell
            if (rad >= 1 && best_i != -1 && best_d < ((rad - 1) * cell) * ((rad - 1) * cell)) break;
            empty_layers = (layer_points == 0) ? empty_layers + 1 : 0;
            ++rad;
            if (rad > kGridMaxRad) {  // 超出预生成半径 → 全云扫描
                for (size_t i = 0; i < n; ++i) {
                    const Vec3 d = (*pts)[i] - q;
                    const double dd = dot(d, d);
                    if (dd < best_d) { best_d = dd; best_i = (int)i; }
                }
                break;
            }
        }
        return {best_i, best_d};
    }
};

// ---- 3×3 对称矩阵特征分解(循环 Jacobi,特征值升序 + 对应特征向量)----
void eigh3(double a[3][3], double evals[3], double evecs[3][3]) {
    double v[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) v[i][j] = (i == j) ? 1.0 : 0.0;
    for (int sweep = 0; sweep < 64; ++sweep) {
        double off = std::abs(a[0][1]) + std::abs(a[0][2]) + std::abs(a[1][2]);
        if (off < 1e-300) break;
        for (int p = 0; p < 2; ++p) {
            for (int qq = p + 1; qq < 3; ++qq) {
                if (std::abs(a[p][qq]) < 1e-300) continue;
                const double theta = (a[qq][qq] - a[p][p]) / (2.0 * a[p][qq]);
                const double t = (theta >= 0.0 ? 1.0 : -1.0) / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0);
                const double s = t * c;
                for (int k = 0; k < 3; ++k) {
                    const double akp = a[k][p], akq = a[k][qq];
                    a[k][p] = c * akp - s * akq;
                    a[k][qq] = s * akp + c * akq;
                }
                for (int k = 0; k < 3; ++k) {
                    const double apk = a[p][k], aqk = a[qq][k];
                    a[p][k] = c * apk - s * aqk;
                    a[qq][k] = s * apk + c * aqk;
                }
                for (int k = 0; k < 3; ++k) {
                    const double vkp = v[k][p], vkq = v[k][qq];
                    v[k][p] = c * vkp - s * vkq;
                    v[k][qq] = s * vkp + c * vkq;
                }
            }
        }
    }
    int idx[3] = {0, 1, 2};
    double d[3] = {a[0][0], a[1][1], a[2][2]};
    std::sort(idx, idx + 3, [&](int x, int y) { return d[x] < d[y]; });
    for (int i = 0; i < 3; ++i) {
        evals[i] = d[idx[i]];
        for (int k = 0; k < 3; ++k) evecs[k][i] = v[k][idx[i]];
    }
}

// ---- 逐点 PCA 法向(27 邻域候选 → 稳定排序取前 k → 协方差最小特征向量)----
std::vector<Vec3> estimate_normals(const std::vector<Vec3>& pts) {
    const size_t n = pts.size();
    std::vector<Vec3> normals(n, {1.0, 0.0, 0.0});
    if (n == 0) return normals;
    GridHash gh(pts, kGridCell);
    const auto& offs = grid_offsets();
    struct Cand { double d2; int idx; };
    std::vector<Cand> cand;
    for (size_t i = 0; i < n; ++i) {
        const Vec3& q = pts[i];
        const Cell3 cq = cell_of(q, kGridCell);
        cand.clear();
        // 偏移序外层、点序内层 → stable_sort(d²) 后即 (d², 偏移序, 点序) 序
        for (const auto& off : offs) {
            auto it = gh.buckets.find(Cell3{cq.x + off[0], cq.y + off[1], cq.z + off[2]});
            if (it == gh.buckets.end()) continue;
            for (int j : it->second) {
                const Vec3 d = pts[j] - q;
                cand.push_back({dot(d, d), j});
            }
        }
        if (cand.empty()) continue;  // 空云兜底(normals 预置 (1,0,0) = numpy eigh(0) 的口径)
        std::stable_sort(cand.begin(), cand.end(), [](const Cand& a, const Cand& b) { return a.d2 < b.d2; });
        const size_t k = std::min<size_t>(kIcpNormalK, cand.size());
        Vec3 mu{0.0, 0.0, 0.0};
        for (size_t j = 0; j < k; ++j) mu = mu + pts[cand[j].idx];
        mu = {mu.x / k, mu.y / k, mu.z / k};
        double cov[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
        for (size_t j = 0; j < k; ++j) {
            const Vec3 d = pts[cand[j].idx] - mu;
            cov[0][0] += d.x * d.x; cov[0][1] += d.x * d.y; cov[0][2] += d.x * d.z;
            cov[1][1] += d.y * d.y; cov[1][2] += d.y * d.z; cov[2][2] += d.z * d.z;
        }
        cov[1][0] = cov[0][1]; cov[2][0] = cov[0][2]; cov[2][1] = cov[1][2];
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) cov[a][b] /= (double)k;
        double evals[3], evecs[3][3];
        eigh3(cov, evals, evecs);
        // **可信门(与 Python estimate_normals 同式,位对齐契约)**:λ1 ≤ kPlanarity·λ2
        // 的邻域退化,最小特征向量方向不定 → 置零向量,GN 侧整点剔除。
        if (!(evals[1] > kPlanarity * evals[2])) {
            normals[i] = {0.0, 0.0, 0.0};
            continue;
        }
        // **符号规范化**:特征向量正负号任意(Jacobi 与 LAPACK 取号不同),而法向有号
        // (参与 p×n 与 −n·(p−ref))→ 翻号会让解翻号、旋转完全错。
        // 规则:|分量| 最大者置正,并列取最先出现的分量(双方同式,可逐位复算)。
        int kmax = 0;
        for (int a = 1; a < 3; ++a)
            if (std::abs(evecs[a][0]) > std::abs(evecs[kmax][0])) kmax = a;
        const double sgn = (evecs[kmax][0] < 0.0) ? -1.0 : 1.0;
        normals[i] = {sgn * evecs[0][0], sgn * evecs[1][0], sgn * evecs[2][0]};
    }
    return normals;
}

// ---- 6×6 对称正定 Cholesky 求解(替代 numpy LU;唯一允许舍入偏差源)----
bool chol_solve6(double A[6][6], const double b[6], double x[6]) {
    double L[6][6] = {{0}};
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j <= i; ++j) {
            double s = A[i][j];
            for (int k = 0; k < j; ++k) s -= L[i][k] * L[j][k];
            if (i == j) {
                if (s <= 0.0) return false;
                L[i][j] = std::sqrt(s);
            } else {
                L[i][j] = s / L[j][j];
            }
        }
    }
    double y[6];
    for (int i = 0; i < 6; ++i) {
        double s = b[i];
        for (int k = 0; k < i; ++k) s -= L[i][k] * y[k];
        y[i] = s / L[i][i];
    }
    for (int i = 5; i >= 0; --i) {
        double s = y[i];
        for (int k = i + 1; k < 6; ++k) s -= L[k][i] * x[k];
        x[i] = s / L[i][i];
    }
    return true;
}

inline void rodrigues(const double w[3], double R[3][3]) {
    const double th = std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
    if (th < 1e-9) {
        R[0][0] = 1.0; R[0][1] = -w[2]; R[0][2] = w[1];
        R[1][0] = w[2]; R[1][1] = 1.0; R[1][2] = -w[0];
        R[2][0] = -w[1]; R[2][1] = w[0]; R[2][2] = 1.0;
        return;
    }
    const double a = w[0] / th, b = w[1] / th, c = w[2] / th;
    const double s = std::sin(th), co = std::cos(th), v = 1.0 - co;
    R[0][0] = co + a * a * v;       R[0][1] = a * b * v - c * s;   R[0][2] = a * c * v + b * s;
    R[1][0] = b * a * v + c * s;    R[1][1] = co + b * b * v;      R[1][2] = b * c * v - a * s;
    R[2][0] = c * a * v - b * s;    R[2][1] = c * b * v + a * s;   R[2][2] = co + c * c * v;
}

// ---- ICP 一步:(ω, v) 解正则化法方程 ----
void estimate_transform_gn(const std::vector<Vec3>& src_in, const std::vector<Vec3>& ref_in,
                           const std::vector<Vec3>& ref_n_in, double R[3][3], double t[3]) {
    // **退化邻域剔除**:estimate_normals 把 λ1/λ2 过小的邻域法向置零(方向不定),
    // 这里显式过滤 —— 零行对 AtA/Atb 贡献本就恒为 0,但白占行数会污染残差口径。
    std::vector<Vec3> src, ref, ref_n;
    src.reserve(src_in.size());
    for (size_t i = 0; i < src_in.size(); ++i) {
        if (dot(ref_n_in[i], ref_n_in[i]) <= 0.0) continue;
        src.push_back(src_in[i]);
        ref.push_back(ref_in[i]);
        ref_n.push_back(ref_n_in[i]);
    }
    double AtA[6][6] = {{0}};
    double Atb[6] = {0};
    const size_t n = src.size();
    if (n == 0) {
        R[0][0] = 1.0; R[0][1] = 0.0; R[0][2] = 0.0;
        R[1][0] = 0.0; R[1][1] = 1.0; R[1][2] = 0.0;
        R[2][0] = 0.0; R[2][1] = 0.0; R[2][2] = 1.0;
        t[0] = t[1] = t[2] = 0.0;
        return;
    }
    for (size_t i = 0; i < n; ++i) {
        const Vec3 cr = cross(src[i], ref_n[i]);
        const Vec3 d = src[i] - ref[i];
        const double A[6] = {cr.x, cr.y, cr.z, ref_n[i].x, ref_n[i].y, ref_n[i].z};
        const double b = -dot(ref_n[i], d);
        for (int r = 0; r < 6; ++r) {
            for (int c = 0; c < 6; ++c) AtA[r][c] += A[r] * A[c];
            Atb[r] += A[r] * b;
        }
    }
    for (int r = 0; r < 6; ++r) AtA[r][r] += kIcpRegLam;
    double x[6] = {0};
    chol_solve6(AtA, Atb, x);
    rodrigues(x, R);
    t[0] = x[3]; t[1] = x[4]; t[2] = x[5];
}

struct IcpResult {
    Mat4 T;        // 位姿 = init_T @ inv(T_delta)(见 icp_odometry 注释;勿改成 T_delta@init_T)
    Mat4 T_delta;  // 点映射 src→ref(ICP 直接解出的量)
    double rmse_final = std::numeric_limits<double>::infinity();
    double overlap = 0.0;
    bool converged = false;
    int iters = 0;
    bool failed = false;
};

// ---- 帧间点面 ICP(init_T = 上一帧链式位姿;seed = 恒速**位姿**增量,仅作迭代起点)----
IcpResult icp_odometry(const std::vector<Vec3>& src, const std::vector<Vec3>& ref, const Mat4& init_T,
                       const Mat4& seed, bool has_seed) {
    IcpResult out;
    mat_identity(out.T);
    mat_identity(out.T_delta);
    if (src.empty() || ref.empty()) {
        out.T = init_T;
        out.failed = true;
        return out;
    }
    const std::vector<Vec3> ref_n = estimate_normals(ref);
    GridHash gh(ref, kGridCell);
    double R_acc[3][3], t_acc[3];
    std::vector<Vec3> cur(src.size());
    if (!has_seed) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) R_acc[i][j] = (i == j) ? 1.0 : 0.0;
            t_acc[i] = 0.0;
        }
        cur = src;
    } else {
        // seed 是位姿增量 → 取逆换成点映射(与 numpy 侧同式;方向错会收敛到次优解)
        Mat4 seed_map;
        mat_inverse_rigid(seed, seed_map);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) R_acc[i][j] = seed_map.m[i][j];
            t_acc[i] = seed_map.m[i][3];
        }
        for (size_t k = 0; k < src.size(); ++k) {
            cur[k] = {R_acc[0][0] * src[k].x + R_acc[0][1] * src[k].y + R_acc[0][2] * src[k].z + t_acc[0],
                      R_acc[1][0] * src[k].x + R_acc[1][1] * src[k].y + R_acc[1][2] * src[k].z + t_acc[1],
                      R_acc[2][0] * src[k].x + R_acc[2][1] * src[k].y + R_acc[2][2] * src[k].z + t_acc[2]};
        }
    }
    std::vector<Vec3> nb(src.size()), nb_n(src.size());
    for (int iter = 0; iter < kIcpMaxIter; ++iter) {
        double se = 0.0;
        for (size_t k = 0; k < cur.size(); ++k) {
            const auto [idx, d2] = gh.nearest(cur[k]);
            (void)d2;
            nb[k] = ref[(size_t)idx];
            nb_n[k] = ref_n[(size_t)idx];
            const double res = std::abs(dot(cur[k] - nb[k], nb_n[k]));
            se += res * res;
        }
        out.rmse_final = std::sqrt(se / (double)cur.size());
        ++out.iters;
        double R[3][3], t[3];
        estimate_transform_gn(cur, nb, nb_n, R, t);
        const double delta = std::sqrt((R[0][0] - 1) * (R[0][0] - 1) + R[0][1] * R[0][1] + R[0][2] * R[0][2] +
                                       R[1][0] * R[1][0] + (R[1][1] - 1) * (R[1][1] - 1) + R[1][2] * R[1][2] +
                                       R[2][0] * R[2][0] + R[2][1] * R[2][1] + (R[2][2] - 1) * (R[2][2] - 1)) +
                             std::sqrt(t[0] * t[0] + t[1] * t[1] + t[2] * t[2]);
        for (size_t k = 0; k < cur.size(); ++k) {
            const Vec3 p = cur[k];
            cur[k] = {R[0][0] * p.x + R[0][1] * p.y + R[0][2] * p.z + t[0],
                      R[1][0] * p.x + R[1][1] * p.y + R[1][2] * p.z + t[1],
                      R[2][0] * p.x + R[2][1] * p.y + R[2][2] * p.z + t[2]};
        }
        double Rn[3][3];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                double s = 0.0;
                for (int k = 0; k < 3; ++k) s += R[i][k] * R_acc[k][j];
                Rn[i][j] = s;
            }
        double tn[3];
        for (int i = 0; i < 3; ++i)
            tn[i] = R[i][0] * t_acc[0] + R[i][1] * t_acc[1] + R[i][2] * t_acc[2] + t[i];
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) R_acc[i][j] = Rn[i][j];
            t_acc[i] = tn[i];
        }
        if (delta < kIcpTolDelta) { out.converged = true; break; }
    }
    Mat4 T_delta;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) T_delta.m[i][j] = R_acc[i][j];
        T_delta.m[i][3] = t_acc[i];
    }
    out.T_delta = T_delta;
    Mat4 inv_delta;
    mat_inverse_rigid(T_delta, inv_delta);
    mat_mul(init_T, inv_delta, out.T);  // 位姿 = init_T @ inv(T_delta)
    // 重叠度:变换后点云在 ref 网格 0.3m 内的比例
    size_t in = 0;
    for (size_t k = 0; k < cur.size(); ++k) {
        const auto [idx, d2] = gh.nearest(cur[k]);
        (void)idx;
        if (d2 < 0.3 * 0.3) ++in;
    }
    out.overlap = (double)in / (double)cur.size();
    out.failed = out.overlap < 0.3;
    return out;
}

std::vector<Vec3> read_velodyne(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    const std::streamsize sz = f.tellg();
    f.seekg(0);
    std::vector<float> buf((size_t)(sz / 4));
    f.read(reinterpret_cast<char*>(buf.data()), sz);
    std::vector<Vec3> pts(buf.size() / 4);
    for (size_t i = 0; i < pts.size(); ++i)
        pts[i] = {buf[i * 4], buf[i * 4 + 1], buf[i * 4 + 2]};
    return pts;
}

}  // namespace


// 调试:导出逐点 PCA 协方差(6 分量)+ 特征值(退化诊断用)
int dump_cov(const char* bin, const char* out) {
    const std::vector<Vec3> pts = voxel_downsample(read_velodyne(bin));
    GridHash gh(pts, kGridCell);
    const auto& offs = grid_offsets();
    std::ofstream f(out);
    struct Cand { double d2; int idx; };
    std::vector<Cand> cand;
    char b[512];
    for (size_t i = 0; i < pts.size(); ++i) {
        const Vec3& q = pts[i];
        const Cell3 cq = cell_of(q, kGridCell);
        cand.clear();
        for (const auto& off : offs) {
            auto it = gh.buckets.find(Cell3{cq.x + off[0], cq.y + off[1], cq.z + off[2]});
            if (it == gh.buckets.end()) continue;
            for (int j : it->second) { const Vec3 d = pts[j] - q; cand.push_back({dot(d, d), j}); }
        }
        std::stable_sort(cand.begin(), cand.end(), [](const Cand& a, const Cand& c) { return a.d2 < c.d2; });
        const size_t k = std::min<size_t>(kIcpNormalK, cand.size());
        Vec3 mu{0, 0, 0};
        for (size_t j = 0; j < k; ++j) mu = mu + pts[cand[j].idx];
        mu = {mu.x / k, mu.y / k, mu.z / k};
        double cov[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
        for (size_t j = 0; j < k; ++j) {
            const Vec3 d = pts[cand[j].idx] - mu;
            cov[0][0] += d.x*d.x; cov[0][1] += d.x*d.y; cov[0][2] += d.x*d.z;
            cov[1][1] += d.y*d.y; cov[1][2] += d.y*d.z; cov[2][2] += d.z*d.z;
        }
        cov[1][0]=cov[0][1]; cov[2][0]=cov[0][2]; cov[2][1]=cov[1][2];
        for (int a2=0;a2<3;++a2) for (int b2=0;b2<3;++b2) cov[a2][b2] /= (double)k;
        double ev[3], evec[3][3];
        double cp[3][3];
        for (int a2=0;a2<3;++a2) for (int b2=0;b2<3;++b2) cp[a2][b2]=cov[a2][b2];
        eigh3(cp, ev, evec);
        std::snprintf(b, sizeof(b), "%zu %zu %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",
                      i, k, cov[0][0], cov[0][1], cov[0][2], cov[1][1], cov[1][2], cov[2][2],
                      ev[0], ev[1], ev[2], evec[0][0], evec[1][0], evec[2][0]);
        f << b;
    }
    return 0;
}


// 调试:单帧对 ICP 逐轮诊断(rmse / 每轮解)
int dump_icp(const char* dir, int a, int b) {
    char n1[512], n2[512];
    std::snprintf(n1, sizeof(n1), "%s/%06d.bin", dir, a);
    std::snprintf(n2, sizeof(n2), "%s/%06d.bin", dir, b);
    const std::vector<Vec3> src = voxel_downsample(read_velodyne(n1));
    const std::vector<Vec3> ref = voxel_downsample(read_velodyne(n2));
    const std::vector<Vec3> ref_n = estimate_normals(ref);
    GridHash gh(ref, kGridCell);
    // 打印零法向数 + 前若干法向(供与 Python 对齐)
    size_t zeros = 0;
    for (const auto& v : ref_n) if (dot(v, v) == 0.0) ++zeros;
    std::printf("ref %zu pts, zero-normals %zu\n", ref.size(), zeros);
    std::vector<Vec3> cur = src;
    for (int it = 0; it < kIcpMaxIter; ++it) {
        std::vector<Vec3> nb(cur.size()), nbn(cur.size());
        double se = 0.0;
        size_t valid = 0;
        for (size_t k = 0; k < cur.size(); ++k) {
            const auto [idx, d2] = gh.nearest(cur[k]);
            (void)d2;
            nb[k] = ref[(size_t)idx];
            nbn[k] = ref_n[(size_t)idx];
            if (dot(nbn[k], nbn[k]) > 0.0) { ++valid; se += dot(cur[k]-nb[k], nbn[k]) * dot(cur[k]-nb[k], nbn[k]); }
        }
        std::printf("it%d rmse_over_valid %.12f valid %zu/%zu\n", it, std::sqrt(se/(double)std::max<size_t>(valid,1)), valid, cur.size());
        double R[3][3], t[3];
        estimate_transform_gn(cur, nb, nbn, R, t);
        std::printf("   R[0]=(%.12f %.12f %.12f) t=(%.12f %.12f %.12f)\n", R[0][0], R[0][1], R[0][2], t[0], t[1], t[2]);
        for (size_t k = 0; k < cur.size(); ++k) {
            const Vec3 pp = cur[k];
            cur[k] = {R[0][0]*pp.x + R[0][1]*pp.y + R[0][2]*pp.z + t[0],
                      R[1][0]*pp.x + R[1][1]*pp.y + R[1][2]*pp.z + t[1],
                      R[2][0]*pp.x + R[2][1]*pp.y + R[2][2]*pp.z + t[2]};
        }
        const double delta = std::sqrt((R[0][0]-1)*(R[0][0]-1)+R[0][1]*R[0][1]+R[0][2]*R[0][2]+
            R[1][0]*R[1][0]+(R[1][1]-1)*(R[1][1]-1)+R[1][2]*R[1][2]+
            R[2][0]*R[2][0]+R[2][1]*R[2][1]+(R[2][2]-1)*(R[2][2]-1)) +
            std::sqrt(t[0]*t[0]+t[1]*t[1]+t[2]*t[2]);
        std::printf("   delta %.12e\n", delta);
        if (delta < kIcpTolDelta) { std::printf("converged it%d\n", it); break; }
    }
    return 0;
}

// 自检(--selftest):不读数据文件,用解析可验的小例子钉死各步语义。
// 供 `slam_diff_test.py --selftest` 在跑对拍前先确认移植版自身没坏(失败 = exit 1)。
// 覆盖:下采样(按原序累加 + 键升序)/ 网格最近邻(rad0 不许早停 + 等距先到者胜)/
//       法向(平面 + 退化置零)/ ICP 一步(纯平移可精确恢复)。
int selftest() {
    int n_fail = 0;
    auto check = [&](const char* name, bool ok) {
        std::printf("  %-34s %s\n", name, ok ? "PASS" : "FAIL");
        if (!ok) ++n_fail;
    };

    // --- 1) 下采样:同一体素内多点取重心,输出按体素键升序 ---
    {
        // 体素 = floor(v/0.5):(0.05,0.05,0.05) 与 (0.05,0.05,0.06) 同格 → 重心 z=0.055;
        // (0.55,0.55,0.55) 格 (1,1,1);(2.05,0.05,0.05) 格 (4,0,0)。
        std::vector<Vec3> pts = {{0.05, 0.05, 0.05}, {0.55, 0.55, 0.55}, {2.05, 0.05, 0.05}, {0.05, 0.05, 0.06}};
        const std::vector<Vec3> d = voxel_downsample(pts);
        bool ok = (d.size() == 3);
        // 键升序 → 格 (0,0,0) 在前:重心 (0.05, 0.05, 0.055)
        if (ok) ok = std::abs(d[0].x - 0.05) < 1e-12 && std::abs(d[0].z - 0.055) < 1e-12;
        if (ok) ok = std::abs(d[1].x - 0.55) < 1e-12 && std::abs(d[2].x - 2.05) < 1e-12;
        check("voxel_downsample(重心+键升序)", ok);
    }

    // --- 2) 网格最近邻:rad0 不许早停 ---
    // 查询 q=(0,0,0) 落在格 (0,0,0)。ref[0]=(0.24,0.24,0) 在**同格**内 d=0.339;
    // ref[1]=(0,0.26,0) 在格 (0,1,0) 即**层 1**,d=0.26 更近。
    // rad0 早停(best_d < 0)若被误用 → 返回 idx0 = 次优解。
    {
        std::vector<Vec3> ref = {{0.24, 0.24, 0.0}, {0.0, 0.26, 0.0}};
        GridHash gh(ref, 0.5);
        const auto [idx, d2] = gh.nearest({0.0, 0.0, 0.0});
        check("GridHash.nearest(rad0 不早停)", idx == 1 && std::abs(d2 - 0.26 * 0.26) < 1e-12);
    }

    // --- 3) 网格最近邻:等距先到者胜(层 0 = 自身格最先扫) ---
    {
        std::vector<Vec3> ref = {{0.1, 0.0, 0.0}, {-0.1, 0.0, 0.0}};  // 等距 d=0.1
        GridHash gh(ref, 0.5);
        const auto [idx, d2] = gh.nearest({0.0, 0.0, 0.0});
        (void)d2;
        // (0.1,0,0) 在格 (0,0,0) = 层 0,(−0.1,0,0) 在格 (−1,0,0) = 层 1;
        // 层 0 先扫 → 等距时先到者 = idx 0(与 numpy 实现同序,Python 侧同锚点已验)
        check("GridHash.nearest(等距先到者胜)", idx == 0);
    }

    // --- 4) 法向:平面 z=0 上稠密点 → 法向 ±z;退化(共线)邻域 → 精确零向量 ---
    {
        std::vector<Vec3> plane;
        for (int a = -3; a <= 3; ++a)
            for (int b = -3; b <= 3; ++b) plane.push_back({a * 0.1, b * 0.1, 0.0});
        const std::vector<Vec3> n = estimate_normals(plane);
        bool ok = true;
        for (size_t i = 0; i < n.size(); ++i) {
            if (std::abs(std::abs(n[i].z) - 1.0) > 1e-9) { ok = false; break; }
        }
        // 退化:两点共线 → λ2 = 0 → 门拒 → 零向量
        std::vector<Vec3> two = {{0.0, 0.0, 0.0}, {0.1, 0.0, 0.0}};
        const std::vector<Vec3> n2 = estimate_normals(two);
        for (size_t i = 0; i < n2.size(); ++i) {
            if (dot(n2[i], n2[i]) != 0.0) ok = false;
        }
        check("estimate_normals(平面±z/退化置零)", ok);
    }

    // --- 5) ICP 一步:纯平移 x+1.0 的三平面场景 → 恢复 Δt=(−1,0,0),ΔR=I ---
    // 点面残差只在法向有约束,三个互相垂直的平面(法向 x/y/z)才让平移可观。
    // src 只保留平移后仍在 ref 覆盖内的点(避免边界点误配)。
    {
        std::vector<Vec3> ref;
        for (int a = -8; a <= 8; ++a)
            for (int b = -8; b <= 8; ++b) {
                ref.push_back({a * 0.25, b * 0.25, 0.0});      // 地面:法向 z
                ref.push_back({-4.0, a * 0.25, b * 0.25});     // 墙 x=−4:法向 x
                ref.push_back({a * 0.25, -4.0, b * 0.25});     // 墙 y=−4:法向 y
            }
        std::vector<Vec3> src;
        for (const Vec3& p : ref) {
            const Vec3 s = {p.x + 1.0, p.y, p.z};
            if (s.x <= 2.0) src.push_back(s);  // 平移后仍在 ref 的 x 覆盖内
        }
        const std::vector<Vec3> rn = estimate_normals(ref);
        GridHash gh(ref, kGridCell);
        std::vector<Vec3> nb(src.size()), nbn(src.size());
        for (size_t k = 0; k < src.size(); ++k) {
            const auto [idx, d2] = gh.nearest(src[k]);
            (void)d2;
            nb[k] = ref[(size_t)idx];
            nbn[k] = rn[(size_t)idx];
        }
        double R[3][3], t[3];
        estimate_transform_gn(src, nb, nbn, R, t);
        // GN 解的是"把 src 移到 ref"的增量 → 真值 (R=I, t=(−1,0,0))。
        // 容差 1e-5 而非 1e-9:解的是**正则化**法方程 (AᵀA+λI)x = Aᵀb,
        // 每轴 ~289 个点 ⇒ 相对偏差 ≈ λ/289 = 3.5e-7(λ=1e-4),是口径本身不是 bug。
        bool ok = std::abs(t[0] + 1.0) < 1e-5 && std::abs(t[1]) < 1e-5 && std::abs(t[2]) < 1e-5;
        for (int i = 0; i < 3 && ok; ++i)
            for (int j = 0; j < 3 && ok; ++j) ok = std::abs(R[i][j] - (i == j ? 1.0 : 0.0)) < 1e-6;
        check("estimate_transform_gn(纯平移恢复)", ok);
    }

    std::printf("[selftest] %s(%d 失败)\n", n_fail ? "FAIL" : "ALL PASS", n_fail);
    return n_fail ? 1 : 0;
}

// 单次 ICP(逐帧对拍用):--icp-one <prev.bin> <cur.bin> <init 16 个数> <seed 16 个数>
// 打印结果 T 的 16 个数(行主序,%.17g)。用途:让 Python 侧链式跑完、把**同一** init/seed
// 交给 C++,从而把"ICP 本身是否位对齐"与"链式反馈是否放大"两件事分开判定。
int icp_one(int argc, char** argv) {
    if (argc < 36) {
        std::fprintf(stderr, "usage: %s --icp-one <prev.bin> <cur.bin> <init 16> <seed 16>\n", argv[0]);
        return 2;
    }
    const std::vector<Vec3> src = voxel_downsample(read_velodyne(argv[2]));
    const std::vector<Vec3> ref = voxel_downsample(read_velodyne(argv[3]));
    Mat4 init, seed;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            init.m[i][j] = std::atof(argv[4 + i * 4 + j]);
            seed.m[i][j] = std::atof(argv[20 + i * 4 + j]);
        }
    }
    const IcpResult r = icp_odometry(src, ref, init, seed, true);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) std::printf("%.17g%c", r.T.m[i][j], (i == 3 && j == 3) ? '\n' : ' ');
    std::fprintf(stderr, "iters %d conv %d overlap %.6f rmse %.17g failed %d\n", r.iters, (int)r.converged,
                 r.overlap, r.rmse_final, (int)r.failed);
    return 0;
}

// 逐帧 ICP 序列(--icp-seq):读文本输入(每帧一行 frame + 16 个 init + 16 个 seed),
// 对每帧做**单次** ICP(prev.bin, cur.bin, init, seed)并输出 T。
//
// **为什么需要它**:链式位姿 T_k = Δ_k·T_{k-1} 对 Δ 的舍入差是**指数放大**的
// (最近邻赋值是离散的:1e-16 的 seed 差会在少数点上翻格 → Δ 跳变 ~1e-5 →
// 下一帧 seed 带着这个差继续)。实测两条纯 Python 链仅把 Δ 的求逆从 LU 换成
// 刚体 Rᵀ,同样按 ~2.4×/帧 发散。故"位对齐"的正确判据是**同一 (prev,cur,init,seed)
// 下单次 ICP 的逐位一致**,而不是 150 帧链式末端的差 —— 后者测的是放大率不是移植正确性。
int icp_seq(const char* dir, const char* in_path, const char* out_path) {
    std::ifstream in(in_path);
    if (!in) { std::fprintf(stderr, "cannot read %s\n", in_path); return 1; }
    std::ofstream out(out_path);
    if (!out) { std::fprintf(stderr, "cannot write %s\n", out_path); return 1; }
    size_t n = 0;
    in >> n;
    std::vector<Vec3> prev_pts;
    int prev_fid = -1;
    for (size_t i = 0; i < n; ++i) {
        int fid = 0;
        Mat4 init, seed;
        in >> fid;
        for (int a = 0; a < 4; ++a)
            for (int b = 0; b < 4; ++b) in >> init.m[a][b];
        for (int a = 0; a < 4; ++a)
            for (int b = 0; b < 4; ++b) in >> seed.m[a][b];
        // prev 帧固定为 fid−1(与 bin/slam_odometry.py 的调用口径一致)
        if (prev_fid != fid - 1) {
            char nm[512];
            std::snprintf(nm, sizeof(nm), "%s/%06d.bin", dir, fid - 1);
            prev_pts = voxel_downsample(read_velodyne(nm));
            prev_fid = fid - 1;
        }
        char nm[512];
        std::snprintf(nm, sizeof(nm), "%s/%06d.bin", dir, fid);
        const std::vector<Vec3> cur = voxel_downsample(read_velodyne(nm));
        const IcpResult r = icp_odometry(prev_pts, cur, init, seed, true);
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%d", fid);
        out << buf;
        for (int a = 0; a < 4; ++a)
            for (int b = 0; b < 4; ++b) {
                std::snprintf(buf, sizeof(buf), " %.17g", r.T.m[a][b]);
                out << buf;
            }
        out << "\n";
        std::fprintf(stderr, "[icp-seq %zu/%zu] fid %d iters %d\n", i + 1, n, fid, r.iters);
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "--selftest") return selftest();
    if (argc >= 2 && std::string(argv[1]) == "--icp-one") return icp_one(argc, argv);
    if (argc >= 5 && std::string(argv[1]) == "--icp-seq") return icp_seq(argv[2], argv[3], argv[4]);
    if (argc >= 5 && std::string(argv[1]) == "--dump-icp") return dump_icp(argv[2], std::atoi(argv[3]), std::atoi(argv[4]));
    if (argc >= 4 && std::string(argv[1]) == "--dump-cov") return dump_cov(argv[2], argv[3]);
    // 调试子命令(对拍定位用):--dump-normals <bin 文件> <out txt>
    // 输出:每行 "i x y z nx ny nz"(下采样后的点 + 法向),供逐点比对。
    if (argc >= 4 && std::string(argv[1]) == "--dump-normals") {
        const std::vector<Vec3> down = voxel_downsample(read_velodyne(argv[2]));
        const std::vector<Vec3> nn = estimate_normals(down);
        std::ofstream f(argv[3]);
        char b[256];
        for (size_t i = 0; i < down.size(); ++i) {
            std::snprintf(b, sizeof(b), "%zu %.17g %.17g %.17g %.17g %.17g %.17g\n", i, down[i].x, down[i].y,
                          down[i].z, nn[i].x, nn[i].y, nn[i].z);
        (void)0;
            f << b;
        }
        std::fprintf(stderr, "[dump] %zu pts → %s\n", down.size(), argv[3]);
        return 0;
    }
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <velodyne_dir> <out_json> [start] [end]\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[1], out_path = argv[2];
    const int start = (argc > 3) ? std::atoi(argv[3]) : 0;
    const int end = (argc > 4) ? std::atoi(argv[4]) : start + 149;

    std::vector<Mat4> poses;
    std::vector<int> ids;
    Mat4 delta_prev;
    bool have_prev = false;
    std::vector<Vec3> prev_down;
    for (int fid = start; fid <= end; ++fid) {
        char name[512];
        std::snprintf(name, sizeof(name), "%s/%06d.bin", dir.c_str(), fid);
        const std::vector<Vec3> raw = read_velodyne(name);
        if (raw.empty()) { std::fprintf(stderr, "[skip] %s\n", name); continue; }
        const std::vector<Vec3> down = voxel_downsample(raw);
        Mat4 T;
        if (!have_prev) {
            have_prev = true;
        } else {
            // init_T = 上一帧链式位姿(**不乘 delta_prev**:seed 已在 ICP 内作为起点应用)
            // **src = 上一帧,ref = 当前帧**(与 bin/slam_odometry.py 的
            // icp_odometry(prev_down, down, ...) 同序;反过来是另一对点面残差,
            // 近恒等运动下看着差不多,实测差 2-3cm,超对拍阈值)。
            const IcpResult r = icp_odometry(prev_down, down, poses.back(), delta_prev, true);
            T = r.T;
            Mat4 inv, d;
            mat_inverse_rigid(poses.back(), inv);
            mat_mul(inv, T, d);
            delta_prev = d;
        }
        poses.push_back(T);
        ids.push_back(fid);
        prev_down = down;
        std::fprintf(stderr, "[%d] %zu pts\n", fid, down.size());
    }

    std::ofstream o(out_path);
    if (!o) { std::fprintf(stderr, "cannot write %s\n", out_path.c_str()); return 1; }
    o << "{\n  \"sensor\": \"velodyne\",\n  \"n_frames\": " << poses.size() << ",\n  \"traj\": [\n";
    char buf[1024];
    for (size_t k = 0; k < poses.size(); ++k) {
        o << "    {\"frame\": " << ids[k] << ", \"T\": [";
        for (int i = 0; i < 4; ++i) {
            o << "[";
            for (int j = 0; j < 4; ++j) {
                std::snprintf(buf, sizeof(buf), "%.17g", poses[k].m[i][j]);
                o << buf << (j == 3 ? "" : ", ");
            }
            o << (i == 3 ? "]" : "], ");
        }
        o << "]}" << (k + 1 == poses.size() ? "\n" : ",\n");
    }
    o << "  ]\n}\n";
    std::fprintf(stderr, "[done] %zu frames → %s\n", poses.size(), out_path.c_str());
    return 0;
}
