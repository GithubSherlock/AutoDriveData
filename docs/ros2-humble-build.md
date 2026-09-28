# ROS2 Humble 源码构建记录(AutoDL 机器,2026-09-28)

> **本文档的定位**:这台机器上「ROS2 到底能不能建起来、建起来要多贵」的**环境侧实测结论 + 可复现步骤**。
> 它**不属于 Plan2.md 的能力线** —— 项目主线(SLAM / 感知 / 建图)是**刻意纯 numpy、不吃 ROS** 的,
> 本文件回答的是「工具链可不可得」;**「值不值得挂上去」另见 [docs/ros2-feasibility.md](ros2-feasibility.md)**
> (结论:**不引入** —— 疼点已被同步执行绕过、教程能力线 16 次选择不用它、代价可量化而收益不可量化)。
> 与 [`Carla_Sim_Tutorial_01..16.md`](Carla_Sim_Tutorial_01.md) 旧 ros-bridge 栈无继承关系,是彻底另起的一条线。

---

## 0 结论先行

| 项 | 实测值 |
|---|---|
| 档1 首轮(rclcpp + rclpy + ros2cli + demo) | 136 个源码包,**7 min 48 s**(`--parallel-workers 8`,已有 21 包缓存) |
| 档1 补 CLI 兄弟包后 | +19 包 = **155 个,1 min 7 s**(增量,已建的跳过) |
| **验收(算数的是这一行)** | `ros2 pkg list` = **144**;`talker` / `listener` 对跑通 **`I heard: [Hello World: 14]`**,SIGINT 正常收尾 |
| 6 个包有 stderr 输出 | **全部无害**:2 条 LTO 警告、2 条 CMake 未用变量、2 条其实是**抓取成功**的记录 |
| 磁盘占用 | 档1 约 56 MB / 全量约 1.2 GB —— **磁盘不是决策变量** |
| ★ 首轮的坑 | **编译成功 ≠ 装好**:TOP 清单漏了 CLI 兄弟包 ⇒ `ros2 run` 不存在(§4) |

---

## 1 环境事实(决定一切的两条)

**(a) `github.com` 是 SNI 级不可达,不是"网络慢"。**

- `git clone https://github.com/...` 有时 TCP 握手会成功(0.23 s),但 TLS 随即死掉(`curl` 报 `code=000`),6 个不同 IP 表现一致
- **不是 DNS**(解析稳定到 `20.205.243.166`)、**不是体积**、改 `/etc/hosts` 换 IP 无效
- 但 `codeload.github.com` / `raw.githubusercontent.com` **通**,含 8 MB 量级的整包传输
- ⇒ 判据是**主机名**(SNI),不是大小。所以 `.../archive/xxx.tar.gz` 这类会 302 到 codeload 的路径**多半能过**,而 `git clone` **必然过不去**

**(b) git 默认没有低速超时,卡住是"静默"的。**

- 症状:传输 0 字节进展,但 `/proc/net/tcp` 里状态是 `01`(**ESTABLISHED**)、重传定时器已武装 —— 内核认为连接好好的
- `--retry N` **只在失败时触发**,对"挂着不动"无效 ⇒ 实测一次挂满 **13 min 43 s**
- 观测到 `130808 ms ≈ 130 s` 恰好是 Linux 默认 `tcp_syn_retries=6` 的退避总和(1+2+4+8+16+32+64)
- **诊断工具**:这台机器 **`ss` 没装**(`command not found`)。用 `/proc/net/tcp` 看状态 + `/proc/PID/io` 采样 `rchar` 判是否真在动

---

## 2 三档成本表(rosdep 直接依赖 → apt 实际新增,全部实测)

| 档 | colcon 源码包 | rosdep 直接依赖 | apt 新增 | 下载 | 装完占盘 |
|---|---|---|---|---|---|
| **档1** rclcpp + rclpy + ros2cli + demo | **136** | 15 | 24 | 13.6 MB | 56 MB |
| **档2** 档1 + tf2 / kdl / robot_state_publisher | **158** | 18 | 39 | 18.4 MB | 67 MB |
| **档3** 全量 104 仓库 | **349** | 61 | 206 | 236 MB | **1.2 GB** |

**三条反直觉的实测结论**:

1. **磁盘不是变量。** 全量也只要 1.2 GB(`/root/autodl-tmp` 有 50 G 空闲)。
   (中途曾把 `apt-cache show` 的 `Size`(字节)与 `Installed-Size`(KB)都按 /1024 读成 MB,误报过"数 GB",已纠。)
2. **大包是 LLVM / mypy,不是 Qt / OGRE / OpenCV。**
   `llvm-14-dev` 264.6 MB、`libllvm14` 104.4、`libclang-common-14-dev` 73.2、`python3-mypy` 58.5。
   **ogre / opencv / pcl / vtk = 0 个包**;Qt 33 个 122 MB。
3. **在 `--dependency-types` 上调优是白费力气。** 去掉 test 依赖只从 206 掉到 203 个包
   (只少 `python3-pygraphviz`、`python3-pytest-mock`、`python3-pytest-timeout`)。
   ⇒ **真正的成本是编译时间 = 源码包数(136 / 158 / 349)**,不是 apt 体积。

**档位之间不是互斥的**:colcon 是增量的。档1 建完再跑档2,只编多出来的 22 个包。所以正确姿势是**档1 → 验收 → 增量升档**。

---

## 3 可复现步骤(四步)

**准备:`~/ros2env.sh`** —— 它在 `~` 下、**不在项目里**,换机器要自建(踢掉 conda / direnv 痕迹):

```bash
#!/bin/bash
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PYTHON_EXE CONDA_SHLVL PYTHONPATH VIRTUAL_ENV
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
hash -r
py=$(command -v python3)
echo "python3 -> $py ($(python3 -V 2>&1))"
[ "$py" = /usr/bin/python3 ] && echo "✔ 系统 python,可以继续" || { echo "✘ 仍指向 conda —— 停,先查 PATH"; return 1 2>/dev/null || exit 1; }
```

它带**自证**:输出必须是 `python3 -> /usr/bin/python3 (Python 3.10.12)` + `✔ 系统 python,可以继续`。
**不看这两行就不算隔离成功**(§4 的 conda 污染条目)。

```bash
source ~/ros2env.sh
cd /root/autodl-tmp/ros2_humble
```

### ① 取源码(104 个仓库)

用 `vcs import --input ros2.repos src`,但**必须先给 git 装上低速超时**,否则单次挂 130 s:

```bash
git config --global http.lowSpeedLimit 1000
git config --global http.lowSpeedTime 60
```

`ros2.repos` 里的 `version:` 钉的是**分支/标签名**(`humble` / `2.6.x` / `v1.0.24` …),**不是 commit SHA**。
⇒ 校验是否偏离必须拿 HEAD 与 `refs/remotes/origin/<version>` 或 `refs/tags/<version>^{commit}` 比
,**不能拿 40 位 SHA 去比分支名**(那样会得出"104 个全偏离"的假象)。正确脚本报 **104/104**,103 分支钉 + 1 标签钉(`eProsima/Fast-CDR` @ `v1.0.24`)。

### ② 装系统依赖(★ 最容易跳过、后果最严重的一步)

`rosdep install` **没有 `--packages-up-to`** —— 档位收窄只能靠 `colcon list` 先算出目录再喂给它:

```bash
# ★ CLI 子命令包必须**逐个列出** —— ros2cli **不依赖**它们(同仓库兄弟包),
#   只写 ros2cli 的话第 ④ 步验收时 ros2 run / ros2 pkg 全都不存在(§4)
CLI="ros2run ros2pkg ros2topic ros2node ros2param ros2service ros2action \
     ros2launch ros2doctor ros2interface ros2lifecycle ros2component ros2multicast"
TOP="rclcpp rclpy ros2cli $CLI demo_nodes_cpp demo_nodes_py"
LEAN="--dependency-types build --dependency-types build_export --dependency-types buildtool \
      --dependency-types buildtool_export --dependency-types exec"
dirs=$(colcon list --packages-up-to $TOP | awk '{print $2}')

rosdep install --simulate $LEAN -y --rosdistro humble \
  --from-paths $dirs --ignore-src \
  --skip-keys "fastcdr rti-connext-dds-6.0.1 urdfdom_headers"
#   ↑ --dependency-types 必须**重复写**成多个 flag。逗号拼一行 / 空格拼一个值
#     都会因 optparse 的 type='choice' 静默返回 **0 个包**(不报错,最坑)
#   ↑ fastcdr 在 src/ 里自编;rti-connext-dds 是非自由 DDS,拿不到

# 确认清单无误后删掉 --simulate 真跑
# ★ 再额外手装一个(rosdep 看不见它,原因见 §5):
sudo apt install -y libfoonathan-memory-dev
```

**跳过这一步的代价 = 一个头文件一个头文件地撞**(见 §4 第一行)。

### ③ 构建

```bash
colcon build --symlink-install \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
  --packages-up-to $TOP --parallel-workers 8
```

- **必须放 tmux / `nohup`**:全量不止几十分钟。**注意这台机器 `tmux` 没装**
- `^Z` 是 **SIGTSTP 挂起,不是退出** —— 会继续占着 `build/`。重跑前先 `kill %2`
- `-DBUILD_TESTING=OFF` 顺带免掉 `mimick_vendor` / `uncrustify_vendor` / `google_benchmark_vendor` 的源码抓取

### ④ 验收(**不是看"编完了"**)

```bash
source install/setup.bash
ros2 pkg list | wc -l                                            # 应为数百,不是 0
(timeout 20 ros2 run demo_nodes_cpp talker &) ; sleep 5
timeout 10 ros2 run demo_nodes_py listener                       # 应打出 I heard: [Hello World: N]
```

---

## 4 ★ 踩坑清单(每条带判据)

| 症状 | 根因 | 判据 / 修法 |
|---|---|---|
| `iceoryx_hoofs`: `fatal error: sys/acl.h: No such file` | **第 ② 步整步没跑**。`sys/acl.h` 属 `libacl1-dev` | `rosdep install --simulate --from-paths src` 报**全量还缺 60 个包**,`libacl1-dev` 在其中。所以这不是"偶发缺一个包" |
| `ros2 run` → `invalid choice: 'run' (choose from 'daemon', 'extension_points', 'extensions')`;`ros2 pkg list` = 0 | **`ros2cli` 不依赖 `ros2run` / `ros2pkg`** —— 同仓库的**兄弟包**,不是依赖。`--packages-up-to ros2cli` 拿不到它们 | `ls install/ \| grep ^ros2` 只有 `ros2cli` 一个,其余 14 个(`ros2run` `ros2pkg` `ros2topic` `ros2node` `ros2param` `ros2service` `ros2action` `ros2launch` `ros2doctor` `ros2interface` `ros2lifecycle` `ros2component` `ros2multicast` `ros2bag`)**src 里有、install 里没有** |
| vendor 包编译时挂 130 s 后 `Connection timed out` | vendor 包是「**先找系统库、找不到才去 github 抓**」;系统库没装 ⇒ 全都去抓 | `log/latest_build/<pkg>/streams.log` 里出现 `Cloning into` |
| 6 个包 "had stderr output" 但显示 Finished | **不致命**:`lto-wrapper: warning: using serial compilation of N LTRANS jobs`(cyclonedds / rclpy)、`CMake Warning: Manually-specified variables were not used: BUILD_TESTING`(iceoryx_*)、以及 libyaml/mimick 的 `Cloning into` 成功记录 | 逐个读 `log/latest_build/<pkg>/stderr.log`;**exit code 0 + 产物存在 = 真过** |
| conda 的 python 3.11 混进构建 | 提示符带 `(autodrivedata)` 就说明没隔离;`rclpy` / `rosidl_generator_py` 会去解析 `python3` | `~/ros2env.sh` 会 unsets `CONDA_*` / `PYTHONPATH` 并把 PATH 钉回系统。验收:`python3 -V` = 3.10.12 且产物是 `cpython-310-*.so` |
| `install/` 里是**指向 `build/` 的软链** | `--symlink-install` 的固有形态 | **别删 `build/`** —— 删了 `install/` 就成了悬空链(实测 `install/rclpy/.../_rclpy_pybind11.cpython-310-*.so` → `build/rclpy/test_rclpy/...`) |
| **新开一个终端 → `bash: ros2: command not found`** | ROS2 在 `/root/autodl-tmp/ros2_humble/install/ros2cli/bin/ros2`,**不在 `/usr/bin`**。工作空间 overlay 是**每个 shell 各自 source** 的,不是装一次全局生效 | `command -v ros2` 为空即此因;`source install/setup.bash` 后必须指向 `.../install/ros2cli/bin/ros2`(实测 `AMENT_PREFIX_PATH` 144 条) |
| **`source ~/ros2env.sh` 反而把 `ros2` 弄没了** | 它 `export PATH=<仅系统路径>` 是**硬覆盖**。先 `setup.bash` 后 `ros2env.sh` ⇒ ros2 的路径被抹掉 | 顺序恒为 **`ros2env.sh` → `setup.bash`**。反了 `command -v ros2` 立刻变空 |

### 4.1 与项目环境共存:conda / direnv / ROS2 三者互斥

**现状(这台机器 `.bashrc` 的既有安排,不是 ROS2 带来的)**:

| 行 | 内容 | 效果 |
|---|---|---|
| `110` | `conda shell.bash hook` | conda 可用 |
| `124` | `conda activate autodrivedata` | **无条件** —— 每个新 shell 的默认就是项目环境 |
| `128` | `eval "$(direnv hook bash)"` | 进含 `.envrc` 的目录(项目)再激活一次 `autodrivedata`(direnv 2.25.2) |

**两者互斥,不是"叠加"**:

| | 项目主线要的 | ROS2 要的 |
|---|---|---|
| python | conda `autodrivedata` 的 **3.11.16** | **系统 `/usr/bin/python3.10`** |
| conda 变量 | 必须在 | `ros2env.sh` **专门 unset 掉** |
| PATH | conda bin 在最前 | 仅系统路径 + `install/` |

⇒ **「每个新终端自动 `source ros2ws.sh`」= 把默认环境从项目环境换成 ROS2 环境**。
项目里任何 `python -m autodrivedata.*` 会拿到系统 python ⇒ pycarla / ultralytics import 失败。
(且 `.bashrc` 改的是**全局默认**,为一条尚未判定去留的验证线付出这个代价不划算。)

**混合态比干净的任一种都糟**:进 `ros2_humble` 得 ROS2 环境,`cd` 回项目目录时 direnv 会**重新激活 conda**,
而 `ros2env.sh` 的 `unset` 它未必回滚 ⇒ 一个「`ros2` 能敲、`python3` 却是 conda 的」shell。

**实测**(一个全新登录 shell,落在 `ros2_humble` 里):

```
CONDA_DEFAULT_ENV=autodrivedata
python3 → /root/miniconda3/bin/python3 (Python 3.10.8)   ← 是 3.10.8,不是系统的 3.10.12
ros2    → 找不到
```

而 `rclpy` 是给 **`/usr/bin/python3.10`** 编的。当前 ROS2 终端能用,只因 `ros2env.sh` 挡着;
哪次忘了 source,就是上面这个混合态。

**推荐做法 —— 不做全局自动,用别名显式进出**:

```bash
echo "alias ros2ws='source ~/ros2ws.sh'" >> ~/.bashrc
```

判据:敲 `ros2ws` 后 `command -v python3` 必须是 `/usr/bin/python3.10`;`conda activate autodrivedata` 负责回到项目环境。
两个环境各自显式进出、永不自作主张。ROS2 若最终判定不用,**删一个 alias 就干净了**,不用去 `.bashrc` 里刨。

**若坚持自动:按目录绑定,别全局改默认** —— direnv 已装,正确形状是"进哪个目录激活哪个":

```bash
cat > /root/autodl-tmp/ros2_humble/.envrc <<'EOF'
export DIRENV_LOG_FORMAT=""
source /root/ros2ws.sh
EOF
cd /root/autodl-tmp/ros2_humble && direnv allow
```

第一行照抄项目 `.envrc` —— conda 激活会把 `unset`/`export` 逐行吐出来刷屏,那句专门压掉它。

★ **这条路有个必须先实测的假设**:`ros2env.sh` 是 `unset` + **硬覆盖 PATH**,而 direnv 只能回滚**它自己记录的那部分**
—— **离开目录能否恢复 conda,不能假设**:

```bash
cd /root/autodl-tmp/ros2_humble
echo "进: python3=$(command -v python3)  ros2=$(command -v ros2)"
cd /root
echo "出: python3=$(command -v python3)  ros2=$(command -v ros2 || echo 无)"
```

判据:出目录后 `python3` 必须回到 `/root/miniconda3/bin/python3`、`ros2` 必须消失。
**恢复不回来就说明这条路有坑,别用它** —— 那意味着你会带着一个半 ROS2 的 shell 到处走。

---

## 5 构建期抓取与镜像注入(已验证生效)

静态扫描 104 个仓库,构建期会去抓第三方源码的只集中在少数 vendor 包,机制分两类:

| 机制 | 包 | 免抓条件 |
|---|---|---|
| `GIT_REPOSITORY`(git clone → **走 github.com,必挂**) | `foonathan_memory_vendor` `libyaml_vendor` `mimick_vendor` `uncrustify_vendor` | 见下 |
| `URL .../archive/...`(302 → codeload,时通时不通) | `spdlog` `yaml-cpp` `console_bridge` `tinyxml` `ogre` `mcap` `assimp` … | 装系统库即免 |

**大部分 vendor 包"先找系统库"**,所以把系统库装齐就**同时**消掉了抓取 —— 这也是 §3 第 ② 步不能跳的第二个理由。那批替代品:

```
libspdlog-dev  libyaml-dev  libconsole-bridge-dev  pybind11-dev  libbenchmark-dev
libtinyxml-dev  libignition-cmake2-dev  libignition-math6-dev  liborocos-kdl-dev  libassimp-dev
```

其中 `libignition-{cmake2,math6}-dev` 正是 `gz_cmake2_vendor` / `gz_math6_vendor` 的系统替代 —— apt 里有,不用抓。

`foonathan_memory_vendor` 稍特殊:它的探测不是 `find_package()` 而是
`cmake --find-package -DNAME=foonathan_memory -DMODE=EXIST`(所以 `grep find_package` 扫不到,会误判成"无条件抓")。
**实测过 `libfoonathan-memory-dev`(0.7.1-4)的文件清单**,确认带
`/usr/lib/x86_64-linux-gnu/cmake/foonathan_memory/foonathan_memory-config.cmake` ⇒ 该探测返回成功 ⇒ **跳过抓取**。
**这是档1 唯一的硬卡点**(Fast-DDS 是 rclcpp 的硬依赖),而它**不在 rosdep 的 60 个包里**
(rosdep 的 key 是 src 里的 vendor 包,被 `--ignore-src` 跳过了)—— 必须手写进 apt 那一行。

### 注入方式(只在构建进程生效)

```bash
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=url.https://gh-proxy.com/https://github.com/.insteadOf
export GIT_CONFIG_VALUE_0=https://github.com/
```

**实测铁证**(真实构建的 `streams.log` 时间戳):

```
libyaml_vendor:  [2.089s] Cloning into 'libyaml-0.2.5'...   [4.812s] HEAD is now at 2c891fc
mimick_vendor:   [1.221s] Cloning into 'mimick-de11f837...'  [2.993s] HEAD is now at de11f83
iceoryx_posh:    Cloning into '.../deps/meta-cmake'...        (CMake ExternalProject 的 git 子进程,同样被覆盖)
```

直连同一个仓库会挂到 130 s 超时 ⇒ **2 秒完成 = 注入生效**。

- 该机制对 **git 子进程**有效,所以 CMake `ExternalProject` / `FetchContent` 里走 `git` 的也一并覆盖(`iceoryx_posh` 即是)
- **不要写进 `~/.gitconfig`**:那会让本仓库(`GithubSherlock/AutoDriveData`)的推拉也过第三方代理。env 方式只影响当前 shell 及其派生的构建进程,正合需要
- 另一条通道(CMake 的 tarball 下载)也验过:gh-proxy 取 spdlog 归档 **200 / 319010 B / 1.36 s**,取 ogre **200 / 128 MB / 11.8 s**;直连同一个返回 302 不跟进(0 字节)—— **github.com 是时通时不通**

---

## 6 未决

- **档2 / 档3 未建**(增量升档成本见 §2)
- ~~ROS2 对本项目的净收益评估~~ → **已完成**:[docs/ros2-feasibility.md](ros2-feasibility.md),结论 **不引入**,
  并给了五条触发重评条件(接真实车 / 出现外部消费方 / `live_studio` 再撞并发瓶颈 / 有漂移明显的真实数据 / 需独立第三方完成某能力)
- 本文件只记环境侧结论;**Plan.md 已冻结,Plan2.md 是能力线,两者都不放这类内容**
