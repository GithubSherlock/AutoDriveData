"""Vulkan 设备枚举探针——CARLA 渲染停摆的一线判据(§5.11f 根因④)。

为什么需要:UE4 渲染线程起不来时的两种表症高度相似(显存恒 0 MiB + RPC 端口能通),
Vulkan/驱动层的问题在 CARLA 日志里**不报错**(只留 `GameThread timed out waiting for
RenderThread` 这种下游症状)。这里直接问加载器"有几个 Vulkan 设备",把驱动层与
UE4 层切开。缺 `libnvidia-gpucomp.so.<驱动版本>` 时本脚本输出 `count=0`。

用法(项目 env 即可,只需 ctypes):
    python -m autodrivedata.sim.probe_vulkan                     # 默认 ICD 全扫
    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json python -m autodrivedata.sim.probe_vulkan
                                                   # lavapipe 软件 ICD 对照(验证探针本身)
    LD_LIBRARY_PATH=<compat> python -m autodrivedata.sim.probe_vulkan   # 带 GPUCOMP 兼容层(carla_server.sh 同款)

退出码:0 = 枚举到设备;2 = 0 个设备(驱动层问题);1 = 建实例失败。
实现注:`VkPhysicalDeviceProperties` 真实尺寸 ~824 B,给 1024 B 缓冲;**探针函数签名要
显式 `c_void_p`**——`vkEnumeratePhysicalDevices` 出的句柄是 Python int,无 argtypes 时
ctypes 按 32 位传 → 指针被截断 → 段错误(曾据此误判"驱动崩了")。
"""

from __future__ import annotations

import ctypes
import sys

PROPS_OFF_NAME = 20  # apiVersion/driverVersion/vendorID/deviceID/deviceType 各 4 B
PROPS_NAME_LEN = 256


class _AppInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("pApplicationName", ctypes.c_char_p),
        ("applicationVersion", ctypes.c_uint32),
        ("pEngineName", ctypes.c_char_p),
        ("engineVersion", ctypes.c_uint32),
        ("apiVersion", ctypes.c_uint32),
    ]


class _InstanceCI(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("pApplicationInfo", ctypes.POINTER(_AppInfo)),
        ("enabledLayerCount", ctypes.c_uint32),
        ("ppEnabledLayerNames", ctypes.c_void_p),
        ("enabledExtensionCount", ctypes.c_uint32),
        ("ppEnabledExtensionNames", ctypes.c_void_p),
    ]


def _u32(buf: ctypes.Array, off: int) -> int:
    return (ctypes.c_uint32 * 1).from_buffer(buf, off)[0]  # type: ignore[no-any-return]


def main() -> int:
    lib = ctypes.CDLL("libvulkan.so.1")
    app = _AppInfo(0, None, b"probe_vulkan", 1, None, 1, (1 << 22) | (3 << 12))  # Vulkan 1.3
    ci = _InstanceCI(0, None, 0, ctypes.pointer(app), 0, None, 0, None)
    inst = ctypes.c_void_p()
    rc = lib.vkCreateInstance(ctypes.byref(ci), None, ctypes.byref(inst))
    print(f"vkCreateInstance -> {rc}")
    if rc != 0:
        return 1

    n = ctypes.c_uint32()
    rc = lib.vkEnumeratePhysicalDevices(ctypes.c_void_p(inst.value), ctypes.byref(n), None)
    print(f"vkEnumeratePhysicalDevices -> {rc}, count={n.value}")
    if n.value == 0:
        print("NO VULKAN DEVICE(驱动层:查 libnvidia-gpucomp / VK_LOADER_DEBUG=all)")
        return 2

    devs = (ctypes.c_void_p * n.value)()
    lib.vkEnumeratePhysicalDevices(ctypes.c_void_p(inst.value), ctypes.byref(n), devs)
    buf = (ctypes.c_ubyte * 1024)()
    for i, d in enumerate(devs):
        ctypes.memset(buf, 0, len(buf))
        lib.vkGetPhysicalDeviceProperties(ctypes.c_void_p(d), ctypes.byref(buf))
        api, vid, dev, dtype = _u32(buf, 0), _u32(buf, 8), _u32(buf, 12), _u32(buf, 16)
        name = bytes(buf[PROPS_OFF_NAME : PROPS_OFF_NAME + PROPS_NAME_LEN]).split(b"\x00")[0]
        print(
            f"[{i}] {name.decode(errors='replace')} api={api >> 22}.{(api >> 12) & 0x3FF}.{api & 0xFFF}"
            f" vendor={hex(vid)} dev={hex(dev)} type={dtype}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
