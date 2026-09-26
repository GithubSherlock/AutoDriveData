/* NVIDIA glcore 在裁剪环境的缺失符号补齐(glibc 2.34+ malloc hooks + Xlib ErrorF)。
   实现与 glibc/Xlib 一致:shim 先加载,符号进全局表供 dlopen 的 glcore 解析。 */
#include <stddef.h>
#include <stdarg.h>
#include <stdio.h>
void *__malloc_hook = NULL;
void *__realloc_hook = NULL;
void *__free_hook = NULL;
void *__memalign_hook = NULL;
void ErrorF(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
}
