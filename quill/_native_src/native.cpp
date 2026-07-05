// quill/_native/native.cpp — the in-tree compiled CPU sort backends.
//
// This single extension module (``quill._native``) folds in the kernels that
// used to live in five separate companion packages (quill-fastsort,
// -parallel, -ips4o, -numa, -simd). They all shared one header-only core
// (quill_core.hpp), so there was never a reason for five wheels — this bundles
// them so `pip install quill-sort` ships the fast path with no extra installs.
//
// Exposed functions (each sorts a writable, C-contiguous numpy buffer of the
// matching dtype IN PLACE, ascending):
//   radix_{i64,u64,i32,u32,f64,f32}   -> parallel MSD radix
//                                        (rust_voracious / rust_parallel_radix / numa)
//   sample_{i64,u64,i32,u32,f64,f32}  -> parallel comparison samplesort (ips4o)
//   serial_{i64,u64,i32,u32,f64,f32}  -> single-threaded radix (simd_companion)
//   numa_topology()                   -> (nodes, cores) on multi-socket Linux, else None
//
// The Quill dispatcher guarantees contiguity/writeability and strips NaN before
// calling, so the kernels assume those invariants. Portable C++17 (macOS/Linux/
// Windows); the only platform-specific code is numa_topology()'s Linux sysfs probe.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cstdio>          // snprintf (numa_topology)
#include "quill_core.hpp"

#if defined(__linux__)
#include <sys/stat.h>
#include <unistd.h>
#endif

using namespace quillcore;

// Acquire a writable, C-contiguous buffer and validate the element size.
static bool get_buffer(PyObject* arg, Py_buffer* v, size_t elem) {
    if (PyObject_GetBuffer(arg, v, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS) < 0)
        return false;
    if (v->itemsize != 0 && (size_t)v->itemsize != elem) {
        PyBuffer_Release(v);
        PyErr_Format(PyExc_TypeError,
                     "quill._native: expected %zu-byte elements, got %zd",
                     elem, (Py_ssize_t)v->itemsize);
        return false;
    }
    if (v->len % (Py_ssize_t)elem != 0) {
        PyBuffer_Release(v);
        PyErr_SetString(PyExc_ValueError, "buffer length not a multiple of itemsize");
        return false;
    }
    return true;
}

// One in-place sort entry point. CALL runs with the GIL released.
#define SORT_FN(NAME, T, CALL)                                            \
static PyObject* NAME(PyObject*, PyObject* arg) {                          \
    Py_buffer v;                                                          \
    if (!get_buffer(arg, &v, sizeof(T))) return NULL;                     \
    size_t n = (size_t)v.len / sizeof(T);                                 \
    T* d = (T*)v.buf;                                                     \
    Py_BEGIN_ALLOW_THREADS                                                \
    CALL;                                                                 \
    Py_END_ALLOW_THREADS                                                  \
    PyBuffer_Release(&v);                                                 \
    Py_RETURN_NONE;                                                       \
}

// parallel MSD radix
SORT_FN(radix_i64, int64_t,  parallel_radix<int64_t>(d, n, 0))
SORT_FN(radix_u64, uint64_t, parallel_radix<uint64_t>(d, n, 0))
SORT_FN(radix_i32, int32_t,  parallel_radix<int32_t>(d, n, 0))
SORT_FN(radix_u32, uint32_t, parallel_radix<uint32_t>(d, n, 0))
SORT_FN(radix_f64, double,   parallel_radix<double>(d, n, 0))
SORT_FN(radix_f32, float,    parallel_radix<float>(d, n, 0))

// parallel comparison samplesort (ips4o)
SORT_FN(sample_i64, int64_t,  parallel_samplesort<int64_t>(d, n, 0))
SORT_FN(sample_u64, uint64_t, parallel_samplesort<uint64_t>(d, n, 0))
SORT_FN(sample_i32, int32_t,  parallel_samplesort<int32_t>(d, n, 0))
SORT_FN(sample_u32, uint32_t, parallel_samplesort<uint32_t>(d, n, 0))
SORT_FN(sample_f64, double,   parallel_samplesort<double>(d, n, 0))
SORT_FN(sample_f32, float,    parallel_samplesort<float>(d, n, 0))

// single-threaded radix (simd companion)
SORT_FN(serial_i64, int64_t,  serial_sort<int64_t>(d, n))
SORT_FN(serial_u64, uint64_t, serial_sort<uint64_t>(d, n))
SORT_FN(serial_i32, int32_t,  serial_sort<int32_t>(d, n))
SORT_FN(serial_u32, uint32_t, serial_sort<uint32_t>(d, n))
SORT_FN(serial_f64, double,   serial_sort<double>(d, n))
SORT_FN(serial_f32, float,    serial_sort<float>(d, n))

// NUMA topology: (nodes, cores) on multi-socket Linux, else None. quill's
// NumaBackend engages only when this yields nodes >= 2.
static PyObject* numa_topology(PyObject*, PyObject*) {
#if defined(__linux__)
    int nodes = 0;
    for (int i = 0; ; ++i) {
        char path[128];
        snprintf(path, sizeof(path), "/sys/devices/system/node/node%d", i);
        struct stat st;
        if (stat(path, &st) == 0 && S_ISDIR(st.st_mode)) nodes++;
        else break;
    }
    if (nodes < 1) Py_RETURN_NONE;
    long cores = sysconf(_SC_NPROCESSORS_ONLN);
    if (cores < 1) cores = 1;
    return Py_BuildValue("(ii)", nodes, (int)cores);
#else
    Py_RETURN_NONE;
#endif
}

#define ENTRY(NAME) {#NAME, NAME, METH_O, "in-place ascending sort"}
static PyMethodDef Methods[] = {
    ENTRY(radix_i64),  ENTRY(radix_u64),  ENTRY(radix_i32),
    ENTRY(radix_u32),  ENTRY(radix_f64),  ENTRY(radix_f32),
    ENTRY(sample_i64), ENTRY(sample_u64), ENTRY(sample_i32),
    ENTRY(sample_u32), ENTRY(sample_f64), ENTRY(sample_f32),
    ENTRY(serial_i64), ENTRY(serial_u64), ENTRY(serial_i32),
    ENTRY(serial_u32), ENTRY(serial_f64), ENTRY(serial_f32),
    {"numa_topology", numa_topology, METH_NOARGS,
     "return (nodes, cores) on multi-socket Linux, else None"},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "_native",
    "Quill bundled CPU sort backends (portable C++).", -1, Methods,
    NULL, NULL, NULL, NULL,
};

PyMODINIT_FUNC PyInit__native(void) { return PyModule_Create(&moduledef); }
