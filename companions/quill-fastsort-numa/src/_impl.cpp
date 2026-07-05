// quill_fastsort_numa._impl — NUMA-aware parallel radix sort backend.
// Exposes the exact API quill/_backends.py::NumaBackend probes:
//   numa_sort_{i64,u64,i32,u32,f64,f32}(ndarray)  — sorts in place, ascending.
// Portable C++17 (macOS / Linux / Windows). Kernel: quillcore::parallel_radix
// (NUMA-aware parallel radix; falls back to plain parallel radix on single-socket).
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "quill_core.hpp"

using namespace quillcore;

// Sort a writable, C-contiguous buffer of element type T in place. The Quill
// dispatcher guarantees contiguity, writeability and NaN-free float data before
// calling, so we only defend against a mismatched itemsize (wrong entry point).
template <class T>
static PyObject* run(PyObject* arg) {
    Py_buffer v;
    if (PyObject_GetBuffer(arg, &v, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS) < 0)
        return NULL;
    if (v.itemsize != 0 && (size_t)v.itemsize != sizeof(T)) {
        PyBuffer_Release(&v);
        PyErr_Format(PyExc_TypeError,
                     "quill_fastsort_numa: expected %zu-byte elements, got %zd",
                     sizeof(T), (Py_ssize_t)v.itemsize);
        return NULL;
    }
    if (v.len % (Py_ssize_t)sizeof(T) != 0) {
        PyBuffer_Release(&v);
        PyErr_SetString(PyExc_ValueError, "buffer length not a multiple of itemsize");
        return NULL;
    }
    size_t n = (size_t)v.len / sizeof(T);
    T* data = (T*)v.buf;
    Py_BEGIN_ALLOW_THREADS
    parallel_radix<T>(data, n, 0);
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&v);
    Py_RETURN_NONE;
}

static PyObject* numa_sort_i64(PyObject*, PyObject* a) { return run<int64_t>(a); }
static PyObject* numa_sort_u64(PyObject*, PyObject* a) { return run<uint64_t>(a); }
static PyObject* numa_sort_i32(PyObject*, PyObject* a) { return run<int32_t>(a); }
static PyObject* numa_sort_u32(PyObject*, PyObject* a) { return run<uint32_t>(a); }
static PyObject* numa_sort_f64(PyObject*, PyObject* a) { return run<double>(a); }
static PyObject* numa_sort_f32(PyObject*, PyObject* a) { return run<float>(a); }

#if defined(__linux__)
#include <sys/stat.h>
#include <unistd.h>
#endif
static PyObject* detect_topology(PyObject*, PyObject*) {
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
    Py_RETURN_NONE;  // no NUMA topology off Linux -> single-socket semantics
#endif
}

static PyMethodDef Methods[] = {
    {"numa_sort_i64", numa_sort_i64, METH_O, "in-place ascending sort, int64"},
    {"numa_sort_u64", numa_sort_u64, METH_O, "in-place ascending sort, uint64"},
    {"numa_sort_i32", numa_sort_i32, METH_O, "in-place ascending sort, int32"},
    {"numa_sort_u32", numa_sort_u32, METH_O, "in-place ascending sort, uint32"},
    {"numa_sort_f64", numa_sort_f64, METH_O, "in-place ascending sort, float64"},
    {"numa_sort_f32", numa_sort_f32, METH_O, "in-place ascending sort, float32"},
    {"detect_topology", detect_topology, METH_NOARGS, "return (nodes, cores) on multi-socket Linux, else None"},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "_impl",
    "Quill NUMA-aware parallel radix sort backend (portable C++).", -1, Methods,
    NULL, NULL, NULL, NULL,
};

PyMODINIT_FUNC PyInit__impl(void) { return PyModule_Create(&moduledef); }
