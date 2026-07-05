// quill_fastsort._impl — multi-threaded MSD-radix sort backend.
// Exposes the exact API quill/_backends.py::RustBackend probes:
//   sort_{i64,f64}(ndarray)  — sorts in place, ascending.
// Portable C++17 (macOS / Linux / Windows). Kernel: quillcore::parallel_radix.
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
                     "quill_fastsort: expected %zu-byte elements, got %zd",
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

static PyObject* sort_i64(PyObject*, PyObject* a) { return run<int64_t>(a); }
static PyObject* sort_f64(PyObject*, PyObject* a) { return run<double>(a); }

static PyMethodDef Methods[] = {
    {"sort_i64", sort_i64, METH_O, "in-place ascending sort, int64"},
    {"sort_f64", sort_f64, METH_O, "in-place ascending sort, float64"},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "_impl",
    "Quill multi-threaded MSD-radix sort backend (portable C++).", -1, Methods,
    NULL, NULL, NULL, NULL,
};

PyMODINIT_FUNC PyInit__impl(void) { return PyModule_Create(&moduledef); }
