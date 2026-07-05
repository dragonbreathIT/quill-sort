/* quill/_native_src/spectre_py.c — CPython glue for the bundled Spectre core.
 *
 * Spectre (spectre_sort.c) is a portable, multi-threaded MSD→LSD radix sort for
 * 32/64-bit signed/unsigned integers. This file wraps its four ascending
 * in-place entry points as a tiny C extension module, ``quill._spectre``,
 * mirroring the buffer-protocol contract quill._native/native.cpp uses:
 *
 *   spectre_{i64,u64,i32,u32}(buf)  ->  sort a writable, C-contiguous numpy
 *                                       buffer of the matching dtype IN PLACE,
 *                                       ascending. Returns None on success.
 *
 * Spectre is C (not C++), so it is compiled as its OWN extension rather than
 * folded into the C++ quill._native — a C source cannot share that module's
 * ``-std=c++17`` compile flags. Both ship inside the same binary wheels, so the
 * bundling story ("no extra install") is identical.
 *
 * The Quill dispatcher (SpectreBackend in _backends.py) guarantees eligibility
 * (1-D, integer dtype, contiguous, n below Spectre's 2^32 limit) before calling.
 * On any nonzero Spectre return code we raise a Python exception; dispatch_sort
 * catches it (BaseException) and falls back to np.sort — the never-lose contract.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>

/* Spectre's public functions are defined in spectre_sort.c, compiled into THIS
 * same module. Declare them plainly (no dllimport): referencing the in-module
 * definitions links directly on every toolchain, with no import thunk. */
extern int spectre_sort_i64(int64_t  *a, int64_t n);
extern int spectre_sort_u64(uint64_t *a, int64_t n);
extern int spectre_sort_i32(int32_t  *a, int64_t n);
extern int spectre_sort_u32(uint32_t *a, int64_t n);

/* SPECTRE_OK=0, SPECTRE_ENOMEM=1, SPECTRE_TOO_BIG=2 (see spectre.h). */
static int spectre__raise(int rc) {
    if (rc == 0) return 0;
    if (rc == 1)      PyErr_SetString(PyExc_MemoryError, "spectre: working allocation failed");
    else if (rc == 2) PyErr_SetString(PyExc_ValueError,  "spectre: n >= 2^32 is unsupported");
    else              PyErr_SetString(PyExc_RuntimeError, "spectre: sort failed");
    return -1;
}

static int spectre__get_buffer(PyObject *arg, Py_buffer *v, size_t elem) {
    if (PyObject_GetBuffer(arg, v, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS) < 0)
        return -1;
    if (v->itemsize != 0 && (size_t)v->itemsize != elem) {
        PyBuffer_Release(v);
        PyErr_Format(PyExc_TypeError,
                     "quill._spectre: expected %zu-byte elements, got %zd",
                     elem, (Py_ssize_t)v->itemsize);
        return -1;
    }
    if (v->len % (Py_ssize_t)elem != 0) {
        PyBuffer_Release(v);
        PyErr_SetString(PyExc_ValueError, "buffer length not a multiple of itemsize");
        return -1;
    }
    return 0;
}

/* One in-place sort entry point. The sort runs with the GIL released, since
 * Spectre spins up its own worker pool and never touches Python objects. */
#define SPECTRE_FN(NAME, T, CALL)                                         \
static PyObject* NAME(PyObject* self, PyObject* arg) {                     \
    (void)self;                                                           \
    Py_buffer v;                                                          \
    if (spectre__get_buffer(arg, &v, sizeof(T)) < 0) return NULL;         \
    int64_t n = (int64_t)(v.len / (Py_ssize_t)sizeof(T));                 \
    T* d = (T*)v.buf;                                                     \
    int rc;                                                               \
    Py_BEGIN_ALLOW_THREADS                                                \
    rc = CALL(d, n);                                                      \
    Py_END_ALLOW_THREADS                                                  \
    PyBuffer_Release(&v);                                                 \
    if (spectre__raise(rc) < 0) return NULL;                             \
    Py_RETURN_NONE;                                                       \
}

SPECTRE_FN(spectre_i64, int64_t,  spectre_sort_i64)
SPECTRE_FN(spectre_u64, uint64_t, spectre_sort_u64)
SPECTRE_FN(spectre_i32, int32_t,  spectre_sort_i32)
SPECTRE_FN(spectre_u32, uint32_t, spectre_sort_u32)

static PyMethodDef Methods[] = {
    {"spectre_i64", spectre_i64, METH_O, "in-place ascending int64 sort"},
    {"spectre_u64", spectre_u64, METH_O, "in-place ascending uint64 sort"},
    {"spectre_i32", spectre_i32, METH_O, "in-place ascending int32 sort"},
    {"spectre_u32", spectre_u32, METH_O, "in-place ascending uint32 sort"},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "_spectre",
    "Spectre: bundled parallel integer radix sort (32/64-bit).", -1, Methods,
    NULL, NULL, NULL, NULL,
};

PyMODINIT_FUNC PyInit__spectre(void) { return PyModule_Create(&moduledef); }
