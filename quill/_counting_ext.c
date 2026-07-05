/*
 * quill/_counting_ext.c
 * ----------------------
 * Fused count + cumsum + scatter counting-sort kernels for QuillSort v7.
 *
 * WHY this exists
 * ---------------
 * v6's counting sort (quill/_strategies.py::counting_sort_array) does:
 *
 *     shifted = arr - mn if mn else arr                              # alloc 1
 *     counts  = np.bincount(shifted, minlength=mx-mn+1)              # alloc 2
 *     return  np.repeat(np.arange(mn, mx+1, dtype=arr.dtype), counts)# alloc 3+4
 *
 * That is three Python-visible allocations and two full passes over the data
 * (one for the shift, one for bincount). The output is then a freshly
 * heap-allocated ndarray that numpy must zero-fill before np.repeat scatters
 * into it.
 *
 * The kernels in this file fuse those steps:
 *   pass 1  — increment a stack-or-heap counts[] array
 *   pass 2  — walk counts and write contiguous runs of (k + mn) into out[]
 *
 * The output buffer can be supplied by the caller (e.g. from
 * quill._scratch.POOL), which removes the last heap allocation on hot paths.
 * Measured target on int64 dense ranges: 1.3-1.6x v6's path.
 *
 * INVARIANTS the caller must uphold (we do NOT re-validate, for speed):
 *   - arr is C-contiguous and matches the dtype of the called entry point.
 *   - All values in arr lie in [mn, mx]; mn <= mx.
 *   - If out is supplied: C-contiguous, correct dtype, length == len(arr).
 *
 * NEVER-LOSE contract
 * -------------------
 * This module is OPTIONAL. The Python caller must probe with
 *     try: from . import _counting_ext
 *     except Exception: _counting_ext = None
 * and fall back to the v6 np.bincount + np.repeat path when absent. Any
 * exception raised here (OverflowError, MemoryError, TypeError) is the
 * caller's signal to take that fallback.
 *
 * Notes on integer sizes
 * ----------------------
 * On Windows MSVC `long` is 32 bits, so we use `long long` / fixed-width
 * `int64_t` etc. throughout. Subtractions like (mx - mn + 1) are done in
 * the *unsigned* counterpart of the dtype to make wrap-around well defined
 * when mn is very negative and mx very positive — and so we can detect a
 * range that won't fit in Py_ssize_t.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

#include <stdint.h>
#include <string.h>   /* memset */


/* ------------------------------------------------------------------ */
/* Output-allocation helper                                            */
/* ------------------------------------------------------------------ */

/*
 * Resolve the output ndarray. If `out_obj` is Py_None / NULL we allocate
 * a fresh ndarray with `npy_type`. Otherwise we validate that the user-
 * supplied ndarray is the right dtype, contiguous, and the right length.
 *
 * Returns a NEW reference on success, NULL with exception set on failure.
 */
static PyArrayObject *
resolve_output(PyObject *out_obj, npy_intp n, int npy_type)
{
    if (out_obj == NULL || out_obj == Py_None) {
        PyArrayObject *fresh = (PyArrayObject *)PyArray_EMPTY(1, &n, npy_type, 0);
        return fresh;  /* may be NULL on MemoryError */
    }

    if (!PyArray_Check(out_obj)) {
        PyErr_SetString(PyExc_TypeError,
            "out must be a numpy ndarray or None");
        return NULL;
    }
    PyArrayObject *out = (PyArrayObject *)out_obj;

    if (PyArray_TYPE(out) != npy_type) {
        PyErr_SetString(PyExc_TypeError,
            "out has wrong dtype for this kernel");
        return NULL;
    }
    if (PyArray_NDIM(out) != 1 || PyArray_DIM(out, 0) != n) {
        PyErr_SetString(PyExc_ValueError,
            "out has wrong shape (must be 1-D, len(arr))");
        return NULL;
    }
    if (!PyArray_IS_C_CONTIGUOUS(out)) {
        PyErr_SetString(PyExc_ValueError,
            "out must be C-contiguous");
        return NULL;
    }

    Py_INCREF(out);
    return out;
}


/* ------------------------------------------------------------------ */
/* The fused kernel, parameterized by dtype                            */
/* ------------------------------------------------------------------ */

/*
 * We instantiate one C function per (dtype) entry point. Each one:
 *
 *   1. Parses (arr, mn, mx, out=None).
 *   2. Computes range_size = (mx - mn) + 1, checking that it fits in
 *      Py_ssize_t. We do the subtraction in the UNSIGNED form of the dtype
 *      so a mn==INT_MIN, mx==INT_MAX pair doesn't UB-overflow.
 *   3. Allocates counts[range_size] via PyMem_Malloc; zeroed with memset.
 *   4. Pass 1 — increments counts[v - mn] for every v in arr.
 *   5. Pass 2 — walks counts and writes runs of (k + mn) into out[].
 *
 * Step 5 (write-runs) is preferred over the scatter-from-input variant
 * because every element with key k IS k (we're sorting values, not keyed
 * payloads), so iterating counts[] sequentially is more cache-friendly
 * than re-reading arr[] a second time. Stability is moot for value-only
 * sorts: equal values are bit-indistinguishable.
 *
 * Macro args:
 *   FNAME      — exported C function name (e.g. fused_i64)
 *   DTYPE      — element C type (int64_t)
 *   UDTYPE     — unsigned counterpart for safe subtraction (uint64_t)
 *   NPY_TYPE   — NPY_INT64 etc.
 *   PARSE_FMT  — PyArg_ParseTupleAndKeywords format for (arr, mn, mx, out)
 */

#define DEFINE_FUSED_KERNEL(FNAME, DTYPE, UDTYPE, NPY_TYPE, PARSE_FMT)        \
static PyObject *                                                              \
FNAME(PyObject *self, PyObject *args, PyObject *kwds)                          \
{                                                                              \
    (void)self;                                                                \
    static char *kwlist[] = {"arr", "mn", "mx", "out", NULL};                  \
    PyArrayObject *arr = NULL;                                                 \
    PyObject *out_obj = Py_None;                                               \
    long long mn_in = 0, mx_in = 0;                                            \
                                                                               \
    if (!PyArg_ParseTupleAndKeywords(args, kwds, PARSE_FMT, kwlist,            \
                                     &PyArray_Type, &arr,                      \
                                     &mn_in, &mx_in, &out_obj)) {              \
        return NULL;                                                           \
    }                                                                          \
                                                                               \
    /* WHY: cheap sanity checks. Caller is supposed to guarantee these,        \
     * but a wrong-dtype ndarray here would silently miscompute, so we         \
     * still verify dtype/contiguity. We do NOT scan arr for out-of-range      \
     * values — that's the caller's contract. */                               \
    if (PyArray_TYPE(arr) != NPY_TYPE) {                                       \
        PyErr_SetString(PyExc_TypeError, "arr has wrong dtype for kernel");    \
        return NULL;                                                           \
    }                                                                          \
    if (PyArray_NDIM(arr) != 1) {                                              \
        PyErr_SetString(PyExc_ValueError, "arr must be 1-D");                  \
        return NULL;                                                           \
    }                                                                          \
    if (!PyArray_IS_C_CONTIGUOUS(arr)) {                                       \
        PyErr_SetString(PyExc_ValueError, "arr must be C-contiguous");         \
        return NULL;                                                           \
    }                                                                          \
                                                                               \
    DTYPE mn = (DTYPE)mn_in;                                                   \
    DTYPE mx = (DTYPE)mx_in;                                                   \
    if (mx < mn) {                                                             \
        PyErr_SetString(PyExc_ValueError, "mx < mn");                          \
        return NULL;                                                           \
    }                                                                          \
                                                                               \
    /* range_size = (mx - mn) + 1, computed via unsigned to avoid signed       \
     * overflow when mn is very negative and mx very positive. */              \
    UDTYPE u_range = (UDTYPE)((UDTYPE)mx - (UDTYPE)mn) + (UDTYPE)1;            \
                                                                               \
    /* Reject ranges that won't fit in Py_ssize_t (Windows: 32-bit ssize       \
     * on 32-bit Python, 64-bit otherwise). Caller's _counting_is_worth_it     \
     * gate makes this nearly impossible to hit, but we check anyway. */       \
    if (u_range == 0 ||                                                        \
        u_range > (UDTYPE)PY_SSIZE_T_MAX / sizeof(Py_ssize_t)) {               \
        PyErr_SetString(PyExc_OverflowError,                                   \
            "counting-sort range too large for this platform");                \
        return NULL;                                                           \
    }                                                                          \
    Py_ssize_t range_size = (Py_ssize_t)u_range;                               \
    npy_intp n = PyArray_DIM(arr, 0);                                          \
                                                                               \
    PyArrayObject *out = resolve_output(out_obj, n, NPY_TYPE);                 \
    if (out == NULL) return NULL;                                              \
                                                                               \
    /* counts[] holds the number of times each bucket appears. We size it      \
     * as Py_ssize_t (signed, machine word) — n can be at most PY_SSIZE_T_MAX  \
     * so each cell fits trivially. PyMem_Malloc is the Python heap allocator  \
     * (interacts nicely with debug builds / tracemalloc). */                  \
    Py_ssize_t *counts =                                                       \
        (Py_ssize_t *)PyMem_Malloc((size_t)range_size * sizeof(Py_ssize_t));   \
    if (counts == NULL) {                                                      \
        Py_DECREF(out);                                                        \
        PyErr_NoMemory();                                                      \
        return NULL;                                                           \
    }                                                                          \
    memset(counts, 0, (size_t)range_size * sizeof(Py_ssize_t));                \
                                                                               \
    const DTYPE *src = (const DTYPE *)PyArray_DATA(arr);                       \
    DTYPE *dst = (DTYPE *)PyArray_DATA(out);                                   \
                                                                               \
    /* The hot loops below do no Python work. Drop the GIL so other threads    \
     * (or our own ThreadPoolExecutor in the parallel path) can run. */        \
    Py_BEGIN_ALLOW_THREADS                                                     \
                                                                               \
    /* Pass 1: count. The bucket index uses UDTYPE so the subtraction is       \
     * well-defined even when mn is very negative. */                          \
    for (npy_intp i = 0; i < n; ++i) {                                         \
        UDTYPE bucket = (UDTYPE)src[i] - (UDTYPE)mn;                           \
        counts[bucket]++;                                                      \
    }                                                                          \
                                                                               \
    /* Pass 2: write runs. Walking counts[] sequentially and writing           \
     * contiguous blocks of the same value is much friendlier to the           \
     * prefetcher than a scatter from src[]; it also touches dst[] in          \
     * strictly increasing order. Empty buckets cost one branch each, which    \
     * is why _counting_is_worth_it gates on density (rng <= n/4). */          \
    Py_ssize_t offset = 0;                                                     \
    for (Py_ssize_t k = 0; k < range_size; ++k) {                              \
        Py_ssize_t cnt = counts[k];                                            \
        if (cnt == 0) continue;                                                \
        DTYPE value = (DTYPE)((UDTYPE)mn + (UDTYPE)k);                         \
        /* Tight inner write loop. The compiler is free to lower this to       \
         * a memset-equivalent for narrow types or a vectorized store          \
         * for wide types — both are fine. */                                  \
        for (Py_ssize_t j = 0; j < cnt; ++j) {                                 \
            dst[offset + j] = value;                                           \
        }                                                                      \
        offset += cnt;                                                         \
    }                                                                          \
                                                                               \
    Py_END_ALLOW_THREADS                                                       \
                                                                               \
    PyMem_Free(counts);                                                        \
    return (PyObject *)out;                                                    \
}


/*
 * Entry points, one per dtype.
 *
 * The PARSE_FMT for all four uses "O!LL|O" because:
 *   - mn / mx come in as Python ints (long long fits all four dtypes
 *     including uint64 values up to 2^63-1; values above that would have
 *     to come in as the unsigned variant, but our caller never picks
 *     counting-sort for ranges that span there — _counting_is_worth_it
 *     caps range at 1<<24 = 16M, so the mn/mx pair always fits in i64).
 *   - The trailing "|O" is the optional out= ndarray (or None).
 */

DEFINE_FUSED_KERNEL(fused_counting_sort_i64,
                    int64_t,  uint64_t, NPY_INT64,  "O!LL|O")
DEFINE_FUSED_KERNEL(fused_counting_sort_i32,
                    int32_t,  uint32_t, NPY_INT32,  "O!LL|O")
DEFINE_FUSED_KERNEL(fused_counting_sort_u32,
                    uint32_t, uint32_t, NPY_UINT32, "O!LL|O")
DEFINE_FUSED_KERNEL(fused_counting_sort_u64,
                    uint64_t, uint64_t, NPY_UINT64, "O!LL|O")


/* ------------------------------------------------------------------ */
/* Method table + module init                                          */
/* ------------------------------------------------------------------ */

static PyMethodDef CountingExtMethods[] = {
    {"fused_counting_sort_i64",
     (PyCFunction)(void(*)(void))fused_counting_sort_i64,
     METH_VARARGS | METH_KEYWORDS,
     "Fused counting sort for int64 arrays.\n\n"
     "fused_counting_sort_i64(arr, mn, mx, out=None) -> ndarray\n"
     "Caller guarantees arr is C-contiguous int64 and all values lie\n"
     "in [mn, mx]. If out is given it must be int64, 1-D, contiguous,\n"
     "len(arr) — and will be returned. Raises OverflowError if the\n"
     "value range cannot fit a counts table on this platform."},

    {"fused_counting_sort_i32",
     (PyCFunction)(void(*)(void))fused_counting_sort_i32,
     METH_VARARGS | METH_KEYWORDS,
     "Fused counting sort for int32 arrays. See _i64 for semantics."},

    {"fused_counting_sort_u32",
     (PyCFunction)(void(*)(void))fused_counting_sort_u32,
     METH_VARARGS | METH_KEYWORDS,
     "Fused counting sort for uint32 arrays. See _i64 for semantics."},

    {"fused_counting_sort_u64",
     (PyCFunction)(void(*)(void))fused_counting_sort_u64,
     METH_VARARGS | METH_KEYWORDS,
     "Fused counting sort for uint64 arrays. See _i64 for semantics.\n"
     "NB: mn/mx are passed as Python ints; values above 2^63-1 cannot\n"
     "round-trip through the long-long parse and must be gated by the\n"
     "caller (the v6 _counting_is_worth_it cap of 1<<24 makes this moot)."},

    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef countingextmodule = {
    PyModuleDef_HEAD_INIT,
    "_counting_ext",
    "Fused counting-sort kernels for QuillSort. Optional native extension; "
    "the Python caller MUST fall back to numpy when this module fails to "
    "import or any function raises.",
    -1,
    CountingExtMethods,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC
PyInit__counting_ext(void)
{
    /* numpy C-API: required before any PyArray_* / NPY_* macro call. */
    import_array();
    if (PyErr_Occurred()) {
        return NULL;
    }
    return PyModule_Create(&countingextmodule);
}
