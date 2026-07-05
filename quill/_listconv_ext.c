/*
 * quill/_listconv_ext.c
 * ---------------------
 * Fast list <-> ndarray conversion for ``quill_sort(list)``.
 *
 * Why this exists
 * ===============
 * np.asarray(list_of_python_ints) and ndarray.tolist() dominate the cost of
 * sorting a Python list with quill — by 3-5x on 1M ints. Both phases are pure
 * CPython-API overhead (per-element type probe + box/unbox). The kernels in
 * this file shortcut that by:
 *   - assuming the list is homogeneous (caller in _listconv.py is responsible
 *     for the fall-back when it isn't);
 *   - reading via the macro-form PyList_GET_ITEM (no bounds check, no GIL
 *     traffic) and writing straight into a pre-allocated numpy buffer.
 *
 * Build wiring (pyproject.toml / setup.py) is a separate phase — the Python
 * wrapper in _listconv.py works fully without this extension.
 *
 * Correctness notes
 * -----------------
 *  - We raise TypeError on the first non-conforming element. The wrapper
 *    catches TypeError/OverflowError and returns None so its caller can take
 *    the generic np.asarray path.
 *  - PyList_GET_ITEM returns a *borrowed* reference; we must not Py_DECREF it.
 *  - PyList_SET_ITEM *steals* a reference; we hand it a fresh PyLong/PyFloat.
 *  - All output arrays are owning C-contiguous numpy arrays — the GIL is held
 *    throughout, so it is safe to write straight into PyArray_DATA.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

/* numpy's C API requires the unique-symbol dance so the imported array_api
 * table doesn't collide with any other extension that also uses numpy in the
 * same process. */
#define PY_ARRAY_UNIQUE_SYMBOL QUILL_LISTCONV_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"


/* ─────────────────────────────────────────────────────────────────────────
 * list -> ndarray
 * ───────────────────────────────────────────────────────────────────────── */

static PyObject *
fast_to_int64(PyObject *self, PyObject *args)
{
    PyObject *list;
    if (!PyArg_ParseTuple(args, "O", &list)) {
        return NULL;
    }
    if (!PyList_Check(list)) {
        PyErr_SetString(PyExc_TypeError, "expected a list");
        return NULL;
    }

    Py_ssize_t n = PyList_GET_SIZE(list);
    npy_intp dims[1] = { (npy_intp) n };
    PyObject *arr = PyArray_SimpleNew(1, dims, NPY_INT64);
    if (arr == NULL) {
        return NULL;
    }

    npy_int64 *out = (npy_int64 *) PyArray_DATA((PyArrayObject *) arr);
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject *item = PyList_GET_ITEM(list, i);  /* borrowed */
        /* Strict PyLong check: bools and numpy scalars are not the fast path
         * (they round-trip cleanly via np.asarray, just slower — the wrapper
         * still gets the right result). */
        if (!PyLong_CheckExact(item)) {
            Py_DECREF(arr);
            PyErr_SetString(PyExc_TypeError,
                            "non-int element in fast_to_int64");
            return NULL;
        }
        int overflow = 0;
        long long v = PyLong_AsLongLongAndOverflow(item, &overflow);
        if (overflow != 0) {
            Py_DECREF(arr);
            PyErr_SetString(PyExc_OverflowError,
                            "int out of range for int64");
            return NULL;
        }
        if (v == -1 && PyErr_Occurred()) {
            Py_DECREF(arr);
            return NULL;
        }
        out[i] = (npy_int64) v;
    }
    return arr;
}


static PyObject *
fast_to_float64(PyObject *self, PyObject *args)
{
    PyObject *list;
    if (!PyArg_ParseTuple(args, "O", &list)) {
        return NULL;
    }
    if (!PyList_Check(list)) {
        PyErr_SetString(PyExc_TypeError, "expected a list");
        return NULL;
    }

    Py_ssize_t n = PyList_GET_SIZE(list);
    npy_intp dims[1] = { (npy_intp) n };
    PyObject *arr = PyArray_SimpleNew(1, dims, NPY_FLOAT64);
    if (arr == NULL) {
        return NULL;
    }

    double *out = (double *) PyArray_DATA((PyArrayObject *) arr);
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject *item = PyList_GET_ITEM(list, i);  /* borrowed */
        double v;
        if (PyFloat_CheckExact(item)) {
            /* AS_DOUBLE is the unchecked macro form; we already verified the
             * type, so the macro is safe and faster than PyFloat_AsDouble. */
            v = PyFloat_AS_DOUBLE(item);
        } else if (PyLong_CheckExact(item)) {
            /* int -> double via the standard converter; raises on
             * out-of-range, which we surface to the wrapper as a miss. */
            v = PyLong_AsDouble(item);
            if (v == -1.0 && PyErr_Occurred()) {
                Py_DECREF(arr);
                return NULL;
            }
        } else {
            Py_DECREF(arr);
            PyErr_SetString(PyExc_TypeError,
                            "non-numeric element in fast_to_float64");
            return NULL;
        }
        out[i] = v;
    }
    return arr;
}


/* ─────────────────────────────────────────────────────────────────────────
 * ndarray -> list
 * ───────────────────────────────────────────────────────────────────────── */

static PyObject *
fast_from_int64(PyObject *self, PyObject *args)
{
    PyObject *obj;
    if (!PyArg_ParseTuple(args, "O", &obj)) {
        return NULL;
    }
    if (!PyArray_Check(obj)) {
        PyErr_SetString(PyExc_TypeError, "expected an ndarray");
        return NULL;
    }
    PyArrayObject *arr = (PyArrayObject *) obj;
    if (PyArray_TYPE(arr) != NPY_INT64) {
        PyErr_SetString(PyExc_TypeError, "expected int64 dtype");
        return NULL;
    }
    if (!PyArray_IS_C_CONTIGUOUS(arr)) {
        PyErr_SetString(PyExc_TypeError, "expected C-contiguous array");
        return NULL;
    }

    Py_ssize_t n = (Py_ssize_t) PyArray_SIZE(arr);
    PyObject *list = PyList_New(n);
    if (list == NULL) {
        return NULL;
    }
    const npy_int64 *data = (const npy_int64 *) PyArray_DATA(arr);
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject *item = PyLong_FromLongLong((long long) data[i]);
        if (item == NULL) {
            Py_DECREF(list);
            return NULL;
        }
        /* SET_ITEM steals the reference we just made — no DECREF here. */
        PyList_SET_ITEM(list, i, item);
    }
    return list;
}


static PyObject *
fast_from_float64(PyObject *self, PyObject *args)
{
    PyObject *obj;
    if (!PyArg_ParseTuple(args, "O", &obj)) {
        return NULL;
    }
    if (!PyArray_Check(obj)) {
        PyErr_SetString(PyExc_TypeError, "expected an ndarray");
        return NULL;
    }
    PyArrayObject *arr = (PyArrayObject *) obj;
    if (PyArray_TYPE(arr) != NPY_FLOAT64) {
        PyErr_SetString(PyExc_TypeError, "expected float64 dtype");
        return NULL;
    }
    if (!PyArray_IS_C_CONTIGUOUS(arr)) {
        PyErr_SetString(PyExc_TypeError, "expected C-contiguous array");
        return NULL;
    }

    Py_ssize_t n = (Py_ssize_t) PyArray_SIZE(arr);
    PyObject *list = PyList_New(n);
    if (list == NULL) {
        return NULL;
    }
    const double *data = (const double *) PyArray_DATA(arr);
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject *item = PyFloat_FromDouble(data[i]);
        if (item == NULL) {
            Py_DECREF(list);
            return NULL;
        }
        PyList_SET_ITEM(list, i, item);
    }
    return list;
}


/* ─────────────────────────────────────────────────────────────────────────
 * Module wiring
 * ───────────────────────────────────────────────────────────────────────── */

static PyMethodDef ListConvMethods[] = {
    {"fast_to_int64",    fast_to_int64,    METH_VARARGS,
     "Convert a homogeneous list of Python ints to an int64 ndarray."},
    {"fast_to_float64",  fast_to_float64,  METH_VARARGS,
     "Convert a homogeneous list of int/float to a float64 ndarray."},
    {"fast_from_int64",  fast_from_int64,  METH_VARARGS,
     "Convert a C-contiguous int64 ndarray to a list[int]."},
    {"fast_from_float64", fast_from_float64, METH_VARARGS,
     "Convert a C-contiguous float64 ndarray to a list[float]."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef listconvmodule = {
    PyModuleDef_HEAD_INIT,
    "quill._listconv_ext",
    "Fast list<->ndarray conversion kernels for quill.",
    -1,
    ListConvMethods,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC
PyInit__listconv_ext(void)
{
    PyObject *m = PyModule_Create(&listconvmodule);
    if (m == NULL) {
        return NULL;
    }
    /* import_array() is a macro that returns NULL on failure — must be called
     * before any PyArray_* API is used. */
    import_array();
    if (PyErr_Occurred()) {
        Py_DECREF(m);
        return NULL;
    }
    return m;
}
