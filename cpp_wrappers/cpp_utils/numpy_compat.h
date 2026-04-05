#ifndef LCA_NUMPY_COMPAT_H
#define LCA_NUMPY_COMPAT_H

#include <numpy/arrayobject.h>

/*
 * NumPy 2.0+ C API: PyArray_DATA / PyArray_NDIM / PyArray_DIM expect PyArrayObject*,
 * not PyObject*. These thin wrappers accept PyObject* from PyArray_FROM_OTF /
 * PyArray_SimpleNew and cast — valid for both NumPy 1.x and 2.x.
 */
static inline void *lca_PyArray_DATA(PyObject *arr) {
    return PyArray_DATA(reinterpret_cast<PyArrayObject *>(arr));
}

static inline int lca_PyArray_NDIM(PyObject *arr) {
    return PyArray_NDIM(reinterpret_cast<PyArrayObject *>(arr));
}

static inline npy_intp lca_PyArray_DIM(PyObject *arr, int i) {
    return PyArray_DIM(reinterpret_cast<PyArrayObject *>(arr), i);
}

#endif /* LCA_NUMPY_COMPAT_H */
