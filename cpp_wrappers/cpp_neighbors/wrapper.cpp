#include <Python.h>
#include "../cpp_utils/numpy_compat.h"
#include "radius_neighbors/neighbors.h"
#include <string>



// docstrings for our module
// *************************

static char module_docstring[] = "This module provides two methods to compute radius neighbors from pointclouds or batch of pointclouds";

static char batch_query_docstring[] = "Method to get radius neighbors in a batch of stacked pointclouds";


// Declare the functions
// *********************

static PyObject *batch_neighbors(PyObject *self, PyObject *args, PyObject *keywds);


// Specify the members of the module
// *********************************

static PyMethodDef module_methods[] = 
{
	{ "batch_query", (PyCFunction)batch_neighbors, METH_VARARGS | METH_KEYWORDS, batch_query_docstring },
	{NULL, NULL, 0, NULL}
};


// Initialize the module
// *********************

static struct PyModuleDef moduledef = 
{
    PyModuleDef_HEAD_INIT,
    "radius_neighbors",		// m_name
    module_docstring,       // m_doc
    -1,                     // m_size
    module_methods,         // m_methods
    NULL,                   // m_reload
    NULL,                   // m_traverse
    NULL,                   // m_clear
    NULL,                   // m_free
};

PyMODINIT_FUNC PyInit_radius_neighbors(void)
{
    import_array();
	return PyModule_Create(&moduledef);
}


// Definition of the batch_subsample method
// **********************************

static PyObject* batch_neighbors(PyObject* self, PyObject* args, PyObject* keywds)
{

	// Manage inputs
	// *************

	// Args containers
	PyObject* queries_obj = NULL;
	PyObject* supports_obj = NULL;
	PyObject* q_batches_obj = NULL;
	PyObject* s_batches_obj = NULL;

	// Keywords containers
	static char* kwlist[] = { "queries", "supports", "q_batches", "s_batches", "radius", NULL };
	float radius = 0.1;

	// Parse the input  
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "OOOO|$f", kwlist, &queries_obj, &supports_obj, &q_batches_obj, &s_batches_obj, &radius))
	{
		PyErr_SetString(PyExc_RuntimeError, "Error parsing arguments");
		return NULL;
	}


	// Interpret the input objects as numpy arrays.
	PyObject* queries_array = PyArray_FROM_OTF(queries_obj, NPY_FLOAT, NPY_IN_ARRAY);
	PyObject* supports_array = PyArray_FROM_OTF(supports_obj, NPY_FLOAT, NPY_IN_ARRAY);
	PyObject* q_batches_array = PyArray_FROM_OTF(q_batches_obj, NPY_INT, NPY_IN_ARRAY);
	PyObject* s_batches_array = PyArray_FROM_OTF(s_batches_obj, NPY_INT, NPY_IN_ARRAY);

	// Verify data was load correctly.
	if (queries_array == NULL)
	{
		Py_XDECREF(queries_array);
		Py_XDECREF(supports_array);
		Py_XDECREF(q_batches_array);
		Py_XDECREF(s_batches_array);
		PyErr_SetString(PyExc_RuntimeError, "Error converting query points to numpy arrays of type float32");
		return NULL;
	}
	if (supports_array == NULL)
	{
		Py_XDECREF(queries_array);
		Py_XDECREF(supports_array);
		Py_XDECREF(q_batches_array);
		Py_XDECREF(s_batches_array);
		PyErr_SetString(PyExc_RuntimeError, "Error converting support points to numpy arrays of type float32");
		return NULL;
	}
	if (q_batches_array == NULL)
	{
		Py_XDECREF(queries_array);
		Py_XDECREF(supports_array);
		Py_XDECREF(q_batches_array);
		Py_XDECREF(s_batches_array);
		PyErr_SetString(PyExc_RuntimeError, "Error converting query batches to numpy arrays of type int32");
		return NULL;
	}
	if (s_batches_array == NULL)
	{
		Py_XDECREF(queries_array);
		Py_XDECREF(supports_array);
		Py_XDECREF(q_batches_array);
		Py_XDECREF(s_batches_array);
		PyErr_SetString(PyExc_RuntimeError, "Error converting support batches to numpy arrays of type int32");
		return NULL;
	}

	// Check that the input array respect the dims
	if ((int)lca_PyArray_NDIM(queries_array) != 2 || (int)lca_PyArray_DIM(queries_array, 1) != 3)
	{
		Py_XDECREF(queries_array);
		Py_XDECREF(supports_array);
		Py_XDECREF(q_batches_array);
		Py_XDECREF(s_batches_array);
		PyErr_SetString(PyExc_RuntimeError, "Wrong dimensions : query.shape is not (N, 3)");
		return NULL;
	}
	if ((int)lca_PyArray_NDIM(supports_array) != 2 || (int)lca_PyArray_DIM(supports_array, 1) != 3)
	{
		Py_XDECREF(queries_array);
		Py_XDECREF(supports_array);
		Py_XDECREF(q_batches_array);
		Py_XDECREF(s_batches_array);
		PyErr_SetString(PyExc_RuntimeError, "Wrong dimensions : support.shape is not (N, 3)");
		return NULL;
	}
	if ((int)lca_PyArray_NDIM(q_batches_array) > 1)
	{
		Py_XDECREF(queries_array);
		Py_XDECREF(supports_array);
		Py_XDECREF(q_batches_array);
		Py_XDECREF(s_batches_array);
		PyErr_SetString(PyExc_RuntimeError, "Wrong dimensions : queries_batches.shape is not (B,) ");
		return NULL;
	}
	if ((int)lca_PyArray_NDIM(s_batches_array) > 1)
	{
		Py_XDECREF(queries_array);
		Py_XDECREF(supports_array);
		Py_XDECREF(q_batches_array);
		Py_XDECREF(s_batches_array);
		PyErr_SetString(PyExc_RuntimeError, "Wrong dimensions : supports_batches.shape is not (B,) ");
		return NULL;
	}
	if ((int)lca_PyArray_DIM(q_batches_array, 0) != (int)lca_PyArray_DIM(s_batches_array, 0))
	{
		Py_XDECREF(queries_array);
		Py_XDECREF(supports_array);
		Py_XDECREF(q_batches_array);
		Py_XDECREF(s_batches_array);
		PyErr_SetString(PyExc_RuntimeError, "Wrong number of batch elements: different for queries and supports ");
		return NULL;
	}

	// Number of points
	int Nq = (int)lca_PyArray_DIM(queries_array, 0);
	int Ns= (int)lca_PyArray_DIM(supports_array, 0);

	// Number of batches
	int Nb = (int)lca_PyArray_DIM(q_batches_array, 0);

	// Zero queries: return empty (0, 1) int32 — same convention as Python batch_neighbors.
	if (Nq == 0)
	{
		npy_intp* empty_dims = new npy_intp[2];
		empty_dims[0] = 0;
		empty_dims[1] = 1;
		PyObject* empty_arr = PyArray_SimpleNew(2, empty_dims, NPY_INT);
		delete[] empty_dims;
		if (empty_arr == NULL)
		{
			Py_XDECREF(queries_array);
			Py_XDECREF(supports_array);
			Py_XDECREF(q_batches_array);
			Py_XDECREF(s_batches_array);
			return NULL;
		}
		Py_XDECREF(queries_array);
		Py_XDECREF(supports_array);
		Py_XDECREF(q_batches_array);
		Py_XDECREF(s_batches_array);
		return Py_BuildValue("N", empty_arr);
	}

	// Call the C++ function
	// *********************

	// Convert PyArray to Cloud C++ class
	vector<PointXYZ> queries;
	vector<PointXYZ> supports;
	vector<int> q_batches;
	vector<int> s_batches;
	queries = vector<PointXYZ>((PointXYZ*)lca_PyArray_DATA(queries_array), (PointXYZ*)lca_PyArray_DATA(queries_array) + Nq);
	supports = vector<PointXYZ>((PointXYZ*)lca_PyArray_DATA(supports_array), (PointXYZ*)lca_PyArray_DATA(supports_array) + Ns);
	q_batches = vector<int>((int*)lca_PyArray_DATA(q_batches_array), (int*)lca_PyArray_DATA(q_batches_array) + Nb);
	s_batches = vector<int>((int*)lca_PyArray_DATA(s_batches_array), (int*)lca_PyArray_DATA(s_batches_array) + Nb);

	// Create result containers
	vector<int> neighbors_indices;

	// Compute results
	//batch_ordered_neighbors(queries, supports, q_batches, s_batches, neighbors_indices, radius);
	batch_nanoflann_neighbors(queries, supports, q_batches, s_batches, neighbors_indices, radius);

	// No in-radius neighbors for any query (max_count == 0): one padding column per query (sentinel = Ns).
	if ((int)neighbors_indices.size() < 1)
	{
		neighbors_indices.assign((size_t)Nq, Ns);
	}

	// Manage outputs
	// **************

	// Maximal number of neighbors
	int max_neighbors = (int)neighbors_indices.size() / Nq;

	// Dimension of output containers (heap; NumPy copies dims — free after SimpleNew)
	npy_intp* neighbors_dims = new npy_intp[2];
	neighbors_dims[0] = (npy_intp)Nq;
	neighbors_dims[1] = (npy_intp)max_neighbors;
	PyObject* res_obj = PyArray_SimpleNew(2, neighbors_dims, NPY_INT);
	delete[] neighbors_dims;
	PyObject* ret = NULL;

	// Fill output array with values
	size_t size_in_bytes = Nq * max_neighbors * sizeof(int);
	memcpy(lca_PyArray_DATA(res_obj), neighbors_indices.data(), size_in_bytes);

	// Merge results
	ret = Py_BuildValue("N", res_obj);

	// Clean up
	// ********

	Py_XDECREF(queries_array);
	Py_XDECREF(supports_array);
	Py_XDECREF(q_batches_array);
	Py_XDECREF(s_batches_array);

	return ret;
}
