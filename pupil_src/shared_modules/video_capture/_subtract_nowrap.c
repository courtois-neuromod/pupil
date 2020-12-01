#include "Python.h"
//#include "math.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"

/*
 * subtract_nowrap.c
 * This is a C code for ufunc that subtract 2 arrays element-wise and set to zero to avoid wrapping of negative results.
 *
 * In this code we only define the ufunc for
 * a single dtype. The computations that must
 * be replaced to create a ufunc for
 * a different function are marked with BEGIN
 * and END.
 *
 * Details explaining the Python-C API can be found under
 * 'Extending and Embedding' and 'Python/C API' at
 * docs.python.org .
 *
 * BUILD with `python3 setup.py build_ext --inplace`
 *
 */

static PyMethodDef MathMethods[] = {
        {NULL, NULL, 0, NULL}
};

/* The loop definition must precede the PyMODINIT_FUNC. */

static void subtract_nowrap_uint8(char **args, npy_intp *dimensions,
                            npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    npy_uint8 *in1 = args[0], *in2 = args[1];
    npy_intp in1_step = steps[0], in2_step = steps[1];

    for (i = 0; i < n; i++) {
        /*BEGIN main ufunc computation*/
        if (*in1 > *in2){
          *in1 -= *in2;
        } else {
          *in1 = 0;
        }

        in1 += in1_step;
        in2 += in2_step;
    }
}

/*This a pointer to the above function*/
PyUFuncGenericFunction funcs[1] = {&subtract_nowrap_uint8};

/* These are the input and return dtypes of logit.*/
static char types[2] = {NPY_UINT8, NPY_UINT8};

static void *data[1] = {NULL};

#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_npufunc",
    NULL,
    -1,
    MathMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit__npufunc(void)
{
    PyObject *m, *subtract_nowrap, *d;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }

    import_array();
    import_umath();

    subtract_nowrap = PyUFunc_FromFuncAndData(funcs, data, types, 1, 2, 0,
                                    PyUFunc_None, "subtract_nowrap",
                                    "subtract_nowrap_docstring", 0);

    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "subtract_nowrap", subtract_nowrap);
    Py_DECREF(subtract_nowrap);

    return m;
}
#else
PyMODINIT_FUNC init_npufunc(void)
{
    PyObject *m, *subtract_nowrap, *d;


    m = Py_InitModule("npufunc", MathMethods);
    if (m == NULL) {
        return;
    }

    import_array();
    import_umath();

     subtract_nowrap = PyUFunc_FromFuncAndData(funcs, data, types, 1, 2, 1,
                                    PyUFunc_None, "subtract_nowrap",
                                    "subtract_nowrap_docstring", 0);

    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "subtract_nowrap", subtract_nowrap);
    Py_DECREF(subtract_nowrap);
}
#endif
