#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include "capi.h"


typedef struct {
    PyObject_HEAD
    PinkTrombone *pinktrombone;
} PinkTromboneObject;


static int
PinkTromboneObject_init(PinkTromboneObject *self, PyObject *args, PyObject *kwds)
{
    int n;
    if (!PyArg_ParseTuple(args, "i", &n)) {
        return -1;
    }
    self->pinktrombone = PinkTrombone_new(n);
    if (self->pinktrombone == NULL) {
        return -1;
    }
    return 0;
}

static void
PinkTromboneObject_dealloc(PinkTromboneObject *self)
{
    if (self->pinktrombone != NULL) {
        PinkTrombone_delete(self->pinktrombone);
        self->pinktrombone = NULL;
    }
}

static PyObject *
PinkTromboneObject_control(PinkTromboneObject *self, PyObject *args)
{
    int ret;
    char *buffer;
    Py_ssize_t length;
    if (PyBytes_AsStringAndSize(args, &buffer, &length) != 0) {
        return NULL;
    }
    ret = PinkTrombone_control(self->pinktrombone, (double *)buffer, length / sizeof (double));
    if (ret != 0) {
        PyErr_SetString(PyExc_TypeError, "size doesn't match");
        return NULL;
    }
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *
PinkTromboneObject_process(PinkTromboneObject *self, PyObject *Py_UNUSED(ignored))
{
    int ret;
    PyObject *result;
    char *buffer;
    Py_ssize_t length;

    length = 512;
    result = PyBytes_FromStringAndSize(NULL, length * sizeof (double));
    if (result == NULL) {
        return NULL;
    }

    if (PyBytes_AsStringAndSize(result, &buffer, &length) != 0) {
        Py_DECREF(result);
        return NULL;
    }
    ret = PinkTrombone_process(self->pinktrombone, (double *)buffer, length / sizeof (double));
    if (ret != 0) {
        Py_DECREF(result);
        PyErr_SetString(PyExc_TypeError, "error");
        return NULL;
    }

    return result;
}

static PyMethodDef PinkTromboneObject_methods[] = {
    {"control", (PyCFunction) PinkTromboneObject_control, METH_O,
     "set control values"
    },
    {"process", (PyCFunction) PinkTromboneObject_process, METH_NOARGS,
     "process"
    },
    {NULL}  /* Sentinel */
};

static PyTypeObject PinkTromboneType = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "voice100_pinktrombone._PinkTrombone",
    .tp_basicsize = sizeof(PinkTromboneObject),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor) PinkTromboneObject_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = PyDoc_STR("Pink Trombone"),
    .tp_methods = PinkTromboneObject_methods,
    .tp_init = (initproc) PinkTromboneObject_init,
    .tp_new = PyType_GenericNew,
};


static PyMethodDef method_defs[] = {
    { NULL, NULL, 0, NULL }
};

static PyModuleDef def = {
    PyModuleDef_HEAD_INIT,
    "_pinktrombone",
    "",
    -1,
    method_defs
};


PyMODINIT_FUNC PyInit__pinktrombone(void)
{
    PyObject *module;

    if (PyType_Ready(&PinkTromboneType) < 0)
        return NULL;

    module = PyModule_Create(&def);

    Py_INCREF(&PinkTromboneType);
    if (PyModule_AddObject(module, "_PinkTrombone", (PyObject *) &PinkTromboneType) < 0) {
        Py_DECREF(&PinkTromboneType);
        Py_DECREF(module);
        return NULL;
    }

    return module;
}
