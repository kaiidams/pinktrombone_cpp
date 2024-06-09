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
  self->pinktrombone = PinkTrombone_new(44);
  if (self->pinktrombone == NULL) {
    return -1;
  }
  printf("init\n");
  return 0;
}

static void
PinkTromboneObject_dealloc(PinkTromboneObject *self)
{
  if (self->pinktrombone != NULL) {
    PinkTrombone_delete(self->pinktrombone);
    printf("dealloc\n");
    self->pinktrombone = NULL;
  }
}
#if 0
static PyObject *
PinkTromboneObject_getservername(PinkTromboneObject *self, void *closure)
{
    Py_INCREF(self->servername);
    return self->servername;
}

static int
PinkTromboneObject_setservername(PinkTromboneObject *self, PyObject *value, void *closure)
{
    PyObject *tmp;
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete servername");
        return -1;
    }
    if (!PyUnicode_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "servername must be a string");
        return -1;
    }
    printf("setservername\n");
    tmp = self->servername;
    Py_INCREF(value);
    self->servername = value;
    Py_DECREF(tmp);
    return 0;
}

static PyObject *
PinkTromboneObject_connect(PinkTromboneObject *self, PyObject *Py_UNUSED(ignored))
{
  PinkTrombone_Connect(self->server);
  printf("connect\n");
  Py_INCREF(Py_None);
  return Py_None;
}

static PyObject *
PinkTromboneObject_continue(PinkTromboneObject *self, PyObject *Py_UNUSED(ignored))
{
  printf("continue\n");
  PinkTrombone_Continue(self->server);
  printf("continue\n");
  Py_INCREF(Py_None);
  return Py_None;
}
#endif

static PyMemberDef PinkTromboneObject_members[] = {
    // {"servername", (getter) PinkTromboneObject_getservername, (setter) PinkTromboneObject_setservername,
    //  "server name", NULL},
    {NULL}  /* Sentinel */
};

static PyMethodDef PinkTromboneObject_methods[] = {
    // {"connect", (PyCFunction) PinkTromboneObject_connect, METH_NOARGS,
    //  "connect to the server"
    // },
    // {"_continue", (PyCFunction) PinkTromboneObject_continue, METH_NOARGS,
    //  "continue"
    // },
    {NULL}  /* Sentinel */
};

static PyTypeObject PinkTromboneType = {
  .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
  .tp_name = "pinktrombone._PinkTrombone",
  .tp_basicsize = sizeof(PinkTromboneObject),
  .tp_itemsize = 0,
  .tp_dealloc = (destructor) PinkTromboneObject_dealloc,
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = PyDoc_STR("Pink Trombone"),
  .tp_methods = PinkTromboneObject_methods,
  .tp_members = PinkTromboneObject_members,
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
  if (PyModule_AddObject(module, "PinkTrombone", (PyObject *) &PinkTromboneType) < 0) {
    Py_DECREF(&PinkTromboneType);
    Py_DECREF(module);
    return NULL;
  }

  return module;
}
