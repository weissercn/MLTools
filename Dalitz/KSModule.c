
#include <Python.h>

/*
 * Another function to be called from Python
 */
static PyObject* py_KS_loop(PyObject* self, PyObject* args)
{
  double x, y;
  PyArg_ParseTuple(args, "dd", &x, &y);
  return Py_BuildValue("d", x*y);
}

/*
 * Bind Python function names to our C functions
 */
static PyMethodDef KSModule_methods[] = { 
  {"KS_loop", py_KS_loop, METH_VARARGS},
  {NULL, NULL}
};

/*
 * Python calls this to let us initialize our module
 */
void initKSModule()
{
  (void) Py_InitModule("KSModule", KSModule_methods);
}
