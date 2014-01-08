#ifndef PY_INTERFACE_LIBRARY
#define PY_INTERFACE_LIBRARY

#include "std_include.h"

static PyObject *
spam_system(PyObject *self, PyObject *args);

extern "C" int square(int x);

extern "C" double sum(double a, double b);

#endif
