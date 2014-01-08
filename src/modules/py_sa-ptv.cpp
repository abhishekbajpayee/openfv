#include "std_include.h"
#include "py_sa-ptv.h"

using namespace std;

static PyObject *
spam_system(PyObject *self, PyObject *args) {

    const char *command;
    int sts;

    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;
    sts = system(command);
    return Py_BuildValue("i", sts);

}

extern "C" int square(int x) {

    return (x*x);

}

extern "C" double sum(double a, double b) {

    cout<<"Creating sum"<<endl;

    return(a+b);

}
