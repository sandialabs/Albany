//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_Pybind11_ParameterList.hpp"
#include "Albany_Interface.hpp"

// Implementation based on:
// https://github.com/trilinos/Trilinos/tree/master/packages/PyTrilinos/src/PyTrilinos_Teuchos_Util.cpp

#if PY_VERSION_HEX >= 0x03000000

#define PyClass_Check(obj) PyObject_IsInstance(obj, (PyObject *)&PyType_Type)
#define PyInt_Check(x) PyLong_Check(x)
#define PyInt_AsLong(x) PyLong_AsLong(x)
#define PyInt_FromLong(x) PyLong_FromLong(x)
#define PyInt_FromSize_t(x) PyLong_FromSize_t(x)
#define PyString_Check(name) PyBytes_Check(name)
#define PyString_FromString(x) PyUnicode_FromString(x)
#define PyString_FromStringAndSize(x,s) PyUnicode_FromStringAndSize(x,s)
#define PyString_Format(fmt, args)  PyUnicode_Format(fmt, args)
#define PyString_AsString(str) PyBytes_AsString(str)
#define PyString_Size(str) PyBytes_Size(str)    
#define PyString_InternFromString(key) PyUnicode_InternFromString(key)
#define Py_TPFLAGS_HAVE_CLASS Py_TPFLAGS_BASETYPE
#define PyString_AS_STRING(x) PyUnicode_AS_STRING(x)
#define PyObject_Compare(x, y) (1-PyObject_RichCompareBool(x, y, Py_EQ))
#define _PyLong_FromSsize_t(x) PyLong_FromSsize_t(x)
#define convertPyStringToChar(pyobj) PyBytes_AsString(PyUnicode_AsASCIIString(pyobj))
#else
#define convertPyStringToChar(pyobj) PyString_AsString(pyobj)
#endif

namespace py = pybind11;

template<typename T>
Teuchos::Array< T > copyNumPyToTeuchosArray(pybind11::array_t<T> array) {

    auto np_array = array.template mutable_unchecked<1>();
    int size = array.shape(0);
    Teuchos::Array< T > av(size);
    for (int i=0; i < size; ++i)
      av[i] = np_array(i);
    return av;
}

template<typename T>
pybind11::array_t<T> copyTeuchosArrayToNumPy(Teuchos::Array< T > & tArray) {

    pybind11::array_t<T> array(tArray.size());
    auto data = array.template mutable_unchecked<1>();
    for (int i=0; i < tArray.size(); ++i)
      data(i) = tArray[i];
    return array;
}

bool setPythonParameter(Teuchos::ParameterList & plist,
			const std::string      & name,
			py::object             value)
{
  py::handle h = value;

  auto &npy_api = py::detail::npy_api::get();

  // Boolean values
  if (PyBool_Check(value.ptr ()))
  {
    if (value == Py_True) plist.set(name,true );
    else                  plist.set(name,false);
  }

  // Integer values
  else if (PyInt_Check(value.ptr ()))
  {
    plist.set(name, h.cast<int>());
  }

  // Floating point values
  else if (PyFloat_Check(value.ptr ()))
  {
    plist.set(name, h.cast<double>());
  }

  // Unicode values
  else if (PyUnicode_Check(value.ptr ()))
  {
    PyObject * pyBytes = PyUnicode_AsASCIIString(value.ptr ());
    if (!pyBytes) return false;
    plist.set(name, std::string(PyBytes_AsString(pyBytes)));
    Py_DECREF(pyBytes);
  }

  // String values
  else if (PyString_Check(value.ptr ()))
  {
    plist.set(name, h.cast<std::string>());
  }

/*
  // Sublist values
  else if (PyObject_TypeCheck(value.ptr(), PyParameterList))
  {
    plist.set(name, *(h.cast<RCP_PyParameterList>()));
  }
*/

  // None object not allowed: this is a python type not usable by
  // Trilinos solver packages, so we reserve it for the
  // getPythonParameter() function to indicate that the requested
  // parameter does not exist in the given Teuchos::ParameterList.
  // For logic reasons, this check must come before the check for
  // Teuchos::ParameterList
  else if (value.ptr () == Py_None)
  {
    return false;
  }

  // All other value types are unsupported
  else
  {
    return false;
  }

  // Successful type conversion
  return true;
}    // setPythonParameter


template <typename T>
bool setPythonParameterArray(Teuchos::ParameterList & plist,
			const std::string      & name,
			pybind11::array_t< T >   value)
{
  auto tArray = copyNumPyToTeuchosArray(value);
  plist.set(name, tArray);
  return true;
}

// **************************************************************** //

py::object getPythonParameter(const Teuchos::ParameterList & plist,
			      const std::string            & name)
{
  // If parameter does not exist, return None
  //if (!plist.isParameter(name)) return Py_BuildValue("");

  // Get the parameter entry.  I now deal with the Teuchos::ParameterEntry
  // objects so that I can query the Teuchos::ParameterList without setting
  // the "used" flag to true.
  const Teuchos::ParameterEntry * entry = plist.getEntryPtr(name);
  // Boolean parameter values
  if (entry->isType< bool >())
  {
    bool value = Teuchos::any_cast< bool >(entry->getAny(false));
    return py::cast(value);
  }
  // Integer parameter values
  else if (entry->isType< int >())
  {
    int value = Teuchos::any_cast< int >(entry->getAny(false));
    return py::cast(value);
  }
  // Double parameter values
  else if (entry->isType< double >())
  {
    double value = Teuchos::any_cast< double >(entry->getAny(false));
    return py::cast(value);
  }
  // String parameter values
  else if (entry->isType< std::string >())
  {
    std::string value = Teuchos::any_cast< std::string >(entry->getAny(false));
    return py::cast(value.c_str());
  }
  // Char * parameter values
  else if (entry->isType< char * >())
  {
    char * value = Teuchos::any_cast< char * >(entry->getAny(false));
    return py::cast(value);
  }

  else if (entry->isArray())
  {
    // try
    // {
    //   Teuchos::Array< bool > tArray =
    //     Teuchos::any_cast< Teuchos::Array< bool > >(entry->getAny(false));
    //   return copyTeuchosArrayToNumPy(tArray);
    // }
    // catch(Teuchos::bad_any_cast &e)
    // {
      try
      {
        Teuchos::Array< int > tArray =
          Teuchos::any_cast< Teuchos::Array< int > >(entry->getAny(false));
        return copyTeuchosArrayToNumPy(tArray);
      }
      catch(Teuchos::bad_any_cast &e)
      {
        try
        {
          Teuchos::Array< long > tArray =
            Teuchos::any_cast< Teuchos::Array< long > >(entry->getAny(false));
          return copyTeuchosArrayToNumPy(tArray);
        }
        catch(Teuchos::bad_any_cast &e)
        {
          try
          {
            Teuchos::Array< float > tArray =
              Teuchos::any_cast< Teuchos::Array< float > >(entry->getAny(false));
            return copyTeuchosArrayToNumPy(tArray);
          }
          catch(Teuchos::bad_any_cast &e)
          {
            try
            {
              Teuchos::Array< double > tArray =
                Teuchos::any_cast< Teuchos::Array< double > >(entry->getAny(false));
              return copyTeuchosArrayToNumPy(tArray);
            }
            catch(Teuchos::bad_any_cast &e)
            {
              // Teuchos::Arrays of type other than int or double are
              // currently unsupported
              //return NULL;
            }
          }
        }
      }
    // }
  }

  // All  other types are unsupported
  //return NULL;
}    // getPythonParameter

// **************************************************************** //

RCP_PyParameterList createRCPPyParameterList() {
    return Teuchos::rcp<PyParameterList>(new PyParameterList());
}

void pyalbany_parameterlist(py::module &m) {
    py::class_<RCP_PyParameterList>(m, "RCPPyParameterList")
        .def(py::init(&createRCPPyParameterList))
        .def("sublist", [](RCP_PyParameterList &m, const std::string &name) {
            if (m->isSublist(name))
                return py::cast(sublist(m,name));
            return py::cast("Invalid sublist name");
        }, py::return_value_policy::reference)
        .def("print", [](RCP_PyParameterList &m) {
            m->print();
        })
        .def("setSublist", [](RCP_PyParameterList &m, const std::string &name, RCP_PyParameterList &sub) {
            m->set(name, *sub);
        })
        .def("isParameter", [](RCP_PyParameterList &m, const std::string &name) {
            return m->isParameter(name);
        })
        .def("get", [](RCP_PyParameterList &m, const std::string &name) {
            if (m->isParameter(name)) {
                return getPythonParameter(*m, name);
            }
            return py::cast("Invalid parameter name");
        })
        .def("set", [](RCP_PyParameterList &m, const std::string &name, py::object value) {
            if (!setPythonParameter(*m,name,value))
                PyErr_SetString(PyExc_TypeError, "ParameterList value type not supported");
        })
        .def("set", [](RCP_PyParameterList &m, const std::string &name, pybind11::array_t<int> value) {
            setPythonParameterArray(*m,name,value);
        })
        .def("set", [](RCP_PyParameterList &m, const std::string &name, pybind11::array_t<long> value) {
            setPythonParameterArray(*m,name,value);
        })
        .def("set", [](RCP_PyParameterList &m, const std::string &name, pybind11::array_t<float> value) {
            setPythonParameterArray(*m,name,value);
        })
        .def("set", [](RCP_PyParameterList &m, const std::string &name, pybind11::array_t<double> value) {
            setPythonParameterArray(*m,name,value);
        });
    m.def("getParameterList", &PyAlbany::getParameterList, "A function which returns an RCP to a parameter list read from file");
}
