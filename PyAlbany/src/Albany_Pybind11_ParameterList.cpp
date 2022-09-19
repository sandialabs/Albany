//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_Pybind11_ParameterList.hpp"
#include "Albany_Interface.hpp"

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
pybind11::array_t<T> copyTeuchosArrayToNumPy(Teuchos::Array< T > & tArray) {

    pybind11::array_t<T> array(tArray.size());
    auto data = array.template mutable_unchecked<1>();
    for (int i=0; i < tArray.size(); ++i)
      data(i) = tArray[i];
    return array;
}

// Implementation based on:
// https://github.com/trilinos/Trilinos/tree/master/packages/PyTrilinos/src/PyTrilinos_Teuchos_Util.cpp
bool setPythonParameter(Teuchos::RCP<Teuchos::ParameterList> plist,
			const std::string      & name,
			py::object             value)
{
  py::handle h = value;

  // Boolean values
  if (PyBool_Check(value.ptr ()))
  {
    if (value == Py_True) plist->set(name,true );
    else                  plist->set(name,false);
  }

  // Integer values
  else if (PyInt_Check(value.ptr ()))
  {
    plist->set(name, h.cast<int>());
  }

  // Floating point values
  else if (PyFloat_Check(value.ptr ()))
  {
    plist->set(name, h.cast<double>());
  }

  // Unicode values
  else if (PyUnicode_Check(value.ptr ()))
  {
    PyObject * pyBytes = PyUnicode_AsASCIIString(value.ptr ());
    if (!pyBytes) return false;
    plist->set(name, std::string(PyBytes_AsString(pyBytes)));
    Py_DECREF(pyBytes);
  }

  // String values
  else if (PyString_Check(value.ptr ()))
  {
    plist->set(name, h.cast<std::string>());
  }

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



// Implementation based on:
// https://github.com/trilinos/Trilinos/tree/master/packages/PyTrilinos/src/PyTrilinos_Teuchos_Util.cpp
py::object getPythonParameter(Teuchos::RCP<Teuchos::ParameterList> plist,
			      const std::string            & name)
{
  // Get the parameter entry.  I now deal with the Teuchos::ParameterEntry
  // objects so that I can query the Teuchos::ParameterList without setting
  // the "used" flag to true.
  const Teuchos::ParameterEntry * entry = plist->getEntryPtr(name);
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
            return py::none();
          }
        }
      }
    }
  }

  // All  other types are unsupported
  return py::none();
}    // getPythonParameter

template<typename T>
Teuchos::Array< T > copyNumPyToTeuchosArray(pybind11::array_t<T> array) {

    auto np_array = array.template mutable_unchecked<1>();
    int size = array.shape(0);
    Teuchos::Array< T > av(size);
    for (int i=0; i < size; ++i)
      av[i] = np_array(i);
    return av;
}


template <typename T>
bool setPythonParameterArray(Teuchos::RCP<Teuchos::ParameterList> plist,
			const std::string      & name,
			pybind11::array_t< T >   value)
{
  auto tArray = copyNumPyToTeuchosArray(value);
  plist->set(name, tArray);
  return true;
}

template <typename T>
void def_ParameterList_member_functions(T cl) {
  cl.def("__setitem__", [](Teuchos::RCP<Teuchos::ParameterList> &m, const std::string &name, Teuchos::ParameterList value) { m->set(name,value);  });
  cl.def("set", [](Teuchos::RCP<Teuchos::ParameterList> &m, const std::string &name, Teuchos::ParameterList value) { m->set(name,value);  });
  cl.def("sublist", [](Teuchos::RCP<Teuchos::ParameterList> &m, const std::string &name) { if (m->isSublist(name)) { return pybind11::cast(sublist(m, name)); } return pybind11::cast("Invalid sublist name"); }, pybind11::return_value_policy::reference);
  cl.def("__setitem__", [](Teuchos::RCP<Teuchos::ParameterList> &m, const std::string &name, pybind11::object value) { setPythonParameter(m,name,value);  });
  cl.def("__getitem__", [](Teuchos::RCP<Teuchos::ParameterList> &m, const std::string &name) {
    // Sublist
    if (m->isSublist(name))
      return py::cast(Teuchos::sublist(m, name));
    return getPythonParameter(m,name);
  });
  cl.def("set", [](Teuchos::RCP<Teuchos::ParameterList> &m, const std::string &name, pybind11::object value) { setPythonParameter(m,name,value);  });
  cl.def("get", [](Teuchos::RCP<Teuchos::ParameterList> &m, const std::string &name) {
    // Sublist
    if (m->isSublist(name))
      return py::cast(Teuchos::sublist(m, name));
    return getPythonParameter(m,name);
  });
}

template<typename T>
void pyalbany_teuchosarray(py::module &m) {
  std::string pyName = std::string("TArray_")+std::string(typeid(T).name());
  py::class_<Teuchos::Array<T>, Teuchos::RCP<Teuchos::Array<T>>>(m, pyName.c_str(), py::buffer_protocol())
    .def_buffer([](Teuchos::Array<T> &a) -> py::buffer_info {
          return py::buffer_info(
              a.data(),
              sizeof(T),
              py::format_descriptor<T>::format(),
              1,
              { a.size() },
              { sizeof(T) }
          );
      });
}

void pyalbany_parameterlist(py::module &m) {
    pyalbany_teuchosarray<int>(m);
    pyalbany_teuchosarray<double>(m);

    py::class_<Teuchos::ParameterList, Teuchos::RCP<Teuchos::ParameterList>> cl(m, "PyParameterList");
    cl.def( pybind11::init( [](){ return new Teuchos::ParameterList(); } ) );
    cl.def( pybind11::init( [](const std::string & a0){ return new Teuchos::ParameterList(a0); } ), "doc" , pybind11::arg("name"));
    cl.def( pybind11::init<const std::string &, const class Teuchos::RCP<const class Teuchos::ParameterListModifier> &>(), pybind11::arg("name"), pybind11::arg("modifier") );

    cl.def( pybind11::init( [](Teuchos::ParameterList const &o){ return new Teuchos::ParameterList(o); } ) );
    cl.def("setName", (class Teuchos::ParameterList & (Teuchos::ParameterList::*)(const std::string &)) &Teuchos::ParameterList::setName, "Set the name of *this list.\n\nC++: Teuchos::ParameterList::setName(const std::string &) --> class Teuchos::ParameterList &", pybind11::return_value_policy::automatic, pybind11::arg("name"));
    cl.def("assign", (class Teuchos::ParameterList & (Teuchos::ParameterList::*)(const class Teuchos::ParameterList &)) &Teuchos::ParameterList::operator=, "Replace the current parameter list with \n\n \n This also replaces the name returned by this->name()\n\nC++: Teuchos::ParameterList::operator=(const class Teuchos::ParameterList &) --> class Teuchos::ParameterList &", pybind11::return_value_policy::automatic, pybind11::arg("source"));
    cl.def("setModifier", (void (Teuchos::ParameterList::*)(const class Teuchos::RCP<const class Teuchos::ParameterListModifier> &)) &Teuchos::ParameterList::setModifier, "C++: Teuchos::ParameterList::setModifier(const class Teuchos::RCP<const class Teuchos::ParameterListModifier> &) --> void", pybind11::arg("modifier"));
    cl.def("setParameters", (class Teuchos::ParameterList & (Teuchos::ParameterList::*)(const class Teuchos::ParameterList &)) &Teuchos::ParameterList::setParameters, "Set the parameters in source.\n\n This function will set the parameters and sublists from\n source into *this, but will not remove\n parameters from *this.  Parameters in *this\n with the same names as those in source will be\n overwritten.\n\nC++: Teuchos::ParameterList::setParameters(const class Teuchos::ParameterList &) --> class Teuchos::ParameterList &", pybind11::return_value_policy::automatic, pybind11::arg("source"));
    cl.def("setParametersNotAlreadySet", (class Teuchos::ParameterList & (Teuchos::ParameterList::*)(const class Teuchos::ParameterList &)) &Teuchos::ParameterList::setParametersNotAlreadySet, "Set the parameters in source that are not already set in\n *this.\n\n Note, this function will set the parameters and sublists from\n source into *this but will not result in parameters\n being removed from *this or in parameters already set in\n *this being overrided.  Parameters in *this with the\n same names as those in source will not be overwritten.\n\nC++: Teuchos::ParameterList::setParametersNotAlreadySet(const class Teuchos::ParameterList &) --> class Teuchos::ParameterList &", pybind11::return_value_policy::automatic, pybind11::arg("source"));
    cl.def("disableRecursiveValidation", (class Teuchos::ParameterList & (Teuchos::ParameterList::*)()) &Teuchos::ParameterList::disableRecursiveValidation, "Disallow recusive validation when this sublist is used in a valid\n parameter list.\n\n This function should be called when setting a sublist in a valid\n parameter list which is broken off to be passed to another object.\n The other object should validate its own list.\n\nC++: Teuchos::ParameterList::disableRecursiveValidation() --> class Teuchos::ParameterList &", pybind11::return_value_policy::automatic);
    cl.def("disableRecursiveModification", (class Teuchos::ParameterList & (Teuchos::ParameterList::*)()) &Teuchos::ParameterList::disableRecursiveModification, "Disallow recursive modification when this sublist is used in a modified\n parameter list.\n\n This function should be called when setting a sublist in a modified\n parameter list which is broken off to be passed to another object.\n The other object should modify its own list.  The parameter list can\n still be modified using a direct call to its modify method.\n\nC++: Teuchos::ParameterList::disableRecursiveModification() --> class Teuchos::ParameterList &", pybind11::return_value_policy::automatic);
    cl.def("disableRecursiveReconciliation", (class Teuchos::ParameterList & (Teuchos::ParameterList::*)()) &Teuchos::ParameterList::disableRecursiveReconciliation, "Disallow recursive reconciliation when this sublist is used in a\n reconciled parameter list.\n\n This function should be called when setting a sublist in a reconciled\n parameter list which is broken off to be passed to another object.\n The other object should reconcile its own list.  The parameter list can\n still be reconciled using a direct call to its reconcile method.\n\nC++: Teuchos::ParameterList::disableRecursiveReconciliation() --> class Teuchos::ParameterList &", pybind11::return_value_policy::automatic);
    cl.def("disableRecursiveAll", (class Teuchos::ParameterList & (Teuchos::ParameterList::*)()) &Teuchos::ParameterList::disableRecursiveAll, "Disallow all recursive modification, validation, and reconciliation when\n this sublist is used in a parameter list.\n\n This function should be called when setting a sublist in a\n parameter list which is broken off to be passed to another object.\n The other object should handle its own list.\n\nC++: Teuchos::ParameterList::disableRecursiveAll() --> class Teuchos::ParameterList &", pybind11::return_value_policy::automatic);
    cl.def("setEntry", (class Teuchos::ParameterList & (Teuchos::ParameterList::*)(const std::string &, const class Teuchos::ParameterEntry &)) &Teuchos::ParameterList::setEntry, "Set a parameter directly as a ParameterEntry. \n \n\n This is required to preserve the isDefault value when reading back\n from XML. KL 7 August 2004 \n\nC++: Teuchos::ParameterList::setEntry(const std::string &, const class Teuchos::ParameterEntry &) --> class Teuchos::ParameterList &", pybind11::return_value_policy::automatic, pybind11::arg("name"), pybind11::arg("entry"));
    cl.def("getEntry", (class Teuchos::ParameterEntry & (Teuchos::ParameterList::*)(const std::string &)) &Teuchos::ParameterList::getEntry, "Retrieves an entry with the name name.\n\n Throws Exceptions::InvalidParameterName if this parameter does\n not exist.\n\nC++: Teuchos::ParameterList::getEntry(const std::string &) --> class Teuchos::ParameterEntry &", pybind11::return_value_policy::automatic, pybind11::arg("name"));
    cl.def("getEntryPtr", (class Teuchos::ParameterEntry * (Teuchos::ParameterList::*)(const std::string &)) &Teuchos::ParameterList::getEntryPtr, "Retrieves the pointer for an entry with the name name if\n  it exists. \n\nC++: Teuchos::ParameterList::getEntryPtr(const std::string &) --> class Teuchos::ParameterEntry *", pybind11::return_value_policy::automatic, pybind11::arg("name"));
    cl.def("getEntryRCP", (class Teuchos::RCP<class Teuchos::ParameterEntry> (Teuchos::ParameterList::*)(const std::string &)) &Teuchos::ParameterList::getEntryRCP, "Retrieves the RCP for an entry with the name name if\n  it exists. \n\nC++: Teuchos::ParameterList::getEntryRCP(const std::string &) --> class Teuchos::RCP<class Teuchos::ParameterEntry>", pybind11::arg("name"));
    cl.def("getModifier", (class Teuchos::RCP<const class Teuchos::ParameterListModifier> (Teuchos::ParameterList::*)() const) &Teuchos::ParameterList::getModifier, "Return the optional modifier object\n\nC++: Teuchos::ParameterList::getModifier() const --> class Teuchos::RCP<const class Teuchos::ParameterListModifier>");
    cl.def("remove", [](Teuchos::ParameterList &o, const std::string & a0) -> bool { return o.remove(a0); }, "", pybind11::arg("name"));
    cl.def("remove", (bool (Teuchos::ParameterList::*)(const std::string &, bool)) &Teuchos::ParameterList::remove, "Remove a parameter (does not depend on the type of the\n parameter).\n\n \n (in) The name of the parameter to remove\n\n \n (in) If true then if the parameter with\n the name name does not exist then a std::exception will be\n thrown!\n\n \n Returns true if the parameter was removed, and\n false if the parameter was not removed (false return\n value possible only if throwIfNotExists==false).\n\nC++: Teuchos::ParameterList::remove(const std::string &, bool) --> bool", pybind11::arg("name"), pybind11::arg("throwIfNotExists"));
    cl.def("name", (const std::string & (Teuchos::ParameterList::*)() const) &Teuchos::ParameterList::name, "The name of this ParameterList.\n\nC++: Teuchos::ParameterList::name() const --> const std::string &", pybind11::return_value_policy::automatic);
    cl.def("isParameter", (bool (Teuchos::ParameterList::*)(const std::string &) const) &Teuchos::ParameterList::isParameter, "Whether the given parameter exists in this list.\n\n Return true if a parameter with name  exists in this\n list, else return false.\n\nC++: Teuchos::ParameterList::isParameter(const std::string &) const --> bool", pybind11::arg("name"));
    cl.def("isSublist", (bool (Teuchos::ParameterList::*)(const std::string &) const) &Teuchos::ParameterList::isSublist, "Whether the given sublist exists in this list.\n\n Return true if a parameter with name  exists in this\n list, and is itself a ParameterList.  Otherwise, return false.\n\nC++: Teuchos::ParameterList::isSublist(const std::string &) const --> bool", pybind11::arg("name"));
    cl.def("numParams", (long (Teuchos::ParameterList::*)() const) &Teuchos::ParameterList::numParams, "Get the number of stored parameters.\n\nC++: Teuchos::ParameterList::numParams() const --> long");
    cl.def("print", (void (Teuchos::ParameterList::*)() const) &Teuchos::ParameterList::print, "Print function to use in debugging in a debugger.\n\n Prints to *VerboseObjectBase::getDefaultOStream() so it will print well\n in parallel.\n\nC++: Teuchos::ParameterList::print() const --> void");
    cl.def("currentParametersString", (std::string (Teuchos::ParameterList::*)() const) &Teuchos::ParameterList::currentParametersString, "Create a single formated std::string of all of the zero-level parameters in this list\n\nC++: Teuchos::ParameterList::currentParametersString() const --> std::string");
    cl.def("begin", (class Teuchos::FilteredIterator<struct std::_Deque_iterator<class Teuchos::StringIndexedOrderedValueObjectContainerBase::KeyObjectPair<class Teuchos::ParameterEntry>, const class Teuchos::StringIndexedOrderedValueObjectContainerBase::KeyObjectPair<class Teuchos::ParameterEntry> &, const class Teuchos::StringIndexedOrderedValueObjectContainerBase::KeyObjectPair<class Teuchos::ParameterEntry> *>, class Teuchos::StringIndexedOrderedValueObjectContainerBase::SelectActive<class Teuchos::ParameterEntry> > (Teuchos::ParameterList::*)() const) &Teuchos::ParameterList::begin, "An iterator pointing to the first entry\n\nC++: Teuchos::ParameterList::begin() const --> class Teuchos::FilteredIterator<struct std::_Deque_iterator<class Teuchos::StringIndexedOrderedValueObjectContainerBase::KeyObjectPair<class Teuchos::ParameterEntry>, const class Teuchos::StringIndexedOrderedValueObjectContainerBase::KeyObjectPair<class Teuchos::ParameterEntry> &, const class Teuchos::StringIndexedOrderedValueObjectContainerBase::KeyObjectPair<class Teuchos::ParameterEntry> *>, class Teuchos::StringIndexedOrderedValueObjectContainerBase::SelectActive<class Teuchos::ParameterEntry> >");
    cl.def("end", (class Teuchos::FilteredIterator<struct std::_Deque_iterator<class Teuchos::StringIndexedOrderedValueObjectContainerBase::KeyObjectPair<class Teuchos::ParameterEntry>, const class Teuchos::StringIndexedOrderedValueObjectContainerBase::KeyObjectPair<class Teuchos::ParameterEntry> &, const class Teuchos::StringIndexedOrderedValueObjectContainerBase::KeyObjectPair<class Teuchos::ParameterEntry> *>, class Teuchos::StringIndexedOrderedValueObjectContainerBase::SelectActive<class Teuchos::ParameterEntry> > (Teuchos::ParameterList::*)() const) &Teuchos::ParameterList::end, "An iterator pointing beyond the last entry\n\nC++: Teuchos::ParameterList::end() const --> class Teuchos::FilteredIterator<struct std::_Deque_iterator<class Teuchos::StringIndexedOrderedValueObjectContainerBase::KeyObjectPair<class Teuchos::ParameterEntry>, const class Teuchos::StringIndexedOrderedValueObjectContainerBase::KeyObjectPair<class Teuchos::ParameterEntry> &, const class Teuchos::StringIndexedOrderedValueObjectContainerBase::KeyObjectPair<class Teuchos::ParameterEntry> *>, class Teuchos::StringIndexedOrderedValueObjectContainerBase::SelectActive<class Teuchos::ParameterEntry> >");
    cl.def("name", (const std::string & (Teuchos::ParameterList::*)(class Teuchos::FilteredIterator<struct std::_Deque_iterator<class Teuchos::StringIndexedOrderedValueObjectContainerBase::KeyObjectPair<class Teuchos::ParameterEntry>, const class Teuchos::StringIndexedOrderedValueObjectContainerBase::KeyObjectPair<class Teuchos::ParameterEntry> &, const class Teuchos::StringIndexedOrderedValueObjectContainerBase::KeyObjectPair<class Teuchos::ParameterEntry> *>, class Teuchos::StringIndexedOrderedValueObjectContainerBase::SelectActive<class Teuchos::ParameterEntry> >) const) &Teuchos::ParameterList::name, "Access to name (i.e., returns i->first)\n\nC++: Teuchos::ParameterList::name(class Teuchos::FilteredIterator<struct std::_Deque_iterator<class Teuchos::StringIndexedOrderedValueObjectContainerBase::KeyObjectPair<class Teuchos::ParameterEntry>, const class Teuchos::StringIndexedOrderedValueObjectContainerBase::KeyObjectPair<class Teuchos::ParameterEntry> &, const class Teuchos::StringIndexedOrderedValueObjectContainerBase::KeyObjectPair<class Teuchos::ParameterEntry> *>, class Teuchos::StringIndexedOrderedValueObjectContainerBase::SelectActive<class Teuchos::ParameterEntry> >) const --> const std::string &", pybind11::return_value_policy::automatic, pybind11::arg("i"));
    cl.def("entry", (const class Teuchos::ParameterEntry & (Teuchos::ParameterList::*)(class Teuchos::FilteredIterator<struct std::_Deque_iterator<class Teuchos::StringIndexedOrderedValueObjectContainerBase::KeyObjectPair<class Teuchos::ParameterEntry>, const class Teuchos::StringIndexedOrderedValueObjectContainerBase::KeyObjectPair<class Teuchos::ParameterEntry> &, const class Teuchos::StringIndexedOrderedValueObjectContainerBase::KeyObjectPair<class Teuchos::ParameterEntry> *>, class Teuchos::StringIndexedOrderedValueObjectContainerBase::SelectActive<class Teuchos::ParameterEntry> >) const) &Teuchos::ParameterList::entry, "Access to ParameterEntry (i.e., returns i->second)\n\nC++: Teuchos::ParameterList::entry(class Teuchos::FilteredIterator<struct std::_Deque_iterator<class Teuchos::StringIndexedOrderedValueObjectContainerBase::KeyObjectPair<class Teuchos::ParameterEntry>, const class Teuchos::StringIndexedOrderedValueObjectContainerBase::KeyObjectPair<class Teuchos::ParameterEntry> &, const class Teuchos::StringIndexedOrderedValueObjectContainerBase::KeyObjectPair<class Teuchos::ParameterEntry> *>, class Teuchos::StringIndexedOrderedValueObjectContainerBase::SelectActive<class Teuchos::ParameterEntry> >) const --> const class Teuchos::ParameterEntry &", pybind11::return_value_policy::automatic, pybind11::arg("i"));
    cl.def("validateParameters", [](Teuchos::ParameterList const &o, const class Teuchos::ParameterList & a0) -> void { return o.validateParameters(a0); }, "", pybind11::arg("validParamList"));
    cl.def("validateParameters", [](Teuchos::ParameterList const &o, const class Teuchos::ParameterList & a0, const int & a1) -> void { return o.validateParameters(a0, a1); }, "", pybind11::arg("validParamList"), pybind11::arg("depth"));
    cl.def("validateParameters", [](Teuchos::ParameterList const &o, const class Teuchos::ParameterList & a0, const int & a1, const enum Teuchos::EValidateUsed & a2) -> void { return o.validateParameters(a0, a1, a2); }, "", pybind11::arg("validParamList"), pybind11::arg("depth"), pybind11::arg("validateUsed"));
    cl.def("validateParameters", (void (Teuchos::ParameterList::*)(const class Teuchos::ParameterList &, const int, const enum Teuchos::EValidateUsed, const enum Teuchos::EValidateDefaults) const) &Teuchos::ParameterList::validateParameters, "Validate the parameters in this list given valid selections in\n the input list.\n\n \n [in] This is the list that the parameters and\n sublist in *this are compared against.\n\n \n [in] Determines the number of levels of depth that the\n validation will recurse into.  A value of depth=0 means that\n only the top level parameters and sublists will be checked.  Default:\n depth = large number.\n\n \n [in] Determines if parameters that have been used are\n checked against those in validParamList.  Default:\n validateDefaults = VALIDATE_DEFAULTS_ENABLED.\n\n \n [in] Determines if parameters set at their\n default values using get(name,defaultVal) are checked against\n those in validParamList.  Default: validateDefaults =\n VALIDATE_DEFAULTS_ENABLED.\n\n If a parameter in *this is not found in validParamList\n then an std::exception of type\n Exceptions::InvalidParameterName will be thrown which will\n contain an excellent error message returned by excpt.what().  If\n the parameter exists but has the wrong type, then an std::exception type\n Exceptions::InvalidParameterType will be thrown.  If the\n parameter exists and has the right type, but the value is not valid then\n an std::exception type Exceptions::InvalidParameterValue will be\n thrown.\n\n Recursive validation stops when:\n\n The maxinum depth is reached\n\n A sublist note in validParamList has been marked with the\n disableRecursiveValidation() function, or\n\n There are not more parameters or sublists left in *this\n\n \n\n A breath-first search is performed to validate all of the parameters in\n one sublist before moving into nested subslist.\n\nC++: Teuchos::ParameterList::validateParameters(const class Teuchos::ParameterList &, const int, const enum Teuchos::EValidateUsed, const enum Teuchos::EValidateDefaults) const --> void", pybind11::arg("validParamList"), pybind11::arg("depth"), pybind11::arg("validateUsed"), pybind11::arg("validateDefaults"));
    cl.def("validateParametersAndSetDefaults", [](Teuchos::ParameterList &o, const class Teuchos::ParameterList & a0) -> void { return o.validateParametersAndSetDefaults(a0); }, "", pybind11::arg("validParamList"));
    cl.def("validateParametersAndSetDefaults", (void (Teuchos::ParameterList::*)(const class Teuchos::ParameterList &, const int)) &Teuchos::ParameterList::validateParametersAndSetDefaults, "Validate the parameters in this list given valid selections in\n the input list and set defaults for those not set.\n\n \n [in] This is the list that the parameters and\n sublist in *this are compared against.\n\n \n [in] Determines the number of levels of depth that the\n validation will recurse into.  A value of depth=0 means that\n only the top level parameters and sublists will be checked.  Default:\n depth = large number.\n\n If a parameter in *this is not found in validParamList\n then an std::exception of type Exceptions::InvalidParameterName will\n be thrown which will contain an excellent error message returned by\n excpt.what().  If the parameter exists but has the wrong type,\n then an std::exception type Exceptions::InvalidParameterType will be\n thrown.  If the parameter exists and has the right type, but the value is\n not valid then an std::exception type\n Exceptions::InvalidParameterValue will be thrown.  If a\n parameter in validParamList does not exist in *this,\n then it will be set at its default value as determined by\n validParamList.\n\n Recursive validation stops when:\n\n The maxinum depth is reached\n\n A sublist note in validParamList has been marked with the\n disableRecursiveValidation() function, or\n\n There are not more parameters or sublists left in *this\n\n \n\n A breath-first search is performed to validate all of the parameters in\n one sublist before moving into nested subslist.\n\nC++: Teuchos::ParameterList::validateParametersAndSetDefaults(const class Teuchos::ParameterList &, const int) --> void", pybind11::arg("validParamList"), pybind11::arg("depth"));
    cl.def("modifyParameterList", [](Teuchos::ParameterList &o, class Teuchos::ParameterList & a0) -> void { return o.modifyParameterList(a0); }, "", pybind11::arg("validParamList"));
    cl.def("modifyParameterList", (void (Teuchos::ParameterList::*)(class Teuchos::ParameterList &, const int)) &Teuchos::ParameterList::modifyParameterList, "Modify the valid parameter list prior to validation.\n\n \n [in,out] The parameter list used as a template for validation.\n\n \n [in] Determines the number of levels of depth that the\n modification will recurse into.  A value of depth=0 means that\n only the top level parameters and sublists will be checked.  Default:\n depth = large number.\n\n We loop over the valid parameter list in this modification routine.  This routine\n adds and/or removes fields in the valid parameter list to match the structure of the\n parameter list about to be validated.  After completion, both parameter lists should\n have the same fields or else an error will be thrown during validation.\n\nC++: Teuchos::ParameterList::modifyParameterList(class Teuchos::ParameterList &, const int) --> void", pybind11::arg("validParamList"), pybind11::arg("depth"));
    cl.def("reconcileParameterList", [](Teuchos::ParameterList &o, class Teuchos::ParameterList & a0) -> void { return o.reconcileParameterList(a0); }, "", pybind11::arg("validParamList"));
    cl.def("reconcileParameterList", (void (Teuchos::ParameterList::*)(class Teuchos::ParameterList &, const bool)) &Teuchos::ParameterList::reconcileParameterList, "Reconcile a parameter list after validation\n\n \n [in,out] The parameter list used as a template for validation.\n\n \n [in] Sweep through the parameter list tree from left to right.\n\n We loop through the valid parameter list in reverse breadth-first order in this reconciliation\n routine.  This routine assumes that the reconciliation routine won't create new sublists as it\n traverses the parameter list.\n\nC++: Teuchos::ParameterList::reconcileParameterList(class Teuchos::ParameterList &, const bool) --> void", pybind11::arg("validParamList"), pybind11::arg("left_to_right"));

    cl.def("__str__", [](Teuchos::ParameterList const &o) -> std::string { std::ostringstream s; s << o; return s.str(); } );

    def_ParameterList_member_functions(cl);    

    m.def("getParameterList", &PyAlbany::getParameterList, "A function which returns an RCP to a parameter list read from file");
}
