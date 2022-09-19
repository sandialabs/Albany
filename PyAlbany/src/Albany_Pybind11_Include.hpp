#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#ifndef BINDER_PYBIND11_TYPE_CASTER
    #define BINDER_PYBIND11_TYPE_CASTER
    PYBIND11_DECLARE_HOLDER_TYPE(T, Teuchos::RCP<T>)
    PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>)
    PYBIND11_DECLARE_HOLDER_TYPE(T, T*)
    PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>)
#endif