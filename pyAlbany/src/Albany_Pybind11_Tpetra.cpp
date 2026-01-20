//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_Pybind11_Tpetra.hpp"
#include "MatrixMarket_Tpetra.hpp"
#include "Albany_LinearOpWithSolveDecorators.hpp"
#include "Albany_TpetraThyraUtils.hpp"

// The implementation of the conversion from numpy array to Kokkos view 
// in both directions is based on:
// https://github.com/sandialabs/compadre/blob/master/pycompadre/pycompadre.cpp

namespace py = pybind11;

template< bool B, class T = void >
using enable_if_t = typename std::enable_if<B,T>::type;

template<typename T>
Teuchos::ArrayView< T > convert_np_to_ArrayView(pybind11::array_t<T> array) {

    int size = array.shape(0);
    Teuchos::ArrayView< T > av(array.mutable_data(0), size);

    return av;
}

// conversion of numpy arrays to kokkos arrays

template<typename ViewType>
void convert_np_to_kokkos_1d(pybind11::array_t<typename ViewType::non_const_value_type> array,  ViewType kokkos_array_device) {

    auto np_array = array.template unchecked<1>();

    auto kokkos_array_host = Kokkos::create_mirror_view(kokkos_array_device);
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, array.shape(0)), [&](int i) {
        kokkos_array_host(i) = np_array(i);
    });
    Kokkos::fence();
    Kokkos::deep_copy(kokkos_array_device, kokkos_array_host);
}

template<typename ViewType>
void convert_np_to_kokkos_2d(pybind11::array_t<typename ViewType::non_const_value_type> array,  ViewType kokkos_array_device) {

    auto np_array = array.template unchecked<2>();

    auto kokkos_array_host = Kokkos::create_mirror_view(kokkos_array_device);
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, array.shape(0)), [&](int i) {
        for (int j=0; j<array.shape(1); ++j) {
            kokkos_array_host(i,j) = np_array(i,j);
        }
    });
    Kokkos::fence();
    Kokkos::deep_copy(kokkos_array_device, kokkos_array_host);
}

// conversion of kokkos arrays to numpy arrays

template<typename T, typename T2=void>
struct cknp1d {
    pybind11::array_t<typename T::value_type> result;
    cknp1d (T kokkos_array_host) {

        const int dim_out_0 = kokkos_array_host.extent(0);
        result = pybind11::array_t<typename T::value_type>(dim_out_0);
        auto data = result.template mutable_unchecked<1>();
        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,dim_out_0), [&](int i) {
            data(i) = kokkos_array_host(i);
        });
        Kokkos::fence();

    }
    pybind11::array_t<typename T::value_type> convert() { return result; }
};

template<typename T>
struct cknp1d<T, enable_if_t<(T::rank!=1)> > {
    pybind11::array_t<typename T::value_type> result;
    cknp1d (T kokkos_array_host) {
        result = pybind11::array_t<typename T::value_type>(0);
    }
    pybind11::array_t<typename T::value_type> convert() { return result; }
};

template<typename T, typename T2=void>
struct cknp2d {
    pybind11::array_t<typename T::value_type> result;
    cknp2d (T kokkos_array_host) {

        const int dim_out_0 = kokkos_array_host.extent(0);
        const int dim_out_1 = kokkos_array_host.extent(1);

        result = pybind11::array_t<typename T::value_type>({dim_out_0,dim_out_1});
        auto data = result.template mutable_unchecked<T::rank>();
        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,dim_out_0), [&](int i) {
            for (int j=0; j<dim_out_1; ++j) {
                data(i,j) = kokkos_array_host(i,j);
            }
        });
        Kokkos::fence();

    }
    pybind11::array_t<typename T::value_type> convert() { return result; }
};

template<typename T>
struct cknp2d<T, enable_if_t<(T::rank!=2)> > {
    pybind11::array_t<typename T::value_type> result;
    cknp2d (T kokkos_array_host) {
        result = pybind11::array_t<typename T::value_type>(0);
    }
    pybind11::array_t<typename T::value_type> convert() { return result; }
};


template<typename T>
pybind11::array_t<typename T::value_type> convert_kokkos_to_np(T kokkos_array_device) {

    // ensure data is accessible
    auto kokkos_array_host =
        Kokkos::create_mirror_view(kokkos_array_device);
    Kokkos::deep_copy(kokkos_array_host, kokkos_array_device);

    pybind11::array_t<typename T::value_type> result;
    if (T::rank==1) {
        result = cknp1d<decltype(kokkos_array_host)>(kokkos_array_host).convert();
    } else if (T::rank==2) {
        result = cknp2d<decltype(kokkos_array_host)>(kokkos_array_host).convert();
    } else {
        result = pybind11::array_t<typename T::value_type>(0);
    }
    return result;

}

RCP_PyMap createRCPPyMapEmpty() {
    return Teuchos::rcp<Tpetra_Map>(new Tpetra_Map());
}

RCP_PyMap createRCPPyMap(int numGlobalEl, int numMyEl, int indexBase, RCP_Teuchos_Comm_PyAlbany comm ) {
    return Teuchos::rcp<Tpetra_Map>(new Tpetra_Map(numGlobalEl, numMyEl, indexBase, comm));
}

RCP_PyMap createRCPPyMapFromView(int numGlobalEl, pybind11::array_t<int> indexList, int indexBase, RCP_Teuchos_Comm_PyAlbany comm ) {
    Kokkos::View<Tpetra_GO*, Kokkos::DefaultExecutionSpace> indexView("map index view", indexList.shape(0));
    convert_np_to_kokkos_1d(indexList, indexView);
    return Teuchos::rcp<Tpetra_Map>(new Tpetra_Map(numGlobalEl, indexView, indexBase, comm));
}

RCP_PyVector createRCPPyVectorEmpty() {
    return Teuchos::rcp<Tpetra_Vector>(new Tpetra_Vector());
}

RCP_PyVector createRCPPyVector1(RCP_PyMap &map, const bool zeroOut) {
    return Teuchos::rcp<Tpetra_Vector>(new Tpetra_Vector(map, zeroOut));
}

RCP_PyVector createRCPPyVector2(RCP_ConstPyMap &map, const bool zeroOut) {
    return Teuchos::rcp<Tpetra_Vector>(new Tpetra_Vector(map, zeroOut));
}

RCP_PyMultiVector createRCPPyMultiVectorEmpty() {
    return Teuchos::rcp<Tpetra_MultiVector>(new Tpetra_MultiVector());
}

RCP_PyMultiVector createRCPPyMultiVector1(RCP_PyMap &map, const int n_cols, const bool zeroOut) {
    return Teuchos::rcp<Tpetra_MultiVector>(new Tpetra_MultiVector(map, n_cols, zeroOut));
}

RCP_PyMultiVector createRCPPyMultiVector2(RCP_ConstPyMap &map, const int n_cols, const bool zeroOut) {
    return Teuchos::rcp<Tpetra_MultiVector>(new Tpetra_MultiVector(map, n_cols, zeroOut));
}

pybind11::array_t<ST> getLocalView(RCP_PyVector &vector) {
    return convert_kokkos_to_np(Kokkos::subview(vector->getLocalViewDevice(Tpetra::Access::ReadOnly), Kokkos::ALL, 0));
}

pybind11::array_t<ST> getLocalView(RCP_PyMultiVector &mvector) {
    return convert_kokkos_to_np(mvector->getLocalViewDevice(Tpetra::Access::ReadOnly));
}

void setLocalView(RCP_PyVector &vector, pybind11::array_t<double> input) {
    auto view = Kokkos::subview(vector->getLocalViewDevice(Tpetra::Access::ReadWrite), Kokkos::ALL, 0);
    convert_np_to_kokkos_1d(input, view);
}

void setLocalView(RCP_PyMultiVector &mvector, pybind11::array_t<double> input) {
    auto view = mvector->getLocalViewDevice(Tpetra::Access::ReadWrite);
    convert_np_to_kokkos_2d(input, view);
}

RCP_PyCrsMatrix createRCPPyCrsMatrixFromFile(RCP_PyMap &map, const std::string& filename) {
  bool mapIsContiguous =
      (static_cast<Tpetra_GO>(map->getMaxAllGlobalIndex()+1-map->getMinAllGlobalIndex()) ==
       static_cast<Tpetra_GO>(map->getGlobalNumElements()));

  TEUCHOS_TEST_FOR_EXCEPTION (!mapIsContiguous, std::runtime_error,
                              "Error! Map needs to be contiguous for the Matrix reader to work.\n");

  Teuchos::RCP<const Tpetra_Map> colMap;
  Teuchos::RCP<const Tpetra_Map> domainMap = map;
  Teuchos::RCP<const Tpetra_Map> rangeMap = map;
  using reader_type = Tpetra::MatrixMarket::Reader<Tpetra_CrsMatrix>;

  return reader_type::readSparseFile (filename, map, colMap, domainMap, rangeMap);
}

pybind11::tuple getRemoteIndexList(RCP_ConstPyMap map, pybind11::array_t<Tpetra_GO> globalIndexes)
{
    auto globalIndexes_av = convert_np_to_ArrayView(globalIndexes);

    Tpetra::LookupStatus result;
    Teuchos::ArrayView< const Tpetra_GO > globalList(globalIndexes_av);

    pybind11::array_t<int> nodeList_np(globalList.size());
    pybind11::array_t<Tpetra_LO> localList_np(globalList.size());

    Teuchos::ArrayView< int >             nodeList(nodeList_np.mutable_data(0), globalList.size());
    Teuchos::ArrayView< Tpetra_LO >       localList(localList_np.mutable_data(0), globalList.size());

    // Call the method
    result = map->getRemoteIndexList(globalList,
                                      nodeList,
                                      localList);
    
    return pybind11::make_tuple(nodeList_np, localList_np, static_cast< long >(result));
}

void pyalbany_map(pybind11::module &m) {
    py::class_<PyMap, Teuchos::RCP<PyMap>>(m, "PyMap")
        .def(py::init(&createRCPPyMapEmpty))
        .def(py::init(&createRCPPyMap))
        .def(py::init(&createRCPPyMapFromView))
        .def("isOneToOne", &PyMap::isOneToOne)
        .def("getIndexBase", &PyMap::getIndexBase)
        .def("getMinLocalIndex", &PyMap::getMinLocalIndex)
        .def("getMaxLocalIndex", &PyMap::getMaxLocalIndex)
        .def("getMinGlobalIndex", &PyMap::getMinGlobalIndex)
        .def("getMaxGlobalIndex", &PyMap::getMaxGlobalIndex)
        .def("getMinAllGlobalIndex", &PyMap::getMinAllGlobalIndex)
        .def("getMaxAllGlobalIndex", &PyMap::getMaxAllGlobalIndex)
        .def("getLocalNumElements", &PyMap::getLocalNumElements)
        .def("getGlobalNumElements", &PyMap::getGlobalNumElements)
        .def("getLocalElement", &PyMap::getLocalElement)
        .def("getGlobalElement", &PyMap::getGlobalElement)
        .def("isNodeGlobalElement", &PyMap::isNodeGlobalElement)
        .def("isNodeLocalElement", &PyMap::isNodeLocalElement)
        .def("isUniform", &PyMap::isUniform)
        .def("isContiguous", &PyMap::isContiguous)
        .def("isDistributed", &PyMap::isDistributed)
        .def("isCompatible", &PyMap::isCompatible)
        .def("isSameAs", &PyMap::isSameAs)
        .def("locallySameAs", &PyMap::locallySameAs)
        .def("getComm", &PyMap::getComm)
        .def("getRemoteIndexList", [](Teuchos::RCP<PyMap> &m, py::array_t<Tpetra_GO> globalIndexes) {
            return getRemoteIndexList(m, globalIndexes);
        });
}

void pyalbany_vector(pybind11::module &m){
    py::class_<Tpetra_Vector, Teuchos::RCP<Tpetra_Vector>>(m, "PyVector")
        .def(py::init(&createRCPPyVector1))
        .def(py::init(&createRCPPyVector2))
        .def(py::init(&createRCPPyVectorEmpty))
        .def("getMap",&Tpetra_Vector::getMap)
        .def("getLocalView",[](Teuchos::RCP<Tpetra_Vector> &m){
            return getLocalView(m);
        })
        .def("setLocalView",[](Teuchos::RCP<Tpetra_Vector> &m, py::array_t<ST> input){
            return setLocalView(m, input);
        })
        .def("putScalar",[](Teuchos::RCP<Tpetra_Vector> &m, ST val) {
            m->putScalar(val);
        })
        // this = alpha vec + beta this
        .def("update",[](Teuchos::RCP<Tpetra_Vector> &m, ST alpha, Teuchos::RCP<Tpetra_Vector> & vec, ST beta) {
            m->update(alpha, *vec, beta);
        })
        // this = alpha this
        .def("scale",[](Teuchos::RCP<Tpetra_Vector> &m, ST alpha) {
            m->scale(alpha);
        })
        .def("dot",[](Teuchos::RCP<Tpetra_Vector> &m, Teuchos::RCP<Tpetra_Vector> &m2){
            return m->dot(*m2);
        });
}

void pyalbany_mvector(pybind11::module &m){
    py::class_<Tpetra_MultiVector, Teuchos::RCP<Tpetra_MultiVector>>(m, "PyMultiVector")
        .def(py::init(&createRCPPyMultiVector1))
        .def(py::init(&createRCPPyMultiVector2))
        .def(py::init(&createRCPPyMultiVectorEmpty))
        .def("getNumVectors",&Tpetra_MultiVector::getNumVectors)
        .def("getMap",&Tpetra_MultiVector::getMap)
        .def("getVector",&Tpetra_MultiVector::getVector)
        .def("getLocalView",[](Teuchos::RCP<Tpetra_MultiVector> &m){
            return getLocalView(m);
        })
        .def("setLocalView",[](Teuchos::RCP<Tpetra_MultiVector> &m, py::array_t<ST> input){
            return setLocalView(m, input);
        })
        .def("putScalar",[](Teuchos::RCP<Tpetra_MultiVector> &m, ST val) {
            m->putScalar(val);
        })        
        // this = alpha this
        .def("scale",[](Teuchos::RCP<Tpetra_MultiVector> &m, ST alpha) {
            m->scale(alpha);
        })
        // this = alpha vec + beta this
        .def("update",[](Teuchos::RCP<Tpetra_MultiVector> &m, ST alpha, Teuchos::RCP<Tpetra_MultiVector> & vec, ST beta) {
            m->update(alpha, *vec, beta);
        });
}

void pyalbany_crsmatrix(pybind11::module &m){
    py::class_<Tpetra_CrsMatrix, Teuchos::RCP<Tpetra_CrsMatrix>>(m, "PyCrsMatrix")
        .def(py::init(&createRCPPyCrsMatrixFromFile))
        .def("getMap",&Tpetra_CrsMatrix::getMap)
        .def("getDomainMap",&Tpetra_CrsMatrix::getDomainMap)
        .def("getRangeMap",&Tpetra_CrsMatrix::getRangeMap)
        .def("apply",[](Teuchos::RCP<Tpetra_CrsMatrix> &m, Teuchos::RCP<Tpetra_MultiVector> &x, Teuchos::RCP<Tpetra_MultiVector> &y, bool trans, double alpha, double beta){
            m->apply(*x, *y, trans ? Teuchos::TRANS : Teuchos::NO_TRANS, alpha, beta);
        })
        .def("solve",[](Teuchos::RCP<Tpetra_CrsMatrix> &m, Teuchos::RCP<Tpetra_MultiVector> &x, Teuchos::RCP<Tpetra_MultiVector> &b, bool trans, Teuchos::RCP<Teuchos::ParameterList> solverOptions){
            Albany::MatrixBased_LOWS solver(Albany::createThyraLinearOp(m));
            solver.initializeSolver(solverOptions);
            Thyra::solve(solver,trans ? Thyra::TRANS : Thyra::NOTRANS, *Albany::createConstThyraMultiVector(b), Albany::createThyraMultiVector(x).ptr());
        })
        .def("apply",[](Teuchos::RCP<Tpetra_CrsMatrix> &m, Teuchos::RCP<Tpetra_Vector> &x, Teuchos::RCP<Tpetra_Vector> &y, bool trans, double alpha, double beta){
            m->apply(*x, *y, trans ? Teuchos::TRANS : Teuchos::NO_TRANS, alpha, beta);
        })
        .def("solve",[](Teuchos::RCP<Tpetra_CrsMatrix> &m, Teuchos::RCP<Tpetra_Vector> &x, Teuchos::RCP<Tpetra_Vector> &b, bool trans, Teuchos::RCP<Teuchos::ParameterList> solverOptions){
            Albany::MatrixBased_LOWS solver(Albany::createThyraLinearOp(m));
            solver.initializeSolver(solverOptions);
            Thyra::solve(solver,trans ? Thyra::TRANS : Thyra::NOTRANS, *Albany::createConstThyraMultiVector(b), Albany::createThyraMultiVector(x).ptr());
        });
}
