//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <vector>
#include <string>

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

namespace PHAL {

template<typename EvalT, typename Traits>
GatherCoordinateVector<EvalT, Traits>::
GatherCoordinateVector(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
  coordVec  (p.get<std::string> ("Coordinate Vector Name"), dl->vertices_vector ),
  worksetSize(0), numVertices(0), numDim(0)
{  
  if (p.isType<bool>("Periodic BC")) periodic = p.get<bool>("Periodic BC");
  else periodic = false;

  if (p.isType<std::string>("Current Displacement Vector Name")){
    std::string strDispVec = p.get<std::string>("Current Displacement Vector Name");
    dispVecName = Teuchos::rcp( new std::string(strDispVec) );
  }
    
  this->addEvaluatedField(coordVec);
  this->setName("Gather Coordinate Vector"+PHX::print<EvalT>());
}

template<typename EvalT, typename Traits>
GatherCoordinateVector<EvalT, Traits>::
GatherCoordinateVector(const Teuchos::ParameterList& p) :
  coordVec         (p.get<std::string>                   ("Coordinate Vector Name"),
                    p.get<Teuchos::RCP<PHX::DataLayout> >("Coordinate Data Layout") ),
  worksetSize(0), numVertices(0), numDim(0)
{  
  if (p.isType<bool>("Periodic BC")) periodic = p.get<bool>("Periodic BC");
  else periodic = false;

  this->addEvaluatedField(coordVec);
  this->setName("Gather Coordinate Vector"+PHX::print<EvalT>());
}

// **********************************************************************
template<typename EvalT, typename Traits> 
void GatherCoordinateVector<EvalT, Traits>::postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(coordVec,fm);

  typename std::vector< typename PHX::template MDField<MeshScalarT,Cell,Vertex,Dim>::size_type > dims;
  coordVec.dimensions(dims); //get dimensions

  worksetSize = dims[0];
  numVertices = dims[1];
  numDim = dims[2];

  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
  if (d.memoizer_active()) memoizer.enable_memoizer();
}

// **********************************************************************
template<typename EvalT, typename Traits>
void GatherCoordinateVector<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  if (memoizer.have_saved_data(workset,this->evaluatedFields())) return;

  unsigned int numCells = workset.numCells;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<const double*> > wsCoords = workset.wsCoords;

#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  if( dispVecName.is_null() ){
    for (std::size_t cell=0; cell < numCells; ++cell) {
      for (std::size_t node = 0; node < numVertices; ++node) {
        for (std::size_t eq=0; eq < numDim; ++eq) { 
          coordVec(cell,node,eq) = wsCoords[cell][node][eq]; 
        }
      }
    }

    // Since Intrepid2 will later perform calculations on the entire workset size
    // and not just the used portion, we must fill the excess with reasonable 
    // values. Leaving this out leads to calculations on singular elements.
    for (std::size_t cell=numCells; cell < worksetSize; ++cell) {
      for (std::size_t node = 0; node < numVertices; ++node) {
        for (std::size_t eq=0; eq < numDim; ++eq) { 
          coordVec(cell,node,eq) = coordVec(0,node,eq); 
        }
      }
    }
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION((workset.stateArrayPtr->count(*dispVecName)==1), std::logic_error,
           "Error: cannot locate " << *dispVecName << " in PHAL::GatherCoordinateVector\n");

    auto dVec = workset.stateArrayPtr->at(*dispVecName).host();

    for (std::size_t cell=0; cell < numCells; ++cell) {
      for (std::size_t node = 0; node < numVertices; ++node) {
        for (std::size_t eq=0; eq < numDim; ++eq) { 
          coordVec(cell,node,eq) = wsCoords[cell][node][eq] + dVec(cell,node,eq);
        }
      }
    }

    // Since Intrepid2 will later perform calculations on the entire workset size
    // and not just the used portion, we must fill the excess with reasonable 
    // values. Leaving this out leads to calculations on singular elements.
    for (std::size_t cell=numCells; cell < worksetSize; ++cell) {
      for (std::size_t node = 0; node < numVertices; ++node) {
        for (std::size_t eq=0; eq < numDim; ++eq) { 
          coordVec(cell,node,eq) = coordVec(0,node,eq) + dVec(cell,node,eq);
        }
      }
    }
  }
#else
 typedef Kokkos::View<MeshScalarT***,PHX::Device> view_type;
 typedef typename view_type::HostMirror host_view_type;
  
// host_view_type coordVecHost = Kokkos::create_mirror_view (coordVec.get_view());
 host_view_type coordVecHost = Kokkos::create_mirror_view (coordVec.get_static_view());

  if( dispVecName.is_null() ){
    for (std::size_t cell=0; cell < numCells; ++cell) {
      for (std::size_t node = 0; node < numVertices; ++node) {
        for (std::size_t eq=0; eq < numDim; ++eq) {
          coordVecHost(cell,node,eq) = wsCoords[cell][node][eq];
        }
      }
    }

    for (std::size_t cell=numCells; cell < worksetSize; ++cell) {
      for (std::size_t node = 0; node < numVertices; ++node) {
        for (std::size_t eq=0; eq < numDim; ++eq) {
          coordVecHost(cell,node,eq) = coordVecHost(0,node,eq);
        }
      }
    }
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION((workset.stateArrayPtr->count(*dispVecName)==1), std::logic_error,
           "Error: cannot locate " << *dispVecName << " in PHAL::GatherCoordinateVector\n");

    auto dVec = workset.stateArrayPtr->at(*dispVecName).host();
    for (std::size_t cell=0; cell < numCells; ++cell) {
      for (std::size_t node = 0; node < numVertices; ++node) {
        for (std::size_t eq=0; eq < numDim; ++eq) {
          coordVecHost(cell,node,eq) = wsCoords[cell][node][eq] + dVec(cell,node,eq);
        }
      }
    }
   for (std::size_t cell=numCells; cell < worksetSize; ++cell) {
      for (std::size_t node = 0; node < numVertices; ++node) {
        for (std::size_t eq=0; eq < numDim; ++eq) {
          coordVecHost(cell,node,eq) = coordVecHost(0,node,eq) + dVec(cell,node,eq);
        }
      }
    }
  } 
  // Kokkos::deep_copy (coordVec.get_view(), coordVecHost);
  Kokkos::deep_copy (coordVec.get_static_view(), coordVecHost);
#endif // ALBANY_KOKKOS_UNDER_DEVELOPMENT
}

} // namespace PHAL
