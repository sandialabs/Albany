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
  numVertices(0), numDim(0), worksetSize(0)
{  
  if (p.isType<bool>("Periodic BC")) periodic = p.get<bool>("Periodic BC");
  else periodic = false;

  if (p.isType<std::string>("Current Displacement Vector Name")){
    std::string strDispVec = p.get<std::string>("Current Displacement Vector Name");
    dispVecName = Teuchos::rcp( new std::string(strDispVec) );
  }
    

  this->addEvaluatedField(coordVec);
  this->setName("Gather Coordinate Vector" );
}

template<typename EvalT, typename Traits>
GatherCoordinateVector<EvalT, Traits>::
GatherCoordinateVector(const Teuchos::ParameterList& p) :
  coordVec         (p.get<std::string>                   ("Coordinate Vector Name"),
                    p.get<Teuchos::RCP<PHX::DataLayout> >("Coordinate Data Layout") ),
  numVertices(0), numDim(0), worksetSize(0)
{  
  if (p.isType<bool>("Periodic BC")) periodic = p.get<bool>("Periodic BC");
  else periodic = false;

  this->addEvaluatedField(coordVec);
  this->setName("Gather Coordinate Vector" );
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
}

// **********************************************************************
template<typename EvalT, typename Traits>
void GatherCoordinateVector<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{ 
  unsigned int numCells = workset.numCells;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > wsCoords = workset.wsCoords;

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
    Albany::StateArray::const_iterator it;
    it = workset.stateArrayPtr->find(*dispVecName);

    TEUCHOS_TEST_FOR_EXCEPTION((it == workset.stateArrayPtr->end()), std::logic_error,
           std::endl << "Error: cannot locate " << *dispVecName << " in PHAL_GatherCoordinateVector_Def" << std::endl);

    Albany::MDArray dVec = it->second;

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
    Albany::StateArray::const_iterator it;
    it = workset.stateArrayPtr->find(*dispVecName);

    TEUCHOS_TEST_FOR_EXCEPTION((it == workset.stateArrayPtr->end()), std::logic_error,
           std::endl << "Error: cannot locate " << *dispVecName << " in PHAL_GatherCoordinateVector_Def" << std::endl);

    Albany::MDArray dVec = it->second;

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
  Kokkos::deep_copy (coordVec.get_static_view(), coordVecHost);

#endif
}
// **********************************************************************
#if defined(ALBANY_MESH_DEPENDS_ON_PARAMETERS) || defined(ALBANY_MESH_DEPENDS_ON_SOLUTION)
template<typename Traits>
GatherCoordinateVector<PHAL::AlbanyTraits::Tangent, Traits>::
GatherCoordinateVector(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
  coordVec         (p.get<std::string> ("Coordinate Vector Name"), dl->vertices_vector ),
  numVertices(0), numDim(0), worksetSize(0)
{  
  if (p.isType<bool>("Periodic BC")) periodic = p.get<bool>("Periodic BC");
  else periodic = false;

  this->addEvaluatedField(coordVec);
  this->setName("Gather Coordinate Vector Tangent");
}

template<typename Traits>
GatherCoordinateVector<PHAL::AlbanyTraits::Tangent, Traits>::
GatherCoordinateVector(const Teuchos::ParameterList& p) :
  coordVec         (p.get<std::string>                   ("Coordinate Vector Name"),
                    p.get<Teuchos::RCP<PHX::DataLayout> >("Coordinate Data Layout") ),
  numVertices(0), numDim(0), worksetSize(0)
{  
  if (p.isType<bool>("Periodic BC")) periodic = p.get<bool>("Periodic BC");
  else periodic = false;

  this->addEvaluatedField(coordVec);
  this->setName("Gather Coordinate Vector  Tangent");
}

// **********************************************************************
template<typename Traits> 
void GatherCoordinateVector<PHAL::AlbanyTraits::Tangent, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(coordVec,fm);

  typename std::vector< typename PHX::template MDField<MeshScalarT,Cell,Vertex,Dim>::size_type > dims;
  coordVec.dimensions(dims); //get dimensions

  worksetSize = dims[0];
  numVertices = dims[1];
  numDim = dims[2];
}
// **********************************************************************
template<typename Traits>
void GatherCoordinateVector<PHAL::AlbanyTraits::Tangent, Traits>::
evaluateFields(typename Traits::EvalData workset)
{ 
  unsigned int numCells = workset.numCells;

  Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > wsCoords = workset.wsCoords;
  const Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> > > > & ws_coord_derivs = workset.ws_coord_derivs;
  std::vector<int> *coord_deriv_indices = workset.coord_deriv_indices;
  int numShapeDerivs = ws_coord_derivs.size();
  int numParams = workset.num_cols_p;

  for (std::size_t cell=0; cell < numCells; ++cell) {
    for (std::size_t node = 0; node < numVertices; ++node) {
      for (std::size_t eq=0; eq < numDim; ++eq) { 
        coordVec(cell,node,eq) = TanFadType(numParams, wsCoords[cell][node][eq]); 
        for (int j=0; j < numShapeDerivs; ++j) { 
          coordVec(cell,node,eq).fastAccessDx((*coord_deriv_indices)[j]) 
               =  ws_coord_derivs[j][cell][node][eq];
        }
      }
    }
  }

  // Since Intrepid2 will later perform calculations on the entire workset size
  // and not just the used portion, we must fill the excess with reasonable 
  // values. Leaving this out leads to calculations on singular elements.
  //
  for (std::size_t cell=numCells; cell < worksetSize; ++cell) {
    for (std::size_t node = 0; node < numVertices; ++node) {
      for (std::size_t eq=0; eq < numDim; ++eq) { 
        coordVec(cell,node,eq) = coordVec(0,node,eq); 
      }
    }
  }
}
#endif
}
