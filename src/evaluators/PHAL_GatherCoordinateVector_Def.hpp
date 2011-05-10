/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


#include <vector>
#include <string>

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

namespace PHAL {

template<typename EvalT, typename Traits>
GatherCoordinateVector<EvalT, Traits>::
GatherCoordinateVector(const Teuchos::ParameterList& p) :
  coordVec         (p.get<std::string>                   ("Coordinate Vector Name"),
                    p.get<Teuchos::RCP<PHX::DataLayout> >("Coordinate Data Layout") ),
  numVertices(0), numDim(0)
{  
  if (p.isType<bool>("Periodic BC")) periodic = p.get<bool>("Periodic BC");
  else periodic = false;

  this->addEvaluatedField(coordVec);
  this->setName("Gather Coordinate Vector"+PHX::TypeString<EvalT>::value);
}

// **********************************************************************
template<typename EvalT, typename Traits> 
void GatherCoordinateVector<EvalT, Traits>::postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(coordVec,fm);

  typename std::vector< typename PHX::template MDField<MeshScalarT,Cell,Vertex,Dim>::size_type > dims;
  coordVec.dimensions(dims); //get dimensions

  numVertices = dims[1];
  numDim = dims[2];
}

// **********************************************************************
template<typename EvalT, typename Traits>
void GatherCoordinateVector<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{ 
  unsigned int numCells = workset.numCells;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > wsCoords = workset.wsCoords;

  for (std::size_t cell=0; cell < numCells; ++cell) {
    const Teuchos::ArrayRCP<int>& nodeID  = workset.wsElNodeID[cell];
    for (std::size_t node = 0; node < numVertices; ++node) {
      const int row_loc = 3*nodeID[node];

      for (std::size_t eq=0; eq < numDim; ++eq) { 
        coordVec(cell,node,eq) = wsCoords[cell][node][eq]; 
      }
    }
  }
}
// **********************************************************************
template<typename Traits>
GatherCoordinateVector<PHAL::AlbanyTraits::Tangent, Traits>::
GatherCoordinateVector(const Teuchos::ParameterList& p) :
  coordVec         (p.get<std::string>                   ("Coordinate Vector Name"),
                    p.get<Teuchos::RCP<PHX::DataLayout> >("Coordinate Data Layout") ),
  numVertices(0), numDim(0)
{  
  if (p.isType<bool>("Periodic BC")) periodic = p.get<bool>("Periodic BC");
  else periodic = false;

  this->addEvaluatedField(coordVec);
  this->setName("Gather Coordinate Vector"
                +PHX::TypeString<PHAL::AlbanyTraits::Tangent>::value);
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

  numVertices = dims[1];
  numDim = dims[2];
}
// **********************************************************************
template<typename Traits>
void GatherCoordinateVector<PHAL::AlbanyTraits::Tangent, Traits>::
evaluateFields(typename Traits::EvalData workset)
{ 
  const Teuchos::ArrayRCP<double> &coordinates = workset.coordinates;
  unsigned int numCells = workset.numCells;

  const Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> >& coord_derivs = workset.coord_derivs;
  std::vector<int> *coord_deriv_indices = workset.coord_deriv_indices;
  int numShapeDerivs = coord_derivs.size();
  int numParams = workset.num_cols_p;

  for (std::size_t cell=0; cell < numCells; ++cell) {
    const Teuchos::ArrayRCP<int>& nodeID  = workset.wsElNodeID[cell];
    for (std::size_t node = 0; node < numVertices; ++node) {
      const int row_loc = 3*nodeID[node];

      for (std::size_t eq=0; eq < numDim; ++eq) { 
        coordVec(cell,node,eq) = FadType(numParams, coordinates[row_loc+eq]); 
        for (int j=0; j < numShapeDerivs; ++j) { 
          coordVec(cell,node,eq).fastAccessDx((*coord_deriv_indices)[j]) 
               =  coord_derivs[j][row_loc+eq];
//               =  ws_coord_derivs[j][cell][node][eq];
        }
      }
    }
  }
}
}

