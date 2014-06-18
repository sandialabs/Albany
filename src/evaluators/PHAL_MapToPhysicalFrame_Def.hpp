//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits>
MapToPhysicalFrame<EvalT, Traits>::
MapToPhysicalFrame(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
  coords_vertices  (p.get<std::string>  ("Coordinate Vector Name"), dl->vertices_vector),
  cubature         (p.get<Teuchos::RCP <Intrepid::Cubature<RealType> > >("Cubature")),
  cellType         (p.get<Teuchos::RCP <shards::CellTopology> > ("Cell Type")),
  coords_qp        (p.get<std::string>  ("Coordinate Vector Name"), dl->qp_gradient)
{
  this->addDependentField(coords_vertices);
  this->addEvaluatedField(coords_qp);

  // Get Dimensions
  std::vector<PHX::DataLayout::size_type> dims;
  dl->qp_gradient->dimensions(dims);

  // Compute cubature points in reference elements
  refPoints.resize(dims[1],dims[2]);
  refWeights.resize(dims[1]);
  cubature->getCubature(refPoints, refWeights); 

  this->setName("MapToPhysicalFrame" );
}

//**********************************************************************
template<typename EvalT, typename Traits>
void MapToPhysicalFrame<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(coords_vertices,fm);
  this->utils.setFieldData(coords_qp,fm);

}

//**********************************************************************
template<typename EvalT, typename Traits>
void MapToPhysicalFrame<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  
  Intrepid::CellTools<RealType>::mapToPhysicalFrame
       (coords_qp, refPoints, coords_vertices, *cellType);
}

//**********************************************************************
}

