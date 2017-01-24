//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits>
MapToPhysicalFrame<EvalT, Traits>::
MapToPhysicalFrame(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
  coords_vertices  (p.get<std::string>  ("Coordinate Vector Name"), dl->vertices_vector),
  cubature         (p.get<Teuchos::RCP <Intrepid2::Cubature<PHX::Device> > >("Cubature")),
  intrepidBasis (p.get<Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> > > ("Intrepid2 Basis") ),
  cellType         (p.get<Teuchos::RCP <shards::CellTopology> > ("Cell Type")),
  coords_qp        (p.get<std::string>  ("Coordinate Vector Name"), dl->qp_gradient)
{
  this->addDependentField(coords_vertices.fieldTag());
  this->addEvaluatedField(coords_qp);

  // Get Dimensions
  std::vector<PHX::DataLayout::size_type> dims;
  dl->qp_gradient->dimensions(dims);

  numQPs = dims[1];
  numDim = dims[2];

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

  // Compute cubature points in reference elements
  refPoints = Kokkos::DynRankView<RealType, PHX::Device>("XXX", numQPs, numDim);
  refWeights = Kokkos::DynRankView<RealType, PHX::Device>("XXX", numQPs);
  cubature->getCubature(refPoints, refWeights); 
}
//**********************************************************************
template <class Scalar, class ArrayPhysPoint, class ArrayRefPoint, class ArrayCell>
void mapToPhysicalFrame(ArrayPhysPoint      &        physPoints,
                                           const ArrayRefPoint &        refPoints,
                                           const ArrayCell     &        cellWorkset,
                                           const shards::CellTopology & cellTopo,
                                           const int &                  whichCell)
{
  int spaceDim  = (int)cellTopo.getDimension();
  int numCells  = cellWorkset.dimension(0);
  //points can be rank-2 (P,D), or rank-3 (C,P,D)
  int numPoints = (refPoints.rank() == 2) ? refPoints.dimension(0) : refPoints.dimension(1);

  // Initialize physPoints
  for(int i = 0; i < physPoints.dimentions(0); i++)
    for(int j = 0; j < physPoints.dimentions(1); j++)  
      for(int k = 0; k < physPoints.dimentions(2); k++)
         physPoints(i,j,k) = 0.0;
  

}
//**********************************************************************
template<typename EvalT, typename Traits>
void MapToPhysicalFrame<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  if (intrepidBasis != Teuchos::null){ 
    Intrepid2::CellTools<PHX::Device>::mapToPhysicalFrame
         (coords_qp.get_view(), refPoints, coords_vertices.get_view(), intrepidBasis);
  }
  else {
    Intrepid2::CellTools<PHX::Device>::mapToPhysicalFrame
         (coords_qp.get_view(), refPoints, coords_vertices.get_view(), *cellType);
  }
  
}

//**********************************************************************
}

