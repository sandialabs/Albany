//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits>
MapToPhysicalFrameSide<EvalT, Traits>::
MapToPhysicalFrameSide(const Teuchos::ParameterList& p,
                       const Teuchos::RCP<Albany::Layouts>& dl_side)
{
  sideSetName = p.get<std::string>("Side Set Name");

  TEUCHOS_TEST_FOR_EXCEPTION (!dl_side->isSideLayouts, Teuchos::Exceptions::InvalidParameter,
                              "Error! The layouts structure does not appear to be that of a side set.\n");

  coords_side_vertices = decltype(coords_side_vertices)(
      p.get<std::string>("Coordinate Vector Vertex Name"), dl_side->vertices_vector);
  coords_side_qp = decltype(coords_side_qp)(
      p.get<std::string>("Coordinate Vector QP Name"), dl_side->qp_coords);

  this->addDependentField(coords_side_vertices);
  this->addEvaluatedField(coords_side_qp);

  // Get Dimensions
  int numSides = dl_side->qp_coords->dimension(1);
  numSideQPs   = dl_side->qp_coords->dimension(2);
  numDim       = dl_side->qp_coords->dimension(3);
  int sideDim  = numDim-1;

  // Compute cubature points in reference elements
  auto cubature = p.get<Teuchos::RCP<Intrepid2::Cubature<PHX::Device>>>("Cubature");
  Kokkos::DynRankView<RealType, PHX::Device> ref_cub_points, ref_weights;
  ref_cub_points = Kokkos::DynRankView<RealType, PHX::Device>("XXX", numSideQPs,sideDim);
  ref_weights = Kokkos::DynRankView<RealType, PHX::Device>("XXX", numSideQPs); // Not needed per se, but need to be passed to the following function call
  cubature->getCubature(ref_cub_points, ref_weights);

  // Index of the vertices on the sides in the numeration of the cell
  Teuchos::RCP<shards::CellTopology> cellType = p.get<Teuchos::RCP <shards::CellTopology> > ("Cell Type");
  numSideVertices.resize(numSides);
  phi_at_cub_points.resize(numSides);
  for (int side=0; side<numSides; ++side)
  {
    // Since sides may be different (and we don't know on which local side this side set is), we build one basis per side.
    auto sideBasis = Albany::getIntrepid2Basis (*cellType->getCellTopologyData(sideDim,side));
    numSideVertices[side] = cellType->getVertexCount(sideDim,side);
    phi_at_cub_points[side] = Kokkos::DynRankView<RealType, PHX::Device>("XXX", numSideVertices[side],numSideQPs);
    sideBasis->getValues(phi_at_cub_points[side], ref_cub_points, Intrepid2::OPERATOR_VALUE);
  }

  this->setName("MapToPhysicalFrameSide" );
}

//**********************************************************************
template<typename EvalT, typename Traits>
void MapToPhysicalFrameSide<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(coords_side_vertices,fm);
  this->utils.setFieldData(coords_side_qp,fm);
}

template<typename EvalT, typename Traits>
void MapToPhysicalFrameSide<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  if (workset.sideSets->find(sideSetName)==workset.sideSets->end())
    return;

  const std::vector<Albany::SideStruct>& sideSet = workset.sideSets->at(sideSetName);
  for (auto const& it_side : sideSet)
  {
    // Get the local data of side and cell
    const int cell = it_side.elem_LID;
    const int side = it_side.side_local_id;

    for (int qp=0; qp<numSideQPs; ++qp)
    {
      for (int dim=0; dim<numDim; ++dim)
      {
        coords_side_qp(cell,side,qp,dim) = 0;
        for (int v=0; v<numSideVertices[side]; ++v)
          coords_side_qp(cell,side,qp,dim) += coords_side_vertices(cell,side,v,dim)*phi_at_cub_points[side](v,qp);
      }
    }
  }
}

} // Namespace PHAL
