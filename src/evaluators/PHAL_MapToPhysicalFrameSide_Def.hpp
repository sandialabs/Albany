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
MapToPhysicalFrameSide<EvalT, Traits>::
MapToPhysicalFrameSide(const Teuchos::ParameterList& p,
                       const Teuchos::RCP<Albany::Layouts>& dl) :
  coords_cell_vertices  (p.get<std::string>  ("Coordinate Vector Name"), dl->vertices_vector),
  coords_side_qp        (p.get<std::string>  ("Coordinate Side QP Vector Name"), dl->side_qp_coords)
{
  this->addDependentField(coords_cell_vertices);
  this->addEvaluatedField(coords_side_qp);

  // Get Dimensions
  int numSides = dl->side_qp_coords->dimension(1);
  numSideQPs   = dl->side_qp_coords->dimension(2);
  cellDim      = dl->side_qp_coords->dimension(3);
  int sideDim  = cellDim-1;

  // Compute cubature points in reference elements
  Teuchos::RCP<Intrepid::Cubature<RealType> > cubature = p.get<Teuchos::RCP <Intrepid::Cubature<RealType> > >("Cubature");
  Intrepid::FieldContainer<RealType> ref_cub_points, ref_weights;
  ref_cub_points.resize(numSideQPs,sideDim);
  ref_weights.resize(numSideQPs); // Not needed per se, but need to be passed to the following function call
  cubature->getCubature(ref_cub_points, ref_weights);

  // Index of the vertices on the sides in the numeration of the cell
  Teuchos::RCP<shards::CellTopology> cellType = p.get<Teuchos::RCP <shards::CellTopology> > ("Cell Type");
  numSideVertices.resize(numSides);
  sideVertices.resize(numSides);
  phi_at_cub_points.resize(numSides);
  shards::CellTopology baseCellType(cellType->getBaseCellTopologyData());
  for (int side=0; side<numSides; ++side)
  {
    // Need to get the subcell exact count, since different sides may have different number of vertices (e.g., Wedge)
    numSideVertices[side] = baseCellType.getVertexCount(sideDim,side);
    sideVertices[side].resize(numSideVertices[side]);
    for (int vertex=0; vertex<numSideVertices[side]; ++vertex)
    {
      // Since it's the base cell type, node=vertex and we can use getNodeMap (there's no getVertexMap)
      sideVertices[side][vertex] = baseCellType.getNodeMap(sideDim,side,vertex);
    }

    // Since sides may be different (and we don't know on which local side we are), we build one basis per side.
    auto sideBasis = Albany::getIntrepidBasis (*baseCellType.getCellTopologyData(sideDim,side));
    phi_at_cub_points[side].resize(numSideVertices[side],numSideQPs);
    sideBasis->getValues(phi_at_cub_points[side], ref_cub_points, Intrepid::OPERATOR_VALUE);
  }

  sideSetName = p.get<std::string>("Side Set Name");

  this->setName("MapToPhysicalFrameSide" );
}

//**********************************************************************
template<typename EvalT, typename Traits>
void MapToPhysicalFrameSide<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(coords_cell_vertices,fm);
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
      for (int dim=0; dim<cellDim; ++dim)
      {
        coords_side_qp(cell,side,qp,dim) = 0;
        for (int v=0; v<numSideVertices[side]; ++v)
          coords_side_qp(cell,side,qp,dim) += coords_cell_vertices(cell,sideVertices[side][v],dim)*phi_at_cub_points[side](v,qp);
      }
    }
  }
}

} // Namespace PHAL
