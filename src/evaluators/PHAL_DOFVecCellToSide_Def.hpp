//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_ParameterList.hpp"
#include "Phalanx_DataLayout.hpp"

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits>
DOFVecCellToSide<EvalT, Traits>::
DOFVecCellToSide(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Albany::Layouts>& dl) :
  sideSetName (p.get<std::string> ("Side Set Name")),
  val_cell    (p.get<std::string> ("Cell Variable Name"), dl->node_vector),
  val_side    (p.get<std::string> ("Side Variable Name"), dl->side_node_vector )
{
  this->addDependentField(val_cell);
  this->addEvaluatedField(val_side);

  this->setName("DOFVecCellToSide" );

  std::vector<PHX::DataLayout::size_type> dims;
  dl->side_node_qp_gradient->dimensions(dims);
  int numSides = dims[1];
  numSideNodes = dims[2];
  int sideDim  = dims[4];

  dl->node_vector->dimensions(dims);
  vecDim = dims[2];

  Teuchos::RCP<shards::CellTopology> cellType;
  cellType = p.get<Teuchos::RCP <shards::CellTopology> > ("Cell Type");

  sideNodes.resize(numSides);
  for (int side=0; side<numSides; ++side)
  {
    // Need to get the subcell exact count, since different sides may have different number of nodes (e.g., Wedge)
    int thisSideNodes = cellType->getNodeCount(sideDim,side);
    sideNodes[side].resize(thisSideNodes);
    for (int node=0; node<thisSideNodes; ++node)
    {
      sideNodes[side][node] = cellType->getNodeMap(sideDim,side,node);
    }
  }
}

//**********************************************************************
template<typename EvalT, typename Traits>
void DOFVecCellToSide<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val_cell,fm);
  this->utils.setFieldData(val_side,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void DOFVecCellToSide<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  if (workset.sideSets->find(sideSetName)==workset.sideSets->end())
    return;

  const std::vector<Albany::SideStruct>& sideSet = workset.sideSets->at(sideSetName);
  for (auto const& it_side : sideSet)
  {
    // Get the local data of side and cell
    const int cell = it_side.elem_LID;
    const int side = it_side.side_local_id;

    for (int dim=0; dim<vecDim; ++dim)
    {
      for (int node=0; node<numSideNodes; ++node)
      {
        val_side(cell,side,node,dim) = val_cell(cell,sideNodes[side][node],dim);
      }
    }
  }
}

} // Namespace PHAL
