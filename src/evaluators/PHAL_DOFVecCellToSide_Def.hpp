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
    sideNodes[side].resize(numSideNodes);
    for (int node=0; node<numSideNodes; ++node)
      sideNodes[side][node] = cellType->getNodeMap(sideDim,side,node);
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
  const Albany::SideSetList& ssList = *(workset.sideSets);
  Albany::SideSetList::const_iterator it_ss = ssList.find(sideSetName);

  if (it_ss==ssList.end())
    return;

  const std::vector<Albany::SideStruct>& sideSet = it_ss->second;
  std::vector<Albany::SideStruct>::const_iterator iter_s;
  for (iter_s=sideSet.begin(); iter_s!=sideSet.end(); ++iter_s)
  {
    // Get the local data of side and cell
    const int cell = iter_s->elem_LID;
    const int side = iter_s->side_local_id;

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
