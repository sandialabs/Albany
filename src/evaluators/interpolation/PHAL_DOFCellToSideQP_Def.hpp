//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_ParameterList.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Shards_CellTopology.hpp"

#include "Albany_DiscretizationUtils.hpp"
#include "PHAL_DOFCellToSideQP.hpp"

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
DOFCellToSideQPBase<EvalT, Traits, ScalarT>::
DOFCellToSideQPBase(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl) :
  sideSetName (p.get<std::string> ("Side Set Name"))
{
  TEUCHOS_TEST_FOR_EXCEPTION (dl->side_layouts.find(sideSetName)==dl->side_layouts.end(), std::runtime_error,
                              "Error! Layout for side set " << sideSetName << " not found.\n");

  if (p.isType<bool>("Enable Memoizer") && p.get<bool>("Enable Memoizer")) {
    memoizer.enable_memoizer();
  }

  Teuchos::RCP<Albany::Layouts> dl_side = dl->side_layouts.at(sideSetName);
  std::string layout_str = p.get<std::string>("Data Layout");

  if (layout_str=="Cell Scalar") {
    val_cell = decltype(val_cell)(p.get<std::string> ("Cell Variable Name"),
        dl->cell_scalar2);
    val_side_qp = decltype(val_side_qp)(p.get<std::string> ("Side Variable Name"),
        dl_side->qp_scalar);

    layout = CELL_SCALAR;
  } else if (layout_str=="Cell Vector") {
    val_cell = decltype(val_cell)(p.get<std::string> ("Cell Variable Name"),
        dl->cell_vector);
    val_side_qp = decltype(val_side_qp)(p.get<std::string> ("Side Variable Name"),
        dl_side->qp_vector);

    layout = CELL_VECTOR;
  } else if (layout_str=="Cell Tensor") {
    val_cell = decltype(val_cell)(p.get<std::string> ("Cell Variable Name"),
        dl->cell_tensor);
    val_side_qp = decltype(val_side_qp)(p.get<std::string> ("Side Variable Name"),
        dl_side->qp_tensor);

    layout = CELL_TENSOR;
  } else if (layout_str=="Node Scalar") {
    val_cell = decltype(val_cell)(p.get<std::string> ("Cell Variable Name"),
        dl->node_scalar);
    val_side_qp = decltype(val_side_qp)(p.get<std::string> ("Side Variable Name"),
        dl_side->qp_scalar);

    layout = NODE_SCALAR;
  } else if (layout_str=="Node Vector") {
    val_cell = decltype(val_cell)(p.get<std::string> ("Cell Variable Name"),
        dl->node_vector);
    val_side_qp = decltype(val_side_qp)(p.get<std::string> ("Side Variable Name"),
        dl_side->qp_vector);

    layout = NODE_VECTOR;
  } else if (layout_str=="Node Tensor") {
    val_cell = decltype(val_cell)(p.get<std::string> ("Cell Variable Name"),
        dl->node_tensor);
    val_side_qp = decltype(val_side_qp)(p.get<std::string> ("Side Variable Name"),
        dl_side->qp_tensor);

    layout = NODE_TENSOR;
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, "Error! Invalid field layout.\n");
  }

  num_side_nodes = dl_side->node_scalar->dimension(2);

  this->addDependentField(val_cell);
  this->addEvaluatedField(val_side_qp);

  this->setName("DOFCellToSideQP");

  if (layout==NODE_SCALAR || layout==NODE_VECTOR || layout==NODE_TENSOR) {
    Teuchos::RCP<shards::CellTopology> cellType;
    cellType = p.get<Teuchos::RCP <shards::CellTopology> > ("Cell Type");

    int sideDim = dl_side->cell_gradient->dimension(2);
    int numSides = dl_side->cell_gradient->dimension(1);
    sideNodes.resize(numSides);
    for (int side=0; side<numSides; ++side) {
      // Need to get the subcell exact count, since different sides may have different number of nodes (e.g., Wedge)
      int thisSideNodes = cellType->getNodeCount(sideDim,side);
      sideNodes[side].resize(thisSideNodes);
      for (int node=0; node<thisSideNodes; ++node) {
        sideNodes[side][node] = cellType->getNodeMap(sideDim,side,node);
      }
    }

    BF = decltype(BF)(p.get<std::string> ("BF Name"), dl_side->node_qp_scalar);
    this->addDependentField(BF);
  }
}

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
void DOFCellToSideQPBase<EvalT, Traits, ScalarT>::
postRegistrationSetup(typename Traits::SetupData /* d */,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val_cell,fm);
  this->utils.setFieldData(val_side_qp,fm);
  if (layout==NODE_SCALAR || layout==NODE_VECTOR || layout==NODE_TENSOR) {
    this->utils.setFieldData(BF,fm);
  }

  val_side_qp.dimensions(dims_side);
}

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
void DOFCellToSideQPBase<EvalT, Traits, ScalarT>::
evaluateFields(typename Traits::EvalData workset)
{
  if (workset.sideSets->find(sideSetName)==workset.sideSets->end()) {
    return;
  }

  if (memoizer.have_stored_data(workset)) {
    return;
  }

  const std::vector<Albany::SideStruct>& sideSet = workset.sideSets->at(sideSetName);
  for (auto const& it_side : sideSet) {
    // Get the local data of side and cell
    const int cell = it_side.elem_LID;
    const int side = it_side.side_local_id;

    switch (layout) {
      case CELL_SCALAR:
        for (int qp=0; qp<dims_side[2]; ++qp) {
          val_side_qp(cell,side,qp) = val_cell(cell);
        }
        break;

      case CELL_VECTOR:
        for (int qp=0; qp<dims_side[2]; ++qp) {
          for (int i=0; i<dims_side[3]; ++i) {
            val_side_qp(cell,side,qp,i) = val_cell(cell,i);
        }}
        break;

      case CELL_TENSOR:
        for (int qp=0; qp<dims_side[2]; ++qp) {
          for (int i=0; i<dims_side[3]; ++i) {
            for (int j=0; j<dims_side[4]; ++j) {
              val_side_qp(cell,side,qp,i,j) = val_cell(cell,i,j);
        }}}
        break;

      case NODE_SCALAR:
        for (int qp=0; qp<dims_side[2]; ++qp) {
          val_side_qp(cell,side,qp) = 0;
          for (int node=0; node<num_side_nodes; ++node) {
            val_side_qp(cell,side,qp) += val_cell(cell,sideNodes[side][node]) * BF(cell,side,node,qp);
        }}
        break;

      case NODE_VECTOR:
        for (int qp=0; qp<dims_side[2]; ++qp) {
          for (int i=0; i<dims_side[3]; ++i) {
            val_side_qp(cell,side,qp,i) = 0;
            for (int node=0; node<num_side_nodes; ++node) {
                val_side_qp(cell,side,qp,i) += val_cell(cell,sideNodes[side][node],i) * BF(cell,side,node,qp);
        }}}
        break;
      case NODE_TENSOR:
        for (int qp=0; qp<dims_side[2]; ++qp) {
          for (int i=0; i<dims_side[3]; ++i) {
            for (int j=0; j<dims_side[4]; ++j) {
              val_side_qp(cell,side,qp,i,j) = 0;
              for (int node=0; node<num_side_nodes; ++node) {
                val_side_qp(cell,side,qp,i,j) += val_cell(cell,sideNodes[side][node],i,j) * BF(cell,side,node,qp);
        }}}}
        break;
      default:
        TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Invalid layout (this error should have happened earlier though).\n");
    }
  }
}

} // Namespace PHAL
