//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "PHAL_ScatterSideEqnResidual.hpp"
#include "Albany_Macros.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_AbstractDiscretization.hpp"
#include "Albany_DistributedParameterLibrary.hpp"
#include "Albany_GlobalLocalIndexer.hpp"

// **********************************************************************
// Base Class Generic Implemtation
// **********************************************************************
namespace PHAL {

template<typename EvalT, typename Traits>
ScatterSideEqnResidualBase<EvalT, Traits>::
ScatterSideEqnResidualBase (const Teuchos::ParameterList& p,
                            const Teuchos::RCP<Albany::Layouts>& dl)
 : sideSetName ( p.get<std::string>("Side Set Name") )
 , residualsAreVolumeFields ( p.get<bool>("Residuals Are Volume Fields") )
 , tensorRank ( p.get<int>("Tensor Rank") )
{
  // Sanity check
  TEUCHOS_TEST_FOR_EXCEPTION ( dl->isSideLayouts, std::logic_error,
    "Error! The Layouts structure must *not* be that of a side set.\n");

  // Name of the PHX computed tag. This is not really a field, just a tag
  // whose evaluation must be requested in order to trigger the whole evaluation tree
  std::string fieldName;
  if (p.isType<std::string>("Scatter Field Name")) {
    fieldName = p.get<std::string>("Scatter Field Name");
  } else {
    fieldName = "Scatter";
  }
  scatter_operation = Teuchos::rcp(new PHX::Tag<ScalarT>(fieldName, dl->dummy));

  // Name of the residual fields(s)
  Teuchos::ArrayRCP<std::string> names;
  if (p.isType<Teuchos::ArrayRCP<std::string>>("Residual Names")) {
    names = p.get< Teuchos::ArrayRCP<std::string> >("Residual Names");
  } else if (p.isType<std::string>("Residual Name")) {
    names = Teuchos::ArrayRCP<std::string>(1,p.get<std::string>("Residual Name"));
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error,
        "Error! You must specify either the std::string 'Residual Name',\n"
        "       or the Teuchos::ArrayRCP<std::string> 'Residual Names'.\n");
  }

  // Store information of all sides, since we don't know which local side id this sideset will be
  auto cellType = p.get<Teuchos::RCP <shards::CellTopology> > ("Cell Type");
  int sideDim = cellType->getDimension()-1;
  int numSides = cellType->getSideCount();
  numSideNodes.resize(numSides);
  sideNodes.resize(numSides);
  for (int side=0; side<numSides; ++side) {
    numSideNodes[side] = cellType->getNodeCount(sideDim,side);
    sideNodes[side].resize(numSideNodes[side]);
    for (int node=0; node<numSideNodes[side]; ++node) {
      sideNodes[side][node] = cellType->getNodeMap(sideDim,side,node);
    }
  }
  numCellNodes = dl->node_scalar->extent_int(1);

  auto res_dl = residualsAreVolumeFields ? dl : dl->side_layouts.at(sideSetName);

  using res_type = PHX::MDField<const ScalarT>;
  if (tensorRank == 0 ) {
    // scalar
    numFields = names.size();
    val.resize(numFields);
    for (int eq = 0; eq < numFields; ++eq) {
      val[eq] = res_type(names[eq], res_dl->node_scalar);
      this->addDependentField(val[eq]);
    }
  } else if (tensorRank == 1 ) {
    // vector
    valVec = res_type(names[0], res_dl->node_vector);
    this->addDependentField(valVec);
    numFields = res_dl->node_vector->extent(2);
  } else if (tensorRank == 2 ) {
    // tensor
    valTensor = res_type (names[0], res_dl->node_tensor);
    this->addDependentField(valTensor);
    numDims = res_dl->node_tensor->extent(2);
    numFields = (res_dl->node_tensor->extent(2))*(res_dl->node_tensor->extent(3));
  }

  if (p.isType<int>("Offset of First DOF")) {
    offset = p.get<int>("Offset of First DOF");
  } else {
    offset = 0;
  }

  this->addEvaluatedField(*scatter_operation);

  this->setName(fieldName+PHX::print<EvalT>());
}

// **********************************************************************
template<typename EvalT, typename Traits>
void ScatterSideEqnResidualBase<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& /* fm */)
{
  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
}

template<typename EvalT, typename Traits>
void ScatterSideEqnResidualBase<EvalT, Traits>::
gatherSideSetNodeGIDs (const Albany::AbstractDiscretization& disc) {
  // Note: you cannot call this function on a per-workset basis, since
  //       it is technically possible for this ws to not have any side
  //       on the Eqn sideSet, and yet have a node on it. Consider
  //       the following element pathc:
  //
  //               1-------2-------3
  //                \  A  / \  C  /
  //                 \   /   \   /
  //                  \ /  B  \ /
  //                   4-------5
  //
  //       If the sideset is at the top, and the ws contains B but not
  //       A nor C, then this ws cannot deduce that node 2 is on the
  //       sideset, since it has no side on the sideset (more generally,
  //       none of its side on the sideset contains the node 2).
  //       Therefore, we need to loop over the whole mesh.
  // Note: the scenario above could still happen at the MPI decomp level;
  //       that is, rank 0 might own element B but not A or C, so it would
  //       not be able to deduce that 2 is on the sideset without a global
  //       all-to-all communication.
  //       However, this scenario cannot happen for the basal sideset of an extruded
  //       mesh (since each column is on a single rank), which is the main
  //       case we are interested in right now. If you are solving a different
  //       problem, or using a different mesh, double check that you are
  //       still ensuring the following: if an MPI rank has a node on the
  //       sideset (in the owned+shared map), then it also has a side containing
  //       that node on that sideset.

  const auto& wsElNodeID = disc.getWsElNodeID();
  const int num_ws = wsElNodeID.size();
  for (int ws=0; ws<num_ws; ++ws) {
    const auto& ssMap = disc.getSideSets(ws);
    if (ssMap.find(this->sideSetName)==ssMap.end()) {
      continue;
    }
    const auto& ss = ssMap.at(this->sideSetName);
    for (const auto& side : ss) {
      const int icell = side.elem_LID;
      const int iside = side.side_local_id;

      const auto& side_nodes = this->sideNodes[iside];

      for (int inode=0; inode<this->numSideNodes[iside]; ++inode) {
        ss_nodes_gids.insert(wsElNodeID[ws][icell][side_nodes[inode]]);
      }
    }
  }

  // Avoid doing this again
  ss_node_gids_gathered = true;
}

template<typename EvalT, typename Traits>
void ScatterSideEqnResidualBase<EvalT, Traits>::
buildSideSetNodeMap (typename Traits::EvalData workset)
{
  const int ws = workset.wsIndex;

  // Do this only once per workset
  if (ss_ws_cell_nodes_lids.find(ws)==ss_ws_cell_nodes_lids.end()) {
    // Gather sideSet node gids only once
    if (not ss_node_gids_gathered) {
      gatherSideSetNodeGIDs(*workset.disc);
    }

    auto& ws_ss_nodes = ss_ws_cell_nodes_lids[ws];

    const auto& wsElNodeID = workset.disc->getWsElNodeID();
    for (unsigned int icell=0; icell<workset.numCells; ++icell) {

      const auto& cell_node_gids = wsElNodeID[workset.wsIndex][icell];
      for (int inode=0; inode<numCellNodes; ++inode) {
        const GO gid = cell_node_gids[inode];
        if (ss_nodes_gids.count(gid)>0) {
          ws_ss_nodes[icell].insert(inode);
        }
      }
    }
  }
}

template<typename EvalT, typename Traits>
void ScatterSideEqnResidualBase<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  buildSideSetNodeMap(workset);

  if (workset.sideSets->find(this->sideSetName)!=workset.sideSets->end()) {
    sideSet = workset.sideSetViews->at(this->sideSetName);
    for (int sideSet_idx = 0; sideSet_idx < sideSet.size; ++sideSet_idx)
    {
      // Get the local data of side and cell
      const int icell = sideSet.elem_LID(sideSet_idx);
      const int iside = sideSet.side_local_id(sideSet_idx);

      if (residualsAreVolumeFields) {
        doEvaluateFieldsCell(workset,icell,iside);
      } else {
        doEvaluateFieldsSide(workset,icell,iside,sideSet_idx);
      }
    }
  }

  // We might need to do something for dofs not on this side set
  doPostEvaluate (workset);
}

template<typename EvalT, typename Traits>
void ScatterSideEqnResidualBase<EvalT, Traits>::
doPostEvaluate(typename Traits::EvalData workset)
{
  auto f = workset.f;
  if (!f.is_null()) {
    // We evaluated the residual. Let's set it to 0 outside the side set
    Teuchos::ArrayRCP<ST> f_nonconstView = Albany::getNonconstLocalData(f);
    const auto& nodeID = workset.wsElNodeEqID;
    auto& ws_ss_nodes = this->ss_ws_cell_nodes_lids[workset.wsIndex];
    for (size_t icell=0; icell<workset.numCells; ++icell) {
      const auto& skip_nodes = ws_ss_nodes[icell];
      for (int inode=0; inode<this->numCellNodes; ++inode) {
        if (skip_nodes.count(inode)==0) {
          // This node is not on the side set. Set jac to 1
          for (int eq = 0; eq < this->numFields; eq++) {
            f_nonconstView[nodeID(icell,inode,this->offset + eq)] = 0.0;
          }
        }
      }
    }
  }
}

template<typename EvalT, typename Traits>
void ScatterSideEqnResidualBase<EvalT, Traits>::
doEvaluateFieldsCellResidual(typename Traits::EvalData workset, int cell, int side)
{
  Teuchos::RCP<Thyra_Vector> f = workset.f;

  const auto& nodeID = workset.wsElNodeEqID;

  //get nonconst (read and write) view of f
  Teuchos::ArrayRCP<ST> f_nonconstView = Albany::getNonconstLocalData(f);

  int numNodes = this->numSideNodes[side];
  const auto& side_nodes = this->sideNodes[side];
  if (this->tensorRank == 0) {
    for (int inode = 0; inode<numNodes; ++inode) {
      int node = side_nodes[inode];
      for (int eq = 0; eq < this->numFields; ++eq)
        f_nonconstView[nodeID(cell,node,this->offset + eq)] += Albany::ADValue((this->val[eq])(cell,node));
    }
  } else if (this->tensorRank == 1) {
    for (int inode = 0; inode<numNodes; ++inode) {
      int node = side_nodes[inode];
      for (int eq = 0; eq < this->numFields; eq++)
        f_nonconstView[nodeID(cell,node,this->offset + eq)] += Albany::ADValue((this->valVec)(cell,node,eq));
    }
  } else if (this->tensorRank == 2) {
    for (int inode = 0; inode<numNodes; ++inode) {
      int node = side_nodes[inode];
      for (int i = 0; i < numDims; i++)
        for (int j = 0; j < numDims; j++)
          f_nonconstView[nodeID(cell,node,this->offset + i*numDims + j)] += Albany::ADValue((this->valTensor)(cell,node,i,j));
    }
  }
}

template<typename EvalT, typename Traits>
void ScatterSideEqnResidualBase<EvalT, Traits>::
doEvaluateFieldsSideResidual(typename Traits::EvalData workset, int cell, int side, int sideSet_idx)
{
  Teuchos::RCP<Thyra_Vector> f = workset.f;

  const auto& nodeID = workset.wsElNodeEqID;

  //get nonconst (read and write) view of f
  Teuchos::ArrayRCP<ST> f_nonconstView = Albany::getNonconstLocalData(f);

  int numNodes = this->numSideNodes[side];
  const auto& side_nodes = this->sideNodes[side];
  if (this->tensorRank == 0) {
    for (int inode = 0; inode<numNodes; ++inode) {
      int node = side_nodes[inode];
      for (int eq = 0; eq < this->numFields; ++eq)
        f_nonconstView[nodeID(cell,node,this->offset + eq)] += Albany::ADValue((this->val[eq])(sideSet_idx,inode));
    }
  } else if (this->tensorRank == 1) {
    for (int inode = 0; inode<numNodes; ++inode) {
      int node = side_nodes[inode];
      for (int eq = 0; eq < this->numFields; eq++)
        f_nonconstView[nodeID(cell,node,this->offset + eq)] += Albany::ADValue((this->valVec)(sideSet_idx,inode,eq));
    }
  } else if (this->tensorRank == 2) {
    for (int inode = 0; inode<numNodes; ++inode) {
      int node = side_nodes[inode];
      for (int i = 0; i < numDims; i++)
        for (int j = 0; j < numDims; j++)
          f_nonconstView[nodeID(cell,node,this->offset + i*numDims + j)] += Albany::ADValue((this->valTensor)(sideSet_idx,inode,i,j));
    }
  }
}

// **********************************************************************
// Specialization: Residual
// **********************************************************************

template<typename Traits>
ScatterSideEqnResidual<AlbanyTraits::Residual, Traits>::
ScatterSideEqnResidual (const Teuchos::ParameterList& p,
                        const Teuchos::RCP<Albany::Layouts>& dl)
 : base_type(p,dl)
{
  // Nothing to do here
}

template<typename Traits>
void ScatterSideEqnResidual<AlbanyTraits::Residual, Traits>::
doEvaluateFieldsCell(typename Traits::EvalData workset, int cell, int side)
{
  this->doEvaluateFieldsCellResidual(workset,cell,side);
}

template<typename Traits>
void ScatterSideEqnResidual<AlbanyTraits::Residual, Traits>::
doEvaluateFieldsSide(typename Traits::EvalData workset, int cell, int side, int sideSet_idx)
{
  this->doEvaluateFieldsSideResidual(workset,cell,side,sideSet_idx);
}

// **********************************************************************

// **********************************************************************
// Specialization: Jacobian
// **********************************************************************

template<typename Traits>
ScatterSideEqnResidual<AlbanyTraits::Jacobian, Traits>::
ScatterSideEqnResidual (const Teuchos::ParameterList& p,
                        const Teuchos::RCP<Albany::Layouts>& dl)
 : base_type(p,dl)
{
  // Nothing to do here
}

template<typename Traits>
void ScatterSideEqnResidual<AlbanyTraits::Jacobian, Traits>::
doPostEvaluate(typename Traits::EvalData workset)
{
  // Loop over all cells in the ws, and if a node is not on the sideSet, set Jac=1
  auto Jac = workset.Jac;
  Teuchos::Array<LO> lrow(1);
  Teuchos::Array<ST> one(1,1.0);
  const auto& nodeID = workset.wsElNodeEqID;
  auto& ws_ss_nodes = this->ss_ws_cell_nodes_lids[workset.wsIndex];
  for (size_t icell=0; icell<workset.numCells; ++icell) {
    const auto& skip_nodes = ws_ss_nodes[icell];
    for (int inode=0; inode<this->numCellNodes; ++inode) {
      if (skip_nodes.count(inode)==0) {
        // This node is not on the side set. Set jac to 1
        for (int eq = 0; eq < this->numFields; eq++) {
          lrow[0] = nodeID(icell,inode,this->offset + eq);
          Albany::setLocalRowValues(Jac,lrow[0],lrow, one);
        }
      }
    }
  }

  // Call the base class one, to handle residual (if needed)
  base_type::doPostEvaluate(workset);
}

template<typename Traits>
void ScatterSideEqnResidual<AlbanyTraits::Jacobian, Traits>::
doEvaluateFieldsCell(typename Traits::EvalData workset, int cell, int side)
{
  if (!workset.f.is_null()) {
    this->doEvaluateFieldsCellResidual(workset,cell,side);
  }

  Teuchos::RCP<Thyra_LinearOp> Jac = workset.Jac;

  int numNodes = this->numSideNodes[side];
  const auto& side_nodes = this->sideNodes[side];

  const auto& nodeID = workset.wsElNodeEqID;
  const int neq = nodeID.extent_int(2);
  const int nunk = neq*numNodes;
  Teuchos::Array<LO> cols(nunk);

  // Local Unks: Loop over nodes in element, Loop over equations per node
  for (int inode=0; inode<numNodes; ++inode){
    const int node = side_nodes[inode];
    for (int eq_col=0; eq_col<neq; eq_col++) {
      cols[neq * inode + eq_col] = nodeID(cell,node,eq_col);
    }
  }
  for (int inode = 0; inode < numNodes; ++inode) {
    const int node = side_nodes[inode];
    for (int eq = 0; eq < this->numFields; eq++) {
      typename Ref<ScalarT const>::type
        valptr = (this->tensorRank == 0 ? this->val[eq](cell,node) :
                  this->tensorRank == 1 ? this->valVec(cell,node,eq) :
                  this->valTensor(cell,node, eq/this->numDims, eq%this->numDims));
      const LO row = nodeID(cell,node,this->offset + eq);

      // Check derivative array is nonzero
      if (valptr.hasFastAccess()) {
        // Sum Jacobian entries all at once
        Albany::addToLocalRowValues(Jac,
          row, cols, Teuchos::arrayView(&(valptr.fastAccessDx(0)), nunk));
      } // has fast access
    }
  }
}

template<typename Traits>
void ScatterSideEqnResidual<AlbanyTraits::Jacobian, Traits>::
doEvaluateFieldsSide(typename Traits::EvalData workset, int cell, int side, int sideSet_idx)
{
  if (!workset.f.is_null()) {
    this->doEvaluateFieldsSideResidual(workset,cell,side,sideSet_idx);
  }
  Teuchos::RCP<Thyra_LinearOp> Jac = workset.Jac;

  int numNodes = this->numSideNodes[side];
  const auto& side_nodes = this->sideNodes[side];

  const auto& nodeID = workset.wsElNodeEqID;
  const int neq = nodeID.extent_int(2);
  const int nunk = neq*numNodes;
  Teuchos::Array<LO> cols(nunk);

  // Local Unks: Loop over nodes in element, Loop over equations per node
  for (int inode=0; inode<numNodes; ++inode){
    const int node = side_nodes[inode];
    for (int eq_col=0; eq_col<neq; eq_col++) {
      cols[neq * inode + eq_col] = nodeID(cell,node,eq_col);
    }
  }

  for (int inode = 0; inode < numNodes; ++inode) {
    const int node = side_nodes[inode];
    for (int eq = 0; eq < this->numFields; eq++) {
      typename Ref<ScalarT const>::type
        valptr = (this->tensorRank == 0 ? this->val[eq](sideSet_idx,inode) :
                  this->tensorRank == 1 ? this->valVec(sideSet_idx,inode,eq) :
                  this->valTensor(sideSet_idx,inode, eq/this->numDims, eq%this->numDims));
      const LO row = nodeID(cell,node,this->offset + eq);

      // Check derivative array is nonzero
      if (valptr.hasFastAccess()) {
        // Sum Jacobian entries all at once
        Albany::addToLocalRowValues(Jac,
          row, cols, Teuchos::arrayView(&(valptr.fastAccessDx(0)), nunk));
      } // has fast access
    }
  }
}

// **********************************************************************
// Specialization: Tangent
// **********************************************************************

template<typename Traits>
ScatterSideEqnResidual<AlbanyTraits::Tangent, Traits>::
ScatterSideEqnResidual (const Teuchos::ParameterList& p,
                        const Teuchos::RCP<Albany::Layouts>& dl)
 : base_type(p,dl)
{
  // Nothing to do here
}

template<typename Traits>
void ScatterSideEqnResidual<AlbanyTraits::Tangent, Traits>::
doEvaluateFieldsCell(typename Traits::EvalData workset, int cell, int side)
{
  if (!workset.f.is_null()) {
    this->doEvaluateFieldsCellResidual(workset,cell,side);
  }

  Teuchos::RCP<Thyra_MultiVector> JV = workset.JV;
  Teuchos::RCP<Thyra_MultiVector> fp = workset.fp;

  Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>> JV_nonconst2dView;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>> fp_nonconst2dView;

  if (!JV.is_null()) {
    JV_nonconst2dView = Albany::getNonconstLocalData(JV);
  }
  if (!fp.is_null()) {
    fp_nonconst2dView = Albany::getNonconstLocalData(fp);
  }

  int numNodes = this->numSideNodes[side];
  const auto& side_nodes = this->sideNodes[side];

  const auto& nodeID = workset.wsElNodeEqID;
  const int neq = nodeID.extent_int(2);
  const int nunk = neq*numNodes;
  Teuchos::Array<LO> cols(nunk);

  // Local Unks: Loop over nodes in element, Loop over equations per node
  for (int inode=0; inode<numNodes; ++inode){
    const int node = side_nodes[inode];
    for (int eq_col=0; eq_col<neq; eq_col++) {
      cols[neq * inode + eq_col] = nodeID(cell,node,eq_col);
    }
  }
  for (int inode = 0; inode < numNodes; ++inode) {
    const int node = side_nodes[inode];
    for (int eq = 0; eq < this->numFields; eq++) {
      typename Ref<ScalarT const>::type valref = (
          this->tensorRank == 0 ? this->val[eq] (cell, node) :
          this->tensorRank == 1 ? this->valVec (cell, node, eq) :
          this->valTensor (cell, node, eq / this->numDims, eq % this->numDims));

      const LO row = nodeID(cell,node,this->offset + eq);

      if (Teuchos::nonnull (JV)) {
        for (int col = 0; col < workset.num_cols_x; col++) {
          JV_nonconst2dView[col][row] += valref.dx(col);
      }}

      if (Teuchos::nonnull (fp)) {
        for (int col = 0; col < workset.num_cols_p; col++) {
          fp_nonconst2dView[col][row] += valref.dx(col + workset.param_offset);
      }}
    }
  }
}

template<typename Traits>
void ScatterSideEqnResidual<AlbanyTraits::Tangent, Traits>::
doEvaluateFieldsSide(typename Traits::EvalData workset, int cell, int side, int sideSet_idx)
{
  if (!workset.f.is_null()) {
    this->doEvaluateFieldsSideResidual(workset,cell,side,sideSet_idx);
  }

  Teuchos::RCP<Thyra_MultiVector> JV = workset.JV;
  Teuchos::RCP<Thyra_MultiVector> fp = workset.fp;

  Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>> JV_nonconst2dView;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>> fp_nonconst2dView;

  if (!JV.is_null()) {
    JV_nonconst2dView = Albany::getNonconstLocalData(JV);
  }
  if (!fp.is_null()) {
    fp_nonconst2dView = Albany::getNonconstLocalData(fp);
  }

  int numNodes = this->numSideNodes[side];
  const auto& side_nodes = this->sideNodes[side];

  const auto& nodeID = workset.wsElNodeEqID;
  const int neq = nodeID.extent_int(2);
  const int nunk = neq*numNodes;
  Teuchos::Array<LO> cols(nunk);

  // Local Unks: Loop over nodes in element, Loop over equations per node
  for (int inode=0; inode<numNodes; ++inode){
    const int node = side_nodes[inode];
    for (int eq_col=0; eq_col<neq; eq_col++) {
      cols[neq * inode + eq_col] = nodeID(cell,node,eq_col);
    }
  }
  for (int inode = 0; inode < numNodes; ++inode) {
    const int node = side_nodes[inode];
    for (int eq = 0; eq < this->numFields; eq++) {
      typename Ref<ScalarT const>::type valref = (
          this->tensorRank == 0 ? this->val[eq] (sideSet_idx, inode) :
          this->tensorRank == 1 ? this->valVec (sideSet_idx, inode, eq) :
          this->valTensor (sideSet_idx, inode, eq / this->numDims, eq % this->numDims));

      const LO row = nodeID(cell,node,this->offset + eq);

      if (Teuchos::nonnull (JV)) {
        for (int col = 0; col < workset.num_cols_x; col++) {
          JV_nonconst2dView[col][row] += valref.dx(col);
      }}

      if (Teuchos::nonnull (fp)) {
        for (int col = 0; col < workset.num_cols_p; col++) {
          fp_nonconst2dView[col][row] += valref.dx(col + workset.param_offset);
      }}
    }
  }
}

// **********************************************************************
// Specialization: DistParamDeriv
// **********************************************************************


template<typename Traits>
ScatterSideEqnResidual<AlbanyTraits::DistParamDeriv, Traits>::
ScatterSideEqnResidual (const Teuchos::ParameterList& p,
                        const Teuchos::RCP<Albany::Layouts>& dl)
 : base_type(p,dl)
{
  // Nothing to do here
}

template<typename Traits>
void ScatterSideEqnResidual<AlbanyTraits::DistParamDeriv, Traits>::
doEvaluateFieldsCell(typename Traits::EvalData workset, int cell, int side)
{
  if (!workset.f.is_null()) {
    this->doEvaluateFieldsCellResidual(workset,cell,side);
  }

  Teuchos::RCP<Thyra_MultiVector> fpV = workset.fpV;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>> fpV_nonconst2dView = Albany::getNonconstLocalData(fpV);

  bool trans = workset.transpose_dist_param_deriv;
  int num_cols = workset.Vp->domain()->dim();

  if(workset.local_Vp[0].size() == 0) {
    // In case the parameter has not been gathered, e.g. parameter is used only in Dirichlet conditions. 
    return;
  }

  int numNodes = this->numSideNodes[side];
  const auto& side_nodes = this->sideNodes[side];
  const auto& local_Vp = workset.local_Vp[cell];

  const auto& nodeID = workset.wsElNodeEqID;
  if (trans) {
    const int neq = nodeID.extent(2);
    const Albany::IDArray&  wsElDofs = workset.distParamLib->get(workset.dist_param_deriv_name)->workset_elem_dofs()[workset.wsIndex];

    const int num_deriv = numNodes;
    for (int i=0; i<num_deriv; i++) {
      for (int col=0; col<num_cols; col++) {
        double val = 0.0;
        for (int inode = 0; inode < numNodes; ++inode) {
          const int node = side_nodes[inode];
          for (int eq = 0; eq < this->numFields; eq++) {
            typename Ref<ScalarT const>::type
                      valref = (this->tensorRank == 0 ? this->val[eq](cell,node) :
                                this->tensorRank == 1 ? this->valVec(cell,node,eq) :
                                this->valTensor(cell,node, eq/this->numDims, eq%this->numDims));
            val += valref.dx(i)*local_Vp[node*neq+eq+this->offset][col];  //numField can be less then neq
          }
        }
        const LO row = wsElDofs(cell,i,0);
        if(row >=0) {
          fpV_nonconst2dView[col][row] += val;
        }
      }
    }
  } else {
    const int num_deriv = local_Vp.size();

    for (int inode = 0; inode < numNodes; ++inode) {
      const int node = side_nodes[inode];
      for (int eq = 0; eq < this->numFields; eq++) {
        typename Ref<ScalarT const>::type
                  valref = (this->tensorRank == 0 ? this->val[eq](cell,node) :
                            this->tensorRank == 1 ? this->valVec(cell,node,eq) :
                            this->valTensor(cell,node, eq/this->numDims, eq%this->numDims));
        const int row = nodeID(cell,node,this->offset + eq);
        for (int col=0; col<num_cols; col++) {
          double val = 0.0;
          for (int i=0; i<num_deriv; ++i) {
            val += valref.dx(i)*local_Vp[i][col];
          }
          fpV_nonconst2dView[col][row] += val;
        }
      }
    }
  }
}

template<typename Traits>
void ScatterSideEqnResidual<AlbanyTraits::DistParamDeriv, Traits>::
doEvaluateFieldsSide(typename Traits::EvalData workset, int cell, int side, int sideSet_idx)
{
  if (!workset.f.is_null()) {
    this->doEvaluateFieldsSideResidual(workset,cell,side,sideSet_idx);
  }

  Teuchos::RCP<Thyra_MultiVector> fpV = workset.fpV;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>> fpV_nonconst2dView = Albany::getNonconstLocalData(fpV);

  bool trans = workset.transpose_dist_param_deriv;
  int num_cols = workset.Vp->domain()->dim();

  if(workset.local_Vp[0].size() == 0) {
    // In case the parameter has not been gathered, e.g. parameter is used only in Dirichlet conditions. 
    return;
  }

  int numNodes = this->numSideNodes[side];
  const auto& side_nodes = this->sideNodes[side];
  const auto& local_Vp = workset.local_Vp[cell];

  const auto& nodeID = workset.wsElNodeEqID;
  if (trans) {
    const int neq = nodeID.extent(2);
    const Albany::IDArray&  wsElDofs = workset.distParamLib->get(workset.dist_param_deriv_name)->workset_elem_dofs()[workset.wsIndex];

    const int num_deriv = numNodes;
    for (int i=0; i<num_deriv; i++) {
      for (int col=0; col<num_cols; col++) {
        double val = 0.0;
        for (int inode = 0; inode < numNodes; ++inode) {
          const int node = side_nodes[inode];
          for (int eq = 0; eq < this->numFields; eq++) {
            typename Ref<ScalarT const>::type
                      valref = (this->tensorRank == 0 ? this->val[eq](sideSet_idx,inode) :
                                this->tensorRank == 1 ? this->valVec(sideSet_idx,inode,eq) :
                                this->valTensor(sideSet_idx,inode, eq/this->numDims, eq%this->numDims));
            val += valref.dx(i)*local_Vp[node*neq+eq+this->offset][col];  //numField can be less then neq
          }
        }
        const LO row = wsElDofs(cell,i,0);
        if(row >=0) {
          fpV_nonconst2dView[col][row] += val;
        }
      }
    }
  } else {
    const int num_deriv = local_Vp.size();

    for (int inode = 0; inode < numNodes; ++inode) {
      const int node = side_nodes[inode];
      for (int eq = 0; eq < this->numFields; eq++) {
        typename Ref<ScalarT const>::type
                  valref = (this->tensorRank == 0 ? this->val[eq](sideSet_idx,inode) :
                            this->tensorRank == 1 ? this->valVec(sideSet_idx,inode,eq) :
                            this->valTensor(sideSet_idx,inode, eq/this->numDims, eq%this->numDims));
        const int row = nodeID(cell,node,this->offset + eq);
        for (int col=0; col<num_cols; col++) {
          double val = 0.0;
          for (int i=0; i<num_deriv; ++i) {
            val += valref.dx(i)*local_Vp[i][col];
          }
          fpV_nonconst2dView[col][row] += val;
        }
      }
    }
  }
}

// **********************************************************************
// Specialization: HessianVec
// **********************************************************************


template<typename Traits>
ScatterSideEqnResidual<AlbanyTraits::HessianVec, Traits>::
ScatterSideEqnResidual (const Teuchos::ParameterList& p,
                        const Teuchos::RCP<Albany::Layouts>& dl)
 : base_type(p,dl)
{
  // Nothing to do here
}

template<typename Traits>
void ScatterSideEqnResidual<AlbanyTraits::HessianVec, Traits>::
doEvaluateFieldsCell(typename Traits::EvalData /* workset */, int /* cell */, int /* side */)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "HessianVec specialization of ScatterSideEqnResidual::doEvaluateFieldsCell is not implemented yet"<< std::endl);
}

template<typename Traits>
void ScatterSideEqnResidual<AlbanyTraits::HessianVec, Traits>::
doEvaluateFieldsSide(typename Traits::EvalData /* workset */, int /* cell */, int /* side */, int /* sideSet_idx */)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "HessianVec specialization of ScatterSideEqnResidual::doEvaluateFieldsSide is not implemented yet"<< std::endl);
}

} // namespace PHAL
