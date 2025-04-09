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
#include "Albany_DualView.hpp"

// **********************************************************************
// Base Class Generic Implementation
// **********************************************************************
namespace PHAL {

template<typename EvalT, typename Traits>
ScatterSideEqnResidualBase<EvalT, Traits>::
ScatterSideEqnResidualBase (const Teuchos::ParameterList& p,
                            const Teuchos::RCP<Albany::Layouts>& dl)
 : sideSetName ( p.get<std::string>("Side Set Name") )
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

  auto res_dl = dl->side_layouts.at(sideSetName);

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
    tensorDim = res_dl->node_tensor->extent(2);
    numFields = tensorDim*tensorDim;
  }
  numSideNodes = res_dl->node_scalar->dimension(1);

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
gatherSideSetNodeGIDs (const Albany::AbstractDiscretization& disc)
{
  // Check for early return
  if (ss_eqns_dofs_lids_gathered) {
    return;
  }

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
  const int num_ws = disc.getNumWorksets();
  const auto dof_mgr = disc.getDOFManager();
  const auto elem_dof_lids = dof_mgr->elem_dof_lids().host();
  constexpr auto ALL = Kokkos::ALL();

  std::set<LO> all_ss_eqns_lids;
  for (int ws=0; ws<num_ws; ++ws) {
    const auto& ssMap = disc.getSideSets(ws);
    if (ssMap.find(this->sideSetName)==ssMap.end()) {
      continue;
    }
    const auto& elem_lids = disc.getElementLIDs_host(ws);
    const auto& ss = ssMap.at(this->sideSetName);
    for (const auto& side : ss) {
      const int icell = side.ws_elem_idx;
      const int elem_LID = elem_lids(icell);
      const int side_pos = side.side_pos;

      const auto dof_lids = Kokkos::subview(elem_dof_lids,elem_LID,ALL);
      for (int eq = 0; eq < this->numFields; eq++) {
        const auto& offsets = dof_mgr->getGIDFieldOffsetsSide(eq+this->offset,side_pos);
        for (auto o : offsets) {
          all_ss_eqns_lids.insert(dof_lids[o]);
        }
      }
    }
  }

  // To speed up calculation at runtime, we now group them by ws
  ss_eqns_dofs_lids.resize(num_ws);
  if (num_ws==1) {
    // All ss eqn dofs are in same ws, just copy and be done
    ss_eqns_dofs_lids[0] = all_ss_eqns_lids;
  } else {
    for (int ws=0; ws<num_ws; ++ws) {
      const auto& elem_lids = disc.getElementLIDs_host(ws);
      for (size_t ie=0; ie<elem_lids.size(); ++ie) {
        const int elem_LID = elem_lids(ie);
        const auto dof_lids = Kokkos::subview(elem_dof_lids,elem_LID,ALL);
        for (int eq = 0; eq < this->numFields; eq++) {
          const auto& offsets = dof_mgr->getGIDFieldOffsets(eq+this->offset);
          for (auto o : offsets) {
            if (all_ss_eqns_lids.count(dof_lids[o]>0)) {
              ss_eqns_dofs_lids[ws].insert(dof_lids[0]);
            }
          }
        }
      }
    }
  }

  // Avoid doing this again
  ss_eqns_dofs_lids_gathered = true;
}

template<typename EvalT, typename Traits>
void ScatterSideEqnResidualBase<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  gatherSideSetNodeGIDs(*workset.disc);

  if (workset.sideSets->find(this->sideSetName)!=workset.sideSets->end()) {
    sideSet = workset.sideSets->at(this->sideSetName);
    doEvaluateFields(workset,sideSet);
  }

  // We might need to do something for dofs not on this side set
  doPostEvaluate (workset);
}

template<typename EvalT, typename Traits>
void ScatterSideEqnResidualBase<EvalT, Traits>::
doPostEvaluate(typename Traits::EvalData workset)
{
  const auto f = workset.f;
  if (f.is_null()) {
    return;
  }
  const auto f_data = Albany::getNonconstLocalData(f);

  Teuchos::RCP<Thyra_Vector const> x = workset.x;
  Teuchos::ArrayRCP<const ST> x_data = Albany::getLocalData(x);

  constexpr auto ALL = Kokkos::ALL();
  const auto dof_mgr = workset.disc->getDOFManager();
  const auto elem_lids = workset.disc->getElementLIDs_host(workset.wsIndex);
  const auto elem_dof_lids = dof_mgr->elem_dof_lids().host();

  const int num_total_eqn = dof_mgr->getNumFields();
  for (size_t icell=0; icell<workset.numCells; ++icell) {
    const auto elem_LID = elem_lids(icell);
    const auto dof_lids = Kokkos::subview(elem_dof_lids,elem_LID,ALL);

    for (int eq = 0; eq < num_total_eqn; eq++) {
      const auto& offsets = dof_mgr->getGIDFieldOffsets(eq);
      for (auto o : offsets) {
        auto lid = dof_lids[o];
        if (ss_eqns_dofs_lids[workset.wsIndex].count(lid)==0) {
          // Set f=x outside the sideset, so we converge in 1 newton iteration
          f_data[lid] = x_data[lid];
        }
      }
    }
  }
}

template<typename EvalT, typename Traits>
void ScatterSideEqnResidualBase<EvalT, Traits>::
doEvaluateFieldsResidual(typename Traits::EvalData workset,
                         const std::vector<Albany::SideStruct>& sides)
{
  //get nonconst residual
  const auto f = workset.f;
  const auto f_data = Albany::getNonconstLocalData(f);

  constexpr auto ALL = Kokkos::ALL();
  const auto dof_mgr = workset.disc->getDOFManager();
  const auto elem_lids = workset.disc->getElementLIDs_host(workset.wsIndex);
  const auto elem_dof_lids = dof_mgr->elem_dof_lids().host();

  // Note: we use the *volume* dof manager, since we need to scatter
  //       in the volume residual vector
  const int numSides = sideSet.size();
  for (int iside=0; iside<numSides; ++iside) {
    const auto& side = sides[iside];
    const auto icell    = side.ws_elem_idx;
    const auto elem_LID = elem_lids(icell);
    const auto dof_lids = Kokkos::subview(elem_dof_lids,elem_LID,ALL);

    for (int eq=0; eq<this->numFields; ++eq) {
      const auto& offsets = dof_mgr->getGIDFieldOffsetsSide(eq+this->offset,side.side_pos);
      for (size_t inode=0; inode<offsets.size(); ++inode) {
        auto res = Albany::ADValue(this->get_resid(iside,inode,eq));
        f_data[dof_lids(offsets[inode])] += res;
      }
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
doEvaluateFields(typename Traits::EvalData workset,
                 const std::vector<Albany::SideStruct>& sideSet)
{
  this->doEvaluateFieldsResidual(workset,sideSet);
}

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
  constexpr auto ALL = Kokkos::ALL();
  const auto dof_mgr = workset.disc->getDOFManager();
  const auto node_dof_mgr = workset.disc->getNodeDOFManager();
  const auto elem_lids = workset.disc->getElementLIDs_host(workset.wsIndex);
  const auto elem_dof_lids = dof_mgr->elem_dof_lids().host();

  // Set J=identity outside of the sideset, so it's not singular
  auto Jac = workset.Jac;
  const int numCellNodes = node_dof_mgr->getGIDFieldOffsets(0).size();
  const int num_total_eqn = dof_mgr->getNumFields();
  const auto& ss_eqn_lids = this->ss_eqns_dofs_lids.at(workset.wsIndex);
  for (size_t icell=0; icell<workset.numCells; ++icell) {
    const auto elem_LID = elem_lids(icell);
    const auto dof_lids = Kokkos::subview(elem_dof_lids,elem_LID,ALL);

    for (int eq = 0; eq < num_total_eqn; eq++) {
      const auto& offsets = dof_mgr->getGIDFieldOffsets(eq);
      for (auto o : offsets) {
        auto lrow = dof_lids[o];
        if (ss_eqn_lids.count(lrow)==0) {
          // LID not on the sideset
          Albany::setLocalRowValue(Jac,lrow,lrow, 1.0);
        }
      }
    }
  }

  // Call the base class one, to handle residual (if needed)
  base_type::doPostEvaluate(workset);
}

template<typename Traits>
void ScatterSideEqnResidual<AlbanyTraits::Jacobian, Traits>::
doEvaluateFields(typename Traits::EvalData workset,
                 const std::vector<Albany::SideStruct>& sideSet)
{
  if (!workset.f.is_null()) {
    this->doEvaluateFieldsResidual(workset,sideSet);
  }
  const auto Jac = workset.Jac;

  constexpr auto ALL = Kokkos::ALL();
  const auto dof_mgr = workset.disc->getDOFManager();
  const auto node_dof_mgr = workset.disc->getNodeDOFManager();
  const auto elem_dof_lids = dof_mgr->elem_dof_lids().host();
  const auto elem_lids = workset.disc->getElementLIDs_host(workset.wsIndex);

  const int numSides = sideSet.size();
  const int numCellNodes = dof_mgr->getGIDFieldOffsets(0).size();
  const int neq = dof_mgr->getNumFields();

  const int nunk = neq*this->numSideNodes;
  Teuchos::Array<LO> cols(nunk);

  for (int iside=0; iside<numSides; ++iside) {
    const auto& side = sideSet[iside];
    const auto icell = side.ws_elem_idx;
    const auto elem_LID = elem_lids(icell);
    const auto dof_lids = Kokkos::subview(elem_dof_lids,elem_LID,ALL);
    const auto side_nodes = node_dof_mgr->getGIDFieldOffsetsSide(0,side.side_pos);

    // Precompute the column indices (same for all the rows in this side)
    // Note: we couple the eq with ALL other dofs on the side, so loop on [0,neq)
    for (int eq=0; eq<neq; ++eq) {
      const auto& offsets = dof_mgr->getGIDFieldOffsetsSide(eq,side.side_pos);
      for (int inode=0; inode<this->numSideNodes; ++inode){
        cols[neq * inode + eq] = dof_lids(offsets[inode]);
      }
    }

    for (int eq=0; eq<this->numFields; ++eq) {
      // Here, we use eq+offset, since we're getting the *row* ids, so only for the SS eqn
      const auto& offsets = dof_mgr->getGIDFieldOffsetsSide(eq+this->offset,side.side_pos);
      for (int inode=0; inode<this->numSideNodes; ++inode) {
        auto res = this->get_resid(iside, inode, eq);
        const LO row = dof_lids(offsets[inode]);
#ifdef ALBANY_DEBUG
        TEUCHOS_TEST_FOR_EXCEPTION (not res.hasFastAccess(), std::runtime_error,
            "[ScatterSideEqnResidual] Error! FAD type does not have fast access.\n");
#endif

        // NOTE: we cannot add a whole row, b/c the FAD has length neq*numCellNodes, but
        //       the jacobian has no column corresponding to dofs not on the side
        for (int col_eq=0; col_eq<neq; ++col_eq) {
          for (int jnode=0; jnode<this->numSideNodes;++jnode) {
            const int deriv = neq*side_nodes[jnode] + col_eq;
            const int col = cols[neq*jnode + col_eq];
            Albany::addToLocalRowValues(Jac,row,1,&col,&res.fastAccessDx(deriv));
          }
        }
      }
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
doEvaluateFields(typename Traits::EvalData workset,
                 const std::vector<Albany::SideStruct>& sideSet)
{
  if (!workset.f.is_null()) {
    this->doEvaluateFieldsResidual(workset,sideSet);
  }

  const auto JV = workset.JV;
  const auto fp = workset.fp;
  const bool do_JV = Teuchos::nonnull(JV);
  const bool do_fp = Teuchos::nonnull(fp);

  // Check for early return
  if (!do_JV && !do_fp) {
    return;
  }

  // Extract raw arrays from multivectors
  using mv_data_t = Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>>;
  mv_data_t JV_data, fp_data;
  if (do_JV)
    JV_data = Albany::getNonconstLocalData(JV);
  if (do_fp)
    fp_data = Albany::getNonconstLocalData(fp);

  const auto elem_lids     = workset.disc->getElementLIDs_host(workset.wsIndex);
  const auto dof_mgr       = workset.disc->getDOFManager();
  const auto elem_dof_lids = dof_mgr->elem_dof_lids().host();

  constexpr auto ALL = Kokkos::ALL();
  const int numSides = sideSet.size();
  for (int iside=0; iside<numSides; ++iside) {
    const auto& side = sideSet[iside];
    const auto icell    = side.ws_elem_idx;
    const auto elem_LID = elem_lids(icell);
    const auto dof_lids = Kokkos::subview(elem_dof_lids,elem_LID,ALL);

    for (int eq = 0; eq < this->numFields; eq++) {
      const auto& offsets = dof_mgr->getGIDFieldOffsetsSide(eq+this->offset,side.side_pos);
      for (int inode = 0; inode < this->numSideNodes; ++inode) {
        auto res = this->get_resid(iside,inode,eq);

        const LO row = dof_lids(offsets[inode]);

        if (do_JV) {
          for (int col = 0; col < workset.num_cols_x; col++) {
            JV_data[col][row] += res.dx(col);
        }}

        if (do_fp) {
          for (int col = 0; col < workset.num_cols_p; col++) {
            fp_data[col][row] += res.dx(col + workset.param_offset);
        }}
      }
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
doEvaluateFields(typename Traits::EvalData workset,
                 const std::vector<Albany::SideStruct>& sideSet)
{
  if (!workset.f.is_null()) {
    this->doEvaluateFieldsResidual(workset,sideSet);
  }

  // Check for early return
  if(workset.local_Vp.size() == 0) {
    // In case the parameter has not been gathered.
    // E.g., parameter is used only in Dirichlet conditions. 
    return;
  }

  const auto fpV = workset.fpV;
  const auto fpV_data = Albany::getNonconstLocalData(fpV);

  const bool trans = workset.transpose_dist_param_deriv;
  const int num_cols = workset.Vp->domain()->dim();

  Albany::DualView<int**>::host_t p_elem_dof_lids;
  if (trans) {
    const auto dist_param = workset.distParamLib->get(workset.dist_param_deriv_name);
    p_elem_dof_lids = dist_param->get_dof_mgr()->elem_dof_lids().host();
  }
  const auto elem_lids     = workset.disc->getElementLIDs_host(workset.wsIndex);
  const auto dof_mgr       = workset.disc->getDOFManager();
  const auto node_dof_mgr  = workset.disc->getNodeDOFManager();
  const auto elem_dof_lids = dof_mgr->elem_dof_lids().host();

  const auto& local_Vp = workset.local_Vp;

  constexpr auto ALL = Kokkos::ALL();
  const int neq = dof_mgr->getNumFields();
  const int numSides = sideSet.size();
  for (int iside=0; iside<numSides; ++iside) {
    const auto& side = sideSet[iside];

    const auto icell = side.ws_elem_idx;
    const auto elem_LID = elem_lids(icell);
    
    const auto& side_nodes = node_dof_mgr->getGIDFieldOffsetsSide(0,side.side_pos);
    const auto dof_lids = Kokkos::subview(elem_dof_lids,elem_LID,ALL);

    if (trans) {
      const int num_deriv = this->numSideNodes;
      for (int i=0; i<num_deriv; ++i) {
        const LO row = p_elem_dof_lids(icell,i);
        if (row<0) {
          continue;
        }

        for (int col=0; col<num_cols; col++) {
          double val = 0.0;
          for (int inode = 0; inode < this->numSideNodes; ++inode) {
            const int node = side_nodes[inode];
            for (int eq = 0; eq < this->numFields; eq++) {
              auto res = this->get_resid(iside,inode,eq);
              val += res.dx(i)*local_Vp(icell,node*neq+eq+this->offset,col);
            }
          }
          fpV_data[col][row] += val;
        }
      }
    } else {
      const int num_deriv = local_Vp.size();

      for (int inode=0; inode<this->numSideNodes; ++inode) {
        for (int eq=0; eq<this->numFields; ++eq) {
          const auto& offsets = dof_mgr->getGIDFieldOffsetsSide(eq+this->offset,side.side_pos);
          auto res = this->get_resid(iside,inode,eq);
          const int row = dof_lids(offsets[inode]);
          for (int col=0; col<num_cols; col++) {
            double val = 0.0;
            for (int i=0; i<num_deriv; ++i) {
              val += res.dx(i)*local_Vp(icell,i,col);
            }
            fpV_data[col][row] += val;
          }
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
doEvaluateFields(typename Traits::EvalData /* workset */,
                 const std::vector<Albany::SideStruct>& /* sideSet */)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
      "ScatterSideEqnResidual<HessianVec> not implemented yet\n");
}

} // namespace PHAL
