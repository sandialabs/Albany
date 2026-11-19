//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "LandIce_ScatterResidual2D.hpp"
#include "PHAL_AlbanyTraits.hpp"

#include "Albany_AbstractDiscretization.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_GlobalLocalIndexer.hpp"

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

namespace PHAL {

using Teuchos::arrayView;

template<typename EvalT, typename Traits>
ScatterResidual2D<EvalT,Traits>::
ScatterResidual2D(const Teuchos::ParameterList& p,
                  const Teuchos::RCP<Albany::Layouts>& dl)
 : Base(p,dl)
{
  TEUCHOS_TEST_FOR_EXCEPTION (this->numFields!=1, std::logic_error,
      "Error! ScatterResidual2D only supports scalar fields.\n");

  fieldLevel = p.get<int>("Field Level");
  meshPart   = p.get<std::string>("Mesh Part");

  if (p.isType<Teuchos::Array<std::string>>("Extruded Param Names")) {
    for (const auto& name : p.get<Teuchos::Array<std::string>>("Extruded Param Names")) {
      extrudedParamNames.insert(name);
    }
  }
}

template<typename EvalT, typename Traits>
ScatterResidualWithExtrudedField<EvalT, Traits>::
ScatterResidualWithExtrudedField(const Teuchos::ParameterList& p,
                                 const Teuchos::RCP<Albany::Layouts>& dl)
 : Base(p,dl)
{
  if (p.isType<int>("Offset 2D Field")) {
    offset2DField = p.get<int>("Offset 2D Field");
  } else {
    offset2DField = numFields-1;
  }
  fieldLevel = p.get<int>("Field Level");

  if (p.isType<Teuchos::Array<std::string>>("Extruded Param Names")) {
    for (const auto& name : p.get<Teuchos::Array<std::string>>("Extruded Param Names")) {
      extrudedParamNames.insert(name);
    }
  }
}

// **********************************************************************
// Specializations: Jacobian
// **********************************************************************

template<>
void ScatterResidual2D<AlbanyTraits::Jacobian, AlbanyTraits>::
evaluateFields(typename AlbanyTraits::EvalData workset)
{
  TEUCHOS_TEST_FOR_EXCEPTION (workset.sideSets.is_null(), std::logic_error,
      "Side sets not properly specified on the mesh.\n");

  // Check for early return
  if (workset.sideSets->count(meshPart)==0) {
    return;
  }

  constexpr auto ALL = Kokkos::ALL();

  const auto ws_elem_lids  = workset.disc->getWsElementLIDs().dev();
  const auto elem_lids     = Kokkos::subview(ws_elem_lids,workset.wsIndex,ALL);
  const auto dof_mgr       = workset.disc->getDOFManager();
  const auto node_dof_mgr  = workset.disc->getNodeDOFManager();
  const auto elem_dof_lids = dof_mgr->elem_dof_lids().dev();

  const auto& layers_data = workset.disc->getMeshStruct()->layers_data;
  const auto  layerOrd = layers_data.cell.lid->layerOrd;
  const auto  numHorizEntities = layers_data.cell.lid->numHorizEntities;
  const int   numLayers = layers_data.cell.lid->numLayers;
  const auto  bot = layers_data.bot_side_pos;
  const auto  top = layers_data.top_side_pos;

  // Cell layer where we'll do the scatter of the 2d residual
  const int fieldLevel = this->fieldLevel;
  const int field_layer = fieldLevel==0 ? 0 : fieldLevel-1;
  const auto field_pos = field_layer==fieldLevel ? bot : top;
  const int offset = this->offset;

  const auto& topo = dof_mgr->get_topology();
  const int numSideNodes = topo.getNodeCount(topo.getDimension()-1,top);
  const int neq = dof_mgr->getNumFields();

  // The first neq*numNodes derivs are for dofs gathered "normally", without
  // any knowledge of column contraction operations.
  const int columnsOffset = neq*this->numNodes;

  const int ncols_volume = neq*this->numNodes;
  const auto& sideSet = workset.sideSetViews->at(meshPart);
  const auto  side_cell = sideSet.ws_elem_idx.view_device();

  // Dof offsets on bot (0) and top (1) sides, ordered as on side $field_pos
  // to guarantee corresponding nodes are vertically aligned.
  Kokkos::View<int***,PHX::Device> side_offsets("side_offsets",neq,2,numSideNodes);
  auto side_offsets_h = Kokkos::create_mirror_view(side_offsets);
  for (int eq=0; eq<neq; ++eq) {
    const auto& bot_offsets = dof_mgr->getGIDFieldOffsetsSide(eq,bot,field_pos);
    const auto& top_offsets = dof_mgr->getGIDFieldOffsetsSide(eq,top,field_pos);
    for (int inode=0; inode<numSideNodes; ++inode) {
      side_offsets_h(eq,0,inode) = bot_offsets[inode];
      side_offsets_h(eq,1,inode) = top_offsets[inode];
    }
  }
  Kokkos::deep_copy(side_offsets,side_offsets_h);

  // Note: top nodes must be parsed in the same 2D order as the field_pos side
  const auto offsets_2d_field = dof_mgr->getGIDFieldOffsetsSideKokkos(offset,field_pos);
  const auto cell_nodes = node_dof_mgr->getGIDFieldOffsetsSideKokkos(0,field_pos);

  Albany::ThyraVDeviceView<ST> f_data;
  const bool scatter_f = Teuchos::nonnull(workset.f);
  if (scatter_f) {
    f_data = workset.f_kokkos;
  }
  auto Jac_kokkos = workset.Jac_kokkos;
  const auto resid = this->device_resid;

  Kokkos::parallel_for(this->getName(),
                       RangePolicy(0,sideSet.size),
                       KOKKOS_LAMBDA(const int iside) {
    const int cell = side_cell(iside);
    // Get column ID of the cell, since it cell might be at the top!
    const int side_elem_LID = elem_lids(cell);
    const int basal_elem_LID = layerOrd ? side_elem_LID % numHorizEntities : side_elem_LID / numLayers;
    const auto layer_elem_LID = [&](const int ilayer) {
      return layerOrd ? basal_elem_LID + ilayer*numHorizEntities : basal_elem_LID*numLayers + ilayer;
    };

    // Set Jac=1 outside of the level where the field is defined
    const ST one = 1;
    for (int ilev=0; ilev<=numLayers; ++ilev) {
      if (ilev==fieldLevel) continue;
      const int ilayer = ilev==0 ? 0 : ilev-1;
      const int tb = ilayer==ilev ? 0 : 1;
      const int elem_LID = layer_elem_LID(ilayer);
      for (int inode=0; inode<numSideNodes; ++inode) {
        const LO lrow = elem_dof_lids(elem_LID,side_offsets(offset,tb,inode));
        Jac_kokkos.replaceValues(lrow,&lrow,1,&one,false,is_atomic);
      }
    }

    // Recall: we scatter a single scalar residual, so no loop on [0,numFields) here.
    const int field_elem_LID = layer_elem_LID(field_layer);
    for (int inode=0; inode<numSideNodes; ++inode) {
      const LO lrow = elem_dof_lids(field_elem_LID,offsets_2d_field(inode));
      auto res = resid.get(cell,cell_nodes(inode),0);
      if (scatter_f) {
        KU::atomic_add<ExecutionSpace>(&f_data(lrow), res.val());
      }
      if (res.hasFastAccess()) {
        // Column-coupled part
        for (int ilev=0; ilev<=numLayers; ++ilev) {
          const int ilayer = ilev==0 ? 0 : ilev-1;
          const int tb = ilayer==ilev ? 0 : 1;
          const int elem_LID = layer_elem_LID(ilayer);
          for (int jnode=0; jnode<numSideNodes; ++jnode) {
            for (int eq=0; eq<neq; ++eq) {
              const LO lcol = elem_dof_lids(elem_LID,side_offsets(eq,tb,jnode));
              const ST val = res.fastAccessDx(columnsOffset + ilev*neq*numSideNodes + neq*jnode + eq);
              Jac_kokkos.sumIntoValues(lrow,&lcol,1,&val,false,is_atomic);
            }
          }
        }
        // Non-column-coupled part
        for (int icol=0; icol<ncols_volume; ++icol) {
          const LO lcol = elem_dof_lids(side_elem_LID,icol);
          const ST val = res.fastAccessDx(icol);
          Jac_kokkos.sumIntoValues(lrow,&lcol,1,&val,false,is_atomic);
        }
      }
    }
  });
}

// **********************************************************************
template<>
void ScatterResidualWithExtrudedField<AlbanyTraits::Jacobian, AlbanyTraits>::
evaluateFields(typename AlbanyTraits::EvalData workset)
{
  // This happens only once
  this->gather_fields_offsets (workset.disc->getDOFManager());

  constexpr auto ALL = Kokkos::ALL();

  const auto& layers_data  = workset.disc->getMeshStruct()->layers_data;
  const auto  bot = layers_data.bot_side_pos;
  const auto  top = layers_data.top_side_pos;

  // Pick element layer that contains the field level
  const auto field_layer = fieldLevel==0 ? 0 : fieldLevel-1;
  const int field_pos = field_layer==fieldLevel ? bot : top;

  const auto& elem_lids     = workset.disc->getElementLIDs_host(workset.wsIndex);
  const auto& dof_mgr       = workset.disc->getDOFManager();
  const auto& elem_dof_lids = dof_mgr->elem_dof_lids().host();
  const auto& node_dof_mgr = workset.disc->getNodeDOFManager();
  
  // Note: grab offsets on top/bot ordered in the same way as on side $field_pos
  //       to guarantee corresponding nodes are vertically aligned.
  const auto& top_nodes = node_dof_mgr->getGIDFieldOffsetsSide(0,top,field_pos);
  const auto& bot_nodes = node_dof_mgr->getGIDFieldOffsetsSide(0,bot,field_pos);

  const auto& field_offsets = dof_mgr->getGIDFieldOffsetsSide(offset2DField,field_pos);

  const int neq = dof_mgr->getNumFields();
  const int nunk = this->numNodes*(neq-1);
  const int numSideNodes = field_offsets.size();
  const int numNodes = this->numNodes;
  const int numFields = this->numFields;
  const int offset = this->offset;
  const int field2D = offset2DField;

  const auto sol_offsets_h = this->m_fields_offsets.host();
  const auto sol_offsets   = this->m_fields_offsets.dev();

  // Offsets of all the dofs except the extruded field
  Kokkos::View<int*,PHX::Device> unk_offsets("unk_offsets",nunk);
  auto unk_offsets_h = Kokkos::create_mirror_view(unk_offsets);
  for (int node=0, i=0; node<numNodes; ++node){
    for (int eq=0; eq<neq; ++eq) {
      if(eq != offset2DField) {
        unk_offsets_h(i++) = sol_offsets_h(node,eq);
      }
    }
  }
  Kokkos::deep_copy(unk_offsets,unk_offsets_h);

  // Top/bottom nodes and dof offsets, ordered as on side $field_pos
  // to guarantee corresponding nodes are vertically aligned.
  Kokkos::View<int**,PHX::Device>  side_nodes("side_nodes",2,numSideNodes);
  Kokkos::View<int***,PHX::Device> side_offsets("side_offsets",numFields,2,numSideNodes);
  auto side_nodes_h   = Kokkos::create_mirror_view(side_nodes);
  auto side_offsets_h = Kokkos::create_mirror_view(side_offsets);
  for (int node2d=0; node2d<numSideNodes; ++node2d) {
    side_nodes_h(0,node2d) = top_nodes[node2d];
    side_nodes_h(1,node2d) = bot_nodes[node2d];
  }
  for (int eq=0; eq<numFields; ++eq) {
    if(eq != offset2DField) {
      const auto& top_offsets = dof_mgr->getGIDFieldOffsetsSide(eq,top,field_pos);
      const auto& bot_offsets = dof_mgr->getGIDFieldOffsetsSide(eq,bot,field_pos);
      for (int node2d=0; node2d<numSideNodes; ++node2d) {
        side_offsets_h(eq,0,node2d) = top_offsets[node2d];
        side_offsets_h(eq,1,node2d) = bot_offsets[node2d];
      }
    }
  }
  Kokkos::deep_copy(side_nodes,side_nodes_h);
  Kokkos::deep_copy(side_offsets,side_offsets_h);

  // Lids of the 2d field in the cell containing the level where the field is
  Kokkos::View<LO**,PHX::Device> field_cols("field_cols",workset.numCells,numNodes);
  auto field_cols_h = Kokkos::create_mirror_view(field_cols);
  for (size_t cell=0; cell<workset.numCells; ++cell ) {
    const int elem_LID = elem_lids(cell);
    const int basal_elem_LID = layers_data.cell.lid->getColumnId(elem_LID);
    const int field_elem_LID = layers_data.cell.lid->getId(basal_elem_LID,field_layer);
    for (int node=0; node<numSideNodes; ++node) {
      field_cols_h(cell,bot_nodes[node]) = elem_dof_lids(field_elem_LID,field_offsets[node]);
      field_cols_h(cell,top_nodes[node]) = elem_dof_lids(field_elem_LID,field_offsets[node]);
    }
  }
  Kokkos::deep_copy(field_cols,field_cols_h);

  const auto ws_elem_lids = workset.disc->getWsElementLIDs().dev();
  const auto elem_lids_d  = Kokkos::subview(ws_elem_lids,workset.wsIndex,ALL);
  const auto elem_dof_lids_d = dof_mgr->elem_dof_lids().dev();

  Albany::ThyraVDeviceView<ST> f_data;
  const bool scatter_f = Teuchos::nonnull(workset.f);
  if (scatter_f) {
    f_data = workset.f_kokkos;
  }
  auto Jac_kokkos = workset.Jac_kokkos;
  const auto resid = this->device_resid;

  Kokkos::parallel_for(this->getName(),
                       RangePolicy(0,workset.numCells),
                       KOKKOS_LAMBDA(const int cell) {
    const auto dof_lids = Kokkos::subview(elem_dof_lids_d,elem_lids_d(cell),ALL);

    // 1. All contributions except the extruded field
    for (int node=0; node<numNodes; ++node) {
      for (int eq=0; eq<numFields; ++eq) {
        if(offset + eq != field2D) {
          auto res = resid.get(cell,node,eq);

          const LO lrow = dof_lids(sol_offsets(node,offset + eq));
          if (scatter_f) {
            KU::atomic_add<ExecutionSpace>(&f_data(lrow), res.val());
          }

          // Need to do derivs one-by-one, since we have a 2-level indirection
          for (int lunk = 0; lunk < nunk; lunk++) {
            const LO lcol = dof_lids(unk_offsets(lunk));
            const ST val = res.fastAccessDx(unk_offsets(lunk));
            Jac_kokkos.sumIntoValues(lrow,&lcol,1,&val,false,is_atomic);
          }
        }
      }
    }

    // 2. Contribution of the extruded field
    for (int node2d=0; node2d<numSideNodes; ++node2d) {
      for (int eq=0; eq<numFields; ++eq) {
        if(eq == field2D) continue;
        for (int tb=0; tb<2; ++tb) {
          const LO lrow = dof_lids(side_offsets(eq,tb,node2d));
          auto res = resid.get(cell,side_nodes(tb,node2d),offset+eq);

          if (scatter_f) {
            KU::atomic_add<ExecutionSpace>(&f_data(lrow), res.val());
          }

          // Need to do derive one-by-one, since they are strided
          for (int i=0; i<numNodes; ++i) {
            const LO lcol = field_cols(cell,i);
            const ST val = res.fastAccessDx(sol_offsets(i,field2D));
            Jac_kokkos.sumIntoValues(lrow,&lcol,1,&val,false,is_atomic);
          }
        }
      }
    }
  });
}

// **********************************************************************
// Specializations: DistParamDeriv and HessianVec
// **********************************************************************
//
// These use the base class implementation, which is not correct for some
// derivatives, and throw for them instead:
//  - The Hessian-vector products H_xx(f)v and H_xp(f)v w.r.t. the solution: the
//    base class only scatters the derivatives w.r.t. the fields of this
//    evaluator, to the dofs of the cell nodes, while the Jacobian
//    specializations above also scatter the derivatives w.r.t. the other
//    fields, w.r.t. the 2D field (to its dofs at the field level), and the
//    column-coupled derivatives.
//  - The derivatives w.r.t. a parameter extruded from one level of the mesh,
//    when they map the parameter dofs to the cell nodes (transposed df/dp,
//    H_px(f)v, H_pp(f)v): the base class maps the parameter derivative of a
//    node to the parameter dof of that node, rather than to the dof at the
//    level of the parameter (see ScatterResidualWithExtrudedParams).
// The other derivatives (df/dp times a direction, H_px(f)v and H_pp(f)v for
// parameters that are not extruded) are computed by the base class.

namespace {

void throwIfHessVecWrtSolution (const PHAL::Workset& workset, const std::string& name)
{
  const auto& hws = workset.hessianWorkset;
  TEUCHOS_TEST_FOR_EXCEPTION (
      !hws.hess_vec_prod_f_xx.is_null() || !hws.hess_vec_prod_f_xp.is_null(), std::logic_error,
      "Error! " << name << ": the Hessian-vector products H_xx(f)v and H_xp(f)v of\n"
      "       the residual are not implemented for an equation of (or depending on) a 2D field\n"
      "       defined at one level of an extruded mesh. Use finite differences instead, setting\n"
      "       Problem->Hessian->\"Use AD for Hessian-vector products (default)\" = false.\n");
}

void throwIfExtrudedParam (const std::set<std::string>& extrudedParamNames,
                           const std::string& pname, const std::string& name)
{
  TEUCHOS_TEST_FOR_EXCEPTION (extrudedParamNames.count(pname)>0, std::logic_error,
      "Error! " << name << ": the derivatives w.r.t. the parameter \"" << pname << "\",\n"
      "       extruded from one level of the mesh, are not implemented for this residual\n"
      "       (transposed df/dp, H_px(f)v, H_pp(f)v).\n");
}

} // anonymous namespace

template<>
void ScatterResidual2D<AlbanyTraits::DistParamDeriv, AlbanyTraits>::
evaluateFields(typename AlbanyTraits::EvalData workset)
{
  if (workset.transpose_dist_param_deriv && workset.local_Vp.size()>0) {
    throwIfExtrudedParam(extrudedParamNames,workset.dist_param_deriv_name,"PHAL::ScatterResidual2D");
  }
  Base::evaluateFields(workset);
}

template<>
void ScatterResidual2D<AlbanyTraits::HessianVec, AlbanyTraits>::
evaluateFields(typename AlbanyTraits::EvalData workset)
{
  throwIfHessVecWrtSolution(workset,"PHAL::ScatterResidual2D");
  const auto& hws = workset.hessianWorkset;
  if (!hws.hess_vec_prod_f_px.is_null() || !hws.hess_vec_prod_f_pp.is_null()) {
    throwIfExtrudedParam(extrudedParamNames,workset.dist_param_deriv_name,"PHAL::ScatterResidual2D");
  }
  Base::evaluateFields(workset);
}

template<>
void ScatterResidualWithExtrudedField<AlbanyTraits::DistParamDeriv, AlbanyTraits>::
evaluateFields(typename AlbanyTraits::EvalData workset)
{
  if (workset.transpose_dist_param_deriv && workset.local_Vp.size()>0) {
    throwIfExtrudedParam(extrudedParamNames,workset.dist_param_deriv_name,
                         "PHAL::ScatterResidualWithExtrudedField");
  }
  Base::evaluateFields(workset);
}

template<>
void ScatterResidualWithExtrudedField<AlbanyTraits::HessianVec, AlbanyTraits>::
evaluateFields(typename AlbanyTraits::EvalData workset)
{
  throwIfHessVecWrtSolution(workset,"PHAL::ScatterResidualWithExtrudedField");
  const auto& hws = workset.hessianWorkset;
  if (!hws.hess_vec_prod_f_px.is_null() || !hws.hess_vec_prod_f_pp.is_null()) {
    throwIfExtrudedParam(extrudedParamNames,workset.dist_param_deriv_name,
                         "PHAL::ScatterResidualWithExtrudedField");
  }
  Base::evaluateFields(workset);
}

} // namespace PHAL
