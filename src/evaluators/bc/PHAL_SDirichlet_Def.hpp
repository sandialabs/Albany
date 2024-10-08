//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_SDIRICHLET_DEF_HPP
#define PHAL_SDIRICHLET_DEF_HPP

#include "PHAL_SDirichlet.hpp"

#include "Albany_CombineAndScatterManager.hpp"
#include "Albany_AbstractDiscretization.hpp"
#include "Albany_Macros.hpp"
#include "Albany_ThyraUtils.hpp"

#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "Teuchos_TestForException.hpp"

//#define DEBUG_OUTPUT

namespace PHAL {

//
// Specialization: Residual
//
template <typename Traits>
SDirichlet<PHAL::AlbanyTraits::Residual, Traits>::
SDirichlet(Teuchos::ParameterList& p)
 : PHAL::DirichletBase<PHAL::AlbanyTraits::Residual, Traits>(p)
{
  // Nothing to do here
}

//
//
//
template <typename Traits>
void
SDirichlet<PHAL::AlbanyTraits::Residual, Traits>::
preEvaluate(typename Traits::EvalData dirichlet_workset)
{
  Teuchos::RCP<Thyra_Vector const> x = dirichlet_workset.x;
  Teuchos::ArrayRCP<ST>            x_view =
      Teuchos::arcp_const_cast<ST>(Albany::getLocalData(x));
  // Grab the vector of node GIDs for this Node Set ID
  const auto& ns_node_elem_pos = dirichlet_workset.nodeSets->at(this->nodeSetID);
  const auto& sol_dof_mgr   = dirichlet_workset.disc->getDOFManager();
  const auto& sol_elem_dof_lids = sol_dof_mgr->elem_dof_lids().host();
  const auto& sol_offsets = sol_dof_mgr->getGIDFieldOffsets(this->offset);
  for (const auto& ep : ns_node_elem_pos) {
    const int ielem = ep.first;
    const int pos   = ep.second;
    const int x_lid = sol_elem_dof_lids(ielem,sol_offsets[pos]);
    x_view[x_lid]   = this->value;
  }
}

//
//
//
template <typename Traits>
void
SDirichlet<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData dirichlet_workset)
{
  Teuchos::RCP<Thyra_Vector> f      = dirichlet_workset.f;
  Teuchos::ArrayRCP<ST>      f_view = Albany::getNonconstLocalData(f);

  const auto& ns_node_elem_pos = dirichlet_workset.nodeSets->at(this->nodeSetID);
  const auto& sol_dof_mgr   = dirichlet_workset.disc->getDOFManager();
  const auto& sol_elem_dof_lids = sol_dof_mgr->elem_dof_lids().host();
  const auto& sol_offsets = sol_dof_mgr->getGIDFieldOffsets(this->offset);
  for (const auto& ep : ns_node_elem_pos) {
    const int ielem = ep.first;
    const int pos   = ep.second;
    const int x_lid = sol_elem_dof_lids(ielem,sol_offsets[pos]);
    f_view[x_lid]   = 0.0;
  }
}

//
// Specialization: Jacobian
//
template <typename Traits>
SDirichlet<PHAL::AlbanyTraits::Jacobian, Traits>::
SDirichlet(Teuchos::ParameterList& p)
 : PHAL::DirichletBase<PHAL::AlbanyTraits::Jacobian, Traits>(p)
{
  // Nothing to do here
}


template <typename Traits>
void
SDirichlet<PHAL::AlbanyTraits::Jacobian, Traits>::
preEvaluate(typename Traits::EvalData dirichlet_workset)
{
  // Check for early return
  if(dirichlet_workset.f.is_null()) {
    return;
  }

  Teuchos::RCP<Thyra_Vector const> x = dirichlet_workset.x;
  Teuchos::ArrayRCP<ST>            x_view =
      Teuchos::arcp_const_cast<ST>(Albany::getLocalData(x));

  const auto& ns_node_elem_pos = dirichlet_workset.nodeSets->at(this->nodeSetID);
  const auto& sol_dof_mgr   = dirichlet_workset.disc->getDOFManager();
  const auto& sol_elem_dof_lids = sol_dof_mgr->elem_dof_lids().host();
  const auto& sol_offsets = sol_dof_mgr->getGIDFieldOffsets(this->offset);
  for (const auto& ep : ns_node_elem_pos) {
    const int ielem = ep.first;
    const int pos   = ep.second;
    const int x_lid = sol_elem_dof_lids(ielem,sol_offsets[pos]);
    x_view[x_lid]   = this->value.val();
  }
}

//
//
//
template <typename Traits>
void
SDirichlet<PHAL::AlbanyTraits::Jacobian, Traits>::
set_row_and_col_is_dbc(typename Traits::EvalData dirichlet_workset)
{
  Teuchos::RCP<const Thyra_LinearOp> J = dirichlet_workset.Jac;

  auto  range_vs  = J->range();
  auto  col_vs    = Albany::getColumnSpace(J);
  auto  domain_vs = range_vs;  // we are assuming this!

  row_is_dbc_ = Thyra::createMember(range_vs);
  col_is_dbc_ = Thyra::createMember(col_vs);
  row_is_dbc_->assign(0.0);
  col_is_dbc_->assign(0.0);

  {
    auto row_is_dbc_data = Albany::getNonconstLocalData(row_is_dbc_);

    const auto& ns_node_elem_pos = dirichlet_workset.nodeSets->at(this->nodeSetID);
    const auto& sol_dof_mgr   = dirichlet_workset.disc->getDOFManager();
    const auto& sol_elem_dof_lids = sol_dof_mgr->elem_dof_lids().host();
    const auto& sol_offsets = sol_dof_mgr->getGIDFieldOffsets(this->offset);
    for (const auto& ep : ns_node_elem_pos) {
      const int ielem = ep.first;
      const int pos   = ep.second;
      const int x_lid = sol_elem_dof_lids(ielem,sol_offsets[pos]);
      row_is_dbc_data[x_lid] = 1.0;
    }
  }

  auto cas_manager = Albany::createCombineAndScatterManager(domain_vs, col_vs);
  cas_manager->scatter(row_is_dbc_, col_is_dbc_, Albany::CombineMode::INSERT);
}

//
//
//
template <typename Traits>
void
SDirichlet<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData dirichlet_workset)
{
  Teuchos::RCP<const Thyra_Vector> x = dirichlet_workset.x;
  Teuchos::RCP<Thyra_Vector>       f = dirichlet_workset.f;
  Teuchos::RCP<Thyra_LinearOp>     J = dirichlet_workset.Jac;

  bool const fill_residual = f != Teuchos::null;

  auto f_view = fill_residual ? Albany::getNonconstLocalData(f) : Teuchos::null;
  auto x_view = fill_residual ?
                    Teuchos::arcp_const_cast<ST>(Albany::getLocalData(x)) :
                    Teuchos::null;

  Teuchos::Array<ST> entries;
  Teuchos::Array<LO> indices;

  this->set_row_and_col_is_dbc(dirichlet_workset);

  auto     col_is_dbc_data = Albany::getLocalData(col_is_dbc_.getConst());
  auto     range_spmd_vs   = Albany::getSpmdVectorSpace(J->range());
  const LO num_local_rows  = range_spmd_vs->localSubDim();

  for (LO local_row = 0; local_row < num_local_rows; ++local_row) {
    Albany::getLocalRowValues(J, local_row, indices, entries);

    auto row_is_dbc = col_is_dbc_data[local_row] > 0;

    if (row_is_dbc && fill_residual == true) {
      f_view[local_row] = 0.0;
      x_view[local_row] = this->value.val();
    }

    const LO num_row_entries = entries.size();

    for (LO row_entry = 0; row_entry < num_row_entries; ++row_entry) {
      auto local_col         = indices[row_entry];
      auto is_diagonal_entry = local_col == local_row;
      if (is_diagonal_entry) { continue; }

      auto col_is_dbc = col_is_dbc_data[local_col] > 0;
      if (row_is_dbc || col_is_dbc) { entries[row_entry] = 0.0; }
    }
    Albany::setLocalRowValues(J, local_row, indices(), entries());
  }
}

//
// Specialization: Tangent
//
template <typename Traits>
SDirichlet<PHAL::AlbanyTraits::Tangent, Traits>::
SDirichlet(Teuchos::ParameterList& p)
 : PHAL::DirichletBase<PHAL::AlbanyTraits::Tangent, Traits>(p)
{
  // Nothing to do here
}

template <typename Traits>
void
SDirichlet<PHAL::AlbanyTraits::Tangent, Traits>::
preEvaluate(typename Traits::EvalData dirichlet_workset)
{
  // Check for early return
  if(dirichlet_workset.f.is_null()) {
    return;
  }

  Teuchos::RCP<Thyra_Vector const> x = dirichlet_workset.x;
  Teuchos::ArrayRCP<ST>            x_view =
      Teuchos::arcp_const_cast<ST>(Albany::getLocalData(x));

  const auto& ns_node_elem_pos = dirichlet_workset.nodeSets->at(this->nodeSetID);
  const auto& sol_dof_mgr   = dirichlet_workset.disc->getDOFManager();
  const auto& sol_elem_dof_lids = sol_dof_mgr->elem_dof_lids().host();
  const auto& sol_offsets = sol_dof_mgr->getGIDFieldOffsets(this->offset);
  for (const auto& ep : ns_node_elem_pos) {
    const int ielem = ep.first;
    const int pos   = ep.second;
    const int x_lid = sol_elem_dof_lids(ielem,sol_offsets[pos]);
    x_view[x_lid]   = this->value.val();
  }
}

//
//
//

template <typename Traits>
void SDirichlet<PHAL::AlbanyTraits::Tangent, Traits>::
evaluateFields(typename Traits::EvalData dirichlet_workset)
{
  Teuchos::RCP<const Thyra_Vector>       x  = dirichlet_workset.x;
  Teuchos::RCP<const Thyra_MultiVector> Vx = dirichlet_workset.Vx;
  Teuchos::RCP<Thyra_Vector>             f  = dirichlet_workset.f;
  Teuchos::RCP<Thyra_MultiVector>       fp = dirichlet_workset.fp;
  Teuchos::RCP<Thyra_MultiVector>       JV = dirichlet_workset.JV;

  Teuchos::ArrayRCP<ST>       f_nonconstView;

  Teuchos::ArrayRCP<Teuchos::ArrayRCP<const ST>> Vx_const2dView;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>>       JV_nonconst2dView;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>>       fp_nonconst2dView;

  if (f != Teuchos::null) {
    f_nonconstView = Albany::getNonconstLocalData(f);
  }
  if (JV != Teuchos::null) {
    JV_nonconst2dView = Albany::getNonconstLocalData(JV);
    Vx_const2dView    = Albany::getLocalData(Vx);
  }
  if (fp != Teuchos::null) {
    fp_nonconst2dView = Albany::getNonconstLocalData(fp);
  }

  const RealType j_coeff = dirichlet_workset.j_coeff;
  const auto& ns_node_elem_pos = dirichlet_workset.nodeSets->at(this->nodeSetID);
  const auto& sol_dof_mgr   = dirichlet_workset.disc->getDOFManager();
  const auto& sol_elem_dof_lids = sol_dof_mgr->elem_dof_lids().host();
  const auto& sol_offsets = sol_dof_mgr->getGIDFieldOffsets(this->offset);
  for (const auto& ep : ns_node_elem_pos) {
    const int ielem = ep.first;
    const int pos   = ep.second;
    const int x_lid = sol_elem_dof_lids(ielem,sol_offsets[pos]);

    if (dirichlet_workset.f != Teuchos::null) {
      f_nonconstView[x_lid] = 0.0;
    }

    if (JV != Teuchos::null) {
      for (int i=0; i<dirichlet_workset.num_cols_x; i++) {
        JV_nonconst2dView[i][x_lid] = j_coeff*Vx_const2dView[i][x_lid];
      }
    }

    if (fp != Teuchos::null) {
      for (int i=0; i<dirichlet_workset.num_cols_p; i++) {
        fp_nonconst2dView[i][x_lid] = -this->value.dx(dirichlet_workset.param_offset+i);
      }
    }
  }
}

//
// Specialization: DistParamDeriv
//
template <typename Traits>
SDirichlet<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
SDirichlet(Teuchos::ParameterList& p)
 : PHAL::DirichletBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p)
{
  // Nothing to do here
}

// **********************************************************************
template <typename Traits>
void SDirichlet<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
preEvaluate(typename Traits::EvalData dirichlet_workset)
{
  bool trans    = dirichlet_workset.transpose_dist_param_deriv;
  if (trans) {
    // For (df/dp)^T*V we zero out corresponding entries in V
    auto Vp_bc = dirichlet_workset.Vp_bc;
    auto data = Albany::getNonconstLocalData(Vp_bc);
    int num_cols = Vp_bc->domain()->dim();

    const auto& ns_node_elem_pos = dirichlet_workset.nodeSets->at(this->nodeSetID);
    const auto& sol_dof_mgr   = dirichlet_workset.disc->getDOFManager();
    const auto& sol_elem_dof_lids = sol_dof_mgr->elem_dof_lids().host();
    const auto& sol_offsets = sol_dof_mgr->getGIDFieldOffsets(this->offset);

    for (const auto& ep : ns_node_elem_pos) {
      const int ielem = ep.first;
      const int pos   = ep.second;
      const int x_lid = sol_elem_dof_lids(ielem,sol_offsets[pos]);
      for (int col = 0; col < num_cols; ++col) {
        data[col][x_lid] = 0.0;
      }
    }
  }
}

// **********************************************************************
template <typename Traits>
void SDirichlet<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  bool trans = dirichletWorkset.transpose_dist_param_deriv;
  if (!trans) {
    // for (df/dp)*V we zero out corresponding entries in df/dp
    auto fpV = dirichletWorkset.fpV;
    auto fpV_data = Albany::getNonconstLocalData(fpV);
    int  num_cols = fpV->domain()->dim();
    const auto& ns_node_elem_pos = dirichletWorkset.nodeSets->at(this->nodeSetID);
    const auto& sol_dof_mgr   = dirichletWorkset.disc->getDOFManager();
    const auto& sol_elem_dof_lids = sol_dof_mgr->elem_dof_lids().host();
    const auto& sol_offsets = sol_dof_mgr->getGIDFieldOffsets(this->offset);

    for (const auto& ep : ns_node_elem_pos) {
      const int ielem = ep.first;
      const int pos   = ep.second;
      const int x_lid = sol_elem_dof_lids(ielem,sol_offsets[pos]);
    
      for (int col = 0; col < num_cols; ++col) {
        fpV_data[col][x_lid] = 0.0;
      }
    }
  }
}

//
// Specialization: HessianVec
//
template <typename Traits>
SDirichlet<PHAL::AlbanyTraits::HessianVec, Traits>::
SDirichlet(Teuchos::ParameterList& p)
 : PHAL::DirichletBase<PHAL::AlbanyTraits::HessianVec, Traits>(p)
{
  // Nothing to do here
}

template <typename Traits>
void
SDirichlet<PHAL::AlbanyTraits::HessianVec, Traits>::
preEvaluate(typename Traits::EvalData dirichlet_workset)
{
  const bool f_multiplier_is_active = !dirichlet_workset.hessianWorkset.f_multiplier.is_null();

  // Check for early return
  if(!f_multiplier_is_active) {
    return;
  }
  auto f_multiplier_data = Albany::getNonconstLocalData(dirichlet_workset.hessianWorkset.f_multiplier);

  const auto& ns_node_elem_pos = dirichlet_workset.nodeSets->at(this->nodeSetID);
  const auto& sol_dof_mgr   = dirichlet_workset.disc->getDOFManager();
  const auto& sol_elem_dof_lids = sol_dof_mgr->elem_dof_lids().host();
  const auto& sol_offsets = sol_dof_mgr->getGIDFieldOffsets(this->offset);

  for (const auto& ep : ns_node_elem_pos) {
    const int ielem = ep.first;
    const int pos   = ep.second;
    const int x_lid = sol_elem_dof_lids(ielem,sol_offsets[pos]);
    f_multiplier_data[x_lid] = 0.;
  }
}

//
//
//
template <typename Traits>
void
SDirichlet<PHAL::AlbanyTraits::HessianVec, Traits>::
evaluateFields(typename Traits::EvalData dirichlet_workset)
{
  const bool f_xx_is_active = !dirichlet_workset.hessianWorkset.hess_vec_prod_f_xx.is_null();
  const bool f_xp_is_active = !dirichlet_workset.hessianWorkset.hess_vec_prod_f_xp.is_null();

  // Check for early return
  if (!f_xx_is_active and !f_xp_is_active) {
    return;
  }

  Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST> > hess_vec_prod_f_xx_data, hess_vec_prod_f_xp_data;

  if(f_xx_is_active)
    hess_vec_prod_f_xx_data = Albany::getNonconstLocalData(dirichlet_workset.hessianWorkset.overlapped_hess_vec_prod_f_xx);
  if(f_xp_is_active)
    hess_vec_prod_f_xp_data = Albany::getNonconstLocalData(dirichlet_workset.hessianWorkset.overlapped_hess_vec_prod_f_xp);

  const auto& ns_node_elem_pos = dirichlet_workset.nodeSets->at(this->nodeSetID);
  const auto& sol_dof_mgr   = dirichlet_workset.disc->getDOFManager();
  const auto& sol_elem_dof_lids = sol_dof_mgr->elem_dof_lids().host();
  const auto& sol_offsets = sol_dof_mgr->getGIDFieldOffsets(this->offset);

  for (const auto& ep : ns_node_elem_pos) {
    const int ielem = ep.first;
    const int pos   = ep.second;
    const int x_lid = sol_elem_dof_lids(ielem,sol_offsets[pos]);

    if(f_xx_is_active)
      hess_vec_prod_f_xx_data[0][x_lid] = 0.;
    if(f_xp_is_active)
      hess_vec_prod_f_xp_data[0][x_lid] = 0.;
  }
}

}  // namespace PHAL

#endif  // PHAL_SDIRICHLET_DEF_HPP
