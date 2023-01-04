//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_SDIRICHLET_DEF_HPP
#define PHAL_SDIRICHLET_DEF_HPP

#include "PHAL_SDirichlet.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "Teuchos_TestForException.hpp"

#include "Albany_CombineAndScatterManager.hpp"
#include "Albany_Macros.hpp"
#include "Albany_ThyraUtils.hpp"

//#define DEBUG_OUTPUT

namespace PHAL {

//
// Specialization: Residual
//
template <typename Traits>
SDirichlet<PHAL::AlbanyTraits::Residual, Traits>::SDirichlet(
    Teuchos::ParameterList& p)
    : PHAL::DirichletBase<PHAL::AlbanyTraits::Residual, Traits>(p)
{
  return;
}

//
//
//
template <typename Traits>
void
SDirichlet<PHAL::AlbanyTraits::Residual, Traits>::preEvaluate(
    typename Traits::EvalData dirichlet_workset)
{
  Teuchos::RCP<Thyra_Vector const> x = dirichlet_workset.x;
  Teuchos::ArrayRCP<ST>            x_view =
      Teuchos::arcp_const_cast<ST>(Albany::getLocalData(x));
  // Grab the vector of node GIDs for this Node Set ID
  std::vector<std::vector<int>> const& ns_nodes =
      dirichlet_workset.nodeSets->find(this->nodeSetID)->second;
  for (size_t ns_node = 0; ns_node < ns_nodes.size(); ns_node++) {
    int const dof = ns_nodes[ns_node][this->offset];
    x_view[dof]   = this->value;
  }
}

//
//
//
template <typename Traits>
void
SDirichlet<PHAL::AlbanyTraits::Residual, Traits>::evaluateFields(
    typename Traits::EvalData dirichlet_workset)
{
  Teuchos::RCP<Thyra_Vector> f      = dirichlet_workset.f;
  Teuchos::ArrayRCP<ST>      f_view = Albany::getNonconstLocalData(f);

  // Grab the vector of node GIDs for this Node Set ID
  std::vector<std::vector<int>> const& ns_nodes =
      dirichlet_workset.nodeSets->find(this->nodeSetID)->second;

  for (size_t ns_node = 0; ns_node < ns_nodes.size(); ns_node++) {
    int const dof = ns_nodes[ns_node][this->offset];
    f_view[dof]   = 0.0;
  }
}

//
// Specialization: Jacobian
//
template <typename Traits>
SDirichlet<PHAL::AlbanyTraits::Jacobian, Traits>::SDirichlet(
    Teuchos::ParameterList& p)
    : PHAL::DirichletBase<PHAL::AlbanyTraits::Jacobian, Traits>(p)
{
}


template <typename Traits>
void
SDirichlet<PHAL::AlbanyTraits::Jacobian, Traits>::preEvaluate(
    typename Traits::EvalData dirichlet_workset)
{
  if(Teuchos::nonnull(dirichlet_workset.f)) {
    Teuchos::RCP<Thyra_Vector const> x = dirichlet_workset.x;
    Teuchos::ArrayRCP<ST>            x_view =
        Teuchos::arcp_const_cast<ST>(Albany::getLocalData(x));
    // Grab the vector of node GIDs for this Node Set ID
    std::vector<std::vector<int>> const& ns_nodes =
        dirichlet_workset.nodeSets->find(this->nodeSetID)->second;
    for (size_t ns_node = 0; ns_node < ns_nodes.size(); ns_node++) {
      int const dof = ns_nodes[ns_node][this->offset];
      x_view[dof]   = this->value.val();
    }
  }
}

//
//
//
template <typename Traits>
void
SDirichlet<PHAL::AlbanyTraits::Jacobian, Traits>::set_row_and_col_is_dbc(
    typename Traits::EvalData dirichlet_workset)
{
  Teuchos::RCP<const Thyra_LinearOp> J = dirichlet_workset.Jac;

  auto  range_vs  = J->range();
  auto  col_vs    = Albany::getColumnSpace(J);
  auto& ns_nodes  = dirichlet_workset.nodeSets->find(this->nodeSetID)->second;
  auto  domain_vs = range_vs;  // we are assuming this!

  row_is_dbc_ = Thyra::createMember(range_vs);
  col_is_dbc_ = Thyra::createMember(col_vs);
  row_is_dbc_->assign(0.0);
  col_is_dbc_->assign(0.0);

  auto row_is_dbc_data = Albany::getNonconstLocalData(row_is_dbc_);

  for (size_t ns_node = 0; ns_node < ns_nodes.size(); ns_node++) {
    auto dof             = ns_nodes[ns_node][this->offset];
    row_is_dbc_data[dof] = 1.0;
  }

  auto cas_manager = Albany::createCombineAndScatterManager(domain_vs, col_vs);
  cas_manager->scatter(row_is_dbc_, col_is_dbc_, Albany::CombineMode::INSERT);
}

//
//
//
template <typename Traits>
void
SDirichlet<PHAL::AlbanyTraits::Jacobian, Traits>::evaluateFields(
    typename Traits::EvalData dirichlet_workset)
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
SDirichlet<PHAL::AlbanyTraits::Tangent, Traits>::SDirichlet(
    Teuchos::ParameterList& p)
    : PHAL::DirichletBase<PHAL::AlbanyTraits::Tangent, Traits>(p)
{
}

template <typename Traits>
void
SDirichlet<PHAL::AlbanyTraits::Tangent, Traits>::preEvaluate(
    typename Traits::EvalData dirichlet_workset)
{
  if(Teuchos::nonnull(dirichlet_workset.f)) {
    Teuchos::RCP<Thyra_Vector const> x = dirichlet_workset.x;
    Teuchos::ArrayRCP<ST>            x_view =
        Teuchos::arcp_const_cast<ST>(Albany::getLocalData(x));
    // Grab the vector of node GIDs for this Node Set ID
    std::vector<std::vector<int>> const& ns_nodes =
        dirichlet_workset.nodeSets->find(this->nodeSetID)->second;
    for (size_t ns_node = 0; ns_node < ns_nodes.size(); ns_node++) {
      int const dof = ns_nodes[ns_node][this->offset];
      x_view[dof]   = this->value.val();
    }
  }
}

//
//
//

template <typename Traits>
void SDirichlet<PHAL::AlbanyTraits::Tangent, Traits>::evaluateFields(
    typename Traits::EvalData dirichlet_workset)
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
  const std::vector<std::vector<int> >& nsNodes = dirichlet_workset.nodeSets->find(this->nodeSetID)->second;

  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
    int lunk = nsNodes[inode][this->offset];

    if (dirichlet_workset.f != Teuchos::null) {
      f_nonconstView[lunk] = 0.0;
    }

    if (JV != Teuchos::null) {
      for (int i=0; i<dirichlet_workset.num_cols_x; i++) {
        JV_nonconst2dView[i][lunk] = j_coeff*Vx_const2dView[i][lunk];
      }
    }

    if (fp != Teuchos::null) {
      for (int i=0; i<dirichlet_workset.num_cols_p; i++) {
        fp_nonconst2dView[i][lunk] = -this->value.dx(dirichlet_workset.param_offset+i);
      }
    }
  }
}

//
// Specialization: DistParamDeriv
//
template <typename Traits>
SDirichlet<PHAL::AlbanyTraits::DistParamDeriv, Traits>::SDirichlet(
    Teuchos::ParameterList& p)
    : PHAL::DirichletBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p)
{
  return;
}

// **********************************************************************
template <typename Traits>
void
SDirichlet<PHAL::AlbanyTraits::DistParamDeriv, Traits>::preEvaluate(
    typename Traits::EvalData dirichletWorkset)
{
  bool trans = dirichletWorkset.transpose_dist_param_deriv;
  if (trans) {
    auto Vp_bc = dirichletWorkset.Vp_bc;
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>> Vp_nonconst2dView = Albany::getNonconstLocalData(Vp_bc);

    int num_cols = Vp_bc->domain()->dim();

    const std::vector<std::vector<int> >& nsNodes =
      dirichletWorkset.nodeSets->find(this->nodeSetID)->second;

    for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
      int lunk = nsNodes[inode][this->offset];

      for (int col=0; col<num_cols; ++col) {
        Vp_nonconst2dView[col][lunk] = 0.0;
       }
    }
  }
}

// **********************************************************************
template <typename Traits>
void
SDirichlet<PHAL::AlbanyTraits::DistParamDeriv, Traits>::evaluateFields(
    typename Traits::EvalData dirichletWorkset)
{
  bool trans    = dirichletWorkset.transpose_dist_param_deriv;

  if (!trans) {
    // for (df/dp)*V we zero out corresponding entries in df/dp
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>> fpV_nonconst2dView =
        Albany::getNonconstLocalData(dirichletWorkset.fpV);

    int  num_cols = dirichletWorkset.fpV->domain()->dim();
    const std::vector<std::vector<int>>& nsNodes =
      dirichletWorkset.nodeSets->find(this->nodeSetID)->second;
    
    for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
      int lunk = nsNodes[inode][this->offset];
      for (int col = 0; col < num_cols; ++col) {
        fpV_nonconst2dView[col][lunk] = 0.0;
      }
    }
  }
}

//
// Specialization: HessianVec
//
template <typename Traits>
SDirichlet<PHAL::AlbanyTraits::HessianVec, Traits>::SDirichlet(
    Teuchos::ParameterList& p)
    : PHAL::DirichletBase<PHAL::AlbanyTraits::HessianVec, Traits>(p)
{
  return;
}

template <typename Traits>
void
SDirichlet<PHAL::AlbanyTraits::HessianVec, Traits>::preEvaluate(
    typename Traits::EvalData dirichlet_workset)
{
  const bool f_multiplier_is_active = !dirichlet_workset.hessianWorkset.f_multiplier.is_null();

  const std::vector<std::vector<int> >& nsNodes = dirichlet_workset.nodeSets->find(this->nodeSetID)->second;

  if(f_multiplier_is_active) {
    auto f_multiplier_data = Albany::getNonconstLocalData(dirichlet_workset.hessianWorkset.f_multiplier);

    for (size_t ns_node = 0; ns_node < nsNodes.size(); ns_node++) {
      int const dof = nsNodes[ns_node][this->offset];
      f_multiplier_data[dof] = 0.;
    }
  }
}

//
//
//
template <typename Traits>
void
SDirichlet<PHAL::AlbanyTraits::HessianVec, Traits>::evaluateFields(
    typename Traits::EvalData dirichlet_workset)
{
  const bool f_xx_is_active = !dirichlet_workset.hessianWorkset.hess_vec_prod_f_xx.is_null();
  const bool f_xp_is_active = !dirichlet_workset.hessianWorkset.hess_vec_prod_f_xp.is_null();

  Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST> > hess_vec_prod_f_xx_data, hess_vec_prod_f_xp_data;

  if(f_xx_is_active)
    hess_vec_prod_f_xx_data = Albany::getNonconstLocalData(dirichlet_workset.hessianWorkset.overlapped_hess_vec_prod_f_xx);
  if(f_xp_is_active)
    hess_vec_prod_f_xp_data = Albany::getNonconstLocalData(dirichlet_workset.hessianWorkset.overlapped_hess_vec_prod_f_xp);

  const std::vector<std::vector<int> >& nsNodes = dirichlet_workset.nodeSets->find(this->nodeSetID)->second;

  for (size_t ns_node = 0; ns_node < nsNodes.size(); ns_node++) {
    int const dof = nsNodes[ns_node][this->offset];

    if(f_xx_is_active)
      hess_vec_prod_f_xx_data[0][dof] = 0.;
    if(f_xp_is_active)
      hess_vec_prod_f_xp_data[0][dof] = 0.;
  }
}

}  // namespace PHAL

#endif  // PHAL_SDIRICHLET_DEF_HPP
