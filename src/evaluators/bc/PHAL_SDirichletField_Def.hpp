//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"

#include "Albany_ThyraUtils.hpp"
#include "Albany_DistributedParameterLibrary.hpp"
#include "Albany_AbstractDiscretization.hpp"
#include "Albany_GlobalLocalIndexer.hpp"

#include "PHAL_SDirichletField.hpp"

#include "Albany_TpetraThyraUtils.hpp"
#include "Albany_Utils.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iomanip>
#include <set>
#include <vector>

// **********************************************************************
// Genereric Template Code for Constructor and PostRegistrationSetup
// **********************************************************************

namespace PHAL {

namespace {

// DIAGNOSTIC: report the dofs that the Dirichlet evaluator actually writes to, keyed by
// the COORDINATES of the node they belong to. The node set stores (elem_lid,node_pos)
// pairs, and the dof lid is looked up through the solution dof manager, so a node set
// that was rebuilt against stale connectivity would still have the right SIZE (which is
// all the set-count diagnostic checks) while resolving to the wrong dofs entirely.
//
// Coordinates are the invariant to check against: the physical boundary the set is
// classified on does not move when the mesh is adapted, so the set of constrained
// coordinates must be the same before and after (up to the vertices added/removed by
// the adaptation itself). The report prints a checksum over the coordinates and the
// bounding box of the constrained nodes; if a rebuild silently remapped the set to
// interior dofs, the bounding box collapses inward and the checksum changes.
//
// Enabled by setting the env var ALBANY_DEBUG_DBC_COORDS to a non-empty value.
template<typename Workset>
void dirichletDofCoordsReport (const Workset& dirichlet_workset,
                               const std::string& nodeSetID,
                               const int offset)
{
  static const bool enabled = (std::getenv("ALBANY_DEBUG_DBC_COORDS")!=nullptr);
  if (not enabled) return;

  const auto& ns_node_elem_pos = dirichlet_workset.nodeSets->at(nodeSetID);
  const auto  disc        = dirichlet_workset.disc;
  const auto& sol_dof_mgr = disc->getDOFManager();
  const auto& node_dof_mgr= disc->getNodeDOFManager();

  const auto& sol_elem_dof_lids  = sol_dof_mgr->elem_dof_lids().host();
  const auto& node_elem_dof_lids = node_dof_mgr->elem_dof_lids().host();
  const auto& sol_offsets  = sol_dof_mgr->getGIDFieldOffsets(offset);
  const auto& node_offsets = node_dof_mgr->getGIDFieldOffsets(0);

  const auto& coords = disc->getCoordinates();
  const int   ndim   = disc->getNumDim();
  const int   nnodes = coords.size()/ndim;

  double lo[3] = { 1e300, 1e300, 1e300};
  double hi[3] = {-1e300,-1e300,-1e300};
  double chksum = 0.0;
  long long nbad_node = 0, nbad_dof = 0;
  const int sol_len = Albany::getLocalSubdim(dirichlet_workset.f->space());

  // Sorted list of (coords,dof lid), so the report is independent of the order in which
  // the node set happens to enumerate its entries.
  std::vector<std::pair<std::array<double,3>,int>> entries;
  entries.reserve(ns_node_elem_pos.size());

  for (const auto& ep : ns_node_elem_pos) {
    const int ielem = ep.first;
    const int pos   = ep.second;
    const int x_lid = sol_elem_dof_lids(ielem,sol_offsets[pos]);
    const int n_lid = node_elem_dof_lids(ielem,node_offsets[pos]);

    if (x_lid<0 or x_lid>=sol_len) { ++nbad_dof; continue; }
    if (n_lid<0 or n_lid>=nnodes)  { ++nbad_node; continue; }

    std::array<double,3> c = {0,0,0};
    for (int d=0; d<ndim; ++d) {
      c[d] = coords[ndim*n_lid+d];
      lo[d] = std::min(lo[d],c[d]);
      hi[d] = std::max(hi[d],c[d]);
      // Order-independent checksum: each coordinate contributes the same regardless of
      // where it appears in the list.
      chksum += c[d]*c[d] + 0.5*std::abs(c[d]);
    }
    entries.push_back({c,x_lid});
  }

  std::sort(entries.begin(),entries.end());

  // Count the distinct dofs actually constrained. A node set that resolves several
  // (elem,pos) pairs to the SAME dof (as it legitimately does, since a node is shared
  // by several elements) is fine; one that resolves to FEWER distinct dofs than there
  // are distinct constrained coordinates is not.
  std::set<int> distinct_dofs;
  std::set<std::array<double,3>> distinct_coords;
  for (const auto& e : entries) {
    distinct_dofs.insert(e.second);
    distinct_coords.insert(e.first);
  }

  auto out = Teuchos::VerboseObjectBase::getDefaultOStream();
  *out << " [dbc coords] node set '" << nodeSetID << "' dof offset " << offset << ":\n"
       << "     (elem,pos) pairs   : " << ns_node_elem_pos.size() << "\n"
       << "     distinct dofs      : " << distinct_dofs.size() << "\n"
       << "     distinct coords    : " << distinct_coords.size() << "\n"
       << "     out-of-range dofs  : " << nbad_dof << "\n"
       << "     out-of-range nodes : " << nbad_node << "\n"
       << "     coord checksum     : " << std::setprecision(17) << chksum << "\n"
       << "     bounding box       : [" << lo[0] << ", " << hi[0] << "]";
  for (int d=1; d<ndim; ++d) {
    *out << " x [" << lo[d] << ", " << hi[d] << "]";
  }
  *out << "\n";

  // First few constrained nodes, sorted by coordinate, so two runs can be diffed
  // directly to see WHICH nodes gained or lost a constraint.
  const int nshow = std::min<int>(10,entries.size());
  for (int i=0; i<nshow; ++i) {
    *out << "     [" << i << "] (" << entries[i].first[0];
    for (int d=1; d<ndim; ++d) *out << ", " << entries[i].first[d];
    *out << ") -> dof lid " << entries[i].second << "\n";
  }
}

} // anonymous namespace

template <typename EvalT, typename Traits>
SDirichletField_Base<EvalT, Traits>::
SDirichletField_Base(Teuchos::ParameterList& p)
 : PHAL::DirichletBase<EvalT, Traits>(p)
{
  // Get field type and corresponding layouts
  field_name = p.get<std::string>("Field Name");
}

template<typename EvalT, typename Traits>
void
SDirichletField_Base<EvalT, Traits>::
preEvaluate(typename Traits::EvalData dirichlet_workset)
{
#ifdef DEBUG_OUTPUT
  Teuchos::RCP<Teuchos::FancyOStream> out = Teuchos::VerboseObjectBase::getDefaultOStream();
  *out << "SDirichletField preEvaluate " << PHX::print<EvalT>() << "\n";
#endif
  Teuchos::RCP<const Thyra_Vector> x = dirichlet_workset.x;
  Teuchos::ArrayRCP<ST> x_view = Teuchos::arcp_const_cast<ST>(Albany::getLocalData(x));

  const auto& p_dof_mgr = dirichlet_workset.disc->getDOFManager(this->field_name);
  const auto& sol_dof_mgr   = dirichlet_workset.disc->getDOFManager();

  //MP: If the parameter is scalar, then the parameter offset is seto to zero. Otherwise the parameter offset is the same of the solution's one.
  const bool isFieldScalar = p_dof_mgr->getNumFields()==1;
  const int  fieldOffset = isFieldScalar ? 0 : this->offset;

  Teuchos::RCP<const Thyra_Vector> pvec = dirichlet_workset.distParamLib->get(this->field_name)->vector();
  Teuchos::ArrayRCP<const ST> p_constView = Albany::getLocalData(pvec);

  const auto& ns_node_elem_pos = dirichlet_workset.nodeSets->at(this->nodeSetID);

  const auto& sol_elem_dof_lids = sol_dof_mgr->elem_dof_lids().host();
  const auto& p_elem_dof_lids   = p_dof_mgr->elem_dof_lids().host();
  const auto& sol_offsets = sol_dof_mgr->getGIDFieldOffsets(this->offset);
  const auto& p_offsets = p_dof_mgr->getGIDFieldOffsets(fieldOffset);
  for (const auto& ep : ns_node_elem_pos) {
    const int ielem = ep.first;
    const int pos   = ep.second;
    const int x_lid = sol_elem_dof_lids(ielem,sol_offsets[pos]);
    const int p_lid = p_elem_dof_lids(ielem,p_offsets[pos]);
    x_view[x_lid] = p_constView[p_lid];
  }
}


// **********************************************************************
// Specialization: Residual
// **********************************************************************
template<typename Traits>
SDirichletField<PHAL::AlbanyTraits::Residual, Traits>::
SDirichletField(Teuchos::ParameterList& p)
 : SDirichletField_Base<PHAL::AlbanyTraits::Residual, Traits>(p)
{
  // Nothing to do here
}

// **********************************************************************
template<typename Traits>
void
SDirichletField<PHAL::AlbanyTraits::Residual, Traits>::
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

  dirichletDofCoordsReport(dirichlet_workset,this->nodeSetID,this->offset);
}

// **********************************************************************
// Specialization: Jacobian
// **********************************************************************
template<typename Traits>
SDirichletField<PHAL::AlbanyTraits::Jacobian, Traits>::
SDirichletField(Teuchos::ParameterList& p)
 : SDirichletField_Base<PHAL::AlbanyTraits::Jacobian, Traits>(p)
{
  // Nothing to do here
}

template <typename Traits>
void
SDirichletField<PHAL::AlbanyTraits::Jacobian, Traits>::
set_row_and_col_is_dbc(typename Traits::EvalData dirichlet_workset)
{
  Teuchos::RCP<const Thyra_LinearOp> J = dirichlet_workset.Jac;

  auto  range_vs  = J->range();
  auto  col_vs    = Albany::getColumnSpace(J);
  auto  domain_vs = range_vs;  // we are assuming this!

  // Check for early return. NOTE: the cached vectors are tied to the vector spaces
  // of the Jacobian they were built from, so they can only be reused if those spaces
  // are still the current ones. After a mesh adaptation the Jacobian is rebuilt on a
  // new (larger) column space, and reusing the old col_is_dbc_ would index it out of
  // bounds in evaluateFields below.
  if (not col_is_dbc_.is_null() and
      col_is_dbc_->space()->isCompatible(*col_vs) and
      row_is_dbc_->space()->isCompatible(*range_vs)) {
    return;
  }

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

// **********************************************************************
template<typename Traits>
void SDirichletField<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData dirichlet_workset)
{
  this->set_row_and_col_is_dbc(dirichlet_workset);

  Teuchos::RCP<Thyra_Vector> f = dirichlet_workset.f;
  bool const fill_residual = f != Teuchos::null;

  Tpetra_Vector::dual_view_type::t_dev fView;
  if (fill_residual)
  {
    auto tf = Albany::getTpetraVector(f);
    fView = tf->getLocalViewDevice(Tpetra::Access::ReadWrite);
  }

  auto tCol2DBC = Albany::getTpetraVector(col_is_dbc_);
  auto col2DBCView = tCol2DBC->getLocalViewDevice(Tpetra::Access::ReadOnly);

  auto tJac = Albany::getTpetraMatrix(dirichlet_workset.Jac);
  auto lJac = tJac->getLocalMatrixDevice();
  auto numLRows = lJac.numRows();

  using range_policy = Kokkos::RangePolicy<PHX::ExecutionSpace>;
  Kokkos::parallel_for("SDirichletField<Jacobian>::evaluateFields",
                        range_policy(0, numLRows), KOKKOS_LAMBDA(const int lrow) {
      const bool isRowDBC = col2DBCView(lrow, 0) > 0;
      if (fill_residual == true && isRowDBC)
        fView(lrow, 0) = 0.0;

      auto lJacRow = lJac.row(lrow);
      auto numCols = lJacRow.length;
      for (LO i = 0; i < numCols; ++i) {
        auto lcol = lJacRow.colidx(i);
        const bool isDiagEntry = lcol == lrow;
        if (isDiagEntry) continue;

        const bool isColDBC = col2DBCView(lcol, 0) > 0;
        if (isRowDBC || isColDBC)
          lJacRow.value(i) = 0.0;
      }
    });
}

// **********************************************************************
// Specialization: Tangent
// **********************************************************************
template<typename Traits>
SDirichletField<PHAL::AlbanyTraits::Tangent, Traits>::
SDirichletField(Teuchos::ParameterList& p) :
  SDirichletField_Base<PHAL::AlbanyTraits::Tangent, Traits>(p) {
}

// **********************************************************************
template<typename Traits>
void SDirichletField<PHAL::AlbanyTraits::Tangent, Traits>::
evaluateFields(typename Traits::EvalData dirichlet_workset) {

  Teuchos::RCP<const Thyra_Vector> x = dirichlet_workset.x;
  Teuchos::RCP<Thyra_Vector>       f = dirichlet_workset.f;

  Teuchos::RCP<const Thyra_MultiVector> Vx = dirichlet_workset.Vx;
  Teuchos::RCP<Thyra_MultiVector>       fp = dirichlet_workset.fp;
  Teuchos::RCP<Thyra_MultiVector>       JV = dirichlet_workset.JV;

  Teuchos::ArrayRCP<const ST> x_constView;
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

    if (f != Teuchos::null) {
      f_nonconstView[x_lid] = 0.0;
    }

    if (JV != Teuchos::null) {
      for (int i=0; i<dirichlet_workset.num_cols_x; i++) {
        JV_nonconst2dView[i][x_lid] = j_coeff*Vx_const2dView[i][x_lid];
      }
    }

    if (fp != Teuchos::null) {
      for (int i=0; i<dirichlet_workset.num_cols_p; i++) {
        fp_nonconst2dView[i][x_lid] = 0;
      }
    }
  }
}

// **********************************************************************
// Specialization: DistParamDeriv
// **********************************************************************
template<typename Traits>
SDirichletField<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
SDirichletField(Teuchos::ParameterList& p)
 : SDirichletField_Base<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p)
{
  // Nothing to do here
}

// **********************************************************************
template<typename Traits>
void SDirichletField<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
preEvaluate(typename Traits::EvalData dirichlet_workset)
{
  bool isFieldParameter =  dirichlet_workset.dist_param_deriv_name == this->field_name;
  TEUCHOS_TEST_FOR_EXCEPTION (isFieldParameter, std::logic_error,
      "Error, SDirichletField cannot handle dirichlet parameter " <<  this->field_name << ", use DirichletField instead.\n");

  bool trans = dirichlet_workset.transpose_dist_param_deriv;

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
template<typename Traits>
void SDirichletField<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset) {

  bool isFieldParameter =  dirichletWorkset.dist_param_deriv_name == this->field_name;
  TEUCHOS_TEST_FOR_EXCEPTION(
      isFieldParameter,
      std::logic_error,
      "Error, SDirichletField cannot handle dirichlet parameter " <<  this->field_name << ", use DirichletField instead." << std::endl);

  bool trans = dirichletWorkset.transpose_dist_param_deriv;

  if (!trans) {
    // for (df/dp)*V we zero out corresponding entries in df/dp
    auto fpV = dirichletWorkset.fpV;
    auto fpV_data = Albany::getNonconstLocalData(fpV);
    int num_cols = fpV->domain()->dim();

    const auto& ns_node_elem_pos = dirichletWorkset.nodeSets->at(this->nodeSetID);
    const auto& sol_dof_mgr   = dirichletWorkset.disc->getDOFManager();
    const auto& sol_elem_dof_lids = sol_dof_mgr->elem_dof_lids().host();
    const auto& sol_offsets = sol_dof_mgr->getGIDFieldOffsets(this->offset);

    for (const auto& ep : ns_node_elem_pos) {
      const int ielem = ep.first;
      const int pos   = ep.second;
      const int x_lid = sol_elem_dof_lids(ielem,sol_offsets[pos]);
      for (int col=0; col<num_cols; ++col) {
        fpV_data[col][x_lid] = 0.0;
      }
    }
  }
}

// **********************************************************************
// Specialization: HessianVec
// **********************************************************************
template<typename Traits>
SDirichletField<PHAL::AlbanyTraits::HessianVec, Traits>::
SDirichletField(Teuchos::ParameterList& p)
 : SDirichletField_Base<PHAL::AlbanyTraits::HessianVec, Traits>(p)
{
  // Nothing to do here
}

// **********************************************************************
template<typename Traits>
void SDirichletField<PHAL::AlbanyTraits::HessianVec, Traits>::
preEvaluate(typename Traits::EvalData dirichlet_workset)
{
  const bool f_multiplier_is_active = !dirichlet_workset.hessianWorkset.f_multiplier.is_null();

  // Check for early return
  if(not f_multiplier_is_active) {
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
    f_multiplier_data[x_lid] = 0;
  }
}

template<typename Traits>
void SDirichletField<PHAL::AlbanyTraits::HessianVec, Traits>::
evaluateFields(typename Traits::EvalData dirichlet_workset)
{
  const bool f_xx_is_active = !dirichlet_workset.hessianWorkset.hess_vec_prod_f_xx.is_null();
  const bool f_xp_is_active = !dirichlet_workset.hessianWorkset.hess_vec_prod_f_xp.is_null();

  // Check for early return
  if (not f_xx_is_active and not f_xp_is_active) {
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

} // namespace PHAL
