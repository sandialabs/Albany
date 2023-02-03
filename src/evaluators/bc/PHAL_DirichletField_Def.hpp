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

#include "PHAL_DirichletField.hpp"

// **********************************************************************
// Genereric Template Code for Constructor and PostRegistrationSetup
// **********************************************************************

namespace PHAL {

template <typename EvalT, typename Traits>
DirichletField_Base<EvalT, Traits>::
DirichletField_Base(Teuchos::ParameterList& p)
 : PHAL::DirichletBase<EvalT, Traits>(p)
{
  // Get field type and corresponding layouts
  field_name = p.get<std::string>("Field Name");
}

// **********************************************************************
// Specialization: Residual
// **********************************************************************
template<typename Traits>
DirichletField<PHAL::AlbanyTraits::Residual, Traits>::
DirichletField(Teuchos::ParameterList& p)
 : DirichletField_Base<PHAL::AlbanyTraits::Residual, Traits>(p)
{
  // Nothing to do here
}

// **********************************************************************
template<typename Traits>
void
DirichletField<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  const auto& p = dirichletWorkset.distParamLib->get(this->field_name);
  const auto& p_dof_mgr = p->get_dof_mgr();

  Teuchos::RCP<const Thyra_Vector> pvec = p->vector();
  Teuchos::ArrayRCP<const ST> p_constView = Albany::getLocalData(pvec);

  Teuchos::RCP<const Thyra_Vector> x = dirichletWorkset.x;
  Teuchos::RCP<Thyra_Vector>       f = dirichletWorkset.f;

  Teuchos::ArrayRCP<const ST> x_constView    = Albany::getLocalData(x);
  Teuchos::ArrayRCP<ST>       f_nonconstView = Albany::getNonconstLocalData(f);

  //MP: If the parameter is scalar, then the parameter offset is seto to zero. Otherwise the parameter offset is the same of the solution's one.
  const bool isFieldScalar = p_dof_mgr->getNumFields()==1;
  const int  fieldOffset = isFieldScalar ? 0 : this->offset;

  const auto& ns_node_elem_pos = dirichletWorkset.nodeSets->at(this->nodeSetID);

  const auto& p_elem_dof_lids = p->get_dof_mgr()->elem_dof_lids().host();
  const auto& p_offsets = p_dof_mgr->getGIDFieldOffsets(fieldOffset);
  const auto& sol_dof_mgr   = dirichletWorkset.disc->getNewDOFManager();
  const auto& sol_elem_dof_lids = sol_dof_mgr->elem_dof_lids().host();
  const auto& sol_offsets = sol_dof_mgr->getGIDFieldOffsets(this->offset);
  for (const auto& ep : ns_node_elem_pos) {
    const int ielem = ep.first;
    const int pos   = ep.second;
    const int x_lid = sol_elem_dof_lids(ielem,sol_offsets[pos]);
    const int p_lid = p_elem_dof_lids(ielem,p_offsets[pos]);
    f_nonconstView[x_lid] = x_constView[x_lid] - p_constView[p_lid];
  }
}

// **********************************************************************
// Specialization: Jacobian
// **********************************************************************
template<typename Traits>
DirichletField<PHAL::AlbanyTraits::Jacobian, Traits>::
DirichletField(Teuchos::ParameterList& p)
 : DirichletField_Base<PHAL::AlbanyTraits::Jacobian, Traits>(p)
{
  // Nothing to do here
}

// **********************************************************************
template<typename Traits>
void DirichletField<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  const auto& p_dof_mgr = dirichletWorkset.disc->getNewDOFManager(this->field_name);
  const auto& sol_dof_mgr   = dirichletWorkset.disc->getNewDOFManager();

  //MP: If the parameter is scalar, then the parameter offset is seto to zero. Otherwise the parameter offset is the same of the solution's one.
  const bool isFieldScalar = p_dof_mgr->getNumFields()==1;
  const int  fieldOffset = isFieldScalar ? 0 : this->offset;

  Teuchos::RCP<const Thyra_Vector> pvec = dirichletWorkset.distParamLib->get(this->field_name)->vector();
  Teuchos::ArrayRCP<const ST> p_constView = Albany::getLocalData(pvec);

  Teuchos::ArrayRCP<const ST> x_constView;
  Teuchos::ArrayRCP<ST>       f_nonconstView;

  Teuchos::RCP<Thyra_LinearOp>     jac = dirichletWorkset.Jac;
  const RealType j_coeff = dirichletWorkset.j_coeff;

  const bool fillResid = dirichletWorkset.f != Teuchos::null;
  if (fillResid) { 
    x_constView = Albany::getLocalData(dirichletWorkset.x);
    f_nonconstView = Albany::getNonconstLocalData(dirichletWorkset.f);
  }
  Teuchos::Array<ST> matrixEntries;
  Teuchos::Array<LO> matrixIndices;

  const auto& ns_node_elem_pos = dirichletWorkset.nodeSets->at(this->nodeSetID);

  const auto& sol_elem_dof_lids = sol_dof_mgr->elem_dof_lids().host();
  const auto& p_elem_dof_lids   = p_dof_mgr->elem_dof_lids().host();
  const auto& sol_offsets = sol_dof_mgr->getGIDFieldOffsets(this->offset);
  const auto& p_offsets = p_dof_mgr->getGIDFieldOffsets(fieldOffset);
  for (const auto& ep : ns_node_elem_pos) {
    const int ielem = ep.first;
    const int pos   = ep.second;
    const int x_lid = sol_elem_dof_lids(ielem,sol_offsets[pos]);

    // Extract the row, zero it out, then put j_coeff on diagonal
    Albany::getLocalRowValues(jac,x_lid,matrixIndices,matrixEntries);
    for (auto& val : matrixEntries) { val = 0.0; }
    Albany::setLocalRowValues(jac, x_lid, matrixIndices(), matrixEntries());
    Albany::setLocalRowValue(jac, x_lid, x_lid, j_coeff);

    if (fillResid) {
      const int p_lid = p_elem_dof_lids(ielem,p_offsets[pos]);
      f_nonconstView[x_lid] = x_constView[x_lid] - p_constView[p_lid];
    }
  }
}

// **********************************************************************
// Specialization: Tangent
// **********************************************************************
template<typename Traits>
DirichletField<PHAL::AlbanyTraits::Tangent, Traits>::
DirichletField(Teuchos::ParameterList& p)
 : DirichletField_Base<PHAL::AlbanyTraits::Tangent, Traits>(p)
{
  // Nothing to do here
}

// **********************************************************************
template<typename Traits>
void DirichletField<PHAL::AlbanyTraits::Tangent, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  const auto& p_dof_mgr = dirichletWorkset.disc->getNewDOFManager(this->field_name);
  const auto& sol_dof_mgr   = dirichletWorkset.disc->getNewDOFManager();

  const bool isFieldScalar = p_dof_mgr->getNumFields()==1;
  const int fieldOffset = isFieldScalar ? 0 : this->offset;

  Teuchos::RCP<const Thyra_Vector> pvec = dirichletWorkset.distParamLib->get(this->field_name)->vector();
  Teuchos::ArrayRCP<const ST> p_constView = Albany::getLocalData(pvec);

  Teuchos::RCP<const Thyra_Vector> x = dirichletWorkset.x;
  Teuchos::RCP<Thyra_Vector>       f = dirichletWorkset.f;

  Teuchos::RCP<const Thyra_MultiVector> Vx = dirichletWorkset.Vx;
  Teuchos::RCP<Thyra_MultiVector>       fp = dirichletWorkset.fp;
  Teuchos::RCP<Thyra_MultiVector>       JV = dirichletWorkset.JV;

  Teuchos::ArrayRCP<const ST> x_constView;
  Teuchos::ArrayRCP<ST>       f_nonconstView;

  Teuchos::ArrayRCP<Teuchos::ArrayRCP<const ST>> Vx_const2dView;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>>       JV_nonconst2dView;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>>       fp_nonconst2dView;

  if (f != Teuchos::null) {
    x_constView = Albany::getLocalData(x);
    f_nonconstView = Albany::getNonconstLocalData(f);
  }
  if (JV != Teuchos::null) {
    JV_nonconst2dView = Albany::getNonconstLocalData(JV);
    Vx_const2dView    = Albany::getLocalData(Vx);
  }

  if (fp != Teuchos::null) {
    fp_nonconst2dView = Albany::getNonconstLocalData(fp);
  }

  const RealType j_coeff = dirichletWorkset.j_coeff;

  const auto& ns_node_elem_pos = dirichletWorkset.nodeSets->at(this->nodeSetID);
  const auto& sol_elem_dof_lids = sol_dof_mgr->elem_dof_lids().host();
  const auto& p_elem_dof_lids   = p_dof_mgr->elem_dof_lids().host();
  const auto& sol_offsets = sol_dof_mgr->getGIDFieldOffsets(this->offset);
  const auto& p_offsets = p_dof_mgr->getGIDFieldOffsets(fieldOffset);
  for (const auto& ep : ns_node_elem_pos) {
    const int ielem = ep.first;
    const int pos   = ep.second;
    const int x_lid = sol_elem_dof_lids(ielem,sol_offsets[pos]);

    if (f != Teuchos::null) {
      const int p_lid = p_elem_dof_lids(ielem,p_offsets[pos]);
      f_nonconstView[x_lid] = x_constView[x_lid] - p_constView[p_lid];
    }

    if (JV != Teuchos::null) {
      for (int i=0; i<dirichletWorkset.num_cols_x; i++) {
        JV_nonconst2dView[i][x_lid] = j_coeff*Vx_const2dView[i][x_lid];
      }
    }

    if (fp != Teuchos::null) {
      for (int i=0; i<dirichletWorkset.num_cols_p; i++) {
        fp_nonconst2dView[i][x_lid] = 0;
      }
    }
  }
}

// **********************************************************************
// Specialization: DistParamDeriv
// **********************************************************************
template<typename Traits>
DirichletField<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
DirichletField(Teuchos::ParameterList& p)
 : DirichletField_Base<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p)
{
  // Nothing to do here
}

// **********************************************************************
template<typename Traits>
void DirichletField<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
preEvaluate(typename Traits::EvalData dirichletWorkset)
{
  bool trans = dirichletWorkset.transpose_dist_param_deriv;

  if (trans) {
    // For (df/dp)^T*V we zero out corresponding entries in V
    auto Vp_data = Albany::getNonconstLocalData(dirichletWorkset.Vp_bc);
    auto Vp_const_data = Albany::getLocalData(dirichletWorkset.Vp);

    bool isFieldParameter =  dirichletWorkset.dist_param_deriv_name == this->field_name;
    int num_cols = dirichletWorkset.Vp->domain()->dim();
    const auto& sol_dof_mgr   = dirichletWorkset.disc->getNewDOFManager();
    const auto& ns_node_elem_pos = dirichletWorkset.nodeSets->at(this->nodeSetID);
    const auto& sol_elem_dof_lids = sol_dof_mgr->elem_dof_lids().host();
    const auto& sol_offsets = sol_dof_mgr->getGIDFieldOffsets(this->offset);

    // Note: it is important to set Vp_bc to Vp at Dirichlet dof nodes even if Vp_bc was already initialized with Vp
    // because other boundary conditions applied before this one could have zeroed it out (if the bcs are applied to the same dofs).
    for (const auto& ep : ns_node_elem_pos) {
      const int ielem = ep.first;
      const int pos   = ep.second;
      const int x_lid = sol_elem_dof_lids(ielem,sol_offsets[pos]);
      for (int col=0; col<num_cols; ++col) {
        Vp_data[col][x_lid] = isFieldParameter ? Vp_const_data[col][x_lid] : 0.0;
      }
    }
  }
}

// **********************************************************************
template<typename Traits>
void DirichletField<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  bool isFieldParameter =  dirichletWorkset.dist_param_deriv_name == this->field_name;
  bool trans = dirichletWorkset.transpose_dist_param_deriv;

  const auto& p_dof_mgr = dirichletWorkset.disc->getNewDOFManager(this->field_name);
  const auto& sol_dof_mgr   = dirichletWorkset.disc->getNewDOFManager();

  const bool isFieldScalar = p_dof_mgr->getNumFields()==1;
  const int fieldOffset = isFieldScalar ? 0 : this->offset;

  Teuchos::RCP<Thyra_MultiVector> fpV = dirichletWorkset.fpV;

  int num_cols = fpV->domain()->dim();

  const auto& ns_node_elem_pos = dirichletWorkset.nodeSets->at(this->nodeSetID);
  const auto& sol_elem_dof_lids = sol_dof_mgr->elem_dof_lids().host();
  const auto& p_elem_dof_lids   = p_dof_mgr->elem_dof_lids().host();
  const auto& sol_offsets = sol_dof_mgr->getGIDFieldOffsets(this->offset);
  const auto& p_offsets = p_dof_mgr->getGIDFieldOffsets(fieldOffset);
  if (trans) {
    // For (df/dp)^T*V we zero out corresponding entries in V
    if(isFieldParameter) {
      auto Vp_data = Albany::getNonconstLocalData(dirichletWorkset.Vp_bc);
      auto fpV_data = Albany::getNonconstLocalData(fpV);
      for (const auto& ep : ns_node_elem_pos) {
        const int ielem = ep.first;
        const int pos   = ep.second;
        const int x_lid = sol_elem_dof_lids(ielem,sol_offsets[pos]);
        const int p_lid = p_elem_dof_lids(ielem,p_offsets[pos]);
        for (int col=0; col<num_cols; ++col) {
          fpV_data[col][p_lid] -= Vp_data[col][x_lid];
          Vp_data[col][x_lid] = 0.0;
        }
      }
    }
  } else {
    // for (df/dp)*V we zero out corresponding entries in df/dp
    auto fpV_data = Albany::getNonconstLocalData(fpV);
    auto Vp_const_data = Albany::getLocalData(dirichletWorkset.Vp);
    if(isFieldParameter) {
      for (const auto& ep : ns_node_elem_pos) {
        const int ielem = ep.first;
        const int pos   = ep.second;
        const int x_lid = sol_elem_dof_lids(ielem,sol_offsets[pos]);
        const int p_lid = p_elem_dof_lids(ielem,p_offsets[pos]);
        for (int col=0; col<num_cols; ++col) {
          fpV_data[col][x_lid] -= Vp_const_data[col][p_lid];
        }
      }
    } else {
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
}

// **********************************************************************
// Specialization: HessianVec
// **********************************************************************
template<typename Traits>
DirichletField<PHAL::AlbanyTraits::HessianVec, Traits>::
DirichletField(Teuchos::ParameterList& p)
 : DirichletField_Base<PHAL::AlbanyTraits::HessianVec, Traits>(p)
{
  // Nothing to do here
}

// **********************************************************************

template<typename Traits>
void DirichletField<PHAL::AlbanyTraits::HessianVec, Traits>::
preEvaluate(typename Traits::EvalData dirichletWorkset)
{
  const bool f_multiplier_is_active = !dirichletWorkset.hessianWorkset.f_multiplier.is_null();

  // Check for early return
  if(not f_multiplier_is_active) {
    return;
  }

  auto f_multiplier_data = Albany::getNonconstLocalData(dirichletWorkset.hessianWorkset.f_multiplier);

  const auto& ns_node_elem_pos = dirichletWorkset.nodeSets->at(this->nodeSetID);

  const auto& sol_dof_mgr   = dirichletWorkset.disc->getNewDOFManager();
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
void DirichletField<PHAL::AlbanyTraits::HessianVec, Traits>::
evaluateFields(typename Traits::EvalData /* dirichletWorkset */)
{
  // Nothing to do here
}

} // namespace PHAL
