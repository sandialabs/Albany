//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "PHAL_DirichletCoordinateFunction.hpp"
#include "Albany_AbstractDiscretization.hpp"
#include "Albany_ThyraUtils.hpp"

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"

// **********************************************************************
// Genereric Template Code for Constructor and PostRegistrationSetup
// **********************************************************************

namespace PHAL {

template <typename EvalT, typename Traits/*, typename cfunc_traits*/>
DirichletCoordFunction_Base<EvalT, Traits/*, cfunc_traits*/>::
DirichletCoordFunction_Base(Teuchos::ParameterList& p)
 : PHAL::DirichletBase<EvalT, Traits>(p)
 , func(p)
{
  // Nothing to do here
}

// **********************************************************************
// Specialization: Residual
// **********************************************************************
template<typename Traits/*, typename cfunc_traits*/>
DirichletCoordFunction<PHAL::AlbanyTraits::Residual, Traits/*, cfunc_traits*/>::
DirichletCoordFunction(Teuchos::ParameterList& p)
 : DirichletCoordFunction_Base<PHAL::AlbanyTraits::Residual, Traits/*, cfunc_traits*/>(p)
{
  // Nothing to do here
}

// **********************************************************************
template<typename Traits/*, typename cfunc_traits*/>
void
DirichletCoordFunction<PHAL::AlbanyTraits::Residual, Traits/*, cfunc_traits*/>::
evaluateFields(typename Traits::EvalData dirichletWorkset) {

  Teuchos::RCP<const Thyra_Vector> x = dirichletWorkset.x;
  Teuchos::RCP<Thyra_Vector>       f = dirichletWorkset.f;

  Teuchos::ArrayRCP<const ST> x_constView    = Albany::getLocalData(x);
  Teuchos::ArrayRCP<ST>       f_nonconstView = Albany::getNonconstLocalData(f);

  RealType time = dirichletWorkset.current_time;
  int number_of_components = this->func.getNumComponents();

  double* coord;
  std::vector<ScalarT> BCVals(number_of_components);

  // Grab the vector off node GIDs for this Node Set ID from the std::map
  const auto& nsNodeCoords = dirichletWorkset.nodeSetCoords->at(this->nodeSetID);
  const auto& ns_node_elem_pos = dirichletWorkset.nodeSets->at(this->nodeSetID);
  const auto& sol_dof_mgr   = dirichletWorkset.disc->getDOFManager();
  const auto& sol_elem_dof_lids = sol_dof_mgr->elem_dof_lids().host();
  std::vector<std::vector<int>> sol_offsets(number_of_components);
  for (int j=0; j<number_of_components; ++j) {
    sol_offsets[j] = sol_dof_mgr->getGIDFieldOffsets(j);
  }

  for (unsigned inode=0; inode<ns_node_elem_pos.size(); ++inode) {
    const int ielem = ns_node_elem_pos[inode].first;
    const int pos   = ns_node_elem_pos[inode].second;

    coord = nsNodeCoords[inode];

    this->func.computeBCs(coord, BCVals, time);

    for(int j = 0; j < number_of_components; j++) {
      const int x_lid = sol_elem_dof_lids(ielem,sol_offsets[j][pos]);
      f_nonconstView[x_lid] = (x_constView[x_lid] - BCVals[j]);
    }
  }
}

// **********************************************************************
// Specialization: Jacobian
// **********************************************************************
template<typename Traits/*, typename cfunc_traits*/>
DirichletCoordFunction<PHAL::AlbanyTraits::Jacobian, Traits/*, cfunc_traits*/>::
DirichletCoordFunction(Teuchos::ParameterList& p)
 : DirichletCoordFunction_Base<PHAL::AlbanyTraits::Jacobian, Traits/*, cfunc_traits*/>(p)
{
  // Nothing to do here
}
// **********************************************************************
template<typename Traits/*, typename cfunc_traits*/>
void DirichletCoordFunction<PHAL::AlbanyTraits::Jacobian, Traits/*, cfunc_traits*/>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  Teuchos::RCP<const Thyra_Vector> x   = dirichletWorkset.x;
  Teuchos::RCP<Thyra_Vector>       f   = dirichletWorkset.f;
  Teuchos::RCP<Thyra_LinearOp>     jac = dirichletWorkset.Jac;

  Teuchos::ArrayRCP<const ST> x_constView;
  Teuchos::ArrayRCP<ST>       f_nonconstView;

  const RealType j_coeff = dirichletWorkset.j_coeff;

  const std::vector<double*>& nsNodeCoords = dirichletWorkset.nodeSetCoords->at(this->nodeSetID);

  Teuchos::Array<ST> matrixEntries;
  Teuchos::Array<LO> matrixIndices;

  bool fillResid = (f != Teuchos::null);
  if (fillResid) {
    x_constView    = Albany::getLocalData(x);
    f_nonconstView = Albany::getNonconstLocalData(f);
  }

  RealType time = dirichletWorkset.current_time;
  int number_of_components = this->func.getNumComponents();

  double* coord;
  std::vector<ScalarT> BCVals(number_of_components);

  const auto& ns_node_elem_pos = dirichletWorkset.nodeSets->at(this->nodeSetID);
  const auto& sol_dof_mgr   = dirichletWorkset.disc->getDOFManager();
  const auto& sol_elem_dof_lids = sol_dof_mgr->elem_dof_lids().host();

  std::vector<std::vector<int>> sol_offsets(number_of_components);
  for (int j=0; j<number_of_components; ++j) {
    sol_offsets[j] = sol_dof_mgr->getGIDFieldOffsets(j);
  }

  for (unsigned inode=0; inode<ns_node_elem_pos.size(); ++inode) {
    const int ielem = ns_node_elem_pos[inode].first;
    const int pos   = ns_node_elem_pos[inode].second;
    coord = nsNodeCoords[inode];

    this->func.computeBCs(coord, BCVals, time);

    for(int j = 0; j < number_of_components; j++) {
      const int x_lid = sol_elem_dof_lids(ielem,sol_offsets[j][pos]);

      // Extract the row, zero it out, then put j_coeff on diagonal
      Albany::getLocalRowValues(jac,x_lid,matrixIndices,matrixEntries);
      for (auto& val : matrixEntries) { val = 0.0; }
      Albany::setLocalRowValues(jac, x_lid, matrixIndices(), matrixEntries());
      Albany::setLocalRowValue(jac, x_lid, x_lid, j_coeff);

      if(fillResid) {
        f_nonconstView[x_lid] = (x_constView[x_lid] - BCVals[j].val());
      }
    }
  }
}

// **********************************************************************
// Specialization: Tangent
// **********************************************************************
template<typename Traits/*, typename cfunc_traits*/>
DirichletCoordFunction<PHAL::AlbanyTraits::Tangent, Traits/*, cfunc_traits*/>::
DirichletCoordFunction(Teuchos::ParameterList& p)
 : DirichletCoordFunction_Base<PHAL::AlbanyTraits::Tangent, Traits/*, cfunc_traits*/>(p)
{
  // Nothing to do here
}
// **********************************************************************
template<typename Traits/*, typename cfunc_traits*/>
void DirichletCoordFunction<PHAL::AlbanyTraits::Tangent, Traits/*, cfunc_traits*/>::
evaluateFields(typename Traits::EvalData dirichletWorkset) {

  Teuchos::RCP<const Thyra_Vector>       x = dirichletWorkset.x;
  Teuchos::RCP<const Thyra_MultiVector> Vx = dirichletWorkset.Vx;
  Teuchos::RCP<Thyra_Vector>             f = dirichletWorkset.f;
  Teuchos::RCP<Thyra_MultiVector>       JV = dirichletWorkset.JV;
  Teuchos::RCP<Thyra_MultiVector>       fp = dirichletWorkset.fp;

  Teuchos::ArrayRCP<const ST> x_constView;
  Teuchos::ArrayRCP<ST>       f_nonconstView;

  Teuchos::ArrayRCP<Teuchos::ArrayRCP<const ST>> Vx_const2dView;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>>       JV_nonconst2dView;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>>       fp_nonconst2dView;

  if (f != Teuchos::null) {
    x_constView     = Albany::getLocalData(x);
    f_nonconstView  = Albany::getNonconstLocalData(f);
  }
  if(JV != Teuchos::null){
    JV_nonconst2dView = Albany::getNonconstLocalData(JV);
    Vx_const2dView    = Albany::getLocalData(Vx);
  }

  if(fp != Teuchos::null){
    fp_nonconst2dView = Albany::getNonconstLocalData(fp);
  }

  const RealType j_coeff = dirichletWorkset.j_coeff;
  const std::vector<double*>& nsNodeCoords = dirichletWorkset.nodeSetCoords->at(this->nodeSetID);

  RealType time = dirichletWorkset.current_time;
  int number_of_components = this->func.getNumComponents();

  double* coord;
  std::vector<ScalarT> BCVals(number_of_components);

  const auto& ns_node_elem_pos = dirichletWorkset.nodeSets->at(this->nodeSetID);
  const auto& sol_dof_mgr   = dirichletWorkset.disc->getDOFManager();
  const auto& sol_elem_dof_lids = sol_dof_mgr->elem_dof_lids().host();

  std::vector<std::vector<int>> sol_offsets(number_of_components);
  for (int j=0; j<number_of_components; ++j) {
    sol_offsets[j] = sol_dof_mgr->getGIDFieldOffsets(j);
  }

  for (unsigned inode=0; inode<ns_node_elem_pos.size(); ++inode) {
    const int ielem = ns_node_elem_pos[inode].first;
    const int pos   = ns_node_elem_pos[inode].second;
    coord = nsNodeCoords[inode];

    this->func.computeBCs(coord, BCVals, time);

    for(int j = 0; j < number_of_components; j++) {
      const int x_lid = sol_elem_dof_lids(ielem,sol_offsets[j][pos]);

      if(f != Teuchos::null) {
        f_nonconstView[x_lid] = (x_constView[x_lid] - BCVals[j].val());
      }

      if(JV != Teuchos::null){
        for(int i = 0; i < dirichletWorkset.num_cols_x; i++){
          JV_nonconst2dView[i][x_lid] = j_coeff * Vx_const2dView[i][x_lid];
        }
      }

      if(fp != Teuchos::null){
        for(int i = 0; i < dirichletWorkset.num_cols_p; i++){
          fp_nonconst2dView[i][x_lid] = -BCVals[j].dx(dirichletWorkset.param_offset + i);
        }
      }
    }
  }
}

// **********************************************************************
// Specialization: DistParamDeriv
// **********************************************************************
template<typename Traits/*, typename cfunc_traits*/>
DirichletCoordFunction<PHAL::AlbanyTraits::DistParamDeriv, Traits/*, cfunc_traits*/>::
DirichletCoordFunction(Teuchos::ParameterList& p)
 : DirichletCoordFunction_Base<PHAL::AlbanyTraits::DistParamDeriv, Traits/*, cfunc_traits*/>(p)
{
  // Nothing to do here
}
// **********************************************************************
template<typename Traits/*, typename cfunc_traits*/>
void DirichletCoordFunction<PHAL::AlbanyTraits::DistParamDeriv, Traits/*, cfunc_traits*/>::
evaluateFields(typename Traits::EvalData dirichletWorkset) {

  // For (df/dp)^T*V we zero out corresponding entries in V
  // while for (df/dp)*V we zero out corresponding entries in df/dp
  bool trans = dirichletWorkset.transpose_dist_param_deriv;
  auto mv = trans ? dirichletWorkset.Vp_bc : dirichletWorkset.fpV;
  auto data = Albany::getNonconstLocalData(mv);
  int num_cols = mv->domain()->dim();

  //
  // We're currently assuming Dirichlet BC's can't be distributed parameters.
  // Thus we don't need to actually evaluate the BC's here.  The code to do
  // so is still here, just commented out for future reference.
  //

  // RealType time = dirichletWorkset.current_time;
  int number_of_components = this->func.getNumComponents();

  // double* coord;
  // std::vector<ScalarT> BCVals(number_of_components);
  const auto& ns_node_elem_pos = dirichletWorkset.nodeSets->at(this->nodeSetID);
  const auto& sol_dof_mgr   = dirichletWorkset.disc->getDOFManager();
  const auto& sol_elem_dof_lids = sol_dof_mgr->elem_dof_lids().host();

  std::vector<std::vector<int>> sol_offsets(number_of_components);
  for (int j=0; j<number_of_components; ++j) {
    sol_offsets[j] = sol_dof_mgr->getGIDFieldOffsets(j);
  }

  for (unsigned inode=0; inode<ns_node_elem_pos.size(); ++inode) {
    const int ielem = ns_node_elem_pos[inode].first;
    const int pos   = ns_node_elem_pos[inode].second;

    for(int j=0; j<number_of_components; ++j) {
      const int x_lid = sol_elem_dof_lids(ielem,sol_offsets[j][pos]);
      for (int col=0; col<num_cols; ++col) {
        data[col][x_lid] = 0;
      }
    }
  }
}

// **********************************************************************
// Specialization: HessianVec
// **********************************************************************
template<typename Traits/*, typename cfunc_traits*/>
DirichletCoordFunction<PHAL::AlbanyTraits::HessianVec, Traits/*, cfunc_traits*/>::
DirichletCoordFunction(Teuchos::ParameterList& p)
 : DirichletCoordFunction_Base<PHAL::AlbanyTraits::HessianVec, Traits/*, cfunc_traits*/>(p)
{
  // Nothing to do here
}
// **********************************************************************
template<typename Traits/*, typename cfunc_traits*/>
void DirichletCoordFunction<PHAL::AlbanyTraits::HessianVec, Traits/*, cfunc_traits*/>::
evaluateFields(typename Traits::EvalData /* dirichletWorkset */)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
      "HessianVec specialization of DirichletCoordFunction::evaluateFields is not implemented yet\n");
}

} // namespace PHAL
