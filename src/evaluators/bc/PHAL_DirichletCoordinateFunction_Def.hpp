//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"

#include "Albany_ThyraUtils.hpp"

// **********************************************************************
// Genereric Template Code for Constructor and PostRegistrationSetup
// **********************************************************************

namespace PHAL {

template <typename EvalT, typename Traits/*, typename cfunc_traits*/>
DirichletCoordFunction_Base<EvalT, Traits/*, cfunc_traits*/>::
DirichletCoordFunction_Base(Teuchos::ParameterList& p) :
  PHAL::DirichletBase<EvalT, Traits>(p),
  func(p) {
}

// **********************************************************************
// Specialization: Residual
// **********************************************************************
template<typename Traits/*, typename cfunc_traits*/>
DirichletCoordFunction<PHAL::AlbanyTraits::Residual, Traits/*, cfunc_traits*/>::
DirichletCoordFunction(Teuchos::ParameterList& p) :
  DirichletCoordFunction_Base<PHAL::AlbanyTraits::Residual, Traits/*, cfunc_traits*/>(p) {
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

  // Grab the vector off node GIDs for this Node Set ID from the std::map
  const std::vector<std::vector<int> >& nsNodes = dirichletWorkset.nodeSets->find(this->nodeSetID)->second;
  const std::vector<double*>& nsNodeCoords = dirichletWorkset.nodeSetCoords->find(this->nodeSetID)->second;

  RealType time = dirichletWorkset.current_time;
  int number_of_components = this->func.getNumComponents();

  double* coord;
  std::vector<ScalarT> BCVals(number_of_components);

  for(unsigned int inode = 0; inode < nsNodes.size(); inode++) {

    coord = nsNodeCoords[inode];

    this->func.computeBCs(coord, BCVals, time);

    for(unsigned int j = 0; j < number_of_components; j++) {
      int offset = nsNodes[inode][j];
      f_nonconstView[offset] = (x_constView[offset] - BCVals[j]);
    }
  }
}

// **********************************************************************
// Specialization: Jacobian
// **********************************************************************
template<typename Traits/*, typename cfunc_traits*/>
DirichletCoordFunction<PHAL::AlbanyTraits::Jacobian, Traits/*, cfunc_traits*/>::
DirichletCoordFunction(Teuchos::ParameterList& p) :
  DirichletCoordFunction_Base<PHAL::AlbanyTraits::Jacobian, Traits/*, cfunc_traits*/>(p) {
}
// **********************************************************************
template<typename Traits/*, typename cfunc_traits*/>
void DirichletCoordFunction<PHAL::AlbanyTraits::Jacobian, Traits/*, cfunc_traits*/>::
evaluateFields(typename Traits::EvalData dirichletWorkset) {

  Teuchos::RCP<const Thyra_Vector> x   = dirichletWorkset.x;
  Teuchos::RCP<Thyra_Vector>       f   = dirichletWorkset.f;
  Teuchos::RCP<Thyra_LinearOp>     jac = dirichletWorkset.Jac;

  Teuchos::ArrayRCP<const ST> x_constView;
  Teuchos::ArrayRCP<ST>       f_nonconstView;

  const RealType j_coeff = dirichletWorkset.j_coeff;

  const std::vector<std::vector<int> >& nsNodes = dirichletWorkset.nodeSets->find(this->nodeSetID)->second;
  const std::vector<double*>& nsNodeCoords = dirichletWorkset.nodeSetCoords->find(this->nodeSetID)->second;

  Teuchos::Array<LO> index(1);
  Teuchos::Array<ST> value(1);
  value[0] = j_coeff;
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

  for(unsigned int inode = 0; inode < nsNodes.size(); inode++) {
    coord = nsNodeCoords[inode];

    this->func.computeBCs(coord, BCVals, time);

    for(unsigned int j = 0; j < number_of_components; j++) {

      int offset = nsNodes[inode][j];
      index[0] = offset;

      // Extract the row, zero it out, then put j_coeff on diagonal
      Albany::getLocalRowValues(jac,offset,matrixIndices,matrixEntries);
      for (auto& val : matrixEntries) { val = 0.0; }
      Albany::setLocalRowValues(jac, offset, matrixIndices(), matrixEntries());
      Albany::setLocalRowValues(jac, offset, index(), value());

      if(fillResid) {
        f_nonconstView[offset] = (x_constView[offset] - BCVals[j].val());
      }
    }
  }
}

// **********************************************************************
// Specialization: Tangent
// **********************************************************************
template<typename Traits/*, typename cfunc_traits*/>
DirichletCoordFunction<PHAL::AlbanyTraits::Tangent, Traits/*, cfunc_traits*/>::
DirichletCoordFunction(Teuchos::ParameterList& p) :
  DirichletCoordFunction_Base<PHAL::AlbanyTraits::Tangent, Traits/*, cfunc_traits*/>(p) {
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
  const std::vector<std::vector<int> >& nsNodes = dirichletWorkset.nodeSets->find(this->nodeSetID)->second;
  const std::vector<double*>& nsNodeCoords = dirichletWorkset.nodeSetCoords->find(this->nodeSetID)->second;

  RealType time = dirichletWorkset.current_time;
  int number_of_components = this->func.getNumComponents();

  double* coord;
  std::vector<ScalarT> BCVals(number_of_components);

  for(unsigned int inode = 0; inode < nsNodes.size(); inode++) {
    coord = nsNodeCoords[inode];

    this->func.computeBCs(coord, BCVals, time);

    for(unsigned int j = 0; j < number_of_components; j++) {

      int offset = nsNodes[inode][j];

      if(f != Teuchos::null) {
        f_nonconstView[offset] = (x_constView[offset] - BCVals[j].val());
      }

      if(JV != Teuchos::null){
        for(int i = 0; i < dirichletWorkset.num_cols_x; i++){
          JV_nonconst2dView[i][offset] = j_coeff * Vx_const2dView[i][offset];
        }
      }

      if(fp != Teuchos::null){
        for(int i = 0; i < dirichletWorkset.num_cols_p; i++){
          fp_nonconst2dView[i][offset] = -BCVals[j].dx(dirichletWorkset.param_offset + i);
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
DirichletCoordFunction(Teuchos::ParameterList& p) :
  DirichletCoordFunction_Base<PHAL::AlbanyTraits::DistParamDeriv, Traits/*, cfunc_traits*/>(p) {
}
// **********************************************************************
template<typename Traits/*, typename cfunc_traits*/>
void DirichletCoordFunction<PHAL::AlbanyTraits::DistParamDeriv, Traits/*, cfunc_traits*/>::
evaluateFields(typename Traits::EvalData dirichletWorkset) {

  Teuchos::RCP<Thyra_MultiVector> fpV = dirichletWorkset.fpV;
  bool trans = dirichletWorkset.transpose_dist_param_deriv;
  int num_cols = fpV->domain()->dim();

  //
  // We're currently assuming Dirichlet BC's can't be distributed parameters.
  // Thus we don't need to actually evaluate the BC's here.  The code to do
  // so is still here, just commented out for future reference.
  //

  const std::vector<std::vector<int> >& nsNodes = dirichletWorkset.nodeSets->find(this->nodeSetID)->second;
  const std::vector<double*>& nsNodeCoords = dirichletWorkset.nodeSetCoords->find(this->nodeSetID)->second;

  // RealType time = dirichletWorkset.current_time;
  int number_of_components = this->func.getNumComponents();

  // double* coord;
  // std::vector<ScalarT> BCVals(number_of_components);

  if (trans) {
    // For (df/dp)^T*V we zero out corresponding entries in V
    Teuchos::RCP<Thyra_MultiVector> Vp = dirichletWorkset.Vp_bc;
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>> Vp_nonconst2dView = Albany::getNonconstLocalData(Vp);
    for(unsigned int inode = 0; inode < nsNodes.size(); inode++) {
      // coord = nsNodeCoords[inode];
      // this->func.computeBCs(coord, BCVals, time);

      for(unsigned int j = 0; j < number_of_components; j++) {
        int offset = nsNodes[inode][j];
        for (int col=0; col<num_cols; ++col) {
          //(*Vp)[col][offset] = 0.0;
          Vp_nonconst2dView[col][offset] = 0.0;
        }
      }
    }
  } else {
    // for (df/dp)*V we zero out corresponding entries in df/dp
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>> fpV_nonconst2dView = Albany::getNonconstLocalData(fpV);
    for(unsigned int inode = 0; inode < nsNodes.size(); inode++) {
      // coord = nsNodeCoords[inode];
      // this->func.computeBCs(coord, BCVals, time);

      for(unsigned int j = 0; j < number_of_components; j++) {
        int offset = nsNodes[inode][j];
        for (int col=0; col<num_cols; ++col) {
          //(*fpV)[col][offset] = 0.0;
          fpV_nonconst2dView[col][offset] = 0.0;
        }
      }
    }
  }
}

} // namespace PHAL
