//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "Tpetra_CrsMatrix.hpp"

#include "Albany_TpetraThyraUtils.hpp"

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

  Teuchos::RCP<Tpetra_Vector> fT = dirichletWorkset.fT;
  Teuchos::RCP<const Tpetra_Vector> xT = Albany::getConstTpetraVector(dirichletWorkset.x);
  Teuchos::ArrayRCP<ST> fT_nonconstView = fT->get1dViewNonConst();
  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();


  // Grab the vector off node GIDs for this Node Set ID from the std::map
  const std::vector<std::vector<int> >& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;
  const std::vector<double*>& nsNodeCoords =
    dirichletWorkset.nodeSetCoords->find(this->nodeSetID)->second;

  RealType time = dirichletWorkset.current_time;
  int number_of_components = this->func.getNumComponents();

  double* coord;
  std::vector<ScalarT> BCVals(number_of_components);

  for(unsigned int inode = 0; inode < nsNodes.size(); inode++) {

    coord = nsNodeCoords[inode];

    this->func.computeBCs(coord, BCVals, time);

    for(unsigned int j = 0; j < number_of_components; j++) {
      int offset = nsNodes[inode][j];
      fT_nonconstView[offset] = (xT_constView[offset] - BCVals[j]);
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

  Teuchos::RCP<Tpetra_Vector> fT = dirichletWorkset.fT;
  Teuchos::ArrayRCP<ST> fT_nonconstView;
  Teuchos::RCP<const Tpetra_Vector> xT = Albany::getConstTpetraVector(dirichletWorkset.x);
  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();
  Teuchos::RCP<Tpetra_CrsMatrix> jacT = dirichletWorkset.JacT;

  const RealType j_coeff = dirichletWorkset.j_coeff;
  const std::vector<std::vector<int> >& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;
  const std::vector<double*>& nsNodeCoords =
    dirichletWorkset.nodeSetCoords->find(this->nodeSetID)->second;

  Teuchos::Array<LO> index(1);
  Teuchos::Array<ST> value(1);
  size_t numEntriesT;
  value[0] = j_coeff;
  Teuchos::Array<ST> matrixEntriesT;
  Teuchos::Array<LO> matrixIndicesT;

  bool fillResid = (fT != Teuchos::null);
  if (fillResid) fT_nonconstView = fT->get1dViewNonConst();

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
      numEntriesT = jacT->getNumEntriesInLocalRow(offset);
      matrixEntriesT.resize(numEntriesT);
      matrixIndicesT.resize(numEntriesT);

      jacT->getLocalRowCopy(offset, matrixIndicesT(), matrixEntriesT(), numEntriesT);
      for (int i=0; i<numEntriesT; i++) matrixEntriesT[i]=0;
      jacT->replaceLocalValues(offset, matrixIndicesT(), matrixEntriesT());

      jacT->replaceLocalValues(offset, index(), value());

      if(fillResid) {
        fT_nonconstView[offset] = (xT_constView[offset] - BCVals[j].val());
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

  Teuchos::RCP<Tpetra_Vector> fT = dirichletWorkset.fT;
  Teuchos::RCP<Tpetra_MultiVector> fpT = dirichletWorkset.fpT;
  Teuchos::RCP<Tpetra_MultiVector> JVT = dirichletWorkset.JVT;
  Teuchos::RCP<const Tpetra_Vector> xT = Albany::getConstTpetraVector(dirichletWorkset.x);
  Teuchos::RCP<const Tpetra_MultiVector> VxT = Albany::getConstTpetraMultiVector(dirichletWorkset.Vx);

  Teuchos::ArrayRCP<const ST> VxT_constView;
  Teuchos::ArrayRCP<ST> fT_nonconstView;
  if (fT != Teuchos::null) fT_nonconstView = fT->get1dViewNonConst();
  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();

  const RealType j_coeff = dirichletWorkset.j_coeff;
  const std::vector<std::vector<int> >& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;
  const std::vector<double*>& nsNodeCoords =
    dirichletWorkset.nodeSetCoords->find(this->nodeSetID)->second;

  RealType time = dirichletWorkset.current_time;
  int number_of_components = this->func.getNumComponents();

  double* coord;
  std::vector<ScalarT> BCVals(number_of_components);

  for(unsigned int inode = 0; inode < nsNodes.size(); inode++) {
    coord = nsNodeCoords[inode];

    this->func.computeBCs(coord, BCVals, time);

    for(unsigned int j = 0; j < number_of_components; j++) {

      int offset = nsNodes[inode][j];

      if(fT != Teuchos::null)
        fT_nonconstView[offset] = (xT_constView[offset] - BCVals[j].val());

      if(JVT != Teuchos::null){
        Teuchos::ArrayRCP<ST> JVT_nonconstView;
        for(int i = 0; i < dirichletWorkset.num_cols_x; i++){
          JVT_nonconstView = JVT->getDataNonConst(i);
          VxT_constView = VxT->getData(i);
          JVT_nonconstView[offset] = j_coeff * VxT_constView[offset];
        }
      }

      if(fpT != Teuchos::null){
        Teuchos::ArrayRCP<ST> fpT_nonconstView;
        for(int i = 0; i < dirichletWorkset.num_cols_p; i++){
          fpT_nonconstView = fpT->getDataNonConst(i);
          fpT_nonconstView[offset] = -BCVals[j].dx(dirichletWorkset.param_offset + i);
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

  Teuchos::RCP<Tpetra_MultiVector> fpVT = dirichletWorkset.fpVT;
  //non-const view of VpT
  Teuchos::ArrayRCP<ST> fpVT_nonconstView;
  bool trans = dirichletWorkset.transpose_dist_param_deriv;
  int num_cols = fpVT->getNumVectors();

  //
  // We're currently assuming Dirichlet BC's can't be distributed parameters.
  // Thus we don't need to actually evaluate the BC's here.  The code to do
  // so is still here, just commented out for future reference.
  //

  const std::vector<std::vector<int> >& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;
  const std::vector<double*>& nsNodeCoords =
    dirichletWorkset.nodeSetCoords->find(this->nodeSetID)->second;

  // RealType time = dirichletWorkset.current_time;
  int number_of_components = this->func.getNumComponents();

  // double* coord;
  // std::vector<ScalarT> BCVals(number_of_components);

  // For (df/dp)^T*V we zero out corresponding entries in V
  if (trans) {
    Teuchos::RCP<Tpetra_MultiVector> VpT = dirichletWorkset.Vp_bcT;
    //non-const view of VpT
    Teuchos::ArrayRCP<ST> VpT_nonconstView;
    for(unsigned int inode = 0; inode < nsNodes.size(); inode++) {
      // coord = nsNodeCoords[inode];
      // this->func.computeBCs(coord, BCVals, time);

      for(unsigned int j = 0; j < number_of_components; j++) {
        int offset = nsNodes[inode][j];
        for (int col=0; col<num_cols; ++col) {
          //(*Vp)[col][offset] = 0.0;
          VpT_nonconstView = VpT->getDataNonConst(col);
          VpT_nonconstView[offset] = 0.0;
        }
      }
    }
  }

  // for (df/dp)*V we zero out corresponding entries in df/dp
  else {
    for(unsigned int inode = 0; inode < nsNodes.size(); inode++) {
      // coord = nsNodeCoords[inode];
      // this->func.computeBCs(coord, BCVals, time);

      for(unsigned int j = 0; j < number_of_components; j++) {
        int offset = nsNodes[inode][j];
        for (int col=0; col<num_cols; ++col) {
          //(*fpV)[col][offset] = 0.0;
          fpVT_nonconstView = fpVT->getDataNonConst(col);
          fpVT_nonconstView[offset] = 0.0;
        }
      }
    }
  }

}

} // namespace PHAL
