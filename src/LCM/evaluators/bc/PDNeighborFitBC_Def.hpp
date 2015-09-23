//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/12/14: only Epetra is SG and MP 

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "Tpetra_CrsMatrix.hpp"

#include "Albany_STKDiscretization.hpp"


#ifdef ALBANY_PERIDIGM
#ifdef ALBANY_EPETRA
#include "PeridigmManager.hpp"
#endif
#endif

// **********************************************************************
// Genereric Template Code for Constructor and PostRegistrationSetup
// **********************************************************************

namespace LCM {

template <typename EvalT, typename Traits>
PDNeighborFitBC_Base<EvalT, Traits>::
PDNeighborFitBC_Base(Teuchos::ParameterList& p) :
  PHAL::DirichletBase<EvalT, Traits>(p), perturbDirichlet(1.0) {
  this->perturbDirichlet = p.get<double>("Perturb Dirichlet");
  this->timeStep = p.get<double>("Time Step");
}

// **********************************************************************
// Specialization: Residual
// **********************************************************************
template<typename Traits>
PDNeighborFitBC<PHAL::AlbanyTraits::Residual, Traits>::
PDNeighborFitBC(Teuchos::ParameterList& p) :
  PDNeighborFitBC_Base<PHAL::AlbanyTraits::Residual, Traits>(p) {

}

// **********************************************************************
template<typename Traits>
void
PDNeighborFitBC<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset) {

#ifdef ALBANY_PERIDIGM
#ifdef ALBANY_EPETRA
  Teuchos::RCP<LCM::PeridigmManager> peridigmManager = LCM::PeridigmManager::self();
#endif
#endif

  Teuchos::RCP<Tpetra_Vector> fT = dirichletWorkset.fT;
  Teuchos::RCP<const Tpetra_Vector> xT = dirichletWorkset.xT;
  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();
  Teuchos::ArrayRCP<ST> fT_nonconstView = fT->get1dViewNonConst();

  const std::vector<std::vector<int>>& nsNodes = dirichletWorkset.nodeSets->find(this->nodeSetID)->second;
  const std::vector<double*>& nsNodeCoords =
    dirichletWorkset.nodeSetCoords->find(this->nodeSetID)->second;
  double* coord;
  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
    int xlunk = nsNodes[inode][0];
    int ylunk = nsNodes[inode][1];
    int zlunk = nsNodes[inode][2];
    // fixme: for now assume that there are always 3 dofs per node so the local id is simply lunk/3
    int localId = xlunk/3;
    coord = nsNodeCoords[inode];

#ifdef ALBANY_PERIDIGM
#ifdef ALBANY_EPETRA
      const double val_x = peridigmManager->getDisplacementNeighborhoodFit(localId,coord,0);
      const double val_y = peridigmManager->getDisplacementNeighborhoodFit(localId,coord,1);
      const double val_z = peridigmManager->getDisplacementNeighborhoodFit(localId,coord,2);
#else
      const double val_x = 0.0;
      const double val_y = 0.0;
      const double val_z = 0.0;
#endif
#else
      const double val_x = 0.0;
      const double val_y = 0.0;
      const double val_z = 0.0;
#endif

      // The scheme below is intended to be consistent with Velocity-Verlet time integration (explicit transient dynamics).
      // Should work for statics and quasi-statics as well.

      double delta_x = xT_constView[xlunk] - val_x;
      double delta_y = xT_constView[ylunk] - val_y;
      double delta_z = xT_constView[zlunk] - val_z;

      double a_x = (2.0/3.0)*delta_x/(this->timeStep*this->timeStep);
      double a_y = (2.0/3.0)*delta_y/(this->timeStep*this->timeStep);
      double a_z = (2.0/3.0)*delta_z/(this->timeStep*this->timeStep);

      fT_nonconstView[xlunk] = -1.5 * this->perturbDirichlet * a_x;
      fT_nonconstView[ylunk] = -1.5 * this->perturbDirichlet * a_y;
      fT_nonconstView[zlunk] = -1.5 * this->perturbDirichlet * a_z;
  }
}

// **********************************************************************
// Specialization: Jacobian
// **********************************************************************
template<typename Traits>
PDNeighborFitBC<PHAL::AlbanyTraits::Jacobian, Traits>::
PDNeighborFitBC(Teuchos::ParameterList& p) :
  PDNeighborFitBC_Base<PHAL::AlbanyTraits::Jacobian, Traits>(p) {
}

// **********************************************************************
template<typename Traits>
void PDNeighborFitBC<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset) {

#ifdef ALBANY_PERIDIGM
#ifdef ALBANY_EPETRA
  Teuchos::RCP<LCM::PeridigmManager> peridigmManager = LCM::PeridigmManager::self();
#endif
#endif

  Teuchos::RCP<Tpetra_Vector> fT = dirichletWorkset.fT;
  Teuchos::RCP<const Tpetra_Vector> xT = dirichletWorkset.xT;
  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();
  Teuchos::RCP<Tpetra_CrsMatrix> jacT = dirichletWorkset.JacT;

  const RealType j_coeff = dirichletWorkset.j_coeff;
  const std::vector<std::vector<int>>& nsNodes = dirichletWorkset.nodeSets->find(this->nodeSetID)->second;
  const std::vector<double*>& nsNodeCoords =
    dirichletWorkset.nodeSetCoords->find(this->nodeSetID)->second;
  double* coord;

  bool fillResid = (fT != Teuchos::null);
  Teuchos::ArrayRCP<ST> fT_nonconstView;                                         
  if (fillResid) fT_nonconstView = fT->get1dViewNonConst();

  Teuchos::Array<LO> index(1);
  Teuchos::Array<ST> value(1); 
  size_t numEntriesT;  
  value[0] = j_coeff; 
  Teuchos::Array<ST> matrixEntriesT; 
  Teuchos::Array<LO> matrixIndicesT; 

  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
      int xlunk = nsNodes[inode][0];
      int ylunk = nsNodes[inode][1];
      int zlunk = nsNodes[inode][2];
      // fixme: for now assume that there are always 3 dofs per node so the local id is simply lunk/3
      int localId = xlunk/3;
      coord = nsNodeCoords[inode];

#ifdef ALBANY_PERIDIGM
#ifdef ALBANY_EPETRA
      const double val_x = peridigmManager->getDisplacementNeighborhoodFit(localId,coord,0);
      const double val_y = peridigmManager->getDisplacementNeighborhoodFit(localId,coord,1);
      const double val_z = peridigmManager->getDisplacementNeighborhoodFit(localId,coord,2);
#else
      const double val_x = 0.0;
      const double val_y = 0.0;
      const double val_z = 0.0;
#endif
#else
      const double val_x = 0.0;
      const double val_y = 0.0;
      const double val_z = 0.0;
#endif

      numEntriesT = jacT->getNumEntriesInLocalRow(xlunk);
      matrixEntriesT.resize(numEntriesT);
      matrixIndicesT.resize(numEntriesT);
      jacT->getLocalRowCopy(xlunk, matrixIndicesT(), matrixEntriesT(), numEntriesT);
      for (int i=0; i<numEntriesT; i++) matrixEntriesT[i]=0;
      jacT->replaceLocalValues(xlunk, matrixIndicesT(), matrixEntriesT());

      index[0] = xlunk;
      jacT->replaceLocalValues(xlunk, index(), value());

      numEntriesT = jacT->getNumEntriesInLocalRow(ylunk);
      matrixEntriesT.resize(numEntriesT); 
      matrixIndicesT.resize(numEntriesT); 
      jacT->getLocalRowCopy(ylunk, matrixIndicesT(), matrixEntriesT(), numEntriesT);
      for (int i=0; i<numEntriesT; i++) matrixEntriesT[i]=0;
      jacT->replaceLocalValues(ylunk, matrixIndicesT(), matrixEntriesT());

      index[0] = ylunk;
      jacT->replaceLocalValues(ylunk, index(), value());

      numEntriesT = jacT->getNumEntriesInLocalRow(zlunk);
      matrixEntriesT.resize(numEntriesT);
      matrixIndicesT.resize(numEntriesT);
      jacT->getLocalRowCopy(zlunk, matrixIndicesT(), matrixEntriesT(), numEntriesT);
      for (int i=0; i<numEntriesT; i++) matrixEntriesT[i]=0;
      jacT->replaceLocalValues(zlunk, matrixIndicesT(), matrixEntriesT());

      index[0] = zlunk;
      jacT->replaceLocalValues(zlunk, index(), value());

      if (fillResid){
        fT_nonconstView[xlunk] = xT_constView[xlunk] - val_x;
        fT_nonconstView[ylunk] = xT_constView[ylunk] - val_y;
        fT_nonconstView[zlunk] = xT_constView[zlunk] - val_z;
      }
  }
}

// **********************************************************************
// Specialization: Tangent
// **********************************************************************
template<typename Traits>
PDNeighborFitBC<PHAL::AlbanyTraits::Tangent, Traits>::
PDNeighborFitBC(Teuchos::ParameterList& p) :
  PDNeighborFitBC_Base<PHAL::AlbanyTraits::Tangent, Traits>(p) {
}

// **********************************************************************
template<typename Traits>
void PDNeighborFitBC<PHAL::AlbanyTraits::Tangent, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset) {

#ifdef ALBANY_PERIDIGM
#ifdef ALBANY_EPETRA
  Teuchos::RCP<LCM::PeridigmManager> peridigmManager = LCM::PeridigmManager::self();
#endif
#endif

  Teuchos::RCP<Tpetra_Vector> fT = dirichletWorkset.fT;
  Teuchos::RCP<Tpetra_MultiVector> fpT = dirichletWorkset.fpT;
  Teuchos::RCP<Tpetra_MultiVector> JVT = dirichletWorkset.JVT;
  Teuchos::RCP<const Tpetra_Vector> xT = dirichletWorkset.xT;
  Teuchos::RCP<const Tpetra_MultiVector> VxT = dirichletWorkset.VxT;
  
  Teuchos::ArrayRCP<const ST> VxT_constView; 
  Teuchos::ArrayRCP<ST> fT_nonconstView;                                         
  if (fT != Teuchos::null) fT_nonconstView = fT->get1dViewNonConst();
  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();                                       

  const RealType j_coeff = dirichletWorkset.j_coeff;
  const std::vector<std::vector<int>>& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;
  const std::vector<double*>& nsNodeCoords =
    dirichletWorkset.nodeSetCoords->find(this->nodeSetID)->second;
  double* coord;

  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
    int xlunk = nsNodes[inode][0];
    int ylunk = nsNodes[inode][1];
    int zlunk = nsNodes[inode][2];
    // fixme: for now assume that there are always 3 dofs per node so the local id is simply lunk/3
    int localId = xlunk/3;
    coord = nsNodeCoords[inode];

#ifdef ALBANY_PERIDIGM
#ifdef ALBANY_EPETRA
      const double val_x = peridigmManager->getDisplacementNeighborhoodFit(localId,coord,0);
      const double val_y = peridigmManager->getDisplacementNeighborhoodFit(localId,coord,1);
      const double val_z = peridigmManager->getDisplacementNeighborhoodFit(localId,coord,2);
#else
      const double val_x = 0.0;
      const double val_y = 0.0;
      const double val_z = 0.0;
#endif
#else
      const double val_x = 0.0;
      const double val_y = 0.0;
      const double val_z = 0.0;
#endif

    if (fT != Teuchos::null) { 
      fT_nonconstView[xlunk] = xT_constView[xlunk] - val_x;
      fT_nonconstView[ylunk] = xT_constView[ylunk] - val_y;
      fT_nonconstView[zlunk] = xT_constView[zlunk] - val_z;
    }
    
    if (JVT != Teuchos::null) {
      Teuchos::ArrayRCP<ST> JVT_nonconstView; 
      for (int i=0; i<dirichletWorkset.num_cols_x; i++) {
        JVT_nonconstView = JVT->getDataNonConst(i); 
        VxT_constView = VxT->getData(i); 
        JVT_nonconstView[xlunk] = j_coeff*VxT_constView[xlunk];
        JVT_nonconstView[ylunk] = j_coeff*VxT_constView[ylunk];
        JVT_nonconstView[zlunk] = j_coeff*VxT_constView[zlunk];
      }
    }
    
    if (fpT != Teuchos::null) {
      Teuchos::ArrayRCP<ST> fpT_nonconstView;                                         
      for (int i=0; i<dirichletWorkset.num_cols_p; i++) {
        fpT_nonconstView = fpT->getDataNonConst(i); 
        fpT_nonconstView[xlunk] = 0;
        fpT_nonconstView[ylunk] = 0;
        fpT_nonconstView[zlunk] = 0;
      }
    }
  }
}

// **********************************************************************
// Specialization: DistParamDeriv
// **********************************************************************
template<typename Traits>
PDNeighborFitBC<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
PDNeighborFitBC(Teuchos::ParameterList& p) :
  PDNeighborFitBC_Base<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p) {
}

// **********************************************************************
template<typename Traits>
void PDNeighborFitBC<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset) {

  Teuchos::RCP<Tpetra_MultiVector> fpVT = dirichletWorkset.fpVT;
  //non-const view of fpVT
  Teuchos::ArrayRCP<ST> fpVT_nonconstView;
  bool trans = dirichletWorkset.transpose_dist_param_deriv;
  int num_cols = fpVT->getNumVectors();

  const std::vector<std::vector<int>>& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;

  // For (df/dp)^T*V we zero out corresponding entries in V
  if (trans) {
    Teuchos::RCP<Tpetra_MultiVector> VpT = dirichletWorkset.Vp_bcT;
    //non-const view of VpT
    Teuchos::ArrayRCP<ST> VpT_nonconstView;
    for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
      int xlunk = nsNodes[inode][0];
      int ylunk = nsNodes[inode][1];
      int zlunk = nsNodes[inode][2];

      for (int col=0; col<num_cols; ++col) {
        //(*Vp)[col][lunk] = 0.0;
        VpT_nonconstView = VpT->getDataNonConst(col); 
        VpT_nonconstView[xlunk] = 0.0;
        VpT_nonconstView[ylunk] = 0.0;
        VpT_nonconstView[zlunk] = 0.0;
       }
    }
  }

  // for (df/dp)*V we zero out corresponding entries in df/dp
  else {
    for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
      int xlunk = nsNodes[inode][0];
      int ylunk = nsNodes[inode][1];
      int zlunk = nsNodes[inode][2];

      for (int col=0; col<num_cols; ++col) {
        //(*fpV)[col][lunk] = 0.0;
        fpVT_nonconstView = fpVT->getDataNonConst(col); 
        fpVT_nonconstView[xlunk] = 0.0;
        fpVT_nonconstView[ylunk] = 0.0;
        fpVT_nonconstView[zlunk] = 0.0;
      }
    }
  }
}

#ifdef ALBANY_SG
// **********************************************************************
template<typename Traits>
PDNeighborFitBC<PHAL::AlbanyTraits::SGResidual, Traits>::
PDNeighborFitBC(Teuchos::ParameterList& p) :
  PDNeighborFitBC_Base<PHAL::AlbanyTraits::SGResidual, Traits>(p) {
    throw("LCM::PDNeighborFitBC not implemented for SG or ENSEMBLE configurations");
}
template<typename Traits>
void PDNeighborFitBC<PHAL::AlbanyTraits::SGResidual, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset) {
}
template<typename Traits>
PDNeighborFitBC<PHAL::AlbanyTraits::SGJacobian, Traits>::
PDNeighborFitBC(Teuchos::ParameterList& p) :
  PDNeighborFitBC_Base<PHAL::AlbanyTraits::SGJacobian, Traits>(p) {
    throw("LCM::PDNeighborFitBC not implemented for SG or ENSEMBLE configurations");
}
template<typename Traits>
void PDNeighborFitBC<PHAL::AlbanyTraits::SGJacobian, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset) {
}
template<typename Traits>
PDNeighborFitBC<PHAL::AlbanyTraits::SGTangent, Traits>::
PDNeighborFitBC(Teuchos::ParameterList& p) :
  PDNeighborFitBC_Base<PHAL::AlbanyTraits::SGTangent, Traits>(p) {
    throw("LCM::PDNeighborFitBC not implemented for SG or ENSEMBLE configurations");
}
template<typename Traits>
void PDNeighborFitBC<PHAL::AlbanyTraits::SGTangent, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset) {
}
#endif

#ifdef ALBANY_ENSEMBLE
template<typename Traits>
PDNeighborFitBC<PHAL::AlbanyTraits::MPResidual, Traits>::
PDNeighborFitBC(Teuchos::ParameterList& p) :
  PDNeighborFitBC_Base<PHAL::AlbanyTraits::MPResidual, Traits>(p) {
    throw("LCM::PDNeighborFitBC not implemented for SG or ENSEMBLE configurations");
}
template<typename Traits>
void PDNeighborFitBC<PHAL::AlbanyTraits::MPResidual, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset) {
}
template<typename Traits>
PDNeighborFitBC<PHAL::AlbanyTraits::MPJacobian, Traits>::
PDNeighborFitBC(Teuchos::ParameterList& p) :
  PDNeighborFitBC_Base<PHAL::AlbanyTraits::MPJacobian, Traits>(p) {
    throw("LCM::PDNeighborFitBC not implemented for SG or ENSEMBLE configurations");
}
template<typename Traits>
void PDNeighborFitBC<PHAL::AlbanyTraits::MPJacobian, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset) {
}
template<typename Traits>
PDNeighborFitBC<PHAL::AlbanyTraits::MPTangent, Traits>::
PDNeighborFitBC(Teuchos::ParameterList& p) :
  PDNeighborFitBC_Base<PHAL::AlbanyTraits::MPTangent, Traits>(p) {
    throw("LCM::PDNeighborFitBC not implemented for SG or ENSEMBLE configurations");
}
template<typename Traits>
void PDNeighborFitBC<PHAL::AlbanyTraits::MPTangent, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset) {
}
#endif

// **********************************************************************
}
