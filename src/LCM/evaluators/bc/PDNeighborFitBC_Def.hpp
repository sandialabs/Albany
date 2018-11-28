//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

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

#include "Albany_ThyraUtils.hpp"

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

  Teuchos::RCP<const Thyra_Vector> x = dirichletWorkset.x;
  Teuchos::RCP<Thyra_Vector>       f = dirichletWorkset.f;

  Teuchos::ArrayRCP<const ST> x_constView    = Albany::getLocalData(x);
  Teuchos::ArrayRCP<ST>       f_nonconstView = Albany::getNonconstLocalData(f);

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

      double delta_x = x_constView[xlunk] - val_x;
      double delta_y = x_constView[ylunk] - val_y;
      double delta_z = x_constView[zlunk] - val_z;

      double a_x = (2.0/3.0)*delta_x/(this->timeStep*this->timeStep);
      double a_y = (2.0/3.0)*delta_y/(this->timeStep*this->timeStep);
      double a_z = (2.0/3.0)*delta_z/(this->timeStep*this->timeStep);

      f_nonconstView[xlunk] = -1.5 * this->perturbDirichlet * a_x;
      f_nonconstView[ylunk] = -1.5 * this->perturbDirichlet * a_y;
      f_nonconstView[zlunk] = -1.5 * this->perturbDirichlet * a_z;
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

  Teuchos::RCP<const Thyra_Vector> x   = dirichletWorkset.x;
  Teuchos::RCP<Thyra_Vector>       f   = dirichletWorkset.f;
  Teuchos::RCP<Thyra_LinearOp>     jac = dirichletWorkset.Jac;

  Teuchos::ArrayRCP<const ST> x_constView = Albany::getLocalData(x);
  Teuchos::ArrayRCP<ST>       f_nonconstView;

  const RealType j_coeff = dirichletWorkset.j_coeff;
  const std::vector<std::vector<int>>& nsNodes = dirichletWorkset.nodeSets->find(this->nodeSetID)->second;
  const std::vector<double*>& nsNodeCoords = dirichletWorkset.nodeSetCoords->find(this->nodeSetID)->second;
  double* coord;

  bool fillResid = (f != Teuchos::null);
  if (fillResid) {
    f_nonconstView = Albany::getNonconstLocalData(f);
  }

  Teuchos::Array<LO> index(1);
  Teuchos::Array<ST> value(1);
  size_t numEntriesT;
  value[0] = j_coeff;
  Teuchos::Array<ST> matrixEntries;
  Teuchos::Array<LO> matrixIndices;

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

      Albany::getLocalRowValues(jac, xlunk, matrixIndices, matrixEntries);
      for (auto& val : matrixEntries) { val = 0.0; }
      Albany::setLocalRowValues(jac, xlunk, matrixIndices(), matrixEntries());
      index[0] = xlunk;
      Albany::setLocalRowValues(jac, xlunk, index(), value());

      Albany::getLocalRowValues(jac, ylunk, matrixIndices, matrixEntries);
      for (auto& val : matrixEntries) { val = 0.0; }
      Albany::setLocalRowValues(jac, ylunk, matrixIndices(), matrixEntries());
      index[0] = ylunk;
      Albany::setLocalRowValues(jac, ylunk, index(), value());

      Albany::getLocalRowValues(jac, zlunk, matrixIndices, matrixEntries);
      for (auto& val : matrixEntries) { val = 0.0; }
      Albany::setLocalRowValues(jac, zlunk, matrixIndices(), matrixEntries());
      index[0] = zlunk;
      Albany::setLocalRowValues(jac, zlunk, index(), value());

      if (fillResid){
        f_nonconstView[xlunk] = x_constView[xlunk] - val_x;
        f_nonconstView[ylunk] = x_constView[ylunk] - val_y;
        f_nonconstView[zlunk] = x_constView[zlunk] - val_z;
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

  Teuchos::RCP<const Thyra_Vector>       x = dirichletWorkset.x;
  Teuchos::RCP<const Thyra_MultiVector> Vx = dirichletWorkset.Vx;

  Teuchos::RCP<Thyra_Vector>       f = dirichletWorkset.f;
  Teuchos::RCP<Thyra_MultiVector> fp = dirichletWorkset.fp;
  Teuchos::RCP<Thyra_MultiVector> JV = dirichletWorkset.JV;

  Teuchos::ArrayRCP<const ST> x_constView = Albany::getLocalData(x);
  Teuchos::ArrayRCP<ST> f_nonconstView;

  Teuchos::ArrayRCP<Teuchos::ArrayRCP<const ST>> Vx_const2dView;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>>       JV_nonconst2dView;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>>       fp_nonconst2dView;

  if (f != Teuchos::null) {
    f_nonconstView = Albany::getNonconstLocalData(f);
  }
  if (JV != Teuchos::null) {
    Vx_const2dView    = Albany::getLocalData(Vx);
    JV_nonconst2dView = Albany::getNonconstLocalData(JV);
  }
  if (fp != Teuchos::null) {
    fp_nonconst2dView = Albany::getNonconstLocalData(fp);
  }

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

    if (f != Teuchos::null) {
      f_nonconstView[xlunk] = x_constView[xlunk] - val_x;
      f_nonconstView[ylunk] = x_constView[ylunk] - val_y;
      f_nonconstView[zlunk] = x_constView[zlunk] - val_z;
    }

    if (JV != Teuchos::null) {
      for (int i=0; i<dirichletWorkset.num_cols_x; i++) {
        JV_nonconst2dView[i][xlunk] = j_coeff*Vx_const2dView[i][xlunk];
        JV_nonconst2dView[i][ylunk] = j_coeff*Vx_const2dView[i][ylunk];
        JV_nonconst2dView[i][zlunk] = j_coeff*Vx_const2dView[i][zlunk];
      }
    }

    if (fp != Teuchos::null) {
      for (int i=0; i<dirichletWorkset.num_cols_p; i++) {
        fp_nonconst2dView[i][xlunk] = 0;
        fp_nonconst2dView[i][ylunk] = 0;
        fp_nonconst2dView[i][zlunk] = 0;
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
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  Teuchos::RCP<Thyra_MultiVector> fpV = dirichletWorkset.fpV;
  bool trans = dirichletWorkset.transpose_dist_param_deriv;
  int num_cols = fpV->domain()->dim();

  const std::vector<std::vector<int>>& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;

  if (trans) {
    // For (df/dp)^T*V we zero out corresponding entries in V
    Teuchos::RCP<Thyra_MultiVector> Vp = dirichletWorkset.Vp_bc;
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>> Vp_nonconst2dView = Albany::getNonconstLocalData(Vp);
    for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
      int xlunk = nsNodes[inode][0];
      int ylunk = nsNodes[inode][1];
      int zlunk = nsNodes[inode][2];

      for (int col=0; col<num_cols; ++col) {
        //(*Vp)[col][lunk] = 0.0;
        Vp_nonconst2dView[col][xlunk] = 0.0;
        Vp_nonconst2dView[col][ylunk] = 0.0;
        Vp_nonconst2dView[col][zlunk] = 0.0;
       }
    }
  } else {
    // for (df/dp)*V we zero out corresponding entries in df/dp
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>> fpV_nonconst2dView = Albany::getNonconstLocalData(fpV);
    for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
      int xlunk = nsNodes[inode][0];
      int ylunk = nsNodes[inode][1];
      int zlunk = nsNodes[inode][2];

      for (int col=0; col<num_cols; ++col) {
        //(*fpV)[col][lunk] = 0.0;
        fpV_nonconst2dView[col][xlunk] = 0.0;
        fpV_nonconst2dView[col][ylunk] = 0.0;
        fpV_nonconst2dView[col][zlunk] = 0.0;
      }
    }
  }
}

// **********************************************************************
} // namespace LCM
