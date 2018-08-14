//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "Teuchos_TestForException.hpp"

#include "Albany_TpetraThyraUtils.hpp"

// **********************************************************************
// Genereric Template Code for Constructor and PostRegistrationSetup
// **********************************************************************

namespace LCM {

template <typename EvalT, typename Traits>
TorsionBC_Base<EvalT, Traits>::TorsionBC_Base(Teuchos::ParameterList& p)
    : PHAL::DirichletBase<EvalT, Traits>(p),
      thetaDot(p.get<RealType>("Theta Dot")),
      X0(p.get<RealType>("X0")),
      Y0(p.get<RealType>("Y0"))
{
}

// **********************************************************************
template <typename EvalT, typename Traits>
void
TorsionBC_Base<EvalT, Traits>::computeBCs(
    double*        coord,
    ScalarT&       Xval,
    ScalarT&       Yval,
    const RealType time)
{
  RealType X(coord[0]);
  RealType Y(coord[1]);
  RealType theta(thetaDot * time);

  // compute displace Xval and Yval. (X0,Y0) is the center of rotation/torsion
  Xval = X0 + (X - X0) * std::cos(theta) - (Y - Y0) * std::sin(theta) - X;
  Yval = Y0 + (X - X0) * std::sin(theta) + (Y - Y0) * std::cos(theta) - Y;

  // a different set of bc, for comparison with analytical solution
  // RealType L = 2.0;
  // Xval = -theta * L * Y;
  // Yval = theta * L * X;
}

// **********************************************************************
// Specialization: Residual
// **********************************************************************
template <typename Traits>
TorsionBC<PHAL::AlbanyTraits::Residual, Traits>::TorsionBC(
    Teuchos::ParameterList& p)
    : TorsionBC_Base<PHAL::AlbanyTraits::Residual, Traits>(p)
{
}

// **********************************************************************
template <typename Traits>
void
TorsionBC<PHAL::AlbanyTraits::Residual, Traits>::evaluateFields(
    typename Traits::EvalData dirichletWorkset)
{
  Teuchos::RCP<Tpetra_Vector>       fT = dirichletWorkset.fT;
  Teuchos::RCP<const Tpetra_Vector> xT =
      Albany::getConstTpetraVector(dirichletWorkset.x);
  Teuchos::ArrayRCP<const ST> xT_constView    = xT->get1dView();
  Teuchos::ArrayRCP<ST>       fT_nonconstView = fT->get1dViewNonConst();

  // Grab the vector off node GIDs for this Node Set ID from the std::map
  const std::vector<std::vector<int>>& nsNodes =
      dirichletWorkset.nodeSets->find(this->nodeSetID)->second;
  const std::vector<double*>& nsNodeCoords =
      dirichletWorkset.nodeSetCoords->find(this->nodeSetID)->second;

  RealType time = dirichletWorkset.current_time;

  int     xlunk, ylunk;  // global and local indicies into unknown vector
  double* coord;
  ScalarT Xval, Yval;

  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
    xlunk = nsNodes[inode][0];
    ylunk = nsNodes[inode][1];
    coord = nsNodeCoords[inode];

    this->computeBCs(coord, Xval, Yval, time);

    fT_nonconstView[xlunk] = xT_constView[xlunk] - Xval;
    fT_nonconstView[ylunk] = xT_constView[ylunk] - Yval;

    // Record DOFs to avoid setting Schwarz BCs on them.
    dirichletWorkset.fixed_dofs_.insert(xlunk);
    dirichletWorkset.fixed_dofs_.insert(ylunk);
  }
}

// **********************************************************************
// Specialization: Jacobian
// **********************************************************************
template <typename Traits>
TorsionBC<PHAL::AlbanyTraits::Jacobian, Traits>::TorsionBC(
    Teuchos::ParameterList& p)
    : TorsionBC_Base<PHAL::AlbanyTraits::Jacobian, Traits>(p)
{
}
// **********************************************************************
template <typename Traits>
void
TorsionBC<PHAL::AlbanyTraits::Jacobian, Traits>::evaluateFields(
    typename Traits::EvalData dirichletWorkset)
{
  Teuchos::RCP<Tpetra_Vector>       fT = dirichletWorkset.fT;
  Teuchos::RCP<const Tpetra_Vector> xT =
      Albany::getConstTpetraVector(dirichletWorkset.x);
  Teuchos::ArrayRCP<const ST>    xT_constView = xT->get1dView();
  Teuchos::RCP<Tpetra_CrsMatrix> jacT         = dirichletWorkset.JacT;

  const RealType                       j_coeff = dirichletWorkset.j_coeff;
  const std::vector<std::vector<int>>& nsNodes =
      dirichletWorkset.nodeSets->find(this->nodeSetID)->second;
  const std::vector<double*>& nsNodeCoords =
      dirichletWorkset.nodeSetCoords->find(this->nodeSetID)->second;

  bool                  fillResid = (fT != Teuchos::null);
  Teuchos::ArrayRCP<ST> fT_nonconstView;
  if (fillResid) fT_nonconstView = fT->get1dViewNonConst();

  RealType time = dirichletWorkset.current_time;

  int     xlunk, ylunk;  // local indicies into unknown vector
  double* coord;
  ScalarT Xval, Yval;

  Teuchos::Array<LO> index(1);
  Teuchos::Array<ST> value(1);
  size_t             numEntriesT;
  value[0] = j_coeff;
  Teuchos::Array<ST> matrixEntriesT;
  Teuchos::Array<LO> matrixIndicesT;

  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
    xlunk = nsNodes[inode][0];
    ylunk = nsNodes[inode][1];
    coord = nsNodeCoords[inode];

    this->computeBCs(coord, Xval, Yval, time);

    // replace jac values for the X dof
    numEntriesT = jacT->getNumEntriesInLocalRow(xlunk);
    matrixEntriesT.resize(numEntriesT);
    matrixIndicesT.resize(numEntriesT);
    jacT->getLocalRowCopy(
        xlunk, matrixIndicesT(), matrixEntriesT(), numEntriesT);
    for (int i = 0; i < numEntriesT; i++) matrixEntriesT[i] = 0;
    jacT->replaceLocalValues(xlunk, matrixIndicesT(), matrixEntriesT());

    index[0] = xlunk;
    jacT->replaceLocalValues(xlunk, index(), value());

    // replace jac values for the y dof
    numEntriesT = jacT->getNumEntriesInLocalRow(ylunk);
    matrixEntriesT.resize(numEntriesT);
    matrixIndicesT.resize(numEntriesT);
    jacT->getLocalRowCopy(
        ylunk, matrixIndicesT(), matrixEntriesT(), numEntriesT);
    for (int i = 0; i < numEntriesT; i++) matrixEntriesT[i] = 0;
    jacT->replaceLocalValues(ylunk, matrixIndicesT(), matrixEntriesT());

    index[0] = ylunk;
    jacT->replaceLocalValues(ylunk, index(), value());

    if (fillResid) {
      fT_nonconstView[xlunk] = xT_constView[xlunk] - Xval.val();
      fT_nonconstView[ylunk] = xT_constView[ylunk] - Yval.val();
    }
  }
}

// **********************************************************************
// Specialization: Tangent
// **********************************************************************
template <typename Traits>
TorsionBC<PHAL::AlbanyTraits::Tangent, Traits>::TorsionBC(
    Teuchos::ParameterList& p)
    : TorsionBC_Base<PHAL::AlbanyTraits::Tangent, Traits>(p)
{
}
// **********************************************************************
template <typename Traits>
void
TorsionBC<PHAL::AlbanyTraits::Tangent, Traits>::evaluateFields(
    typename Traits::EvalData dirichletWorkset)
{
  Teuchos::RCP<Tpetra_Vector>       fT  = dirichletWorkset.fT;
  Teuchos::RCP<Tpetra_MultiVector>  fpT = dirichletWorkset.fpT;
  Teuchos::RCP<Tpetra_MultiVector>  JVT = dirichletWorkset.JVT;
  Teuchos::RCP<const Tpetra_Vector> xT =
      Albany::getConstTpetraVector(dirichletWorkset.x);
  Teuchos::RCP<const Tpetra_MultiVector> VxT =
      Albany::getConstTpetraMultiVector(dirichletWorkset.Vx);

  Teuchos::ArrayRCP<const ST> VxT_constView;
  Teuchos::ArrayRCP<ST>       fT_nonconstView;
  if (fT != Teuchos::null) fT_nonconstView = fT->get1dViewNonConst();
  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();

  const RealType                       j_coeff = dirichletWorkset.j_coeff;
  const std::vector<std::vector<int>>& nsNodes =
      dirichletWorkset.nodeSets->find(this->nodeSetID)->second;
  const std::vector<double*>& nsNodeCoords =
      dirichletWorkset.nodeSetCoords->find(this->nodeSetID)->second;

  RealType time = dirichletWorkset.current_time;

  int     xlunk, ylunk;  // global and local indicies into unknown vector
  double* coord;
  ScalarT Xval, Yval;
  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
    xlunk = nsNodes[inode][0];
    ylunk = nsNodes[inode][1];
    coord = nsNodeCoords[inode];

    this->computeBCs(coord, Xval, Yval, time);

    if (fT != Teuchos::null) {
      fT_nonconstView[xlunk] = xT_constView[xlunk] - Xval.val();
      fT_nonconstView[ylunk] = xT_constView[ylunk] - Yval.val();
    }

    if (JVT != Teuchos::null) {
      Teuchos::ArrayRCP<ST> JVT_nonconstView;
      for (int i = 0; i < dirichletWorkset.num_cols_x; i++) {
        JVT_nonconstView        = JVT->getDataNonConst(i);
        VxT_constView           = VxT->getData(i);
        JVT_nonconstView[xlunk] = j_coeff * VxT_constView[xlunk];
        JVT_nonconstView[ylunk] = j_coeff * VxT_constView[ylunk];
      }
    }

    if (fpT != Teuchos::null) {
      Teuchos::ArrayRCP<ST> fpT_nonconstView;
      for (int i = 0; i < dirichletWorkset.num_cols_p; i++) {
        fpT_nonconstView        = fpT->getDataNonConst(i);
        fpT_nonconstView[xlunk] = -Xval.dx(dirichletWorkset.param_offset + i);
        fpT_nonconstView[ylunk] = -Yval.dx(dirichletWorkset.param_offset + i);
      }
    }
  }
}

// **********************************************************************
// Specialization: DistParamDeriv
// **********************************************************************
template <typename Traits>
TorsionBC<PHAL::AlbanyTraits::DistParamDeriv, Traits>::TorsionBC(
    Teuchos::ParameterList& p)
    : TorsionBC_Base<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p)
{
}
// **********************************************************************
template <typename Traits>
void
TorsionBC<PHAL::AlbanyTraits::DistParamDeriv, Traits>::evaluateFields(
    typename Traits::EvalData dirichletWorkset)
{
  Teuchos::RCP<Tpetra_MultiVector> fpVT = dirichletWorkset.fpVT;
  Teuchos::ArrayRCP<ST>            fpVT_nonconstView;
  bool trans    = dirichletWorkset.transpose_dist_param_deriv;
  int  num_cols = fpVT->getNumVectors();

  //
  // We're currently assuming Dirichlet BC's can't be distributed parameters.
  // Thus we don't need to actually evaluate the BC's here.  The code to do
  // so is still here, just commented out for future reference.
  //

  const std::vector<std::vector<int>>& nsNodes =
      dirichletWorkset.nodeSets->find(this->nodeSetID)->second;
  const std::vector<double*>& nsNodeCoords =
      dirichletWorkset.nodeSetCoords->find(this->nodeSetID)->second;

  // RealType time = dirichletWorkset.current_time;

  int xlunk, ylunk;  // global and local indicies into unknown vector
  // double* coord;
  // ScalarT Xval, Yval;

  // For (df/dp)^T*V we zero out corresponding entries in V
  if (trans) {
    Teuchos::RCP<Tpetra_MultiVector> VpT = dirichletWorkset.Vp_bcT;
    Teuchos::ArrayRCP<ST>            VpT_nonconstView;
    for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
      xlunk = nsNodes[inode][0];
      ylunk = nsNodes[inode][1];
      // coord = nsNodeCoords[inode];

      // this->computeBCs(coord, Xval, Yval, time);

      for (int col = 0; col < num_cols; ++col) {
        //(*Vp)[col][xlunk] = 0.0;
        //(*Vp)[col][ylunk] = 0.0;
        VpT_nonconstView        = VpT->getDataNonConst(col);
        VpT_nonconstView[xlunk] = 0.0;
        VpT_nonconstView[ylunk] = 0.0;
      }
    }
  }

  // for (df/dp)*V we zero out corresponding entries in df/dp
  else {
    for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
      xlunk = nsNodes[inode][0];
      ylunk = nsNodes[inode][1];
      // coord = nsNodeCoords[inode];

      // this->computeBCs(coord, Xval, Yval, time);

      for (int col = 0; col < num_cols; ++col) {
        //(*fpV)[col][xlunk] = 0.0;
        //(*fpV)[col][ylunk] = 0.0;
        fpVT_nonconstView        = fpVT->getDataNonConst(col);
        fpVT_nonconstView[xlunk] = 0.0;
        fpVT_nonconstView[ylunk] = 0.0;
      }
    }
  }
}

}  // namespace LCM
