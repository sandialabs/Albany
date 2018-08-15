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

namespace LCM {

template <typename EvalT, typename Traits>
TorsionBC_Base<EvalT, Traits>::
TorsionBC_Base(Teuchos::ParameterList& p) :
  PHAL::DirichletBase<EvalT, Traits>(p),
  thetaDot(p.get<RealType>("Theta Dot")),
  X0(p.get<RealType>("X0")),
  Y0(p.get<RealType>("Y0"))
{
}

// **********************************************************************
template<typename EvalT, typename Traits>
void
TorsionBC_Base<EvalT, Traits>::
computeBCs(double* coord, ScalarT& Xval, ScalarT& Yval,
           const RealType time)
{
  RealType X(coord[0]);
  RealType Y(coord[1]);
  RealType theta(thetaDot*time);

  // compute displace Xval and Yval. (X0,Y0) is the center of rotation/torsion
  Xval = X0 + (X-X0) * std::cos(theta) - (Y-Y0) * std::sin(theta) - X;
  Yval = Y0 + (X-X0) * std::sin(theta) + (Y-Y0) * std::cos(theta) - Y;

  // a different set of bc, for comparison with analytical solution
  //RealType L = 2.0;
  //Xval = -theta * L * Y;
  //Yval = theta * L * X;
}

// **********************************************************************
// Specialization: Residual
// **********************************************************************
template<typename Traits>
TorsionBC<PHAL::AlbanyTraits::Residual, Traits>::
TorsionBC(Teuchos::ParameterList& p) :
  TorsionBC_Base<PHAL::AlbanyTraits::Residual, Traits>(p)
{
}

// **********************************************************************
template<typename Traits>
void
TorsionBC<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  Teuchos::RCP<const Thyra_Vector> x = dirichletWorkset.x;
  Teuchos::RCP<Thyra_Vector>       f = dirichletWorkset.f;

  Teuchos::ArrayRCP<const ST> x_constView    = Albany::getLocalData(x);
  Teuchos::ArrayRCP<ST>       f_nonconstView = Albany::getNonconstLocalData(f);

  // Grab the vector off node GIDs for this Node Set ID from the std::map
  const std::vector<std::vector<int>>& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;
  const std::vector<double*>& nsNodeCoords =
    dirichletWorkset.nodeSetCoords->find(this->nodeSetID)->second;

  RealType time = dirichletWorkset.current_time;

  int xlunk, ylunk; // global and local indicies into unknown vector
  double* coord;
  ScalarT Xval, Yval;

  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
    xlunk = nsNodes[inode][0];
    ylunk = nsNodes[inode][1];
    coord = nsNodeCoords[inode];

    this->computeBCs(coord, Xval, Yval, time);

    f_nonconstView[xlunk] = x_constView[xlunk] - Xval;
    f_nonconstView[ylunk] = x_constView[ylunk] - Yval;

    // Record DOFs to avoid setting Schwarz BCs on them.
    dirichletWorkset.fixed_dofs_.insert(xlunk);
    dirichletWorkset.fixed_dofs_.insert(ylunk);
  }
}

// **********************************************************************
// Specialization: Jacobian
// **********************************************************************
template<typename Traits>
TorsionBC<PHAL::AlbanyTraits::Jacobian, Traits>::
TorsionBC(Teuchos::ParameterList& p) :
  TorsionBC_Base<PHAL::AlbanyTraits::Jacobian, Traits>(p)
{
}
// **********************************************************************
template<typename Traits>
void TorsionBC<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  Teuchos::RCP<const Thyra_Vector> x   = dirichletWorkset.x;
  Teuchos::RCP<Thyra_Vector>       f   = dirichletWorkset.f;
  Teuchos::RCP<Thyra_LinearOp>     jac = dirichletWorkset.Jac;

  Teuchos::ArrayRCP<const ST> x_constView = Albany::getLocalData(x);
  Teuchos::ArrayRCP<ST>       f_nonconstView;


  const RealType j_coeff = dirichletWorkset.j_coeff;
  const std::vector<std::vector<int>>& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;
  const std::vector<double*>& nsNodeCoords =
    dirichletWorkset.nodeSetCoords->find(this->nodeSetID)->second;

  bool fillResid = (f != Teuchos::null);
  if (fillResid) {
    f_nonconstView = Albany::getNonconstLocalData(f);
  }

  RealType time = dirichletWorkset.current_time;

  int xlunk, ylunk; // local indicies into unknown vector
  double* coord;
  ScalarT Xval, Yval;

  Teuchos::Array<LO> index(1);
  Teuchos::Array<ST> value(1);
  size_t numEntries;
  value[0] = j_coeff;
  Teuchos::Array<ST> matrixEntries;
  Teuchos::Array<LO> matrixIndices;


  for (unsigned int inode = 0; inode < nsNodes.size(); inode++)
  {
    xlunk = nsNodes[inode][0];
    ylunk = nsNodes[inode][1];
    coord = nsNodeCoords[inode];

    this->computeBCs(coord, Xval, Yval, time);

    // replace jac values for the X dof
    Albany::getLocalRowValues(jac, xlunk, matrixIndices, matrixEntries);
    for (auto& val : matrixEntries) { val = 0.0; }
    Albany::setLocalRowValues(jac, xlunk, matrixIndices(), matrixEntries());
    index[0] = xlunk;
    Albany::setLocalRowValues(jac, xlunk, index(), value());

    // replace jac values for the y dof
    Albany::getLocalRowValues(jac, ylunk, matrixIndices, matrixEntries);
    for (auto& val : matrixEntries) { val = 0.0; }
    Albany::setLocalRowValues(jac, ylunk, matrixIndices(), matrixEntries());
    index[0] = ylunk;
    Albany::setLocalRowValues(jac, ylunk, index(), value());

    if (fillResid)
    {
      f_nonconstView[xlunk] = x_constView[xlunk] - Xval.val();
      f_nonconstView[ylunk] = x_constView[ylunk] - Yval.val();
    }
  }
}

// **********************************************************************
// Specialization: Tangent
// **********************************************************************
template<typename Traits>
TorsionBC<PHAL::AlbanyTraits::Tangent, Traits>::
TorsionBC(Teuchos::ParameterList& p) :
  TorsionBC_Base<PHAL::AlbanyTraits::Tangent, Traits>(p)
{
}
// **********************************************************************
template<typename Traits>
void TorsionBC<PHAL::AlbanyTraits::Tangent, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  Teuchos::RCP<const Thyra_Vector>       x = dirichletWorkset.x;
  Teuchos::RCP<const Thyra_MultiVector> Vx = dirichletWorkset.Vx;
  Teuchos::RCP<Thyra_Vector>       f = dirichletWorkset.f;
  Teuchos::RCP<Thyra_MultiVector> fp = dirichletWorkset.fp;
  Teuchos::RCP<Thyra_MultiVector> JV = dirichletWorkset.JV;

  Teuchos::ArrayRCP<const ST> x_constView = Albany::getLocalData(x);
  Teuchos::ArrayRCP<ST>       f_nonconstView;

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

  RealType time = dirichletWorkset.current_time;

  int xlunk, ylunk; // global and local indicies into unknown vector
  double* coord;
  ScalarT Xval, Yval;
  for (unsigned int inode = 0; inode < nsNodes.size(); inode++)
  {
    xlunk = nsNodes[inode][0];
    ylunk = nsNodes[inode][1];
    coord = nsNodeCoords[inode];

    this->computeBCs(coord, Xval, Yval, time);

    if (f != Teuchos::null)
    {
      f_nonconstView[xlunk] = x_constView[xlunk] - Xval.val();
      f_nonconstView[ylunk] = x_constView[ylunk] - Yval.val();
    }

    if (JV != Teuchos::null) {
      for (int i=0; i<dirichletWorkset.num_cols_x; i++)
      {
        JV_nonconst2dView[i][xlunk] = j_coeff*Vx_const2dView[i][xlunk];
        JV_nonconst2dView[i][ylunk] = j_coeff*Vx_const2dView[i][ylunk];
      }
    }

    if (fp != Teuchos::null) {
      for (int i=0; i<dirichletWorkset.num_cols_p; i++)
      {
        fp_nonconst2dView[i][xlunk] = -Xval.dx(dirichletWorkset.param_offset+i);
        fp_nonconst2dView[i][ylunk] = -Yval.dx(dirichletWorkset.param_offset+i);
      }
    }

  }
}

// **********************************************************************
// Specialization: DistParamDeriv
// **********************************************************************
template<typename Traits>
TorsionBC<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
TorsionBC(Teuchos::ParameterList& p) :
  TorsionBC_Base<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p)
{
}
// **********************************************************************
template<typename Traits>
void TorsionBC<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  Teuchos::RCP<Thyra_MultiVector> fpV = dirichletWorkset.fpV;
  bool trans = dirichletWorkset.transpose_dist_param_deriv;
  int num_cols = fpV->domain()->dim();

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

  int xlunk, ylunk; // global and local indicies into unknown vector
  // double* coord;
  // ScalarT Xval, Yval;

  if (trans) {
    // For (df/dp)^T*V we zero out corresponding entries in V
    Teuchos::RCP<Thyra_MultiVector> Vp = dirichletWorkset.Vp_bc;
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>> Vp_nonconst2dView = Albany::getNonconstLocalData(Vp);
    for (unsigned int inode = 0; inode < nsNodes.size(); inode++)
    {
      xlunk = nsNodes[inode][0];
      ylunk = nsNodes[inode][1];
      // coord = nsNodeCoords[inode];

      // this->computeBCs(coord, Xval, Yval, time);

      for (int col=0; col<num_cols; ++col) {
        //(*Vp)[col][xlunk] = 0.0;
        //(*Vp)[col][ylunk] = 0.0;
        Vp_nonconst2dView[col][xlunk] = 0.0;
        Vp_nonconst2dView[col][ylunk] = 0.0;
      }
    }
  } else {
    // for (df/dp)*V we zero out corresponding entries in df/dp
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>> fpV_nonconst2dView = Albany::getNonconstLocalData(fpV);
    for (unsigned int inode = 0; inode < nsNodes.size(); inode++)
    {
      xlunk = nsNodes[inode][0];
      ylunk = nsNodes[inode][1];
      // coord = nsNodeCoords[inode];

      // this->computeBCs(coord, Xval, Yval, time);

      for (int col=0; col<num_cols; ++col) {
        //(*fpV)[col][xlunk] = 0.0;
        //(*fpV)[col][ylunk] = 0.0;
        fpV_nonconst2dView[col][xlunk] = 0.0;
        fpV_nonconst2dView[col][ylunk] = 0.0;
      }
    }
  }
}

} // namespace LCM
