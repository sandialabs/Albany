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
KfieldBC_Base<EvalT, Traits>::
KfieldBC_Base(Teuchos::ParameterList& p) :
offset(p.get<int>("Equation Offset")),
  PHAL::DirichletBase<EvalT, Traits>(p),
  mu(p.get<RealType>("Shear Modulus")),
  nu(p.get<RealType>("Poissons Ratio"))
{
  KIval  = p.get<RealType>("KI Value");
  KIIval = p.get<RealType>("KII Value");

  KI  =  KIval;
  KII = KIIval;

  KI_name  = p.get< std::string >("Kfield KI Name");
  KII_name = p.get< std::string >("Kfield KII Name");

  // Set up values as parameters for parameter library
  Teuchos::RCP<ParamLib> paramLib = p.get< Teuchos::RCP<ParamLib>>
    ("Parameter Library", Teuchos::null);

  this->registerSacadoParameter(KI_name, paramLib);
  this->registerSacadoParameter(KII_name, paramLib);

  timeValues = p.get<Teuchos::Array<RealType>>("Time Values").toVector();
  KIValues = p.get<Teuchos::Array<RealType>>("KI Values").toVector();
  KIIValues = p.get<Teuchos::Array<RealType>>("KII Values").toVector();


  TEUCHOS_TEST_FOR_EXCEPTION( !(timeValues.size() == KIValues.size()),
                              Teuchos::Exceptions::InvalidParameter,
                              "Dimension of \"Time Values\" and \"KI Values\" do not match" );

  TEUCHOS_TEST_FOR_EXCEPTION( !(timeValues.size() == KIIValues.size()),
                              Teuchos::Exceptions::InvalidParameter,
                              "Dimension of \"Time Values\" and \"KII Values\" do not match" );



}

// **********************************************************************
template<typename EvalT, typename Traits>
typename KfieldBC_Base<EvalT, Traits>::ScalarT&
KfieldBC_Base<EvalT, Traits>::
getValue(const std::string &n)
{
  if (n == KI_name)
    return KI;
 // else if (n== timeValues)
//        return timeValues;
  else
        return KII;
}

// **********************************************************************
template<typename EvalT, typename Traits>
void
KfieldBC_Base<EvalT, Traits>::
computeBCs(double* coord, ScalarT& Xval, ScalarT& Yval, RealType time)
{

  TEUCHOS_TEST_FOR_EXCEPTION( time > timeValues.back(),
                                      Teuchos::Exceptions::InvalidParameter,
                                      "Time is growing unbounded!" );

  RealType X, Y, R, theta;
  ScalarT coeff_1, coeff_2;
  RealType tau = 6.283185307179586;
  ScalarT KI_X, KI_Y, KII_X, KII_Y;

  X = coord[0];
  Y = coord[1];
  R = std::sqrt(X*X + Y*Y);
  theta = std::atan2(Y,X);

  ScalarT KIFunctionVal, KIIFunctionVal;
  RealType KIslope, KIIslope;
  unsigned int Index(0);

  while( timeValues[Index] < time )
    Index++;

  if (Index == 0) {
    KIFunctionVal  = KIValues[Index];
    KIIFunctionVal = KIIValues[Index];
  }
  else
  {
    KIslope = ( KIValues[Index] - KIValues[Index - 1] ) / ( timeValues[Index] - timeValues[Index - 1] );
    KIFunctionVal = KIValues[Index-1] + KIslope * ( time - timeValues[Index - 1] );

    KIIslope = ( KIIValues[Index] - KIIValues[Index - 1] ) / ( timeValues[Index] - timeValues[Index - 1] );
    KIIFunctionVal = KIIValues[Index-1] + KIIslope * ( time - timeValues[Index - 1] );
  }

  coeff_1 = ( KI*KIFunctionVal / mu ) * std::sqrt( R / tau );
  coeff_2 = ( KII*KIIFunctionVal / mu ) * std::sqrt( R / tau );

  KI_X  = coeff_1 * ( 1.0 - 2.0 * nu + std::sin( theta / 2.0 ) * std::sin( theta / 2.0 ) ) * std::cos( theta / 2.0 );
  KI_Y  = coeff_1 * ( 2.0 - 2.0 * nu - std::cos( theta / 2.0 ) * std::cos( theta / 2.0 ) ) * std::sin( theta / 2.0 );

  KII_X = coeff_2 * ( 2.0 - 2.0 * nu + std::cos( theta / 2.0 ) * std::cos( theta / 2.0 ) ) * std::sin( theta / 2.0 );
  KII_Y = coeff_2 * (-1.0 + 2.0 * nu + std::sin( theta / 2.0 ) * std::sin( theta / 2.0 ) ) * std::cos( theta / 2.0 );

  Xval = KI_X + KII_X;
  Yval = KI_Y + KII_Y;

/*
//  JTO: I am going to leave this here for now...
     std::cout << "================" << std::endl;
     std::cout.precision(15);
     std::cout << "X : " << X << ", Y: " << Y << ", R: " << R << std::endl;
//     std::cout << "Node : " << nsNodes[inode] << std::endl;
     std::cout << "KI : " << KI << ", KII: " << KII << std::endl;
     std::cout << "theta: " << theta << std::endl;
     std::cout << "coeff_1: " << coeff_1 << ", coeff_2: " << coeff_2 << std::endl;
     std::cout << "KI_X: " << KI_X << ", KI_Y: " << KI_Y << std::endl;
     std::cout << "Xval: " << Xval << ", Yval: " << Yval << std::endl;
     std::cout << "nu: " << nu << std::endl;
//     std::cout << "dx: " << (*x)[xlunk] << std::endl;
//     std::cout << "dy: " << (*x)[ylunk] << std::endl;
//     std::cout << "fx: " << ((*x)[xlunk] - Xval) << std::endl;
//     std::cout << "fy: " << ((*x)[ylunk] - Yval) << std::endl;
     std::cout << "sin(theta/2): " << std::sin( theta / 2.0 ) << std::endl;
     std::cout << "cos(theta/2): " << std::cos( theta / 2.0 ) << std::endl;

     */
}

// **********************************************************************
// Specialization: Residual
// **********************************************************************
template<typename Traits>
KfieldBC<PHAL::AlbanyTraits::Residual, Traits>::
KfieldBC(Teuchos::ParameterList& p) :
  KfieldBC_Base<PHAL::AlbanyTraits::Residual, Traits>(p)
{
}

// **********************************************************************
template<typename Traits>
void
KfieldBC<PHAL::AlbanyTraits::Residual, Traits>::
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
KfieldBC<PHAL::AlbanyTraits::Jacobian, Traits>::
KfieldBC(Teuchos::ParameterList& p) :
  KfieldBC_Base<PHAL::AlbanyTraits::Jacobian, Traits>(p)
{
}
// **********************************************************************
template<typename Traits>
void KfieldBC<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  Teuchos::RCP<const Thyra_Vector> x   = dirichletWorkset.x;
  Teuchos::RCP<Thyra_Vector>       f   = dirichletWorkset.f;
  Teuchos::RCP<Thyra_LinearOp>     jac = dirichletWorkset.Jac;

  Teuchos::ArrayRCP<const ST> x_constView = Albany::getLocalData(x);
  Teuchos::ArrayRCP<ST>       f_nonconstView;

  RealType time = dirichletWorkset.current_time;

  const RealType j_coeff = dirichletWorkset.j_coeff;
  const std::vector<std::vector<int>>& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;
  const std::vector<double*>& nsNodeCoords =
    dirichletWorkset.nodeSetCoords->find(this->nodeSetID)->second;

  bool fillResid = (f != Teuchos::null);
  if (fillResid) {
    f_nonconstView = Albany::getNonconstLocalData(f);
  }

  int xlunk, ylunk; // local indicies into unknown vector
  double* coord;

  ScalarT Xval, Yval;
  Teuchos::Array<LO> index(1);
  Teuchos::Array<ST> value(1);
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
KfieldBC<PHAL::AlbanyTraits::Tangent, Traits>::
KfieldBC(Teuchos::ParameterList& p) :
  KfieldBC_Base<PHAL::AlbanyTraits::Tangent, Traits>(p)
{
}
// **********************************************************************
template<typename Traits>
void KfieldBC<PHAL::AlbanyTraits::Tangent, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  Teuchos::RCP<const Thyra_Vector>      x  = dirichletWorkset.x;
  Teuchos::RCP<const Thyra_MultiVector> Vx = dirichletWorkset.Vx;
  Teuchos::RCP<Thyra_Vector>            f  = dirichletWorkset.f;
  Teuchos::RCP<Thyra_MultiVector>       fp = dirichletWorkset.fp;
  Teuchos::RCP<Thyra_MultiVector>       JV = dirichletWorkset.JV;

  Teuchos::ArrayRCP<const ST> x_constView = Albany::getLocalData(x);
  Teuchos::ArrayRCP<ST>       f_nonconstView;

  Teuchos::ArrayRCP<Teuchos::ArrayRCP<const ST>> Vx_const2dView;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>>       JV_nonconst2dView;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>>       fp_nonconst2dView;

  if (f != Teuchos::null) {
    f_nonconstView = Albany::getNonconstLocalData(f);
  }
  if (JV != Teuchos::null) {
    JV_nonconst2dView = Albany::getNonconstLocalData(JV);
    Vx_const2dView = Albany::getLocalData(Vx);
  }
  if (JV != Teuchos::null) {
    fp_nonconst2dView = Albany::getNonconstLocalData(fp);
  }

  RealType time = dirichletWorkset.current_time;

  const RealType j_coeff = dirichletWorkset.j_coeff;
  const std::vector<std::vector<int>>& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;
  const std::vector<double*>& nsNodeCoords =
    dirichletWorkset.nodeSetCoords->find(this->nodeSetID)->second;

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
KfieldBC<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
KfieldBC(Teuchos::ParameterList& p) :
  KfieldBC_Base<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p)
{
}
// **********************************************************************
template<typename Traits>
void KfieldBC<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
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

  // RealType time = dirichletWorkset.current_time;

  const std::vector<std::vector<int>>& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;
  const std::vector<double*>& nsNodeCoords =
    dirichletWorkset.nodeSetCoords->find(this->nodeSetID)->second;

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

