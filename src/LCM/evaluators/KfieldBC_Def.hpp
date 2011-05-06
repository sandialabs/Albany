/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "Albany_AbstractDiscretization.hpp"

// **********************************************************************
// Genereric Template Code for Constructor and PostRegistrationSetup
// **********************************************************************

namespace LCM {

// **********************************************************************
// Specialization: Residual
// **********************************************************************
template<typename Traits>
KfieldBC<PHAL::AlbanyTraits::Residual, Traits>::
KfieldBC(Teuchos::ParameterList& p) :
  PHAL::DirichletBase<PHAL::AlbanyTraits::Residual, Traits>(p),
  mu(p.get<RealType>("Shear Modulus")),
  nu(p.get<RealType>("Poissons Ratio"))
{
  KIval  = p.get<RealType>("KI Value");
  KIIval = p.get<RealType>("KII Value");

  KI = KIval;
  KII = KIIval;

  std::string KI_name  = p.get< std::string >("Kfield KI Name");
  std::string KII_name = p.get< std::string >("Kfield KII Name");

  // Set up values as parameters for parameter library
  Teuchos::RCP<ParamLib> paramLib = p.get< Teuchos::RCP<ParamLib> >
    ("Parameter Library", Teuchos::null);
  
  new Sacado::ParameterRegistration<PHAL::AlbanyTraits::Residual, SPL_Traits> (KI_name, this, paramLib);
  new Sacado::ParameterRegistration<PHAL::AlbanyTraits::Residual, SPL_Traits> (KII_name, this, paramLib);
}
// **********************************************************************
template<typename Traits>
typename KfieldBC<PHAL::AlbanyTraits::Residual, Traits>::ScalarT&
KfieldBC<PHAL::AlbanyTraits::Residual, Traits>::getValue(const std::string &n)
{
  return KI;
}
// **********************************************************************
template<typename Traits>
void 
KfieldBC<PHAL::AlbanyTraits::Residual, Traits>::evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  Teuchos::RCP<Epetra_Vector> f = dirichletWorkset.f;
  Teuchos::RCP<const Epetra_Vector> x = dirichletWorkset.x;
  // Grab the vector off node GIDs for this Node Set ID from the std::map
  const std::vector<int>& nsNodes = dirichletWorkset.nodeSets->find(this->nodeSetID)->second;

  int xgunk, ygunk, xlunk, ylunk, clunk; // global and local indicies into unknown vector
  double* coord;
  RealType Xval, Yval, X, Y, R, theta, coeff_1, coeff_2;
  RealType tau = 6.283185307179586;
  RealType KI_X, KI_Y, KII_X, KII_Y;
  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
    xgunk = nsNodes[inode] * this->neq;
    ygunk = nsNodes[inode] * this->neq + 1;
    xlunk = f->Map().LID(xgunk);
    ylunk = f->Map().LID(ygunk);
    clunk =  3*(xlunk/this->neq);
    coord = &(dirichletWorkset.coordinates[clunk]);
    
    X = coord[0];
    Y = coord[1];
    R = std::sqrt(X*X + Y*Y);
    theta = std::atan2(Y,X);
    
    coeff_1 = ( KI / mu ) * std::sqrt( R / tau );
    coeff_2 = ( KII / mu ) * std::sqrt( R / tau );
    
    KI_X  = coeff_1 * ( 1.0 - 2.0 * nu + std::sin( theta / 2.0 ) * std::sin( theta / 2.0 ) ) * std::cos( theta / 2.0 );  
    KI_Y  = coeff_1 * ( 2.0 - 2.0 * nu - std::cos( theta / 2.0 ) * std::cos( theta / 2.0 ) ) * std::sin( theta / 2.0 );  
  
    KII_X = coeff_2 * ( 2.0 - 2.0 * nu + std::cos( theta / 2.0 ) * std::cos( theta / 2.0 ) ) * std::sin( theta / 2.0 );  
    KII_Y = coeff_2 * (-1.0 + 2.0 * nu + std::sin( theta / 2.0 ) * std::sin( theta / 2.0 ) ) * std::cos( theta / 2.0 );  
    
    Xval = KI_X + KII_X;
    Yval = KI_Y + KII_Y;
    
    (*f)[xlunk] = ((*x)[xlunk] - Xval);
    (*f)[ylunk] = ((*x)[ylunk] - Yval);

//  JTO: I am going to leave this here for now...
//     std::cout << "================" << std::endl;
//     std::cout.precision(15);
//     std::cout << "X : " << X << ", Y: " << Y << ", R: " << R << std::endl;
//     std::cout << "KI : " << KI << ", KII: " << KII << std::endl;
//     std::cout << "theta: " << theta << std::endl;
//     std::cout << "coeff_1: " << coeff_1 << ", coeff_2: " << coeff_2 << std::endl;
//     std::cout << "KI_X: " << KI_X << ", KI_Y: " << KI_Y << std::endl;
//     std::cout << "Xval: " << Xval << ", Yval: " << Yval << std::endl;
//     std::cout << "dx: " << (*x)[xlunk] << std::endl;
//     std::cout << "dy: " << (*x)[ylunk] << std::endl;
//     std::cout << "fx: " << ((*x)[xlunk] - Xval) << std::endl;
//     std::cout << "fy: " << ((*x)[ylunk] - Yval) << std::endl;
//     std::cout << "sin(theta/2): " << std::sin( theta / 2.0 ) << std::endl;
//     std::cout << "cos(theta/2): " << std::cos( theta / 2.0 ) << std::endl;
  }
}

// **********************************************************************
// Specialization: Jacobian
// **********************************************************************
template<typename Traits>
KfieldBC<PHAL::AlbanyTraits::Jacobian, Traits>::
KfieldBC(Teuchos::ParameterList& p) :
  PHAL::DirichletBase<PHAL::AlbanyTraits::Jacobian, Traits>(p),
  mu(p.get<RealType>("Shear Modulus")),
  nu(p.get<RealType>("Poissons Ratio"))
{
  KIval  = p.get<RealType>("KI Value");
  KIIval = p.get<RealType>("KII Value");

  KI = KIval;
  KII = KIIval;

  std::string KI_name  = p.get< std::string >("Kfield KI Name");
  std::string KII_name = p.get< std::string >("Kfield KII Name");

  // Set up values as parameters for parameter library
  Teuchos::RCP<ParamLib> paramLib = p.get< Teuchos::RCP<ParamLib> >
    ("Parameter Library", Teuchos::null);
  
  new Sacado::ParameterRegistration<PHAL::AlbanyTraits::Jacobian, SPL_Traits> (KI_name, this, paramLib);
  new Sacado::ParameterRegistration<PHAL::AlbanyTraits::Jacobian, SPL_Traits> (KII_name, this, paramLib);
}
// **********************************************************************
template<typename Traits>
typename KfieldBC<PHAL::AlbanyTraits::Jacobian, Traits>::ScalarT&
KfieldBC<PHAL::AlbanyTraits::Jacobian, Traits>::getValue(const std::string &n)
{
  return KI;
}
// **********************************************************************
template<typename Traits>
void KfieldBC<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{

  Teuchos::RCP<Epetra_Vector> f = dirichletWorkset.f;
  Teuchos::RCP<Epetra_CrsMatrix> jac = dirichletWorkset.Jac;
  Teuchos::RCP<const Epetra_Vector> x = dirichletWorkset.x;
  const RealType j_coeff = dirichletWorkset.j_coeff;
  const std::vector<int>& nsNodes = dirichletWorkset.nodeSets->find(this->nodeSetID)->second;

  const Epetra_Map& map = jac->RowMap();

  RealType* matrixEntries;
  int*    matrixIndices;
  int     numEntries;
  RealType diag=j_coeff;
  bool fillResid = (f != Teuchos::null);

  int xgunk, ygunk, xlunk, ylunk, clunk; // global and local indicies into unknown vector
  double* coord;
  RealType Xval, Yval; 
  RealType X, Y, R, theta, coeff_1, coeff_2;
  RealType tau = 6.283185307179586;
  RealType KI_X, KI_Y, KII_X, KII_Y;
  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) 
  {
    //const unsigned nodeid = nsNodes[inode];
    xgunk = nsNodes[inode] * this->neq;
    ygunk = nsNodes[inode] * this->neq + 1;
    xlunk = map.LID(xgunk);
    ylunk = map.LID(ygunk);
    clunk =  3*(xlunk/this->neq);
    coord = &(dirichletWorkset.coordinates[clunk]);
    
    X = coord[0];
    Y = coord[1];
    R = std::sqrt(X*X + Y*Y);
    theta = std::atan2(Y,X);
    
    coeff_1 = ( KI.val() / mu ) * std::sqrt( R / tau );
    coeff_2 = ( KII.val() / mu ) * std::sqrt( R / tau );
    
    KI_X  = coeff_1 * ( 1.0 - 2.0 * nu + std::sin( theta / 2.0 ) * std::sin( theta / 2.0 ) ) * std::cos( theta / 2.0 );  
    KI_Y  = coeff_1 * ( 2.0 - 2.0 * nu - std::cos( theta / 2.0 ) * std::cos( theta / 2.0 ) ) * std::sin( theta / 2.0 );  
    
    KII_X = coeff_2 * ( 2.0 - 2.0 * nu + std::cos( theta / 2.0 ) * std::cos( theta / 2.0 ) ) * std::sin( theta / 2.0 );  
    KII_Y = coeff_2 * (-1.0 + 2.0 * nu + std::sin( theta / 2.0 ) * std::sin( theta / 2.0 ) ) * std::cos( theta / 2.0 );  
    
    Xval = KI_X + KII_X;
    Yval = KI_Y + KII_Y;
    
    // replace jac values for the X dof 
    jac->ExtractMyRowView(xlunk, numEntries, matrixEntries, matrixIndices);
    for (int i=0; i<numEntries; i++) matrixEntries[i]=0;
    jac->ReplaceMyValues(xlunk, 1, &diag, &xlunk);

    // replace jac values for the y dof
    jac->ExtractMyRowView(ylunk, numEntries, matrixEntries, matrixIndices);
    for (int i=0; i<numEntries; i++) matrixEntries[i]=0;
    jac->ReplaceMyValues(ylunk, 1, &diag, &ylunk);
    
    if (fillResid)
    {
      (*f)[xlunk] = ((*x)[xlunk] - Xval);
      (*f)[ylunk] = ((*x)[ylunk] - Yval);
    } 
  }
}

// **********************************************************************
// Specialization: Tangent
// **********************************************************************
template<typename Traits>
KfieldBC<PHAL::AlbanyTraits::Tangent, Traits>::
KfieldBC(Teuchos::ParameterList& p) :
  PHAL::DirichletBase<PHAL::AlbanyTraits::Tangent, Traits>(p),
  mu(p.get<RealType>("Shear Modulus")),
  nu(p.get<RealType>("Poissons Ratio"))
{
  KIval  = p.get<RealType>("KI Value");
  KIIval = p.get<RealType>("KII Value");

  KI = KIval;
  KII = KIIval;

  std::string KI_name  = p.get< std::string >("Kfield KI Name");
  std::string KII_name = p.get< std::string >("Kfield KII Name");

  // Set up values as parameters for parameter library
  Teuchos::RCP<ParamLib> paramLib = p.get< Teuchos::RCP<ParamLib> >
    ("Parameter Library", Teuchos::null);
  
  new Sacado::ParameterRegistration<PHAL::AlbanyTraits::Tangent, SPL_Traits> (KI_name, this, paramLib);
  new Sacado::ParameterRegistration<PHAL::AlbanyTraits::Tangent, SPL_Traits> (KII_name, this, paramLib);
}

// **********************************************************************
template<typename Traits>
typename KfieldBC<PHAL::AlbanyTraits::Tangent, Traits>::ScalarT&
KfieldBC<PHAL::AlbanyTraits::Tangent, Traits>::getValue(const std::string &n)
{
  return KI;
}
// **********************************************************************
template<typename Traits>
void KfieldBC<PHAL::AlbanyTraits::Tangent, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  //TEST_FOR_EXCEPTION(true, std::runtime_error, "In Kfield Tangent, not supported yet.");
  Teuchos::RCP<Epetra_Vector> f = dirichletWorkset.f;
  Teuchos::RCP<Epetra_MultiVector> fp = dirichletWorkset.fp;
  Teuchos::RCP<Epetra_MultiVector> JV = dirichletWorkset.JV;
  Teuchos::RCP<const Epetra_Vector> x = dirichletWorkset.x;
  Teuchos::RCP<const Epetra_MultiVector> Vx = dirichletWorkset.Vx;
  const RealType j_coeff = dirichletWorkset.j_coeff;
  const std::vector<int>& nsNodes = dirichletWorkset.nodeSets->find(this->nodeSetID)->second;

  const Epetra_BlockMap& map = x->Map();
  bool fillResid = (f != Teuchos::null);

  int xgunk, ygunk, xlunk, ylunk, clunk; // global and local indicies into unknown vector
  double* coord;
  RealType Xval, Yval, X, Y, R, theta, coeff_1, coeff_2;
  RealType tau = 6.283185307179586;
  RealType KI_X, KI_Y, KII_X, KII_Y;
  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) 
  {
    //const unsigned nodeid = nsNodes[inode];
    xgunk = nsNodes[inode] * this->neq;
    ygunk = nsNodes[inode] * this->neq + 1;
    xlunk = map.LID(xgunk);
    ylunk = map.LID(ygunk);
    clunk =  3*(xlunk/this->neq);
    coord = &(dirichletWorkset.coordinates[clunk]);
    
    X = coord[0];
    Y = coord[1];
    R = std::sqrt(X*X + Y*Y);
    theta = std::atan2(Y,X);
    
    coeff_1 = ( KI.val() / mu ) * std::sqrt( R / tau );
    coeff_2 = ( KII.val() / mu ) * std::sqrt( R / tau );
    
    KI_X  = coeff_1 * ( 1.0 - 2.0 * nu + std::sin( theta / 2.0 ) * std::sin( theta / 2.0 ) ) * std::cos( theta / 2.0 );  
    KI_Y  = coeff_1 * ( 2.0 - 2.0 * nu - std::cos( theta / 2.0 ) * std::cos( theta / 2.0 ) ) * std::sin( theta / 2.0 );  
    
    KII_X = coeff_2 * ( 2.0 - 2.0 * nu + std::cos( theta / 2.0 ) * std::cos( theta / 2.0 ) ) * std::sin( theta / 2.0 );  
    KII_Y = coeff_2 * (-1.0 + 2.0 * nu + std::sin( theta / 2.0 ) * std::sin( theta / 2.0 ) ) * std::cos( theta / 2.0 );  
    
    Xval = KI_X + KII_X;
    Yval = KI_Y + KII_Y;

    if (fillResid)
    {
      (*f)[xlunk] = ((*x)[xlunk] - Xval);
      (*f)[ylunk] = ((*x)[ylunk] - Yval);
    } 

    for (int i=0; i<dirichletWorkset.num_cols_x; i++)
    {
      (*JV)[i][xlunk] = j_coeff*(*Vx)[i][xlunk];
      (*JV)[i][ylunk] = j_coeff*(*Vx)[i][ylunk];
    }
    
    for (int i=0; i<dirichletWorkset.num_cols_p; i++)
    {
      (*fp)[i][xlunk] = -this->value.dx(dirichletWorkset.param_offset+i);
      (*fp)[i][ylunk] = -this->value.dx(dirichletWorkset.param_offset+i);
    }

  }
}

// **********************************************************************
// Specialization: Stochastic Galerkin Residual
// **********************************************************************
template<typename Traits>
KfieldBC<PHAL::AlbanyTraits::SGResidual, Traits>::
KfieldBC(Teuchos::ParameterList& p) :
  PHAL::DirichletBase<PHAL::AlbanyTraits::SGResidual, Traits>(p),
  mu(p.get<RealType>("Shear Modulus")),
  nu(p.get<RealType>("Poissons Ratio"))
{
  KIval  = p.get<RealType>("KI Value");
  KIIval = p.get<RealType>("KII Value");

  KI = KIval;
  KII = KIIval;

  std::string KI_name  = p.get< std::string >("Kfield KI Name");
  std::string KII_name = p.get< std::string >("Kfield KII Name");

  // Set up values as parameters for parameter library
  Teuchos::RCP<ParamLib> paramLib = p.get< Teuchos::RCP<ParamLib> >
    ("Parameter Library", Teuchos::null);
  
  new Sacado::ParameterRegistration<PHAL::AlbanyTraits::SGResidual, SPL_Traits> (KI_name, this, paramLib);
  new Sacado::ParameterRegistration<PHAL::AlbanyTraits::SGResidual, SPL_Traits> (KII_name, this, paramLib);
}
// **********************************************************************
template<typename Traits>
typename KfieldBC<PHAL::AlbanyTraits::SGResidual, Traits>::ScalarT&
KfieldBC<PHAL::AlbanyTraits::SGResidual, Traits>::getValue(const std::string &n)
{
  return KI;
}
// **********************************************************************
template<typename Traits>
void KfieldBC<PHAL::AlbanyTraits::SGResidual, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  TEST_FOR_EXCEPTION(true, std::runtime_error, "In Kfield SGResidual, not supported yet.");
//   Teuchos::RCP< Stokhos::VectorOrthogPoly<Epetra_Vector> > f = 
//     dirichletWorkset.sg_f;
//   Teuchos::RCP< const Stokhos::VectorOrthogPoly<Epetra_Vector> > x = 
//     dirichletWorkset.sg_x;
//   const std::vector<int>& nsNodes = dirichletWorkset.nodeSets->find(this->nodeSetID)->second;

//   int nblock = x->size();
//   int gunk, lunk; // global and local indicies into unknown vector
//   for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
//       gunk = nsNodes[inode] * this->neq + this->offset;
//       lunk = (*f)[0].Map().LID(gunk);
//       for (int block=0; block<nblock; block++)
// 	(*f)[block][lunk] = ((*x)[block][lunk] - this->value.coeff(block));
//   }
}

// **********************************************************************
// Specialization: Stochastic Galerkin Jacobian
// **********************************************************************
template<typename Traits>
KfieldBC<PHAL::AlbanyTraits::SGJacobian, Traits>::
KfieldBC(Teuchos::ParameterList& p) :
  PHAL::DirichletBase<PHAL::AlbanyTraits::SGJacobian, Traits>(p),
  mu(p.get<RealType>("Shear Modulus")),
  nu(p.get<RealType>("Poissons Ratio"))
{
  KIval  = p.get<RealType>("KI Value");
  KIIval = p.get<RealType>("KII Value");

  KI = KIval;
  KII = KIIval;

  std::string KI_name  = p.get< std::string >("Kfield KI Name");
  std::string KII_name = p.get< std::string >("Kfield KII Name");

  // Set up values as parameters for parameter library
  Teuchos::RCP<ParamLib> paramLib = p.get< Teuchos::RCP<ParamLib> >
    ("Parameter Library", Teuchos::null);
  
  new Sacado::ParameterRegistration<PHAL::AlbanyTraits::SGJacobian, SPL_Traits> (KI_name, this, paramLib);
  new Sacado::ParameterRegistration<PHAL::AlbanyTraits::SGJacobian, SPL_Traits> (KII_name, this, paramLib);
}
// **********************************************************************
template<typename Traits>
typename KfieldBC<PHAL::AlbanyTraits::SGJacobian, Traits>::ScalarT&
KfieldBC<PHAL::AlbanyTraits::SGJacobian, Traits>::getValue(const std::string &n)
{
  return KI;
}
// **********************************************************************
template<typename Traits>
void KfieldBC<PHAL::AlbanyTraits::SGJacobian, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  TEST_FOR_EXCEPTION(true, std::runtime_error, "In Kfield SGJacobian, not supported yet.");
//   Teuchos::RCP< Stokhos::VectorOrthogPoly<Epetra_Vector> > f = 
//     dirichletWorkset.sg_f;
//   Teuchos::RCP< Stokhos::VectorOrthogPoly<Epetra_CrsMatrix> > jac = 
//     dirichletWorkset.sg_Jac;
//   Teuchos::RCP<const Stokhos::VectorOrthogPoly<Epetra_Vector> > x = 
//     dirichletWorkset.sg_x;
//   const RealType j_coeff = dirichletWorkset.j_coeff;
//   const std::vector<int>& nsNodes = dirichletWorkset.nodeSets->find(this->nodeSetID)->second;

//   const Epetra_Map& map = (*jac)[0].RowMap();

//   RealType* matrixEntries;
//   int*    matrixIndices;
//   int     numEntries;
//   int nblock = 0;
//   if (f != Teuchos::null)
//     nblock = f->size();
//   int nblock_jac = jac->size();
//   RealType diag=j_coeff;
//   bool fillResid = (f != Teuchos::null);

//   for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
//     const unsigned nodeid = nsNodes[inode];
//       const int gunk = nodeid * this->neq + this->offset;
//       int lunk = map.LID(gunk);
//       for (int block=0; block<nblock_jac; block++) {
// 	(*jac)[block].ExtractMyRowView(lunk, numEntries, matrixEntries, 
// 				       matrixIndices);
// 	for (int i=0; i<numEntries; i++) 
// 	  matrixEntries[i]=0;
//       }
//       (*jac)[0].ReplaceMyValues(lunk, 1, &diag, &lunk);
//       if (fillResid) {
// 	for (int block=0; block<nblock; block++)
// 	  (*f)[block][lunk] = 
// 	    (*x)[block][lunk] - this->value.val().coeff(block);
//       }
//   }
}

// **********************************************************************
// Specialization: Multi-point Residual
// **********************************************************************
template<typename Traits>
KfieldBC<PHAL::AlbanyTraits::MPResidual, Traits>::
KfieldBC(Teuchos::ParameterList& p) :
  PHAL::DirichletBase<PHAL::AlbanyTraits::MPResidual, Traits>(p),
  mu(p.get<RealType>("Shear Modulus")),
  nu(p.get<RealType>("Poissons Ratio"))
{
  KIval  = p.get<RealType>("KI Value");
  KIIval = p.get<RealType>("KII Value");

  KI = KIval;
  KII = KIIval;

  std::string KI_name  = p.get< std::string >("Kfield KI Name");
  std::string KII_name = p.get< std::string >("Kfield KII Name");

  // Set up values as parameters for parameter library
  Teuchos::RCP<ParamLib> paramLib = p.get< Teuchos::RCP<ParamLib> >
    ("Parameter Library", Teuchos::null);
  
  new Sacado::ParameterRegistration<PHAL::AlbanyTraits::MPResidual, SPL_Traits> (KI_name, this, paramLib);
  new Sacado::ParameterRegistration<PHAL::AlbanyTraits::MPResidual, SPL_Traits> (KII_name, this, paramLib);
}
// **********************************************************************
template<typename Traits>
typename KfieldBC<PHAL::AlbanyTraits::MPResidual, Traits>::ScalarT&
KfieldBC<PHAL::AlbanyTraits::MPResidual, Traits>::getValue(const std::string &n)
{
  return KI;
}
// **********************************************************************
template<typename Traits>
void KfieldBC<PHAL::AlbanyTraits::MPResidual, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  TEST_FOR_EXCEPTION(true, std::runtime_error, "In Kfield MPResidual, not supported yet.");
//   Teuchos::RCP< Stokhos::ProductContainer<Epetra_Vector> > f = 
//     dirichletWorkset.mp_f;
//   Teuchos::RCP< const Stokhos::ProductContainer<Epetra_Vector> > x = 
//     dirichletWorkset.mp_x;
//   const std::vector<int>& nsNodes = dirichletWorkset.nodeSets->find(this->nodeSetID)->second;

//   int nblock = x->size();
//   int gunk, lunk; // global and local indicies into unknown vector
//   for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
//       gunk = nsNodes[inode] * this->neq + this->offset;
//       lunk = (*f)[0].Map().LID(gunk);
//       for (int block=0; block<nblock; block++)
// 	(*f)[block][lunk] = ((*x)[block][lunk] - this->value.coeff(block));
//   }
}

// **********************************************************************
// Specialization: Multi-point Jacobian
// **********************************************************************
template<typename Traits>
KfieldBC<PHAL::AlbanyTraits::MPJacobian, Traits>::
KfieldBC(Teuchos::ParameterList& p) :
  PHAL::DirichletBase<PHAL::AlbanyTraits::MPJacobian, Traits>(p),
  mu(p.get<RealType>("Shear Modulus")),
  nu(p.get<RealType>("Poissons Ratio"))
{
  KIval  = p.get<RealType>("KI Value");
  KIIval = p.get<RealType>("KII Value");

  KI = KIval;
  KII = KIIval;

  std::string KI_name  = p.get< std::string >("Kfield KI Name");
  std::string KII_name = p.get< std::string >("Kfield KII Name");

  // Set up values as parameters for parameter library
  Teuchos::RCP<ParamLib> paramLib = p.get< Teuchos::RCP<ParamLib> >
    ("Parameter Library", Teuchos::null);
  
  new Sacado::ParameterRegistration<PHAL::AlbanyTraits::MPJacobian, SPL_Traits> (KI_name, this, paramLib);
  new Sacado::ParameterRegistration<PHAL::AlbanyTraits::MPJacobian, SPL_Traits> (KII_name, this, paramLib);
}

// **********************************************************************
template<typename Traits>
typename KfieldBC<PHAL::AlbanyTraits::MPJacobian, Traits>::ScalarT&
KfieldBC<PHAL::AlbanyTraits::MPJacobian, Traits>::getValue(const std::string &n)
{
  return KI;
}
// **********************************************************************
template<typename Traits>
void KfieldBC<PHAL::AlbanyTraits::MPJacobian, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  TEST_FOR_EXCEPTION(true, std::runtime_error, "In Kfield MPJacobian, not supported yet.");

//   Teuchos::RCP< Stokhos::ProductContainer<Epetra_Vector> > f = 
//     dirichletWorkset.mp_f;
//   Teuchos::RCP< Stokhos::ProductContainer<Epetra_CrsMatrix> > jac = 
//     dirichletWorkset.mp_Jac;
//   Teuchos::RCP<const Stokhos::ProductContainer<Epetra_Vector> > x = 
//     dirichletWorkset.mp_x;
//   const RealType j_coeff = dirichletWorkset.j_coeff;
//   const std::vector<int>& nsNodes = dirichletWorkset.nodeSets->find(this->nodeSetID)->second;

//   const Epetra_Map& map = (*jac)[0].RowMap();

//   RealType* matrixEntries;
//   int*    matrixIndices;
//   int     numEntries;
//   int nblock = 0;
//   if (f != Teuchos::null)
//     nblock = f->size();
//   int nblock_jac = jac->size();
//   RealType diag=j_coeff;
//   bool fillResid = (f != Teuchos::null);

//   for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
//     const unsigned nodeid = nsNodes[inode];
//       const int gunk = nodeid * this->neq + this->offset;
//       int lunk = map.LID(gunk);
//       for (int block=0; block<nblock_jac; block++) {
// 	(*jac)[block].ExtractMyRowView(lunk, numEntries, matrixEntries, 
// 				       matrixIndices);
// 	for (int i=0; i<numEntries; i++) 
// 	  matrixEntries[i]=0;
// 	(*jac)[block].ReplaceMyValues(lunk, 1, &diag, &lunk);
//       }
//       if (fillResid) {
// 	for (int block=0; block<nblock; block++)
// 	  (*f)[block][lunk] = 
// 	    (*x)[block][lunk] - this->value.val().coeff(block);
//       }
//   }
}

} // namespace LCM

