//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Schwarz_CoupledJacobian.hpp"
#include "Teuchos_ParameterListExceptions.hpp"
#include "Teuchos_TestForException.hpp"
//#include "Tpetra_LocalMap.h"

LCM::Schwarz_CoupledJacobian::Schwarz_CoupledJacobian(Teuchos::Array<Teuchos::RCP<const Tpetra_Map> > disc_maps, 
					   Teuchos::RCP<const Tpetra_Map> coupled_disc_map, 
					   const Teuchos::RCP<const Teuchos_Comm>& commT)
{
  n_models_ = disc_maps.size();
  disc_maps_.resize(n_models_); 
  for (int m=0; m<n_models_; m++)
    disc_maps_[m] = Teuchos::rcp(new Tpetra_Map(*disc_maps[m]));  
  domain_map_ = range_map_ = coupled_disc_map;
  commT_ = commT;
  b_use_transpose_ = false;
  b_initialized_ = false;
}

LCM::Schwarz_CoupledJacobian::~Schwarz_CoupledJacobian()
{
}


//! Initialize the operator with everything needed to apply it
void LCM::Schwarz_CoupledJacobian::initialize(Teuchos::Array<Teuchos::RCP<Tpetra_CrsMatrix> > jacs) 
{
  // Set member variables
  /*jacs_.resize(n_models_); 
  for (int m=0; m<n_models_; m++)
    jacs_[m] = Teuchos::rcp(new Tpetra_CrsMatrix(*jacs[m]));  */
}


//! Returns the result of a Tpetra_Operator applied to a Tpetra_MultiVector X in Y.
void LCM::Schwarz_CoupledJacobian::apply(const Tpetra_MultiVector& X, Tpetra_MultiVector& Y, 
                                       Teuchos::ETransp mode,
                                       ST alpha, ST beta) const
{ 
  std::cout << "In LCM::Schwarz_CoupledJacobian::Apply! \n" << std::endl; 

  //FIXME: fill in!
    // Jacobian Matrix is:
    //
    //                   Phi                    Psi[i]                            -Eval[i]
    //          | ------------------------------------------------------------------------------------------|
    //          |                      |                             |                                      |
    // Poisson  |    Jac_poisson       |   M*diag(dn/d{Psi[i](x)})   |        -M*col(dn/dEval[i])           |
    //          |                      |                             |                                      |
    //          | ------------------------------------------------------------------------------------------|
    //          |                      |                             |                                      |
    // Schro[j] |  M*diag(-Psi[j](x))  | delta(i,j)*[ H-Eval[i]*M ]  |        delta(i,j)*M*Psi[i](x)        |    
    //          |                      |                             |                                      |
    //          | ------------------------------------------------------------------------------------------|
    //          |                      |                             |                                      |
    // Norm[j]  |    0                 | -delta(i,j)*(M+M^T)*Psi[i]  |                   0                  |
    //          |                      |                             |                                      |
    //          | ------------------------------------------------------------------------------------------|
    //
    //
    //   Where:
    //       n = quantum density function which depends on dimension
    
    // Scratch:  val = sum_ij x_i * M_ij * x_j
    //         dval/dx_k = sum_j!=k M_kj * x_j + sum_i!=k x_i * M_ik + 2* M_kk * x_k
    //                   = sum_i (M_ki + M_ik) * x_i
    //                   = sum_i (M + M^T)_ki * x_i  == k-th el of (M + M^T)*x
    //   So d(x*M*x)/dx = (M+M^T)*x in matrix form

    // Do multiplication block-wise
  
}


