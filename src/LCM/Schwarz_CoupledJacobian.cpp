//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Schwarz_CoupledJacobian.hpp"
#include "Teuchos_ParameterListExceptions.hpp"
#include "Teuchos_TestForException.hpp"
//#include "Tpetra_LocalMap.h"

#define WRITE_TO_MATRIX_MARKET

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
  //FIXME: add parameter list argument, member parameters for specifying boundary conditions.
  //These can be stored in an array of Tpetra_CrsMatrices like the jacobians.
  // Set member variables
  jacs_.resize(n_models_); 
  for (int m=0; m<n_models_; m++)
    jacs_[m] = jacs[m];  

#ifdef WRITE_TO_MATRIX_MARKET
  std::cout << "In LCM::Schwarz_CoupledJacobian::initialize! \n"; 
//write individual model jacobians to matrix market for debug
  Tpetra_MatrixMarket_Writer::writeSparseFile("Jac0.mm", jacs[0]);
  if (n_models_ > 1) 
    Tpetra_MatrixMarket_Writer::writeSparseFile("Jac1.mm", jacs[1]);
#endif
}


//! Returns the result of a Tpetra_Operator applied to a Tpetra_MultiVector X in Y.
void LCM::Schwarz_CoupledJacobian::apply(const Tpetra_MultiVector& X, Tpetra_MultiVector& Y, 
                                       Teuchos::ETransp mode,
                                       ST alpha, ST beta) const
{ 
  std::cout << "In LCM::Schwarz_CoupledJacobian::apply! \n"; 

#ifdef WRITE_TO_MATRIX_MARKET
  //writing to MatrixMarket file for debug -- initial X where we will set Y = Jac*X
  Tpetra_MatrixMarket_Writer::writeDenseFile("X.mm", X);
#endif

  //FIXME: fill in!
    // Jacobian Matrix is (for e.g., 3 domaian coupling):
    //
    //                   x1                        x2                              x3           ....
    //          | ------------------------------------------------------------------------------------------|
    //          |                      |                             |                                      |
    //      x1  |        Jac1          |           ??                |             ??                       |
    //          |                      |                             |                                      |
    //          | ------------------------------------------------------------------------------------------|
    //          |                      |                             |                                      |
    //      x2  |         ??           |           Jac2              |             ??                       |    
    //          |                      |                             |                                      |
    //          | ------------------------------------------------------------------------------------------|
    //          |                      |                             |                                      |
    //      x3  |         ??           |           ??                |            Jac3                      |
    //          |                      |                             |                                      |
    //          | ------------------------------------------------------------------------------------------|
    //      :
    //      :
    
    // Do multiplication block-wise
  
#ifdef WRITE_TO_MATRIX_MARKET
  //writing to MatrixMarket file for debug -- final solution Y (after all the operations to set Y = Jac*X
  Tpetra_MatrixMarket_Writer::writeDenseFile("Y.mm", Y);
#endif
}


