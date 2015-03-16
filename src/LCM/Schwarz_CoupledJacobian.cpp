//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Schwarz_CoupledJacobian.hpp"
#include "Teuchos_ParameterListExceptions.hpp"
#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Albany_Utils.hpp"
//#include "Tpetra_LocalMap.h"

//#define WRITE_TO_MATRIX_MARKET

static int c3 = 0; 
static int c4 = 0; 

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
  char name[100];  //create string for file name
  sprintf(name, "Jac0_%i.mm", c3);
//write individual model jacobians to matrix market for debug
  Tpetra_MatrixMarket_Writer::writeSparseFile(name, jacs[0]);
  c3++; 
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
  char name[100];  //create string for file name
  sprintf(name, "X_%i.mm", c4);
  Tpetra_MatrixMarket_Writer::writeDenseFile(name, X);
#endif

  //FIXME: fill in!
    // Jacobian Matrix is (for e.g., 3 domain coupling):
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
    //
   /* Albany::printTpetraVector(*out << "\nX:\n", X.getVector(0));
    std::cout << "Jacobian:" << std::endl;
    jacs_[0]->describe(*out, Teuchos::VERB_HIGH);
    Albany::printTpetraVector(*out << "\nY:\n", Y.getVector(0));
    */
    if (n_models_ == 1) {
      jacs_[0]->apply(X, Y); 
    }
    else 
      std::cout << "WARNING: LCM::Schwarz_CoupledJacbian::apply() method only implemented for 1 model right now! \n"; 
      
  
#ifdef WRITE_TO_MATRIX_MARKET
  //writing to MatrixMarket file for debug -- final solution Y (after all the operations to set Y = Jac*X
  sprintf(name, "Y_%i.mm", c4);
  Tpetra_MatrixMarket_Writer::writeDenseFile(name, Y);
  c4++; 
#endif
}


