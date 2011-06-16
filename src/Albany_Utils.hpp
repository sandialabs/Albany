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


#ifndef ALBANY_UTILS_H
#define ALBANY_UTILS_H

#ifdef ALBANY_MPI
  #define Albany_MPI_Comm MPI_Comm
  #define Albany_MPI_COMM_WORLD MPI_COMM_WORLD
  #define Albany_MPI_COMM_NULL MPI_COMM_NULL
  #include "Epetra_MpiComm.h"
  #include "Teuchos_DefaultMpiComm.hpp"
#else
  #define Albany_MPI_Comm int
  #define Albany_MPI_COMM_WORLD 0  // This is compatible with Dakota
  #define Albany_MPI_COMM_NULL 99
  #include "Epetra_SerialComm.h"
  #include "Teuchos_DefaultSerialComm.hpp"
#endif
#include "Teuchos_RCP.hpp"

namespace Albany {

  const Albany_MPI_Comm getMpiCommFromEpetraComm(const Epetra_Comm& ec);

  Albany_MPI_Comm getMpiCommFromEpetraComm(Epetra_Comm& ec);

  Teuchos::RCP<Epetra_Comm> createEpetraCommFromMpiComm(const Albany_MPI_Comm& mc);
  Teuchos::RCP<Teuchos::Comm<int> > createTeuchosCommFromMpiComm(const Albany_MPI_Comm& mc);

  //! Utility to make a string out of a string + int: strint("dog",2) = "dog 2"
  std::string strint(const std::string s, const int i);

  //! Returns true of the given string is a valid initialization string of the format "initial value 1.54"
  bool isValidInitString(const std::string& initString);

  //! Converts a double to an initialization string:  doubleToInitString(1.54) = "initial value 1.54"
  std::string doubleToInitString(double val);

  //! Converts an init string to a double:  initStringToDouble("initial value 1.54") = 1.54
  double initStringToDouble(const std::string& initString);
}
#endif //ALBANY_UTILS
