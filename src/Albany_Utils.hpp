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
  #include "Epetra_MpiComm.h"
  #include "Teuchos_DefaultMpiComm.hpp"
#else
  typedef int MPI_Comm;
  #define MPI_COMM_WORLD 1
  #include "Epetra_SerialComm.h"
  #include "Teuchos_DefaultSerialComm.hpp"
#endif
#include "Teuchos_RCP.hpp"

namespace Albany {

  const MPI_Comm getMpiCommFromEpetraComm(const Epetra_Comm& ec);

  MPI_Comm getMpiCommFromEpetraComm(Epetra_Comm& ec);

  Teuchos::RCP<Epetra_Comm> createEpetraCommFromMpiComm(const MPI_Comm& mc);
  Teuchos::RCP<Teuchos::Comm<int> > createTeuchosCommFromMpiComm(const MPI_Comm& mc);
}
#endif //ALBANY_UTILS
