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


#ifndef ALBANY_INITIALCONDITION_HPP
#define ALBANY_INITIALCONDITION_HPP

#include <string>
#include "Teuchos_ParameterList.hpp"
#include "Epetra_Vector.h"

namespace Albany {

void InitialCondition(const Teuchos::RCP<Epetra_Vector>& u,
                      const unsigned int local_len_x,
                      const unsigned int local_len_y,
                      Teuchos::ParameterList& p);

void InitialCondition(const Teuchos::RCP<Epetra_Vector>& u,
                      const unsigned int global_len_x,
                      const unsigned int global_len_y,
                      const double alpha, 
                      const double beta, 
                      const std::string &name);
}
#endif
