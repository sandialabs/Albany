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


#ifndef ALBANY_ANALYTICFUNCTION_HPP
#define ALBANY_ANALYTICFUNCTION_HPP

#include <string>

#include "Epetra_Vector.h"


namespace Albany {

void AnalyticFunction(Epetra_Vector& u, 
                       const unsigned int l,
                       const unsigned int h,
                       const double t, 
                       const double alpha, 
                       const double beta, 
                       const std::string name);

bool ValidIdentifier(const std::string name);

}

#endif
