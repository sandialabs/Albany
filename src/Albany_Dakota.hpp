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


#ifndef ALBANY_DAKOTA_H
#define ALBANY_DAKOTA_H

#ifdef ALBANY_DAKOTA

#include "Albany_SolverFactory.hpp"

#include "TriKota_Driver.hpp"
#include "TriKota_DirectApplicInterface.hpp"

/** \brief Main routine to drive ModelEvaluator application with Dakota */
int Albany_Dakota();

#else // ALBANY_DAKOTA
int Albany_Dakota()
{
  std::cout << "\nDakota requested but not compiled in!\n" << std::endl;
  return 999;
}
#endif  // ALBANY_DAKOTA
#endif //ALBANY_DAKOTA_H
