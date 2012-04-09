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

#include "Albany_ExodusOutput.hpp"

#include "Albany_STKDiscretization.hpp"

#include "Teuchos_TimeMonitor.hpp"

namespace Albany {

ExodusOutput::ExodusOutput(const Teuchos::RCP<AbstractDiscretization> &disc) :
   stkDisc_(Teuchos::rcp_dynamic_cast<STKDiscretization>(disc)),
   exoOutTime_(Teuchos::TimeMonitor::getNewTimer("Albany: Output to Exodus"))
{
  // Nothing to oo
}

void ExodusOutput::writeSolution(double stamp, const Epetra_Vector &solution)
{
   Teuchos::TimeMonitor exoOutTimer(*exoOutTime_);
   stkDisc_->outputToExodus(solution, stamp);
}

}
