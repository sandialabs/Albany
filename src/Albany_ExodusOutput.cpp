//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Albany_ExodusOutput.hpp"

#include "Albany_STKDiscretization.hpp"

#include "Petra_Converters.hpp"

#include "Teuchos_TimeMonitor.hpp"

namespace Albany {

ExodusOutput::ExodusOutput(const Teuchos::RCP<AbstractDiscretization> &disc) :
   stkDisc_(Teuchos::rcp_dynamic_cast<STKDiscretization>(disc)),
   exoOutTime_(Teuchos::TimeMonitor::getNewTimer("Albany: Output to Exodus"))
{
  // Nothing to oo
}

void ExodusOutput::writeSolution(double stamp, const Epetra_Vector &solution, const bool overlapped)
{
   Teuchos::TimeMonitor exoOutTimer(*exoOutTime_);
   stkDisc_->outputToExodus(solution, stamp, overlapped);
}

void ExodusOutput::writeSolutionT(double stamp, const Tpetra_Vector &solution, const bool overlapped)
{
  const Teuchos::RCP<const Epetra_Map> map_epetra =
    overlapped ? stkDisc_->getOverlapMap() : stkDisc_->getMap();

  Epetra_Vector solution_epetra(*map_epetra, false);
  Petra::TpetraVector_To_EpetraVector(
      Teuchos::rcpFromRef(solution),
      solution_epetra,
      Teuchos::rcpFromRef(map_epetra->Comm()));

  this->writeSolution(stamp, solution_epetra, overlapped);
}

}
