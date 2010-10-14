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


#include "Albany_NOXObserver.hpp"

#ifdef ALBANY_IOSS
  #include "Albany_STKDiscretization.hpp"
#endif

using namespace std;

Albany_NOXObserver::Albany_NOXObserver(
     const Teuchos::RCP<Albany_VTK> vtk_,
     const Teuchos::RCP<Albany::Application> &app_) : 
  vtk(vtk_),
  app(app_),
  disc(app_->getDiscretization())

{
   if (vtk != Teuchos::null) { vtk->updateGeometry (disc); }

   exooutTime = Teuchos::TimeMonitor::getNewTimer("Albany: Output to Exodus");
}

void Albany_NOXObserver::observeSolution(
    const Epetra_Vector& solution)
{
   if (vtk != Teuchos::null) {
     vtk->visualizeField (solution, disc);
   }

   app->getStateMgr().updateStates();;

#ifdef ALBANY_IOSS
  if (solution.Map().Comm().MyPID()==0)
    cout << "Albany::NOXObserver calling exodus output " << endl;

  Albany::STKDiscretization* stkDisc =
    dynamic_cast<Albany::STKDiscretization*>(disc.get());

  {
    Teuchos::TimeMonitor exooutTimer(*exooutTime); //start timer


    stkDisc->outputToExodus(solution);
  }
#endif
  
}
