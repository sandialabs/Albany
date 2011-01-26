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


#include "Albany_RythmosObserver.hpp"
#include "Thyra_DefaultProductVector.hpp"
#include "Thyra_VectorBase.hpp"
#ifdef ALBANY_IOSS
  #include "Albany_STKDiscretization.hpp"
#endif

using namespace std;

Albany_RythmosObserver::Albany_RythmosObserver(
     const Teuchos::RCP<Albany_VTK> vtk_,
     const Teuchos::RCP<Albany::Application> &app_) : 
  disc(app_->getDiscretization()),
  app(app_),
  vtk(vtk_)
{
   if (vtk != Teuchos::null) { vtk->updateGeometry (disc); }

    exooutTime = Teuchos::TimeMonitor::getNewTimer("Albany: Output to Exodus");
}

void Albany_RythmosObserver::observeCompletedTimeStep(
    const Rythmos::StepperBase<Scalar> &stepper,
    const Rythmos::StepControlInfo<Scalar> &stepCtrlInfo,
    const int timeStepIter
    )
{
  //cout << "ALBANY OBSERVER CALLED step=" <<  timeStepIter 
  //     << ",  time=" << stepper.getStepStatus().time << endl;

  Teuchos::RCP<const Thyra::DefaultProductVector<double> > solnandsens = 
    Teuchos::rcp_dynamic_cast<const Thyra::DefaultProductVector<double> >
      (stepper.getStepStatus().solution);

  Teuchos::RCP<const Thyra::VectorBase<double> > solution;
  if (solnandsens != Teuchos::null) {
    solution = solnandsens->getVectorBlock(0);
  }
  else {
    solution = stepper.getStepStatus().solution;
  }

  const Epetra_Vector soln= *(Thyra::get_Epetra_Vector(*disc->getMap(), solution));
  if (vtk != Teuchos::null) vtk->visualizeField (soln, disc);

#ifdef ALBANY_IOSS
  Albany::STKDiscretization* stkDisc =
    dynamic_cast<Albany::STKDiscretization*>(disc.get());

  {
    Teuchos::TimeMonitor exooutTimer(*exooutTime); //start timer

    std::vector<std::vector<double> > states = app->getStateMgr().getElementAveragedStates();
    stkDisc->outputToExodus(soln,states);
  }
#endif

  app->getStateMgr().updateStates();;
}
