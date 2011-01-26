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
  app(app_),
  disc(app_->getDiscretization()),
  vtk(vtk_)
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

#ifdef ALBANY_IOSS
  // if (solution.Map().Comm().MyPID()==0)
  //   cout << "Albany::NOXObserver calling exodus output " << endl;

  Albany::STKDiscretization* stkDisc =
    dynamic_cast<Albany::STKDiscretization*>(disc.get());

  {
    Teuchos::TimeMonitor exooutTimer(*exooutTime); //start timer

    std::vector<std::vector<double> > states = averageStates(app->getStateMgr().getStateVariables());

    stkDisc->outputToExodus(solution, states);
  }
#endif

   // This must come at the end since it renames the New state 
   // as the Old state in preparation for the next step
   app->getStateMgr().updateStates();;

}

std::vector<std::vector<double> >
Albany_NOXObserver::averageStates(const std::vector<Albany::StateVariables>& stateVariables) {
  std::vector<std::vector<double> > states;
  int numStates = stateVariables[0].size();
  if (numStates==0) return states;

  int numWorksets = stateVariables.size();

  int containerSize = stateVariables[0].begin()->second->dimension(0);
  int numQP  = stateVariables[0].begin()->second->dimension(1);
  int numDim  = stateVariables[0].begin()->second->dimension(2);
  int numDim2 = stateVariables[0].begin()->second->dimension(3);
  int numScalarStates = numDim * numDim2; // 2D stress tensor

  states.resize(numWorksets*containerSize);
  for (int i=0; i<numWorksets*containerSize;i++)  states[i].resize(numScalarStates);

  cout << "QQQ numWorksets " <<  numWorksets << "  numScalarStates " << numScalarStates << "  containersize " << containerSize << endl;

  for (int i=0; i< numWorksets; i++) {
    const Intrepid::FieldContainer<RealType>& fc = *(stateVariables[i].begin()->second);
    for (int j=0; j< containerSize; j++) {
      for (int k=0; k< numQP; k++) {
        for (int l=0; l< numDim; l++) {
          for (int m=0; m< numDim2; m++) {
             states[i*containerSize + j][m+l*numDim] += fc(j,k,l,m)/numQP;
          }
        }
      }
    }
  }
  return states;
}
