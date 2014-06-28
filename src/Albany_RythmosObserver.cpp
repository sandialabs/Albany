//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_RythmosObserver.hpp"

#include "Rythmos_StepperBase.hpp"

#include "Thyra_DefaultProductVector.hpp"
#include "Thyra_VectorBase.hpp"
#include "Thyra_EpetraThyraWrappers.hpp"

#include "Epetra_Vector.h"

#include "Teuchos_Ptr.hpp"

Albany_RythmosObserver::Albany_RythmosObserver(
     const Teuchos::RCP<Albany::Application> &app_) :
  impl(app_),
  initial_step(true)
{
  // Nothing to do
}

void Albany_RythmosObserver::observeStartTimeStep(
    const Rythmos::StepperBase<ScalarType> &stepper,
    const Rythmos::StepControlInfo<ScalarType> &stepCtrlInfo,
    const int timeStepIter
    )
{
  if(initial_step)

    initial_step = false;

  else

    return;

std::cout << "AGS:: Hack do not observe first step" << std::endl; return;


  // Print the initial condition

  Teuchos::RCP<const Thyra::DefaultProductVector<ScalarType> > solnandsens = 
    Teuchos::rcp_dynamic_cast<const Thyra::DefaultProductVector<ScalarType> >
      (stepper.getStepStatus().solution);
  Teuchos::RCP<const Thyra::DefaultProductVector<ScalarType> > solnandsens_dot = 
    Teuchos::rcp_dynamic_cast<const Thyra::DefaultProductVector<ScalarType> >
      (stepper.getStepStatus().solutionDot);

  Teuchos::RCP<const Thyra::VectorBase<ScalarType> > solution;
  Teuchos::RCP<const Thyra::VectorBase<ScalarType> > solution_dot;
  if (solnandsens != Teuchos::null) {
    solution = solnandsens->getVectorBlock(0);
    solution_dot = solnandsens_dot->getVectorBlock(0);
  }
  else {
    solution = stepper.getStepStatus().solution;
    solution_dot = stepper.getStepStatus().solutionDot;
  }

  // Time should be zero unless we are restarting
  const ScalarType time = impl.getTimeParamValueOrDefault(stepper.getStepStatus().time);

  const Epetra_Vector soln= *(Thyra::get_Epetra_Vector(impl.getNonOverlappedMap(), solution));

  if(solution_dot != Teuchos::null){
    const Epetra_Vector soln_dot= *(Thyra::get_Epetra_Vector(impl.getNonOverlappedMap(), solution_dot));
    impl.observeSolution(time, soln, Teuchos::constOptInArg(soln_dot));
  }
  else {
    impl.observeSolution(time, soln, Teuchos::null);
  }

}

void Albany_RythmosObserver::observeCompletedTimeStep(
    const Rythmos::StepperBase<ScalarType> &stepper,
    const Rythmos::StepControlInfo<ScalarType> &stepCtrlInfo,
    const int timeStepIter
    )
{
  //cout << "ALBANY OBSERVER CALLED step=" <<  timeStepIter 
  //     << ",  time=" << stepper.getStepStatus().time << endl;

  Teuchos::RCP<const Thyra::DefaultProductVector<ScalarType> > solnandsens = 
    Teuchos::rcp_dynamic_cast<const Thyra::DefaultProductVector<ScalarType> >
      (stepper.getStepStatus().solution);
  Teuchos::RCP<const Thyra::DefaultProductVector<ScalarType> > solnandsens_dot = 
    Teuchos::rcp_dynamic_cast<const Thyra::DefaultProductVector<ScalarType> >
      (stepper.getStepStatus().solutionDot);

  Teuchos::RCP<const Thyra::VectorBase<ScalarType> > solution;
  Teuchos::RCP<const Thyra::VectorBase<ScalarType> > solution_dot;
  if (solnandsens != Teuchos::null) {
    solution = solnandsens->getVectorBlock(0);
// Next line bombs with BDF
    solution_dot = solnandsens_dot->getVectorBlock(0);
  }
  else {
    solution = stepper.getStepStatus().solution;
    solution_dot = stepper.getStepStatus().solutionDot;
  }

  const ScalarType time = impl.getTimeParamValueOrDefault(stepper.getStepStatus().time);

  const Epetra_Vector soln= *(Thyra::get_Epetra_Vector(impl.getNonOverlappedMap(), solution));

  // Comment out this section for BDF
  if(solution_dot != Teuchos::null){
    const Epetra_Vector soln_dot= *(Thyra::get_Epetra_Vector(impl.getNonOverlappedMap(), solution_dot));
    impl.observeSolution(time, soln, Teuchos::constOptInArg(soln_dot));
  }
  else {
    impl.observeSolution(time, soln, Teuchos::null);
  }

  // Add this section for output of sensitivities instead of soution
  /*
  if (solnandsens != Teuchos::null) {
    std::cout << "AGS: Albany_RythmosObserver writing out sensitivity vectors: " 
              << solnandsens->productSpace()->numBlocks()-1 << std::endl;
    Teuchos::RCP<const Thyra::VectorBase<ScalarType> > sensvec;
    for (int i=1; i < solnandsens->productSpace()->numBlocks(); i++) {
      sensvec = solnandsens->getVectorBlock(i);
      const Epetra_Vector sens= *(Thyra::get_Epetra_Vector(impl.getNonOverlappedMap(), sensvec));
      // tweak time a little bit
      impl.observeSolution(time*(1.0 + 1.0e-4*i), sens, Teuchos::null);
    }
  }
  else  impl.observeSolution(time, soln, Teuchos::null);
  */
}
