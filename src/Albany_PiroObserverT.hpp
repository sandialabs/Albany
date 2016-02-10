//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/12/14: no Epetra!

#ifndef ALBANY_PIROOBSERVERT_HPP
#define ALBANY_PIROOBSERVERT_HPP

#include "Piro_ObserverBase.hpp"

#include "Albany_DataTypes.hpp"
#include "Albany_Application.hpp"
#include "Albany_ObserverImpl.hpp"

#include "Teuchos_RCP.hpp"

namespace Albany {

class PiroObserverT : public Piro::ObserverBase<ST> {
public:
  explicit PiroObserverT(const Teuchos::RCP<Albany::Application> &app, 
                         Teuchos::RCP<const Thyra::ModelEvaluator<double>> model=Teuchos::null);

  virtual void observeSolution(
      const Thyra::VectorBase<ST> &solution);

  virtual void observeSolution(
      const Thyra::VectorBase<ST> &solution,
      const ST stamp);

  virtual void observeSolution(
      const Thyra::VectorBase<ST> &solution,
      const Thyra::VectorBase<ST> &solution_dot,
      const ST stamp);

  virtual void observeSolution(
      const Thyra::MultiVectorBase<ST> &solution,
      const ST stamp);

private:
  void observeSolutionImpl(
      const Thyra::VectorBase<ST> &solution,
      const ST defaultStamp);

  void observeSolutionImpl(
      const Thyra::VectorBase<ST> &solution,
      const Thyra::VectorBase<ST> &solution_dot,
      const ST defaultStamp);

  void observeSolutionImpl(
      const Thyra::MultiVectorBase<ST> &solution,
      const ST defaultStamp);

  void observeTpetraSolutionImpl(
      const Tpetra_Vector &solution,
      Teuchos::Ptr<const Tpetra_Vector> solution_dot,
      const ST defaultStamp);

  // The following function is for calculating / printing responses every step.
  // It is currently not implemented for the case of an Teuchos::RCP<const Thyra::MultiVectorBase<ST>>
  // argument; this may be desired at some point in the future. 
  void observeResponse(
      const ST defaultStamp,  
      Teuchos::RCP<const Thyra::VectorBase<ST>> solution,
      Teuchos::RCP<const Thyra::VectorBase<ST>> solution_dot = Teuchos::null); 

  ObserverImpl impl_;

  Teuchos::RCP<const Thyra::ModelEvaluator<double> > model_; 

protected: 

  bool observe_responses_;  

};

} // namespace Albany

#endif /*ALBANY_PIROOBSERVERT_HPP*/
