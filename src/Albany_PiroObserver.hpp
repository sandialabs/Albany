//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_PIRO_OBSERVER_HPP
#define ALBANY_PIRO_OBSERVER_HPP

#include <string>

#include "Piro_ObserverBase.hpp"

#include "Albany_DataTypes.hpp"
#include "Albany_Application.hpp"
#include "Albany_ObserverImpl.hpp"

#include "Teuchos_RCP.hpp"

namespace Albany {

class PiroObserver : public Piro::ObserverBase<ST> {
public:
  explicit PiroObserver(const Teuchos::RCP<Albany::Application> &app, 
                         Teuchos::RCP<const Thyra_ModelEvaluator> model = Teuchos::null);

  virtual void observeSolution(
      const Thyra_Vector& solution);

  virtual void observeSolution(
      const Thyra_Vector& solution,
      const Thyra_MultiVector& solution_dxdp);

  virtual void observeSolution(
      const Thyra_Vector& solution,
      const ST stamp);
  
  virtual void observeSolution(
      const Thyra_Vector& solution,
      const Thyra_MultiVector& solution_dxdp,
      const ST stamp);

  virtual void observeSolution(
      const Thyra_Vector& solution,
      const Thyra_Vector& solution_dot,
      const ST stamp);
  
  virtual void observeSolution(
      const Thyra_Vector& solution,
      const Thyra_MultiVector& solution_dxdp,
      const Thyra_Vector& solution_dot,
      const ST stamp);
  
  virtual void observeSolution(
      const Thyra_Vector& solution,
      const Thyra_Vector& solution_dot,
      const Thyra_Vector& solution_dotdot,
      const ST stamp);

  virtual void observeSolution(
      const Thyra_Vector& solution,
      const Thyra_MultiVector& solution_dxdp,
      const Thyra_Vector& solution_dot,
      const Thyra_Vector& solution_dotdot,
      const ST stamp);
  
  virtual void observeSolution(
      const Thyra_MultiVector& solution,
      const ST stamp);
  
  virtual void observeSolution(
      const Thyra_MultiVector& solution,
      const Thyra_MultiVector& solution_dxdp,
      const ST stamp);

  virtual void parameterChanged(
      const std::string& param);

private:
  void observeSolutionImpl(
      const Thyra_Vector& solution,
      const ST defaultStamp);

  void observeSolutionImpl(
      const Thyra_Vector& solution,
      const Thyra_MultiVector& solution_dxdp,
      const ST defaultStamp);

  void observeSolutionImpl(
      const Thyra_Vector& solution,
      const Thyra_Vector& solution_dot,
      const ST defaultStamp);
  
  void observeSolutionImpl(
      const Thyra_Vector& solution,
      const Thyra_MultiVector& solution_dxdp,
      const Thyra_Vector& solution_dot,
      const ST defaultStamp);
  
  void observeSolutionImpl(
      const Thyra_Vector& solution,
      const Thyra_Vector& solution_dot,
      const Thyra_Vector& solution_dotdot,
      const ST defaultStamp);

  void observeSolutionImpl(
      const Thyra_Vector& solution,
      const Thyra_MultiVector& solution_dxdp,
      const Thyra_Vector& solution_dot,
      const Thyra_Vector& solution_dotdot,
      const ST defaultStamp);

  void observeSolutionImpl(
      const Thyra_MultiVector &solution,
      const ST defaultStamp);

  void observeSolutionImpl(
      const Thyra_MultiVector &solution,
      const Thyra_MultiVector &solution_dxdp,
      const ST defaultStamp);

  void observeTpetraSolutionImpl(
      const Tpetra_Vector &solution,
      Teuchos::Ptr<const Tpetra_Vector> solution_dot,
      Teuchos::Ptr<const Tpetra_Vector> solution_dotdot,
      const ST defaultStamp);
  
  // The following function is for calculating / printing responses every step.
  // It is currently not implemented for the case of an Teuchos::RCP<const Thyra_MultiVector>
  // argument; this may be desired at some point in the future. 
  void observeResponse(
      const ST defaultStamp,  
      Teuchos::RCP<const Thyra_Vector> solution,
      Teuchos::RCP<const Thyra_Vector> solution_dot = Teuchos::null, 
      Teuchos::RCP<const Thyra_Vector> solution_dotdot = Teuchos::null); 

  ObserverImpl impl_;

  Teuchos::RCP<const Thyra_ModelEvaluator> model_; 

protected: 

  bool observe_responses_;  
  
  int stepper_counter_;  
  
  Teuchos::RCP<Teuchos::FancyOStream> out; 

  int observe_responses_every_n_steps_; 

  bool firstResponseObtained;
  bool calculateRelativeResponses;
  std::vector< std::vector<double> > storedResponses;
  Teuchos::Array<unsigned int> relative_responses;
  std::vector<bool> is_relative;
  const double tol = 1e-15;
};

} // namespace Albany

#endif // ALBANY_PIRO_OBSERVER_HPP
