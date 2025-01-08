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

  void observeSolution (const Thyra_Vector& x)
  {
    observeSolution(x,zero());
  }

  void observeSolution (const Thyra_Vector& x,
                        const Thyra_MultiVector& dxdp)
  {
    observeSolution(x,dxdp);
  }

  void observeSolution (const Thyra_Vector& x,
                        const ST stamp);
                    
  void observeSolution (const Thyra_Vector& x,
                        const Thyra_MultiVector& dxdp,
                        const ST stamp);

  void observeSolution (const Thyra_Vector& x,
                        const Thyra_Vector& x_dot,
                        const ST stamp);
  
  void observeSolution (const Thyra_Vector& x,
                        const Thyra_MultiVector& dxdp,
                        const Thyra_Vector& x_dot,
                        const ST stamp);
  
  void observeSolution (const Thyra_Vector& x,
                        const Thyra_Vector& x_dot,
                        const Thyra_Vector& x_dotdot,
                        const ST stamp);

  void observeSolution (const Thyra_Vector& x,
                        const Thyra_MultiVector& dxdp,
                        const Thyra_Vector& x_dot,
                        const Thyra_Vector& x_dotdot,
                        const ST stamp);
  
  void observeSolution (const Thyra_MultiVector& solution,
                        const ST stamp);
  
  void observeSolution (const Thyra_MultiVector& solution,
                        const Thyra_MultiVector& solution_dxdp,
                        const ST stamp);

  void parameterChanged (const std::string& param) { impl_.parameterChanged(param); }

protected:

  void observeSolutionImpl (const Teuchos::RCP<const Thyra_Vector>& x,
                            const Teuchos::RCP<const Thyra_Vector>& x_dot,
                            const Teuchos::RCP<const Thyra_Vector>& x_dotdot,
                            const Teuchos::RCP<const Thyra_MultiVector>& dxdp,
                            const ST defaultStamp);
  
  // The following function is for calculating / printing responses every step.
  // It is currently not implemented for the case of an Teuchos::RCP<const Thyra_MultiVector>
  // argument; this may be desired at some point in the future. 
  void observeResponse(const ST defaultStamp,  
                       Teuchos::RCP<const Thyra_Vector> x,
                       Teuchos::RCP<const Thyra_Vector> x_dot,
                       Teuchos::RCP<const Thyra_Vector> x_dotdot);

  ObserverImpl impl_;

  Teuchos::RCP<const Thyra_ModelEvaluator> model_; 

protected: 

  static ST zero () { return Teuchos::ScalarTraits<ST>::zero(); }

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
