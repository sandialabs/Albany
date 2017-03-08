//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/12/14: no Epetra!

#if !defined(Schwarz_PiroObserverT_hpp)
#define Schwarz_PiroObserverT_hpp

#include "Piro_ObserverBase.hpp"

#include "Albany_DataTypes.hpp"
#include "Albany_Application.hpp"
#include "Schwarz_Coupled.hpp"
#include "Schwarz_ObserverImpl.hpp"
#include "Teuchos_RCP.hpp"

namespace LCM {

class Schwarz_PiroObserverT: public Piro::ObserverBase<ST> {
public:
  explicit Schwarz_PiroObserverT(
      Teuchos::RCP<SchwarzCoupled> const & cs_model);

  virtual void observeSolution(
      Thyra::VectorBase<ST> const & solution);

  virtual void observeSolution(
      Thyra::VectorBase<ST> const & solution,
      ST const stamp);

  virtual void observeSolution(
      Thyra::VectorBase<ST> const & solution,
      Thyra::VectorBase<ST> const & solution_dot,
      ST const stamp);

protected:
  int n_models_;

private:
  void observeSolutionImpl(
      Thyra::VectorBase<ST> const & solution,
      ST const default_stamp);

  void observeSolutionImpl(
      Thyra::VectorBase<ST> const & solution,
      Thyra::VectorBase<ST> const & solution_dot,
      ST const default_stamp);

  void observeTpetraSolutionImpl(
      Teuchos::Array<Teuchos::RCP<Tpetra_Vector const>> solutions,
      Teuchos::Array<Teuchos::RCP<Tpetra_Vector const>> solutions_dot,
      ST const default_stamp);

  Teuchos::RCP<ObserverImpl>
  impl_;

  Teuchos::ArrayRCP<Teuchos::RCP<Albany::Application>>
  apps_;

};

} // namespace Albany

#endif // Schwarz_PiroObserverT_hpp
