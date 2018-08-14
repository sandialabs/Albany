//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#if !defined(Schwarz_PiroObserver_hpp)
#define Schwarz_PiroObserver_hpp

#include "Albany_Application.hpp"
#include "Albany_DataTypes.hpp"
#include "Piro_ObserverBase.hpp"
#include "Schwarz_Coupled.hpp"
#include "Schwarz_ObserverImpl.hpp"
#include "Teuchos_RCP.hpp"

namespace LCM {

class Schwarz_PiroObserver : public Piro::ObserverBase<ST>
{
 public:
  explicit Schwarz_PiroObserver(Teuchos::RCP<SchwarzCoupled> const& cs_model);

  virtual void
  observeSolution(Thyra::VectorBase<ST> const& solution);

  virtual void
  observeSolution(Thyra::VectorBase<ST> const& solution, ST const stamp);

  virtual void
  observeSolution(
      Thyra::VectorBase<ST> const& solution,
      Thyra::VectorBase<ST> const& solution_dot,
      ST const                     stamp);

 private:
  void
  observeSolutionImpl(
      Thyra::VectorBase<ST> const& solution,
      ST const                     default_stamp);

  void
  observeSolutionImpl(
      Thyra::VectorBase<ST> const& solution,
      Thyra::VectorBase<ST> const& solution_dot,
      ST const                     default_stamp);

  void
  observeTpetraSolutionImpl(
      Teuchos::Array<Teuchos::RCP<Tpetra_Vector const>> solutions,
      Teuchos::Array<Teuchos::RCP<Tpetra_Vector const>> solutions_dot,
      ST const                                          default_stamp);

  int n_models_;

  Teuchos::RCP<ObserverImpl> impl_;

  Teuchos::ArrayRCP<Teuchos::RCP<Albany::Application>> apps_;
};

}  // namespace LCM

#endif  // Schwarz_PiroObserver_hpp
