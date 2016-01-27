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
  explicit PiroObserverT(const Teuchos::RCP<Albany::Application> &app);

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

  ObserverImpl impl_;

};

} // namespace Albany

#endif /*ALBANY_PIROOBSERVERT_HPP*/
