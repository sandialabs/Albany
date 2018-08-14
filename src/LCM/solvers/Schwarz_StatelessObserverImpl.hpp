//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#if !defined(LCM_Schwarz_StatelessObserverImpl_hpp)
#define LCM_Schwarz_StatelessObserverImpl_hpp

#include "Albany_Application.hpp"
#include "Albany_DataTypes.hpp"

namespace LCM {

class StatelessObserverImpl
{
 public:
  explicit StatelessObserverImpl(
      Teuchos::ArrayRCP<Teuchos::RCP<Albany::Application>>& apps);

  virtual ~StatelessObserverImpl();

  RealType
  getTimeParamValueOrDefault(RealType default_value) const;

  virtual void
  observeSolutionT(
      double                                            stamp,
      Teuchos::Array<Teuchos::RCP<Tpetra_Vector const>> non_overlapped_solution,
      Teuchos::Array<Teuchos::RCP<Tpetra_Vector const>>
          non_overlapped_solution_dot);

  StatelessObserverImpl(StatelessObserverImpl const&) = delete;
  StatelessObserverImpl&
  operator=(StatelessObserverImpl const&) = delete;

 protected:
  Teuchos::ArrayRCP<Teuchos::RCP<Albany::Application>> apps_;

  int n_models_;

 private:
  Teuchos::RCP<Teuchos::Time> sol_out_time_;
};

}  // namespace LCM

#endif  // LCM_Schwarz_StatelessObserverImpl_hpp
