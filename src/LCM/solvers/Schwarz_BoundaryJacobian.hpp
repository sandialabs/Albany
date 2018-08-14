//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_SchwarzBoundaryJacobian_hpp)
#define LCM_SchwarzBoundaryJacobian_hpp

#include <iostream>

#include "Teuchos_Comm.hpp"
#include "Teuchos_RCP.hpp"
#include "Tpetra_CrsMatrix.hpp"
#include "Tpetra_Import.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_Operator.hpp"
#include "Tpetra_Vector.hpp"

#include "Albany_Application.hpp"
#include "Albany_DataTypes.hpp"
#include "MiniTensor.h"

namespace LCM {

///
/// \brief A Tpetra operator that evaluates the Jacobian of a
/// LCM coupled Schwarz Multiscale problem.
/// Each Jacobian couples one single application to another.
///

class Schwarz_BoundaryJacobian : public Tpetra_Operator
{
 public:
  Schwarz_BoundaryJacobian(
      Teuchos::RCP<Teuchos_Comm const> const&                     comm,
      Teuchos::ArrayRCP<Teuchos::RCP<Albany::Application>> const& ca,
      Teuchos::Array<Teuchos::RCP<Tpetra_CrsMatrix>>              jacs,
      int const this_app_index    = 0,
      int const coupled_app_index = 0);

  ~Schwarz_BoundaryJacobian();

  /// Initialize the operator with everything needed to apply it
  // FIXME: add arguments
  void
  initialize();

  /// Returns the result of a Tpetra_Operator applied to a
  /// Tpetra_MultiVector X in Y.
  virtual void
  apply(
      Tpetra_MultiVector const& X,
      Tpetra_MultiVector&       Y,
      Teuchos::ETransp          mode  = Teuchos::NO_TRANS,
      ST                        alpha = Teuchos::ScalarTraits<ST>::one(),
      ST                        beta = Teuchos::ScalarTraits<ST>::zero()) const;

  /// Returns explicit matrix representation of operator if available.
  Teuchos::RCP<Tpetra_CrsMatrix>
  getExplicitOperator() const;

  /// Returns the current UseTranspose setting.
  virtual bool
  hasTransposeApply() const
  {
    return b_use_transpose_;
  }

  /// Returns the Tpetra_Map object associated with the domain of this operator.
  virtual Teuchos::RCP<Tpetra_Map const>
  getDomainMap() const
  {
    return domain_map_;
  }

  /// Returns the Tpetra_Map object associated with the range of this operator.
  virtual Teuchos::RCP<Tpetra_Map const>
  getRangeMap() const
  {
    return range_map_;
  }

  void
  setThisAppIndex(int const tai)
  {
    this_app_index_ = tai;
  }

  int
  getThisAppIndex() const
  {
    return this_app_index_;
  }

  void
  setCoupledAppIndex(int const cai)
  {
    coupled_app_index_ = cai;
  }

  int
  getCoupledAppIndex() const
  {
    return coupled_app_index_;
  }

  Albany::Application const&
  getApplication(int const app_index)
  {
    return *(coupled_apps_[app_index]);
  }

  Albany::Application const&
  getApplication(int const app_index) const
  {
    return *(coupled_apps_[app_index]);
  }

 private:
  Teuchos::ArrayRCP<Teuchos::RCP<Albany::Application>> coupled_apps_;

  Teuchos::Array<Teuchos::RCP<Tpetra_CrsMatrix>> jacs_;

  int this_app_index_;

  int coupled_app_index_;

  Teuchos::Array<Teuchos::RCP<Tpetra_Map const>> disc_maps_;

  Teuchos::RCP<Tpetra_Map const> domain_map_;

  Teuchos::RCP<Tpetra_Map const> range_map_;

  Teuchos::RCP<Teuchos_Comm const> comm_;

  bool b_use_transpose_;

  bool b_initialized_;

  int n_models_;
};

}  // namespace LCM
#endif  // LCM_SchwarzBoundaryJacobian_hpp
