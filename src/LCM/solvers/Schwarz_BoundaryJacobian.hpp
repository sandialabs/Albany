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

#include "Albany_Application.hpp"
#include "Albany_DataTypes.hpp"
#include "MiniTensor.h"

namespace LCM {

///
/// \brief A Thyra operator that evaluates the Jacobian of a
/// LCM coupled Schwarz Multiscale problem.
/// Each Jacobian couples one single application to another.
///

class Schwarz_BoundaryJacobian : public Thyra_LinearOp
{
 public:
  Schwarz_BoundaryJacobian(
      Teuchos::RCP<Teuchos_Comm const> const&                     comm,
      Teuchos::ArrayRCP<Teuchos::RCP<Albany::Application>> const& ca,
      Teuchos::Array<Teuchos::RCP<Thyra_LinearOp>>                jacs,
      int const this_app_index    = 0,
      int const coupled_app_index = 0);

  ~Schwarz_BoundaryJacobian() = default;

  /// Initialize the operator with everything needed to apply it
  // FIXME: add arguments
  void
  initialize();

  //! Overrides Thyra::LinearOpBase purely virtual method
  /// Returns the result of a Thyra_LinearOp applied to a
  /// Thyra_MultiVector X in Y.
  virtual void
  applyImpl(
      const Thyra::EOpTransp                 M_trans,
      const Thyra_MultiVector&               X,
      const Teuchos::Ptr<Thyra_MultiVector>& Y,
      const ST alpha = Teuchos::ScalarTraits<ST>::one(),
      const ST beta  = Teuchos::ScalarTraits<ST>::zero()) const;

  //! Overrides Thyra::LinearOpBase purely virtual method
  bool opSupportedImpl(Thyra::EOpTransp /*M_trans*/) const
  {
    // The underlying scalar type is not complex, and we support transpose, so
    // we support everything.
    return true;
  }

  /// Returns explicit matrix representation of operator if available.
  Teuchos::RCP<Thyra_LinearOp>
  getExplicitOperator() const;

  //! Overrides Thyra::LinearOpBase purely virtual method
  Teuchos::RCP<const Thyra_VectorSpace>
  domain() const
  {
    return domain_vs_;
  }

  //! Overrides Thyra::LinearOpBase purely virtual method
  Teuchos::RCP<const Thyra_VectorSpace>
  range() const
  {
    return range_vs_;
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

  Teuchos::Array<Teuchos::RCP<Thyra_LinearOp>> jacs_;

  int this_app_index_;

  int coupled_app_index_;

  Teuchos::Array<Teuchos::RCP<Thyra_VectorSpace const>> disc_maps_;

  Teuchos::RCP<Thyra_VectorSpace const> domain_vs_;

  Teuchos::RCP<Thyra_VectorSpace const> range_vs_;

  Teuchos::RCP<Teuchos_Comm const> comm_;

  int n_models_;
};

}  // namespace LCM
#endif  // LCM_SchwarzBoundaryJacobian_hpp
