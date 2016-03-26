//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_VariationalModel_h)
#define LCM_VariationalModel_h

#include "AbstractModel.hpp"

namespace LCM
{

///
/// Variational Constitutive Model Base Class
///
template<typename EvalT, typename Traits,
typename StateVars, typename InternalVars>
class VariationalModel : public AbstractModel<EvalT, Traits>
{
public:

  using ScalarT = typename AbstractModel<EvalT, Traits>::ScalarT;
  using MeshScalarT = typename AbstractModel<EvalT, Traits>::MeshScalarT;
  using Workset = typename AbstractModel<EvalT, Traits>::Workset;

  using FieldMap = typename AbstractModel<EvalT, Traits>::FieldMap;
  using DataLayoutMap = typename AbstractModel<EvalT, Traits>::DataLayoutMap;

  ///
  /// Constructor
  ///
  VariationalModel(
      Teuchos::ParameterList * p,
      Teuchos::RCP<Albany::Layouts> const & dl)
  {
    return;
  }

  ///
  /// Virtual destructor
  ///
  virtual
  ~VariationalModel()
  {
    return;
  };

  ///
  /// No copy constructor
  ///
  VariationalModel(VariationalModel const &) = delete;

  ///
  /// No copy assignment
  ///
  VariationalModel &
  operator=(VariationalModel const &) = delete;

  ///
  /// Methods to compute the state (e.g. energy, gradient, Hessian)
  ///
  virtual
  ScalarT
  computeEnergy(
      StateVars const & state_vars,
      InternalVars const & internal_vars) = 0;

  virtual
  StateVars
  computeFlux(
      StateVars const & state_vars,
      InternalVars const & internal_vars) = 0;

  virtual
  void
  computeHessian(
      StateVars const & state_vars,
      InternalVars const & internal_vars) = 0;
};

} // namespace LCM

#endif // LCM_VariationalModel_h
