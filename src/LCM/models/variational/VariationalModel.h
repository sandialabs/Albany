//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_VariationalModel_h)
#define LCM_VariationalModel_h

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace LCM
{

///
/// Variational Constitutive Model Base Class
///
template<typename EvalT, typename Traits>
class VariationalModel
{
public:

  using ScalarT = typename EvalT::ScalarT;
  using MeshScalarT = typename EvalT::MeshScalarT;
  using Workset = typename Traits::EvalData;
  using FieldMap = std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>>>;
  using DataLayoutMap = std::map<std::string, Teuchos::RCP<PHX::DataLayout>>;

  ///
  /// Constructor
  ///
  VariationalModel(
      Teuchos::ParameterList * p,
      Teuchos::RCP<Albany::Layouts> const & dl);

  ///
  /// Virtual destructor
  ///
  virtual
  ~VariationalModel()
  {};

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
  void
  computeEnergy(
      Workset workset,
      FieldMap dep_fields,
      FieldMap eval_fields) = 0;

  virtual
  void
  computeGradient(
      Workset workset,
      FieldMap dep_fields,
      FieldMap eval_fields) = 0;

  virtual
  void
  computeHessian(
      Workset workset,
      FieldMap dep_fields,
      FieldMap eval_fields) = 0;

  ///
  /// Return a map to the dependent fields
  ///
  DataLayoutMap
  getDependentFieldMap()
  {
    return dep_field_layout_map_;
  }

  ///
  /// Return a map to the evaluated fields
  ///
  DataLayoutMap
  getEvaluatedFieldMap()
  {
    return eval_field_layout_map_;
  }

protected:

  DataLayoutMap
  dep_field_layout_map_;

  DataLayoutMap
  eval_field_layout_map_;

};

} // namespace LCM

#endif // LCM_VariationalModel_h
