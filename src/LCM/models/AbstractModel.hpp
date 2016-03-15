//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_AbstractModel_hpp)
#define LCM_AbstractModel_hpp

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace LCM
{

///
/// Abstract Model Base Class
///
template<typename EvalT, typename Traits>
class AbstractModel
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
  AbstractModel(
      Teuchos::ParameterList * p,
      Teuchos::RCP<Albany::Layouts> const & dl)
  {
    // extract number of integration points and dimensions
    std::vector<PHX::DataLayout::size_type>
    dims;

    dl->qp_tensor->dimensions(dims);

    num_pts_ = dims[1];
    num_dims_ = dims[2];

    field_name_map_ =
        p->get<Teuchos::RCP<std::map<std::string, std::string>>>("Name Map");

    return;
  }

  ///
  /// Virtual Destructor
  ///
  virtual
  ~AbstractModel()
  {
    return;
  };

  ///
  /// No copy constructor
  ///
  AbstractModel(AbstractModel const &) = delete;

  ///
  /// No copy assignment
  ///
  AbstractModel &
  operator=(AbstractModel const &) = delete;

  ///
  /// Compute the state (e.g. energy, stress, tangent)
  ///
  virtual
  void
  computeState(
      Workset workset,
      FieldMap dep_fields,
      FieldMap eval_fields) = 0;

  ///
  /// Return a map to the dependent fields
  ///
  virtual
  DataLayoutMap
  getDependentFieldMap() final
  {
    return dep_field_map_;
  }

  ///
  /// Return a map to the evaluated fields
  ///
  virtual
  DataLayoutMap
  getEvaluatedFieldMap() final
  {
    return eval_field_map_;
  }

  ///
  /// Convenience function to set dependent fields.
  ///
  virtual
  void
  setDependentField(
      std::string const & field_name,
      Teuchos::RCP<PHX::DataLayout> const & field) final
  {
    dep_field_map_.insert(std::make_pair(field_name, field));
  }

  ///
  /// Convenience function to set evaluated fields.
  ///
  virtual
  void
  setEvaluatedField(
      std::string const & field_name,
      Teuchos::RCP<PHX::DataLayout> const & field) final
  {
    eval_field_map_.insert(std::make_pair(field_name, field));
  }

  ///
  /// state variable registration helpers
  ///
  virtual
  std::string
  getStateVarName(int state_var) final
  {
    return state_var_names_[state_var];
  }

  virtual
  Teuchos::RCP<PHX::DataLayout>
  getStateVarLayout(int state_var) final
  {
    return state_var_layouts_[state_var];
  }

  virtual
  std::string
  getStateVarInitType(int state_var) final
  {
    return state_var_init_types_[state_var];
  }

  virtual
  double
  getStateVarInitValue(int state_var) final
  {
    return state_var_init_values_[state_var];
  }

  virtual
  bool
  getStateVarOldStateFlag(int state_var) final
  {
    return state_var_old_state_flags_[state_var];
  }

  virtual
  bool
  getStateVarOutputFlag(int state_var) final
  {
    return state_var_output_flags_[state_var];
  }

  virtual
  int getNumStateVariables() final
  {
    return num_state_variables_;
  }

protected:

  ///
  /// Number of State Variables
  ///
  int
  num_state_variables_{0};

  ///
  /// Number of dimensions
  ///
  int
  num_dims_{0};

  ///
  /// Number of integration points
  ///
  int
  num_pts_{0};

  ///
  /// Map of field names
  ///
  Teuchos::RCP<std::map<std::string, std::string>>
  field_name_map_;

  std::vector<std::string>
  state_var_names_;

  std::vector<Teuchos::RCP<PHX::DataLayout>>
  state_var_layouts_;

  std::vector<std::string>
  state_var_init_types_;

  std::vector<double>
  state_var_init_values_;

  std::vector<bool>
  state_var_old_state_flags_;

  std::vector<bool>
  state_var_output_flags_;

  DataLayoutMap
  dep_field_map_;

  DataLayoutMap
  eval_field_map_;
};

} // namespace LCM

#endif // LCM_AbstractModel_hpp
