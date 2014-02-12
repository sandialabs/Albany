//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_ConstitutiveModel_hpp)
#define LCM_ConstitutiveModel_hpp

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace LCM
{

//! \brief Constitutive Model Base Class
template<typename EvalT, typename Traits>
class ConstitutiveModel
{
public:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  ///
  /// Constructor
  ///
  ConstitutiveModel(Teuchos::ParameterList* p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  ///
  /// Virtual Destructor
  ///
  virtual
  ~ConstitutiveModel()
  {};

  ///
  /// Method to compute the state (e.g. energy, stress, tangent)
  ///
  virtual
  void
  computeState(
      typename Traits::EvalData workset,
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > dep_fields,
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > eval_fields) = 0;

  ///
  /// Optional Method to volume average the pressure
  ///
  void
  computeVolumeAverage(
      typename Traits::EvalData workset,
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > dep_fields,
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > eval_fields);

  ///
  /// Return a map to the dependent fields
  ///
  std::map<std::string, Teuchos::RCP<PHX::DataLayout> >
  getDependentFieldMap()
  {
    return dep_field_map_;
  }

  ///
  /// Return a map to the evaluated fields
  ///
  std::map<std::string, Teuchos::RCP<PHX::DataLayout> >
  getEvaluatedFieldMap()
  {
    return eval_field_map_;
  }

  ///
  /// state variable registration helpers
  ///
  std::string getStateVarName(int state_var);
  Teuchos::RCP<PHX::DataLayout> getStateVarLayout(int state_var);
  std::string getStateVarInitType(int state_var);
  double getStateVarInitValue(int state_var);
  bool getStateVarOldStateFlag(int state_var);
  bool getStateVarOutputFlag(int state_var);

  ///
  /// Retrive the number of state variables
  ///
  int getNumStateVariables()
  {
    return num_state_variables_;
  }

  ///
  /// Integration point location flag
  ///
  bool
  getIntegrationPointLocationFlag()
  {
    return need_integration_pt_locations_;
  }

  ///
  /// Integration point location set method
  ///
  void
  setCoordVecField(PHX::MDField<MeshScalarT, Cell, QuadPoint, Dim> coord_vec)
  {
    coord_vec_ = coord_vec;
  }

  ///
  /// set the Temperature field
  ///
  void
  setTemperatureField(PHX::MDField<ScalarT, Cell, QuadPoint> temperature)
  {
    temperature_ = temperature;
  }

  ///
  /// set the Temperature field
  ///
  void
  setDamageField(PHX::MDField<ScalarT, Cell, QuadPoint> damage)
  {
    damage_ = damage;
  }

  ///
  /// set the Weights field
  ///
  void
  setWeightsField(PHX::MDField<MeshScalarT, Cell, QuadPoint> weights)
  {
    weights_ = weights;
  }

protected:

  ///
  /// Number of State Variables
  ///
  int num_state_variables_;

  ///
  /// Number of dimensions
  ///
  std::size_t num_dims_;

  ///
  /// Number of integration points
  ///
  std::size_t num_pts_;

  ///
  /// flag for integration point locations
  ///
  bool need_integration_pt_locations_;

  ///
  /// flag that the energy needs to be computed
  ///
  bool compute_energy_;

  ///
  /// flag that the tangent needs to be computed
  ///
  bool compute_tangent_;

  ///
  /// Bool for temperature
  ///
  bool have_temperature_;

  ///
  /// Bool for damage
  ///
  bool have_damage_;

  ///
  /// optional integration point locations field
  ///
  PHX::MDField<MeshScalarT, Cell, QuadPoint, Dim> coord_vec_;

  ///
  /// optional temperature field
  ///
  PHX::MDField<ScalarT, Cell, QuadPoint> temperature_;

  ///
  /// optional damage field
  ///
  PHX::MDField<ScalarT, Cell, QuadPoint> damage_;

  ///
  /// optional weights field
  ///
  PHX::MDField<MeshScalarT, Cell, QuadPoint> weights_;

  ///
  /// Map of field names
  ///
  Teuchos::RCP<std::map<std::string, std::string> > field_name_map_;

  std::vector<std::string> state_var_names_;
  std::vector<Teuchos::RCP<PHX::DataLayout> > state_var_layouts_;
  std::vector<std::string> state_var_init_types_;
  std::vector<double> state_var_init_values_;
  std::vector<bool> state_var_old_state_flags_;
  std::vector<bool> state_var_output_flags_;

  std::map<std::string, Teuchos::RCP<PHX::DataLayout> > dep_field_map_;
  std::map<std::string, Teuchos::RCP<PHX::DataLayout> > eval_field_map_;

  ///
  /// Thermal Expansion Coefficient
  ///
  RealType expansion_coeff_;

  ///
  /// Reference Temperature
  ///
  RealType ref_temperature_;

  ///
  /// Heat Capacity
  ///
  RealType heat_capacity_;

  ///
  /// Density
  ///
  RealType density_;

private:
  ///
  /// Private to prohibit copying
  ///
  ConstitutiveModel(const ConstitutiveModel&);

  ///
  /// Private to prohibit copying
  ///
  ConstitutiveModel& operator=(const ConstitutiveModel&);

};
}

#endif
