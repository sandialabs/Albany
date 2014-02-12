//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_ConstitutiveModelInterface_hpp)
#define LCM_ConstitutiveModelInterface_hpp

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

#include "models/ConstitutiveModel.hpp"

namespace LCM {

  //! \brief Struct to store state variable registration information
  struct StateVariableRegistrationStruct {
  public:
    std::string name;
    Teuchos::RCP<PHX::DataLayout> data_layout;
    std::string init_type;
    double init_value;
    bool register_old_state;
    bool output_to_exodus;
  };

  /// \brief Constitutive Model Interface
  template<typename EvalT, typename Traits>
  class ConstitutiveModelInterface : public PHX::EvaluatorWithBaseImpl<Traits>,
                                     public PHX::EvaluatorDerived<EvalT, Traits>  {

  public:

    ///
    /// Constructor
    ///
    ConstitutiveModelInterface(Teuchos::ParameterList& p,
                               const Teuchos::RCP<Albany::Layouts>& dl);

    ///
    /// Phalanx method to allocate space
    ///
    void postRegistrationSetup(typename Traits::SetupData d,
                               PHX::FieldManager<Traits>& vm);

    ///
    /// Implementation of physics
    ///
    void evaluateFields(typename Traits::EvalData d);

    ///
    /// Populate the state variable registration struct
    ///
    void fillStateVariableStruct(int state_var);

    ///
    /// Retrive the number of model state variables
    ///
    int getNumStateVars() { return model_->getNumStateVariables(); }

    ///
    /// Initialization routine
    ///
    void initializeModel(Teuchos::ParameterList* p,
                         const Teuchos::RCP<Albany::Layouts>& dl);

    ///
    /// Retrive SV name from the state variable registration struct
    ///
    std::string getName() { return sv_struct_.name; }

    ///
    /// Retrive SV layout from the state variable registration struct
    ///
    Teuchos::RCP<PHX::DataLayout> getLayout() { return sv_struct_.data_layout; }

    ///
    /// Retrive SV init type from the state variable registration struct
    ///
    std::string getInitType() { return sv_struct_.init_type; }

    ///
    /// Retrive SV init value from the state variable registration struct
    ///
    double getInitValue() { return sv_struct_.init_value; }

    ///
    /// Retrive SV state flag from the state variable registration struct
    ///
    double getStateFlag() { return sv_struct_.register_old_state; }

    ///
    /// Retrive SV output flag from the state variable registration struct
    ///
    double getOutputFlag() { return sv_struct_.output_to_exodus; }

  private:

    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;

    ///
    /// Dependent MDFields
    ///
    std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > dep_fields_map_;

    ///
    /// Evaluated MDFields
    ///
    std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > eval_fields_map_;

    ///
    /// Constitutive Model
    ///
    Teuchos::RCP<LCM::ConstitutiveModel<EvalT,Traits> > model_;

    ///
    /// State Variable Registration Struct
    ///
    StateVariableRegistrationStruct sv_struct_;

    ///
    /// Optional field for integration point locations
    ///
    PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim> coord_vec_;

    ///
    /// Optional temperature field
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint> temperature_;

    ///
    /// Optional damage field
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint> damage_;

    ///
    /// Optional Integration Weights
    ///
    PHX::MDField<MeshScalarT,Cell,QuadPoint> weights_;

    ///
    /// flag to indicate we have temperature
    ///
    bool have_temperature_;

    ///
    /// flag to indicate we have damage
    ///
    bool have_damage_;

    ///
    /// flag to volume average the pressure
    ///
    bool volume_average_pressure_;
  };

}

#endif
