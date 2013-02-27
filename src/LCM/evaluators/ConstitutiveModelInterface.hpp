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
    // FIXME: get rid of trailing underscore here
    std::string name_;
    Teuchos::RCP<PHX::DataLayout> data_layout_;
    std::string init_type_;
    double init_value_;
    bool register_old_state_;
    bool output_to_exodus_;
  };

  /// \brief Constitutive Model Interface
  template<typename EvalT, typename Traits>
  class ConstitutiveModelInterface : public PHX::EvaluatorWithBaseImpl<Traits>,
                                     public PHX::EvaluatorDerived<EvalT, Traits>  {

  public:

    ///
    /// Constructor
    ///
    ConstitutiveModelInterface(const Teuchos::ParameterList& p,
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
    int getNumStateVars() { return model_->num_state_variables_; }

    ///
    /// Initialization routine
    ///
    void initializeModel(const Teuchos::ParameterList* p, 
                         const Teuchos::RCP<Albany::Layouts>& dl);

    ///
    /// State Variable Registration Struct
    ///
    StateVariableRegistrationStruct sv_struct_;

  private:

    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;

    ///
    /// Dependent MDFields
    ///
    std::vector<Teuchos::RCP<PHX::MDField<ScalarT> > > dependent_fields_;

    ///
    /// Evaluated MDFields
    ///
    std::vector<Teuchos::RCP<PHX::MDField<ScalarT> > > evaluated_fields_;

    ///
    /// Constitutive Model
    ///
    Teuchos::RCP<LCM::ConstitutiveModel<EvalT,Traits> > model_;

  };
    
}

#endif
