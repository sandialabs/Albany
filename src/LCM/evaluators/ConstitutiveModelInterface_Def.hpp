//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_RCP.hpp"
#include "Phalanx_DataLayout.hpp"

#include "models/NeohookeanModel.hpp"
#include "models/J2Model.hpp"

namespace LCM {

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  ConstitutiveModelInterface<EvalT, Traits>::
  ConstitutiveModelInterface(const Teuchos::ParameterList& p,
                             const Teuchos::RCP<Albany::Layouts>& dl)
  {
    this->initializeModel(p.get<Teuchos::ParameterList*>("Material Parameters"),dl);

    // construct the dependent fields
    std::map<std::string, Teuchos::RCP<PHX::DataLayout> > 
      dependent_map = model_->getDependentFieldMap();
    typename std::map<std::string, Teuchos::RCP<PHX::DataLayout> >::iterator miter;
    for ( miter = dependent_map.begin(); 
          miter != dependent_map.end(); 
          ++miter ) {
      Teuchos::RCP<PHX::MDField<ScalarT> > temp_field = 
        Teuchos::rcp( new PHX::MDField<ScalarT>(miter->first,miter->second) );
      std::cout << "\n";
      temp_field->print(std::cout);
      dependent_fields_.push_back(temp_field);
      std::cout << dependent_fields_.size() << std::endl;
    }

    // register dependent fields
    typename std::vector<Teuchos::RCP<PHX::MDField<ScalarT> > >::iterator viter;
    for ( viter = dependent_fields_.begin(); 
          viter != dependent_fields_.end(); 
          ++viter ) {
      this->addDependentField(**viter);
    }

    // construct the evaluated fields
    std::map<std::string, Teuchos::RCP<PHX::DataLayout> > 
      evalMap = model_->getEvaluatedFieldMap();
    for ( miter = evalMap.begin(); 
          miter != evalMap.end(); 
          ++miter ) {
      Teuchos::RCP<PHX::MDField<ScalarT> > temp_field = 
        Teuchos::rcp( new PHX::MDField<ScalarT>(miter->first,miter->second) );
      std::cout << "\n";
      temp_field->print(std::cout);
      evaluated_fields_.push_back(temp_field);
      std::cout << evaluated_fields_.size() << std::endl;
    }

    // register dependent fields
    for ( viter = evaluated_fields_.begin(); 
          viter != evaluated_fields_.end(); 
          ++viter ) {
      this->addEvaluatedField(**viter);
    }

    this->setName("ConstitutiveModelInterface"+PHX::TypeString<EvalT>::value);
  }

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void ConstitutiveModelInterface<EvalT, Traits>::
  postRegistrationSetup(typename Traits::SetupData d,
                        PHX::FieldManager<Traits>& fm)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(dependent_fields_.size() == 0, std::logic_error,
                               "something is wrong in the LCM::CMI");
    TEUCHOS_TEST_FOR_EXCEPTION(evaluated_fields_.size() == 0, std::logic_error,
                               "something is wrong in the LCM::CMI");
    // dependent fields
    typename std::vector<Teuchos::RCP<PHX::MDField<ScalarT> > >::iterator viter;
    for ( viter = dependent_fields_.begin(); 
          viter != dependent_fields_.end(); 
          ++viter ) {
      std::cout << "\n";
      (**viter).print(std::cout);
      this->utils.setFieldData(**viter,fm);
    }

    // evaluated fields
    for ( viter = evaluated_fields_.begin(); 
          viter != evaluated_fields_.end(); 
          ++viter ) {
      std::cout << "\n";
      (**viter).print(std::cout);
      this->utils.setFieldData(**viter,fm);
    }
  }

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void ConstitutiveModelInterface<EvalT, Traits>::
  evaluateFields(typename Traits::EvalData workset)
  {
    std::cout << "\n Calling the Constitutive model" << std::endl;
    model_->computeState(workset, dependent_fields_, evaluated_fields_);
  }

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void ConstitutiveModelInterface<EvalT, Traits>::
  fillStateVariableStruct(int state_var)
  {
    sv_struct_.name_               = model_->getStateVarName(state_var);
    sv_struct_.data_layout_        = model_->getStateVarLayout(state_var);
    sv_struct_.init_type_          = model_->getStateVarInitType(state_var);
    sv_struct_.init_value_         = model_->getStateVarInitValue(state_var);
    sv_struct_.register_old_state_ = model_->getStateVarOldStateFlag(state_var);
    sv_struct_.output_to_exodus_   = model_->getStateVarOutputFlag(state_var);
  }

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void ConstitutiveModelInterface<EvalT, Traits>::
  initializeModel(const Teuchos::ParameterList* p,
                  const Teuchos::RCP<Albany::Layouts>& dl)
  {
    std::string model_name = 
      p->sublist("Material Model").get<std::string>("Model Name");

    if ( model_name == "Neohookean" ) {
      this->model_ = Teuchos::rcp( new LCM::NeohookeanModel<EvalT,Traits>(p,dl) );
    } else if ( model_name == "J2" ) {
      this->model_ = Teuchos::rcp( new LCM::J2Model<EvalT,Traits>(p,dl) );
    } else {
      std::cout << "error!" << std::endl;
    }
  }

  //----------------------------------------------------------------------------
}

