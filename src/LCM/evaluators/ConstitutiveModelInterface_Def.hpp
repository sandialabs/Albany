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
#include "models/AnisotropicHyperelasticDamageModel.hpp"
#include "models/GursonModel.hpp"

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
      dep_fields_map_.insert( std::make_pair(miter->first,temp_field) );
    }

    // register dependent fields
    typename std::map<std::string,Teuchos::RCP<PHX::MDField<ScalarT> > >::iterator it;
    for ( it = dep_fields_map_.begin(); 
          it != dep_fields_map_.end(); 
          ++it ) {
      this->addDependentField(*(it->second));
    }

    // construct the evaluated fields
    std::map<std::string, Teuchos::RCP<PHX::DataLayout> > 
      eval_map = model_->getEvaluatedFieldMap();
    for ( miter = eval_map.begin(); 
          miter != eval_map.end(); 
          ++miter ) {
      Teuchos::RCP<PHX::MDField<ScalarT> > temp_field = 
        Teuchos::rcp( new PHX::MDField<ScalarT>(miter->first,miter->second) );
      eval_fields_map_.insert( std::make_pair(miter->first,temp_field) );
    }

    // register dependent fields
    for ( it = eval_fields_map_.begin(); 
          it != eval_fields_map_.end(); 
          ++it ) {
      this->addEvaluatedField(*(it->second));
    }

    this->setName("ConstitutiveModelInterface"+PHX::TypeString<EvalT>::value);
  }

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void ConstitutiveModelInterface<EvalT, Traits>::
  postRegistrationSetup(typename Traits::SetupData d,
                        PHX::FieldManager<Traits>& fm)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(dep_fields_map_.size() == 0, std::logic_error,
                               "something is wrong in the LCM::CMI");
    TEUCHOS_TEST_FOR_EXCEPTION(eval_fields_map_.size() == 0, std::logic_error,
                               "something is wrong in the LCM::CMI");
    // dependent fields
    typename std::map<std::string,Teuchos::RCP<PHX::MDField<ScalarT> > >::iterator it;
    for ( it = dep_fields_map_.begin(); 
          it != dep_fields_map_.end(); 
          ++it ) {
      this->utils.setFieldData(*(it->second),fm);
    }

    // evaluated fields
    for ( it = eval_fields_map_.begin(); 
          it != eval_fields_map_.end(); 
          ++it ) {
      this->utils.setFieldData(*(it->second),fm);
    }
  }

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void ConstitutiveModelInterface<EvalT, Traits>::
  evaluateFields(typename Traits::EvalData workset)
  {
    model_->computeState(workset, dep_fields_map_, eval_fields_map_);
  }

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void ConstitutiveModelInterface<EvalT, Traits>::
  fillStateVariableStruct(int state_var)
  {
    sv_struct_.name               = model_->getStateVarName(state_var);
    sv_struct_.data_layout        = model_->getStateVarLayout(state_var);
    sv_struct_.init_type          = model_->getStateVarInitType(state_var);
    sv_struct_.init_value         = model_->getStateVarInitValue(state_var);
    sv_struct_.register_old_state = model_->getStateVarOldStateFlag(state_var);
    sv_struct_.output_to_exodus   = model_->getStateVarOutputFlag(state_var);
  }

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void ConstitutiveModelInterface<EvalT, Traits>::
  initializeModel(Teuchos::ParameterList* p,
                  const Teuchos::RCP<Albany::Layouts>& dl)
  {
    std::string model_name = 
      p->sublist("Material Model").get<std::string>("Model Name");

    if ( model_name == "Neohookean" ) {
      this->model_ = Teuchos::rcp( new LCM::NeohookeanModel<EvalT,Traits>(p,dl) );
    } else if ( model_name == "J2" ) {
      this->model_ = Teuchos::rcp( new LCM::J2Model<EvalT,Traits>(p,dl) );
    } else if ( model_name == "AHD" ) {
      this->model_ = Teuchos::rcp( new LCM::AnisotropicHyperelasticDamageModel<EvalT,Traits>(p,dl) );
    } else if ( model_name == "Gurson" ) {
      this->model_ = Teuchos::rcp( new LCM::GursonModel<EvalT,Traits>(p,dl) );
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION(true, 
                                 std::logic_error, 
                                 "Undefined material model name");
    }
  }

  //----------------------------------------------------------------------------
}

