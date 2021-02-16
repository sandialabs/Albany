//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <stack>

#include "Teuchos_VerboseObject.hpp"

#include "PHAL_Setup.hpp"

namespace PHAL {

Setup::Setup() :
    _setupEvals(Teuchos::rcp(new StringSet())),
    _enableMemoization(false),
    _dep2EvalFields(Teuchos::rcp(new StringMap())),
    _savedFields(Teuchos::rcp(new StringSet())),
    _unsavedFields(Teuchos::rcp(new StringSet())),
    _rebootEvals(Teuchos::rcp(new StringSet())),
    _rebootFields(Teuchos::rcp(new StringSet())),
    _enableMemoizationForParams(false),
    _isParamsSetsSaved(false),
    _unsavedParams(Teuchos::rcp(new StringSet())),
    _unsavedParamsEvals(Teuchos::rcp(new StringSet())),
    _savedFieldsWOParams(Teuchos::rcp(new StringSet())),
    _unsavedFieldsWParams(Teuchos::rcp(new StringSet())) {
}

void Setup::init_problem_params(const Teuchos::RCP<Teuchos::ParameterList> problemParams) {
  _enableMemoization = problemParams->get<bool>("Use MDField Memoization", false);
  _enableMemoizationForParams = problemParams->get<bool>("Use MDField Memoization For Parameters", false);
  if (_enableMemoizationForParams) _enableMemoization = true;
}

void Setup::init_disc_params(const Teuchos::ParameterList& discParams) {
  _numLayers = discParams.get<int>("NumLayers");
}

void Setup::init_unsaved_param(const std::string& param) {
  if (_enableMemoizationForParams) {
    auto out = Teuchos::VerboseObjectBase::getDefaultOStream();
    *out << "Disabling memoization for " << param << " and its dependencies." << std::endl;
    _unsavedParams->insert(param);
    _unsavedParamsEvals = Teuchos::rcp(new StringSet(*_setupEvals));
  }
}

bool Setup::memoizer_active() const {
  return _enableMemoization;
}

bool Setup::memoizer_for_params_active() const {
  return _enableMemoizationForParams;
}

void Setup::reboot_memoizer() {
  if (_enableMemoization) {
    auto out = Teuchos::VerboseObjectBase::getDefaultOStream();
    *out << "Rebooting memoizer." << std::endl;
    _rebootEvals = Teuchos::rcp(new StringSet(*_setupEvals));
  }
}

void Setup::pre_eval() {
  if (_enableMemoizationForParams) {
    // If the MDFields haven't been computed yet, everything will be computed
    // anyways so let's skip memoization pre_eval()
    if (_setupEvals->empty()) return;

    // If a parameter has changed and the saved/unsaved string sets haven't
    // been created yet then create the sets
    if ((_unsavedParamsEvals->size() == _setupEvals->size()) && (!_isParamsSetsSaved)) {
      // Save param string sets for later use
      update_fields_with_unsaved_params();
      _isParamsSetsSaved = true;
    }
  }
}

void Setup::insert_eval(const std::string& eval) {
  _setupEvals->insert(eval);
}

bool Setup::contain_eval(const std::string& eval) const {
  return _setupEvals->count(eval) > 0;
}

void Setup::fill_field_dependencies(const std::vector<Teuchos::RCP<PHX::FieldTag>>& depFields,
    const std::vector<Teuchos::RCP<PHX::FieldTag>>& evalFields, const bool saved) {
  if (_enableMemoization) {
    // Fill dependencies as Dependency -> list of Evaluated
    for (const auto & depField: depFields)
      for (const auto & evalField: evalFields) {
        auto && dep2EvalFields = *_dep2EvalFields;
        dep2EvalFields[depField->identifier()].insert(evalField->identifier());
      }

    // Fill MDField lists based on whether it should be saved/unsaved
    auto fields = saved ? _savedFields : _unsavedFields;
    for (const auto & evalField: evalFields)
      fields->insert(evalField->identifier());
  }
}

void Setup::update_fields() {
  if (_enableMemoization) {
    update_fields(_savedFields, _unsavedFields);
    if (_enableMemoizationForParams)
      update_fields_with_unsaved_params();
  }
}

void Setup::check_fields(const std::vector<Teuchos::RCP<PHX::FieldTag>>& fields) const {
  if (_enableMemoization) {
    StringSet missingFields;
    for (const auto & field: fields) {
      const auto & fieldId = field->identifier();
      if (_savedFields->count(fieldId) == 0 && _unsavedFields->count(fieldId) == 0)
        missingFields.insert(fieldId);
    }
    if (!missingFields.empty()) {
      std::ostringstream os;
      for (const auto & missingField: missingFields) {
        os << missingField << "\n";
      }
      TEUCHOS_TEST_FOR_EXCEPTION(true,
          std::logic_error, "The following fields could not be found:\n" + os.str());
    }
  }
}

void Setup::print(std::ostream& os) const {
  os << "************* Phalanx Setup **************" << std::endl;
  os << "************ Evaluation Types ************" << std::endl;
  for (const auto & eval: *_setupEvals)
    os << "  " << eval << std::endl;
  os << std::endl;

  if (_enableMemoization) {
    os << "********** MDField Dependencies **********" << std::endl;
    for (const auto & depField: *_dep2EvalFields) {
      os << "  " << depField.first << " is a dependency of:" << std::endl;
      for (const auto & evalField: depField.second)
        os << "    " << evalField << std::endl;
    }
    os << std::endl;
    print_fields(os);
  }
  os << "******************************************" << std::endl;
}

void Setup::print_fields(std::ostream& os) const {
    os << "**************** MDFields ****************" << std::endl;
  if (_enableMemoization) print_fields(os, _savedFields, _unsavedFields);
  if (_enableMemoizationForParams) {
    os << "**** MDFields with parameter changes *****" << std::endl;
    print_fields(os, _savedFieldsWOParams, _unsavedFieldsWParams);
  }
}

Teuchos::RCP<const StringSet> Setup::get_saved_fields(const std::string& eval) const {
  // If reboot memoizer active, use empty set of fields
  if (_enableMemoization && _rebootEvals->count(eval) > 0) {
    _rebootEvals->erase(eval);
    return _rebootFields;
  }

  if (_enableMemoizationForParams && !_unsavedParams->empty()) {
    // Always load params if evaluation type is DistParamDeriv
    if (eval.find("DistParamDeriv") != std::string::npos)
      return _savedFieldsWOParams;

    // If a parameter has changed, use saved fields w/o parameter
    if (_unsavedParamsEvals->count(eval) > 0) {
      _unsavedParamsEvals->erase(eval);
      return _savedFieldsWOParams;
    }
  }

  return _savedFields;
}

void Setup::update_fields(Teuchos::RCP<StringSet> savedFields,
    Teuchos::RCP<StringSet> unsavedFields) {
  if (_enableMemoization) {
    // Start with list of unsaved fields
    std::stack<std::string> unsavedStack;
    for (const auto & unsavedField: *unsavedFields)
      unsavedStack.push(unsavedField);

    // Continue until all unsaved fields have been removed
    while(!unsavedStack.empty()) {
      const auto iter = _dep2EvalFields->find(unsavedStack.top());
      unsavedStack.pop();

      // If unsaved field is used to evaluate fields, add evaluated fields to list of unsaved
      if (iter != _dep2EvalFields->end()) {
        for (const auto & evalField: iter->second) {
          savedFields->erase(evalField);
          unsavedFields->insert(evalField);
          unsavedStack.push(evalField);
        }
      }
    }
  }
}

void Setup::update_fields_with_unsaved_params() {
  if (_enableMemoizationForParams && !_unsavedParams->empty()) {
    // Copy saved/unsaved MDFields
    _savedFieldsWOParams = Teuchos::rcp(new StringSet(*_savedFields));
    _unsavedFieldsWParams = Teuchos::rcp(new StringSet(*_unsavedFields));

    // If saved field has been changed, add field to list of unsaved
    for (const auto & savedField: *_savedFields)
      for (const auto & unsavedParam: *_unsavedParams)
        if (savedField.find(unsavedParam) != std::string::npos) {
          _unsavedFieldsWParams->insert(savedField);
          _savedFieldsWOParams->erase(savedField);
        }

    // Update list of saved/unsaved fields
    update_fields(_savedFieldsWOParams, _unsavedFieldsWParams);
  }
}

void Setup::print_fields(std::ostream& os, Teuchos::RCP<StringSet> savedFields,
    Teuchos::RCP<StringSet> unsavedFields) const {
  if (_enableMemoization) {
    os << "Saved fields:" << std::endl;
    for (const auto & savedField: *savedFields)
      os << "  " << savedField << std::endl;
    os << std::endl;

    os << "Unsaved fields:" << std::endl;
    for (const auto & unsavedField: *unsavedFields)
      os << "  " << unsavedField << std::endl;
    os << std::endl;
  }
}

int Setup::get_num_layers() const {
  return _numLayers;
}

} // namespace PHAL
