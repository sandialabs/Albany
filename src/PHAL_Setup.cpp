//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <iostream>
#include <stack>

#include "PHAL_Setup.hpp"

namespace PHAL {

Setup::Setup() :
    _setupEvals(Teuchos::rcp(new StringSet())),
    _dep2EvalFields(Teuchos::rcp(new StringMap())),
    _savedFields(Teuchos::rcp(new StringSet())),
    _unsavedFields(Teuchos::rcp(new StringSet())),
    _enableMemoization(false)
{
}

void Setup::init_problem_params(const Teuchos::RCP<Teuchos::ParameterList> problemParams)
{
  _enableMemoization = problemParams->get<bool>("Use MDField Memoization", false);
}

bool Setup::memoizer_active() const
{
  return _enableMemoization;
}

void Setup::insert_eval(const std::string& eval)
{
  _setupEvals->insert(eval);
}

bool Setup::contain_eval(const std::string& eval) const
{
  return _setupEvals->count(eval) > 0;
}

void Setup::fill_field_dependencies(const std::vector<Teuchos::RCP<PHX::FieldTag>>& depFields,
    const std::vector<Teuchos::RCP<PHX::FieldTag>>& evalFields, const bool saved)
{
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

void Setup::update_unsaved_fields()
{
  if (_enableMemoization) {
    // Start with list of unsaved fields
    std::stack<std::string> unsavedStack;
    for (const auto & unsavedField: *_unsavedFields)
      unsavedStack.push(unsavedField);

    // Continue until all unsaved fields have been removed
    while(!unsavedStack.empty()) {
      const auto iter = _dep2EvalFields->find(unsavedStack.top());
      unsavedStack.pop();

      // If unsaved field is used to evaluate fields, add evaluated fields to list of unsaved
      if (iter != _dep2EvalFields->end()) {
        for (const auto & evalField: iter->second) {
          _savedFields->erase(evalField);
          _unsavedFields->insert(evalField);
          unsavedStack.push(evalField);
        }
      }
    }
  }
}

Teuchos::RCP<const StringSet> Setup::get_saved_fields() const
{
  return _savedFields;
}

void Setup::check_fields(const std::vector<Teuchos::RCP<PHX::FieldTag>>& fields) const
{
  if (_enableMemoization) {
    for (const auto & field: fields) {
      const auto & fieldId = field->identifier();
      TEUCHOS_TEST_FOR_EXCEPTION(
          _savedFields->count(fieldId) == 0 &&
          _unsavedFields->count(fieldId) == 0,
          std::logic_error, fieldId + " could not be found!\n");
    }
  }
}

void Setup::print_field_dependencies() const
{
  if (_enableMemoization) {
    std::cout << "******************************************" << std::endl;
    std::cout << "Phalanx MDField dependencies: " << std::endl;
    std::cout << "Eval:";
    for (const auto & eval: *_setupEvals)
      std::cout << " " << eval;
    std::cout << std::endl;

    std::cout << "Saved fields:" << std::endl;
    for (const auto & savedField: *_savedFields)
      std::cout << "  " << savedField << std::endl;
    std::cout << std::endl;

    std::cout << "Unsaved fields:" << std::endl;
    for (const auto & unsavedField: *_unsavedFields)
      std::cout << "  " << unsavedField << std::endl;
    std::cout << std::endl;

    std::cout << "Field dependencies:" << std::endl;
    for (const auto & depField: *_dep2EvalFields) {
      std::cout << "  " << depField.first << " is a dependency of:" << std::endl;
      for (const auto & evalField: depField.second)
        std::cout << "    " << evalField << std::endl;
    }
    std::cout << "******************************************" << std::endl;
  }
}

} // namespace PHAL
