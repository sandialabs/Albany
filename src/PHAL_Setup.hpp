//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_SETUP_HPP_
#define PHAL_SETUP_HPP_

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Phalanx_FieldTag.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"

namespace PHAL {

typedef std::unordered_set<std::string> StringSet;
typedef std::unordered_map<std::string, StringSet> StringMap;

//! PHAL::Setup is used to pass application level data into phalanx evaluators during
//! postRegistrationSetup
class Setup {
public:

  //! Constructor - kept clean to keep setup as optional
  Setup();

  //! Pass problem parameters into Setup to access during postRegistrationSetup
  void init_problem_params(const Teuchos::RCP<Teuchos::ParameterList> problemParams);

  //! Check if memoization is activated
  bool memoizer_active() const;

  //! Insert Eval (e.g. Residual, Jacobian)
  void insert_eval(const std::string& eval);

  //! Determine if Eval (e.g. Residual, Jacobian) exists
  bool contain_eval(const std::string& eval) const;

  //! Store MDField identifiers in order to identify field dependencies in the FieldManager
  //! "saved" is used to specify whether an MDField should be saved for memoization
  void fill_field_dependencies(const std::vector<Teuchos::RCP<PHX::FieldTag>>& depFields,
      const std::vector<Teuchos::RCP<PHX::FieldTag>>& evalFields, const bool saved = true);

  //! Update list of saved/unsaved MDFields based on field dependencies
  void update_unsaved_fields();

  //! Get list of saved MDFields
  Teuchos::RCP<const StringSet> get_saved_fields() const;

  //! Compare list of saved/unsaved MDFields to input
  //! (used to ensure all MDFields have been gathered by fill_field_dependencies())
  void check_fields(const std::vector<Teuchos::RCP<PHX::FieldTag>>& fields) const;

  //! Print all MDField lists for debug
  void print_field_dependencies() const;

private:
  const Teuchos::RCP<StringSet> _setupEvals;
  const Teuchos::RCP<StringMap> _dep2EvalFields;
  const Teuchos::RCP<StringSet> _savedFields, _unsavedFields;
  bool _enableMemoization;
};

} // namespace PHAL

#endif /* PHAL_SETUP_HPP_ */
