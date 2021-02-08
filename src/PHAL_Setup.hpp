//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_SETUP_HPP_
#define PHAL_SETUP_HPP_

#include <iostream>
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

  //! Pass discretization parameters into Setup to access during postRegistrationSetup
  void init_disc_params(const Teuchos::ParameterList discParams);

  //! Pass unsaved parameter into Setup to change unsaved/saved fields
  void init_unsaved_param(const std::string& param);

  //! Check if memoization is activated
  bool memoizer_active() const;

  //! Check if memoization for parameters is activated
  bool memoizer_for_params_active() const;

  //! Reboot memoizer (all fields are unsaved on next eval)
  void reboot_memoizer();

  //! Setup data before app evaluation functions are called
  void pre_eval();

  //! Insert evaluation type name
  void insert_eval(const std::string& eval);

  //! Determine if evaluation type name exists
  bool contain_eval(const std::string& eval) const;

  //! Store MDField identifiers in order to identify field dependencies in the FieldManager
  //! "saved" is used to specify whether an MDField should be saved for memoization
  void fill_field_dependencies(const std::vector<Teuchos::RCP<PHX::FieldTag>>& depFields,
      const std::vector<Teuchos::RCP<PHX::FieldTag>>& evalFields, const bool saved = true);

  //! Update list of _saved/_unsaved MDFields based on _unsaved MDFields and field dependencies
  void update_fields();

  //! Compare list of saved/unsaved MDFields to input
  //! (used to ensure all MDFields have been gathered by fill_field_dependencies())
  void check_fields(const std::vector<Teuchos::RCP<PHX::FieldTag>>& fields) const;

  //! Print Setup information
  void print(std::ostream& os) const;

  //! Print list of _saved/_unsaved MDFields
  void print_fields(std::ostream& os) const;

  //! Get list of saved MDFields
  Teuchos::RCP<const StringSet> get_saved_fields(const std::string& eval) const;

  //! Get number of layers in extruded mesh
  int get_num_layers() const;

private:
  //! Update list of saved/unsaved MDFields based on unsaved MDFields and field dependencies
  void update_fields(Teuchos::RCP<StringSet> savedFields, Teuchos::RCP<StringSet> unsavedFields);

  //! Update list of saved/unsaved MDFields with unsaved parameters
  void update_fields_with_unsaved_params();

  //! Print list of _saved/_unsaved MDFields
  void print_fields(std::ostream& os, Teuchos::RCP<StringSet> savedFields,
      Teuchos::RCP<StringSet> unsavedFields) const;

  //! Used to ensure postRegistrationSetup only occurs once
  const Teuchos::RCP<StringSet> _setupEvals;

  //! Data structures for general memoization
  bool _enableMemoization;
  const Teuchos::RCP<StringMap> _dep2EvalFields;
  const Teuchos::RCP<StringSet> _savedFields, _unsavedFields;

  //! Data structures for rebooting memoizer
  Teuchos::RCP<StringSet> _rebootEvals;
  const Teuchos::RCP<StringSet> _rebootFields;

  //! Data structures for memoization of parameters that change occasionally
  bool _enableMemoizationForParams, _isParamsSetsSaved;
  const Teuchos::RCP<StringSet> _unsavedParams;
  Teuchos::RCP<StringSet> _unsavedParamsEvals;
  Teuchos::RCP<StringSet> _savedFieldsWOParams, _unsavedFieldsWParams;

  //! Number of layers in extruded mesh
  int _numLayers;
};

} // namespace PHAL

#endif /* PHAL_SETUP_HPP_ */
