//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <locale>

#include "MaterialDatabase.h"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_TestForException.hpp"
#include "Albany_Utils.hpp"

LCM::MaterialDatabase::
MaterialDatabase(
    std::string const & input_file,
    Teuchos::RCP<Teuchos::Comm<int> const> & tcomm)
{
  if (input_file.length() == 0) {
    return;
  }

  if (tcomm->getRank() == 0) {
    std::cout << "Initializing material database from ";
    std::cout << input_file << std::endl;
  }

  Teuchos::updateParametersFromXmlFileAndBroadcast(
      input_file,
      Teuchos::ptrFromRef(data_),
      *tcomm);

  // Check for and set element block and materials sublists
  TEUCHOS_TEST_FOR_EXCEPTION(
      data_.isSublist("Materials") == false,
      Teuchos::Exceptions::InvalidParameter,
      std::endl
      << "Material Database Error: Materials sublist required" << std::endl);

  TEUCHOS_TEST_FOR_EXCEPTION(
      data_.isSublist("ElementBlocks") == false,
      Teuchos::Exceptions::InvalidParameter,
      std::endl
      << "Material Database Error: ElementBlocks sublist required" <<
      std::endl);

  p_materials_list_ = &(data_.sublist("Materials"));
  p_eb_list_ = &(data_.sublist("ElementBlocks"));

  return;
}

LCM::MaterialDatabase::
~MaterialDatabase()
{
  return;
}

template<typename T>
T
LCM::MaterialDatabase::
getParam(std::string const & param_name)
{
  return data_.get<T>(param_name);
}

template<typename T>
T
LCM::MaterialDatabase::
getParam(std::string const & param_name, T def_value)
{
  return data_.get<T>(param_name, def_value);
}

bool
LCM::MaterialDatabase::
isParam(std::string const & param_name)
{
  return data_.isParameter(param_name);
}

template<typename T>
T
LCM::MaterialDatabase::
getMaterialParam(
    std::string const & material_name,
    std::string const & param_name)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
      p_materials_list_ == nullptr,
      Teuchos::Exceptions::InvalidParameter,
      std::endl
      << "MaterialDB Error! param required but no DB." << std::endl);

  TEUCHOS_TEST_FOR_EXCEPTION(
      material_name.length() == 0,
      Teuchos::Exceptions::InvalidParameter,
      std::endl
      << "MaterialDB Error! Empty material name" << std::endl);

  TEUCHOS_TEST_FOR_EXCEPTION(
      p_materials_list_->isSublist(material_name) == false,
      Teuchos::Exceptions::InvalidParameter,
      std::endl
      << "MaterialDB Error! Invalid material name "
      << material_name << std::endl);

  Teuchos::ParameterList &
  sublist = p_materials_list_->sublist(material_name);

  return sublist.get<T>(param_name);
}

template<typename T>
T
LCM::MaterialDatabase::
getMaterialParam(
    std::string const & material_name,
    std::string const & param_name,
    T def_value)
{
  if (p_materials_list_ == nullptr) return def_value;

  TEUCHOS_TEST_FOR_EXCEPTION(
      material_name.length() == 0,
      Teuchos::Exceptions::InvalidParameter,
      std::endl
      << "MaterialDB Error! Empty material name" << std::endl);

  TEUCHOS_TEST_FOR_EXCEPTION(
      p_materials_list_->isSublist(material_name) == false,
      Teuchos::Exceptions::InvalidParameter,
      std::endl
      << "MaterialDB Error! Invalid material name "
      << material_name << std::endl);

  Teuchos::ParameterList &
  sublist = p_materials_list_->sublist(material_name);

  return sublist.get<T>(param_name, def_value);
}

bool
LCM::MaterialDatabase::
isMaterialParam(
    std::string const & material_name,
    std::string const & param_name)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
      p_materials_list_ == nullptr,
      Teuchos::Exceptions::InvalidParameter,
      std::endl
      << "MaterialDB Error! param required but no DB." << std::endl);

  if (p_materials_list_->isSublist(material_name) == false) return false;

  Teuchos::ParameterList &
  sublist = p_materials_list_->sublist(material_name);

  return sublist.isParameter(param_name);
}

template<typename T>
T
LCM::MaterialDatabase::
getElementBlockParam(
    std::string const & eb_name,
    std::string const & param_name)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
      p_eb_list_ == nullptr,
      Teuchos::Exceptions::InvalidParameter,
      std::endl
      << "MaterialDB Error! param required but no DB." << std::endl);

  TEUCHOS_TEST_FOR_EXCEPTION(
      eb_name.length() == 0,
      Teuchos::Exceptions::InvalidParameter,
      std::endl
      << "MaterialDB Error! Empty element block name" << std::endl);

  std::string const &
  new_name = translateDBSublistName(p_eb_list_, eb_name);

  TEUCHOS_TEST_FOR_EXCEPTION(
      new_name.length() == 0,
      Teuchos::Exceptions::InvalidParameter,
      std::endl
      << "MaterialDB Error! Invalid element block name \""
      << eb_name << "\"."<< std::endl);

  // This call returns the sublist for the particular EB within the "ElementBlocks" list
  Teuchos::ParameterList &
  sublist = p_eb_list_->sublist(new_name);

  if (sublist.isParameter(param_name) == true) {
    return sublist.get<T>(param_name);
  }

  //check if related material exists (it always should)
  TEUCHOS_TEST_FOR_EXCEPTION(
      sublist.isParameter("material") == false,
      Teuchos::Exceptions::InvalidParameter,
      std::endl
      << "MaterialDB Error! Param " << param_name
      << " not found in " << eb_name << " list and there"
      << " is no related material." << std::endl);

  //Parameter not directly in element block sublist, so try related material
  std::string const &
  material_name = sublist.get<std::string>("material");

  TEUCHOS_TEST_FOR_EXCEPTION(
      p_materials_list_->isSublist(material_name) == false,
      Teuchos::Exceptions::InvalidParameter,
      std::endl
      << "MaterialDB Error! Param " << param_name
      << " not found in " << eb_name << " list, and related"
      << " material " << material_name << "is invalid." << std::endl);

  Teuchos::ParameterList &
  mat_sublist = p_materials_list_->sublist(material_name);

  TEUCHOS_TEST_FOR_EXCEPTION(
      mat_sublist.isParameter(param_name) == false,
      Teuchos::Exceptions::InvalidParameter,
      std::endl
      << "MaterialDB Error! Param " << param_name
      << " not found in " << eb_name << " list or related"
      << " material " << material_name << "list." << std::endl);

  return mat_sublist.get<T>(param_name);
}

std::string
LCM::MaterialDatabase::
translateDBSublistName(
    Teuchos::ParameterList * p_list,
    std::string const & list_name)
{
  //
  // NOTE: STK Ioss lowercases all names in the Exodus file,
  // including element block names. Let's lowercase the names
  // used for the search so users are not confounded when
  // they name the materials using mixed case when they enter
  // mixed case names in as element blocks in Cubit.
  //
  Teuchos::ParameterList::ConstIterator
  i;

  for (i = p_list->begin(); i != p_list->end(); ++i) {

    std::string
    name = p_list->name(i);

    Teuchos::ParameterEntry const &
    entry = p_list->entry(i);

    if (list_name == name && entry.isList() == true) return p_list->name(i);


    // Try to lowercase the list entry
    std::transform(name.begin(), name.end(), name.begin(),
        [](int c) { return std::tolower(c); });

    if (list_name == name && entry.isList() == true) return p_list->name(i);

  }

  return ""; // return string of length zero

}

Teuchos::ParameterList &
LCM::MaterialDatabase::
getElementBlockSublist(
    std::string const & eb_name,
    std::string const & sublist_name)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
      p_eb_list_ == nullptr,
      Teuchos::Exceptions::InvalidParameter,
      std::endl
      << "MaterialDB Error! material sublist requested but no DB."
      << std::endl);

  TEUCHOS_TEST_FOR_EXCEPTION(
      eb_name.length() == 0,
      Teuchos::Exceptions::InvalidParameter,
      std::endl
      << "MaterialDB Error! Empty element block name." << std::endl);

  std::string const &
  new_name = translateDBSublistName(p_eb_list_, eb_name);

  TEUCHOS_TEST_FOR_EXCEPTION(
      new_name.length() == 0,
      Teuchos::Exceptions::InvalidParameter,
      std::endl
      << "MaterialDB Error! Invalid element block name \""
      << eb_name << "\"."<< std::endl);

  // Sublist for the particular EB within the "ElementBlocks" list
  Teuchos::ParameterList &
  sublist = p_eb_list_->sublist(new_name);

  if (sublist.isSublist(sublist_name) == true) {
    return sublist.sublist(sublist_name);
  }

  // Didn't find the requested sublist directly in the EB sublist.
  // Drill down to the material next.

  //check if related material exists (it always should)
  TEUCHOS_TEST_FOR_EXCEPTION(
      sublist.isParameter("material") == false,
      Teuchos::Exceptions::InvalidParameter,
      std::endl
      << "MaterialDB Error! Param " << sublist_name
      << " not found in " << eb_name << " list and there"
      << " is no related material." << std::endl);

  //Parameter not directly in element block sublist, so try related material
  std::string const &
  material_name = sublist.get<std::string>("material");

  TEUCHOS_TEST_FOR_EXCEPTION(
      p_materials_list_->isSublist(material_name) == false,
      Teuchos::Exceptions::InvalidParameter,
      std::endl
      << "MaterialDB Error! Param " << sublist_name
      << " not found in " << eb_name << " list, and related"
      << " material " << material_name << "is invalid." << std::endl);

  Teuchos::ParameterList &
  mat_sublist = p_materials_list_->sublist(material_name);

  // In case the entire material sublist is desired
  if (material_name == sublist_name) {
    return mat_sublist;
  }

  // Does the requested sublist appear in the material sublist?
  TEUCHOS_TEST_FOR_EXCEPTION(
      mat_sublist.isParameter(sublist_name) == false,
      Teuchos::Exceptions::InvalidParameter,
      std::endl
      << "MaterialDB Error! Param " << sublist_name
      << " not found in " << eb_name << " list or related"
      << " material " << material_name << "list." << std::endl);

  // If so, return the requested sublist
  return mat_sublist.sublist(sublist_name);
}

template<typename T>
T
LCM::MaterialDatabase::
getElementBlockParam(
    std::string const & eb_name,
    std::string const & param_name,
    T def_value)
{
  if (p_eb_list_ == nullptr) return def_value;

  TEUCHOS_TEST_FOR_EXCEPTION(
      eb_name.length() == 0,
      Teuchos::Exceptions::InvalidParameter,
      std::endl << "MaterialDB Error! Empty element block name" << std::endl);

  std::string const &
  new_name = translateDBSublistName(p_eb_list_, eb_name);

  //check if element block exists - if not return default
  if (new_name.length() == 0) return def_value;

  Teuchos::ParameterList &
  sublist = p_eb_list_->sublist(new_name);

  if (sublist.isParameter(param_name) == true) {
    return sublist.get<T>(param_name);
  }

  //check if related material exists - if not return default
  if (sublist.isParameter("material") == false) {
    return def_value;
  }

  //Parameter not directly in element block sublist, so try related material
  std::string const &
  material_name = sublist.get<std::string>("material");

  TEUCHOS_TEST_FOR_EXCEPTION(!p_materials_list_->isSublist(material_name),
      Teuchos::Exceptions::InvalidParameter, std::endl
      << "MaterialDB Error! Param " << param_name
      << " not found in " << eb_name << " list, and related"
      << " material " << material_name << "is invalid." << std::endl);

  Teuchos::ParameterList &
  mat_sublist = p_materials_list_->sublist(material_name);

  return mat_sublist.get<T>(param_name, def_value);
}

bool
LCM::MaterialDatabase::
isElementBlockParam(
    std::string const & eb_name,
    std::string const & param_name)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
      p_eb_list_ == NULL,
      Teuchos::Exceptions::InvalidParameter,
      std::endl << "MaterialDB Error! param required but no DB." << std::endl);

  std::string const &
  new_name = translateDBSublistName(p_eb_list_, eb_name);

  if (new_name.length() == 0) return false;

  Teuchos::ParameterList &
  sublist = p_eb_list_->sublist(new_name);

  if (sublist.isParameter(param_name) == true) return true;

  //check if related material exists (it always should)
  if (sublist.isParameter("material") == false) return false;

  //Parameter not directly in element block sublist, so try related material
  std::string const &
  material_name = sublist.get<std::string>("material");

  if (p_materials_list_->isSublist(material_name) == false) return false;

  Teuchos::ParameterList &
  mat_sublist = p_materials_list_->sublist(material_name);

  return mat_sublist.isParameter(param_name);
}

bool
LCM::MaterialDatabase::
isElementBlockSublist(
    std::string const & eb_name,
    std::string const & sublist_name)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
      p_eb_list_ == NULL,
      Teuchos::Exceptions::InvalidParameter,
      std::endl << "MaterialDB Error! param required but no DB." << std::endl);

  std::string const &
  new_name = translateDBSublistName(p_eb_list_, eb_name);

  if (new_name.length() == 0) return false;

  Teuchos::ParameterList &
  sublist = p_eb_list_->sublist(new_name);

  if (sublist.isParameter(sublist_name) == true) return true;

  //check if related material exists (it always should)
  if (sublist.isParameter("material") == false) return false;

  //Parameter not directly in element block sublist, so try related material
  std::string const &
  material_name = sublist.get<std::string>("material");

  if (p_materials_list_->isSublist(material_name) == false) return false;

  Teuchos::ParameterList &
  mat_sublist = p_materials_list_->sublist(material_name);

  return mat_sublist.isSublist(sublist_name);
}

template<typename T>
std::vector<T>
LCM::MaterialDatabase::
getAllMatchingParams(std::string const & param_name)
{
  std::vector<T>
  results;

  getAllMatchingParams_helper(param_name, results, data_);

  return results;
}

template<typename T>
void
LCM::MaterialDatabase::
getAllMatchingParams_helper(
    std::string const & param_name,
    std::vector<T> & results,
    Teuchos::ParameterList& list)
{
  Teuchos::ParameterList *
  list_type{nullptr};

  T *
  param_type{nullptr};

  Teuchos::ParameterList::ConstIterator
  it;

  for (it = list.begin(); it != list.end(); it++) {

    if (it->second.isList() == true) {

      Teuchos::ParameterList &
      sublist = it->second.getValue(list_type);

      getAllMatchingParams_helper(param_name, results, sublist);

      continue;
    }

    if (it->second.isType<T>() == true && it->first == param_name)
      results.push_back(it->second.getValue(param_type));
  }

  return;
}

// Explicit instantiation of functions above; otherwise they never
// seems to get instantiated by compiler

//double
template double
LCM::MaterialDatabase::
getParam<double>(
    std::string const & param_name);

template double
LCM::MaterialDatabase::
getParam<double>(
    std::string const & param_name,
    double def_val);

template double
LCM::MaterialDatabase::
getMaterialParam<double>(
    std::string const & material_name,
    std::string const & param_name);

template double
LCM::MaterialDatabase::
getMaterialParam<double>(
    std::string const & material_name,
    std::string const & param_name,
    double def_val);

template double
LCM::MaterialDatabase::
getElementBlockParam<double>(
    std::string const & material_name,
    std::string const & param_name);

template double
LCM::MaterialDatabase::
getElementBlockParam<double>(
    std::string const & material_name,
    std::string const & param_name,
    double def_val);

//int
template int
LCM::MaterialDatabase::
getParam<int>(
    std::string const & param_name);

template int
LCM::MaterialDatabase::
getParam<int>(
    std::string const & param_name,
    int def_val);

template int
LCM::MaterialDatabase::
getMaterialParam<int>(
    std::string const & material_name,
    std::string const & param_name);

template int
LCM::MaterialDatabase::
getMaterialParam<int>(
    std::string const & material_name,
    std::string const & param_name,
    int def_val);

template int
LCM::MaterialDatabase::
getElementBlockParam<int>(
    std::string const & material_name,
    std::string const & param_name);

template int
LCM::MaterialDatabase::
getElementBlockParam<int>(
    std::string const & material_name,
    std::string const & param_name,
    int def_val);

//bool
template bool
LCM::MaterialDatabase::
getParam<bool>(
    std::string const & param_name);

template bool
LCM::MaterialDatabase::
getParam<bool>(
    std::string const & param_name,
    bool def_val);

template bool
LCM::MaterialDatabase::
getMaterialParam<bool>(
    std::string const & material_name,
    std::string const & param_name);

template bool
LCM::MaterialDatabase::
getMaterialParam<bool>(
    std::string const & material_name,
    std::string const & param_name,
    bool def_val);

template bool
LCM::MaterialDatabase::
getElementBlockParam<bool>(
    std::string const & material_name,
    std::string const & param_name);

template bool
LCM::MaterialDatabase::
getElementBlockParam<bool>(
    std::string const & material_name,
    std::string const & param_name,
    bool def_val);

//string
template std::string
LCM::MaterialDatabase::
getParam<std::string>(
    std::string const & param_name);

template std::string
LCM::MaterialDatabase::
getParam<std::string>(
    std::string const & param_name,
    std::string def_val);

template std::string
LCM::MaterialDatabase::
getMaterialParam<std::string>(
    std::string const & material_name,
    std::string const & param_name);

template std::string
LCM::MaterialDatabase::
getMaterialParam<std::string>(
    std::string const & material_name,
    std::string const & param_name,
    std::string def_val);


template std::string
LCM::MaterialDatabase::
getElementBlockParam<std::string>(
    std::string const & material_name,
    std::string const & param_name);

template std::string
LCM::MaterialDatabase::
getElementBlockParam<std::string>(
    std::string const & material_name,
    std::string const & param_name,
    std::string def_val);

template std::vector<std::string>
LCM::MaterialDatabase::
getAllMatchingParams(
    std::string const & param_name);

template std::vector<bool>
LCM::MaterialDatabase::
getAllMatchingParams(
    std::string const & param_name);

//
//
//
Teuchos::RCP<LCM::MaterialDatabase>
LCM::createMaterialDatabase(
    Teuchos::RCP<Teuchos::ParameterList> const & params,
    Teuchos::RCP<Teuchos_Comm const> & commT)
{
  bool
  is_valid_material_db = params->isType<std::string>("MaterialDB Filename");

  TEUCHOS_TEST_FOR_EXCEPTION(
      is_valid_material_db == false,
      std::logic_error,
      "A required material database cannot be found.");

  std::string
  filename = params->get<std::string>("MaterialDB Filename");

  return Teuchos::rcp(new LCM::MaterialDatabase(filename, commT));
}

