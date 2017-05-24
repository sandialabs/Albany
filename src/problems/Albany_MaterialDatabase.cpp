//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_MaterialDatabase.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#if defined(ALBANY_YAML)
#include "Teuchos_YamlParameterListHelpers.hpp"
#endif // ALBANY_YAML
#include "Teuchos_TestForException.hpp"
#include "Albany_Utils.hpp"

/* TODO: there is still a lot duplication in the code below.
   many of the safety checks could be consolidated into helper functions,
   and common behavior among Material, Element Block, Node Set, and Side Set too.
  */

Albany::MaterialDatabase::
MaterialDatabase(
    std::string const& input_file,
    Teuchos::RCP<Teuchos::Comm<int> const>& tcomm)
{
  if (input_file.length() == 0) {
    return;
  }

  if (tcomm->getRank() == 0) {
    std::cout << "Initializing material database from ";
    std::cout << input_file << std::endl;
  }

#if defined(ALBANY_YAML)
  auto input_extension = Albany::getFileExtension(input_file);
  if (input_extension == "yaml") {
    Teuchos::updateParametersFromYamlFileAndBroadcast(
        input_file, Teuchos::ptrFromRef(data_), *tcomm);
  } else {
    Teuchos::updateParametersFromXmlFileAndBroadcast(
        input_file, Teuchos::ptrFromRef(data_), *tcomm);
  }
#else
  Teuchos::updateParametersFromXmlFileAndBroadcast(
      input_file, Teuchos::ptrFromRef(data_), *tcomm);
#endif // ALBANY_YAML

  // Check for and set element block and materials sublists
  ALBANY_ASSERT(data_.isSublist("Materials"),
      "\nMaterial Database Error: Materials sublist required\n");
  ALBANY_ASSERT(data_.isSublist("ElementBlocks"),
      "\nMaterial Database Error: ElementBlocks sublist required\n");

  // The presence of NodeSet and SideSet info in the material database optional

  p_materials_list_ = &(data_.sublist("Materials"));
  p_eb_list_ = &(data_.sublist("ElementBlocks"));

  if(data_.isSublist("NodeSets")) {
    p_ns_list_ = &(data_.sublist("NodeSets"));
  }

  if(data_.isSublist("SideSets")) {
    p_ss_list_ = &(data_.sublist("SideSets"));
  }
}

Albany::MaterialDatabase::
~MaterialDatabase()
{
}

bool
Albany::MaterialDatabase::
isParam(std::string const& param_name)
{
  return data_.isParameter(param_name);
}

template<typename T> T
Albany::MaterialDatabase::
getParam(std::string const& param_name)
{
  return data_.get<T>(param_name);
}

template<typename T> T
Albany::MaterialDatabase::
getParam(std::string const& param_name, T def_value)
{
  return data_.get<T>(param_name, def_value);
}

bool
Albany::MaterialDatabase::
isMaterialParam(std::string const& material_name, std::string const& param_name)
{
  ALBANY_ASSERT(p_materials_list_,
      "\nMaterialDB Error! param required but no DB.\n");

  if (!p_materials_list_->isSublist(material_name)) return false;
  auto& sublist = p_materials_list_->sublist(material_name);
  return sublist.isParameter(param_name);
}

template<typename T> T
Albany::MaterialDatabase::
getMaterialParam(std::string const& material_name, std::string const& param_name)
{
  ALBANY_ASSERT(p_materials_list_,
      "\nMaterialDB Error! param required but no DB.\n");

  ALBANY_ASSERT(!material_name.empty(),
      "\nMaterialDB Error! Empty material name\n");

  ALBANY_ASSERT(p_materials_list_->isSublist(material_name),
		  "\nMaterialDB Error! Invalid material name " << material_name << '\n');

  auto& sublist = p_materials_list_->sublist(material_name);
  return sublist.get<T>(param_name);
}

template<typename T> T
Albany::MaterialDatabase::
getMaterialParam(std::string const& material_name, std::string const& param_name, T def_value)
{
  if (!p_materials_list_) return def_value;

  ALBANY_ASSERT(!material_name.empty(),
		  "\nMaterialDB Error! Empty material name\n");

  ALBANY_ASSERT(p_materials_list_->isSublist(material_name),
		  "\nMaterialDB Error! Invalid material name " << material_name << '\n');

  Teuchos::ParameterList& sublist = p_materials_list_->sublist(material_name);

  return sublist.get<T>(param_name, def_value);
}

bool
Albany::MaterialDatabase::
isElementBlockParam(std::string const& eb_name, std::string const& param_name)
{
  ALBANY_ASSERT(p_eb_list_,
		  "\nMaterialDB Error! param required but no DB.\n");

  auto new_name = translateDBSublistName(p_eb_list_, eb_name);
  if (new_name.empty()) return false;

  auto& sublist = p_eb_list_->sublist(new_name);

  if (sublist.isParameter(param_name)) return true;

  //check if related material exists (it always should)
  if(!sublist.isParameter("material")) return false;

  //Parameter not directly in element block sublist, so try related material
  auto material_name = sublist.get<std::string>("material");
  if(!p_materials_list_->isSublist(material_name)) return false;

  auto& mat_sublist = p_materials_list_->sublist(material_name);
  return mat_sublist.isParameter(param_name);
}

template<typename T> T
Albany::MaterialDatabase::
getElementBlockParam(std::string const& eb_name, std::string const& param_name)
{
  ALBANY_ASSERT(p_eb_list_,
		  "\nMaterialDB Error! param required but no DB.\n");

  ALBANY_ASSERT(!eb_name.empty(),
		  "\nMaterialDB Error! Empty element block name\n");

  auto new_name = translateDBSublistName(p_eb_list_, eb_name);

  ALBANY_ASSERT(!new_name.empty(),
      "\nMaterialDB Error! Invalid element block name \""
      << eb_name << "\".\n");

  // This call returns the sublist for the particular EB within the "ElementBlocks" list
  auto& sublist = p_eb_list_->sublist(new_name);

  if (sublist.isParameter(param_name)) {
    return sublist.get<T>(param_name);
  }

  //check if related material exists (it always should)
  ALBANY_ASSERT(sublist.isParameter("material"),
		  "\nMaterialDB Error! Param " << param_name
		  << " not found in " << eb_name << " list and there"
		  << " is no related material.\n");

  //Parameter not directly in element block sublist, so try related material
  auto material_name = sublist.get<std::string>("material");

  ALBANY_ASSERT(p_materials_list_->isSublist(material_name),
		     "\nMaterialDB Error! Param " << param_name
		     << " not found in " << eb_name << " list, and related"
		     << " material " << material_name << " is invalid.\n");

  auto& mat_sublist = p_materials_list_->sublist(material_name);
  ALBANY_ASSERT(mat_sublist.isParameter(param_name),
		     "\nMaterialDB Error! Param " << param_name
		     << " not found in " << eb_name << " list or related"
		     << " material " << material_name << " list.\n");
  return mat_sublist.get<T>(param_name);
}

template<typename T> T
Albany::MaterialDatabase::
getElementBlockParam(std::string const& eb_name, std::string const& param_name, T def_value)
{
  if (!p_eb_list_) return def_value;

  ALBANY_ASSERT(!eb_name.empty(),
		  "\nMaterialDB Error! Empty element block name\n");

  auto new_name = translateDBSublistName(p_eb_list_, eb_name);

  //check if element block exists - if not return default
  if (new_name.empty()) return def_value;

  auto& sublist = p_eb_list_->sublist(new_name);

  if (sublist.isParameter(param_name)) {
    return sublist.get<T>(param_name);
  }

  //check if related material exists - if not return default
  if(!sublist.isParameter("material")) return def_value;

  //Parameter not directly in element block sublist, so try related material
  auto& material_name = sublist.get<std::string>("material");

  ALBANY_ASSERT(p_materials_list_->isSublist(material_name),
		     "\nMaterialDB Error! Param " << param_name
		     << " not found in " << eb_name << " list, and related"
		     << " material " << material_name << " is invalid.\n");

  auto& mat_sublist = p_materials_list_->sublist(material_name);
  return mat_sublist.get<T>(param_name, def_value);
}

bool
Albany::MaterialDatabase::
isElementBlockSublist(std::string const& eb_name, std::string const& sublist_name)
{
  ALBANY_ASSERT(p_eb_list_,
		  "\nMaterialDB Error! param required but no DB.\n");

  auto new_name = translateDBSublistName(p_eb_list_, eb_name);

  if (new_name.empty()) return false;

  auto& sublist = p_eb_list_->sublist(new_name);

  if (sublist.isParameter(sublist_name)) return true;

  //check if related material exists (it always should)
  if (!sublist.isParameter("material")) return false;

  //Parameter not directly in element block sublist, so try related material
  auto material_name = sublist.get<std::string>("material");
  if (!p_materials_list_->isSublist(material_name)) return false;

  auto& mat_sublist = p_materials_list_->sublist(material_name);
  return mat_sublist.isSublist(sublist_name);
}

Teuchos::ParameterList&
Albany::MaterialDatabase::
getElementBlockSublist(std::string const& eb_name, std::string const& sublist_name)
{
  ALBANY_ASSERT(p_eb_list_,
		  "\nMaterialDB Error! param required but no DB.\n");

  ALBANY_ASSERT(!eb_name.empty(),
		  "\nMaterialDB Error! Empty element block name\n");

  auto new_name = translateDBSublistName(p_eb_list_, eb_name);

  ALBANY_ASSERT(!new_name.empty(),
      "\nMaterialDB Error! Invalid element block name \""
      << eb_name << "\".\n");

  // This call returns the sublist for the particular EB within the "ElementBlocks" list
  auto& sublist = p_eb_list_->sublist(new_name);

  if (sublist.isSublist(sublist_name)) {
    return sublist.sublist(sublist_name);
  }

  // Didn't find the requested sublist directly in the EB sublist.
  // Drill down to the material next.

  //check if related material exists (it always should)
  ALBANY_ASSERT(sublist.isParameter("material"),
		  "\nMaterialDB Error! Param " << sublist_name
		  << " not found in " << eb_name << " list and there"
		  << " is no related material.\n");

  //Parameter not directly in element block sublist, so try related material
  auto& material_name = sublist.get<std::string>("material");

  ALBANY_ASSERT(p_materials_list_->isSublist(material_name),
		     "\nMaterialDB Error! Param " << sublist_name
		     << " not found in " << eb_name << " list, and related"
		     << " material " << material_name << " is invalid.\n");

  auto& mat_sublist = p_materials_list_->sublist(material_name);

  // In case the entire material sublist is desired
  if (material_name == sublist_name) {
    return mat_sublist;
  }

  // Does the requested sublist appear in the material sublist?
  ALBANY_ASSERT(mat_sublist.isParameter(sublist_name),
		     "\nMaterialDB Error! Sublist " << sublist_name
		     << " not found in " << eb_name << " list or related"
		     << " material " << material_name << " list.\n");

  // If so, return the requested sublist
  return mat_sublist.sublist(sublist_name);
}

template<typename T> std::vector<T>
Albany::MaterialDatabase::
getAllMatchingParams(std::string const& param_name)
{
  std::vector<T> results;
  getAllMatchingParams_helper(param_name, results, data_);
  return results;
}

bool Albany::MaterialDatabase::
isNodeSetParam(std::string const& ns_name, std::string const& param_name)
{
  ALBANY_ASSERT(p_ns_list_,
      "\nMaterialDB Error! param required but no DB.\n");

  if(!p_ns_list_->isSublist(ns_name)) return false;
  auto& sublist = p_ns_list_->sublist(ns_name);
  return sublist.isParameter(param_name);
}

template<typename T> T
Albany::MaterialDatabase::
getNodeSetParam(std::string const& ns_name, std::string const& param_name)
{
  ALBANY_ASSERT(p_ns_list_,
      "\nMaterialDB Error! param required but no DB.\n");

  ALBANY_ASSERT(!ns_name.empty(),
      "\nMaterialDB Error! Empty node set name\n");

  ALBANY_ASSERT(p_ns_list_->isSublist(ns_name),
		  "\nMaterialDB Error! Invalid node set name " << ns_name << '\n');

  auto& sublist = p_ns_list_->sublist(ns_name);
  return sublist.get<T>(param_name);
}

template<typename T> T
Albany::MaterialDatabase::
getNodeSetParam(std::string const& ns_name, std::string const& param_name, T def_value)
{
  if (!p_ns_list_) return def_value;

  ALBANY_ASSERT(!ns_name.empty(),
      "\nMaterialDB Error! Empty node set name\n");

  ALBANY_ASSERT(p_ns_list_->isSublist(ns_name),
		  "\nMaterialDB Error! Invalid node set name " << ns_name << '\n');

  auto& sublist = p_ns_list_->sublist(ns_name);
  return sublist.get<T>(param_name, def_value);
}

bool Albany::MaterialDatabase::
isSideSetParam(std::string const& ss_name, std::string const& param_name)
{
  ALBANY_ASSERT(p_ss_list_,
      "\nMaterialDB Error! param required but no DB.\n");

  if(!p_ss_list_->isSublist(ss_name)) return false;
  auto& sublist = p_ss_list_->sublist(ss_name);
  return sublist.isParameter(param_name);
}

template<typename T> T
Albany::MaterialDatabase::
getSideSetParam(std::string const& ss_name, std::string const& param_name)
{
  ALBANY_ASSERT(p_ss_list_,
      "\nMaterialDB Error! param required but no DB.\n");

  ALBANY_ASSERT(!ss_name.empty(),
      "\nMaterialDB Error! Empty side set name\n");

  ALBANY_ASSERT(p_ss_list_->isSublist(ss_name),
		  "\nMaterialDB Error! Invalid side set name " << ss_name << '\n');

  auto& sublist = p_ss_list_->sublist(ss_name);
  return sublist.get<T>(param_name);
}

template<typename T> T
Albany::MaterialDatabase::
getSideSetParam(std::string const& ss_name, std::string const& param_name, T def_value)
{
  if (!p_ss_list_) return def_value;

  ALBANY_ASSERT(!ss_name.empty(),
      "\nMaterialDB Error! Empty side set name\n");

  ALBANY_ASSERT(p_ss_list_->isSublist(ss_name),
		  "\nMaterialDB Error! Invalid side set name " << ss_name << '\n');

  auto& sublist = p_ss_list_->sublist(ss_name);
  return sublist.get<T>(param_name, def_value);
}

template<typename T> void
Albany::MaterialDatabase::
getAllMatchingParams_helper(std::string const& param_name, std::vector<T>& results, Teuchos::ParameterList& list)
{
  Teuchos::ParameterList* list_type{nullptr};
  T* param_type{nullptr};
  for (auto it = list.begin(); it != list.end(); ++it) {
    if (it->second.isList()) {
      Teuchos::ParameterList& sublist = it->second.getValue(list_type);
      getAllMatchingParams_helper(param_name, results, sublist);
      continue;
    }
    if( it->second.isType<T>() && it->first == param_name )
      results.push_back( it->second.getValue(param_type) );
  }
}

std::string
Albany::MaterialDatabase::
translateDBSublistName(Teuchos::ParameterList* list, std::string const& listname)
{
  // NOTE: STK Ioss lowercases all names in the Exodus file,
  // including element block names. Let's lowercase the names
  // used for the search so users are not confounded when
  // they name the materials using mixed case when they enter
  // mixed case names in as element blocks in Cubit.
  for(auto i = list->begin(); i != list->end(); ++i ) {
    std::string name_i = list->name(i);
    const Teuchos::ParameterEntry &entry_i = list->entry(i);
    if(listname == name_i && entry_i.isList()){ // found it
      return list->name(i);
    }
    // Try to lowercase the list entry
    std::transform(name_i.begin(), name_i.end(), name_i.begin(), ::tolower);
    if(listname == name_i && entry_i.isList()){ // found it
      return list->name(i);
    }
  }
  return {}; // return string of length zero
}

Teuchos::RCP<Albany::MaterialDatabase>
Albany::createMaterialDatabase(
    Teuchos::RCP<Teuchos::ParameterList> const & params,
    Teuchos::RCP<Teuchos_Comm const> & commT)
{
  ALBANY_ASSERT(params->isType<std::string>("MaterialDB Filename"),
     "A required material database cannot be found.");

  auto filename = params->get<std::string>("MaterialDB Filename");
  return Teuchos::rcp(new Albany::MaterialDatabase(filename, commT));
}

// Explicit instantiation of functions above
#define ALBANY_INST(T) \
template T \
Albany::MaterialDatabase:: \
getParam<T>(std::string const& param_name); \
template T \
Albany::MaterialDatabase:: \
getParam<T>(std::string const& param_name, T def_val); \
template T \
Albany::MaterialDatabase:: \
getMaterialParam<T>(std::string const& material_name, std::string const& param_name); \
template T \
Albany::MaterialDatabase:: \
getMaterialParam<T>(std::string const& material_name, std::string const& param_name, T def_val); \
template T \
Albany::MaterialDatabase:: \
getElementBlockParam<T>(std::string const& material_name, std::string const& param_name); \
template T \
Albany::MaterialDatabase:: \
getElementBlockParam<T>(std::string const& material_name, std::string const& param_name, T def_val); \
template T \
Albany::MaterialDatabase:: \
getNodeSetParam<T>(std::string const& ns_name, std::string const& param_name); \
template T \
Albany::MaterialDatabase:: \
getNodeSetParam<T>(std::string const& ns_name, std::string const& param_name, T def_val); \
template T \
Albany::MaterialDatabase:: \
getSideSetParam<T>(std::string const& ss_name, std::string const& param_name); \
template T \
Albany::MaterialDatabase:: \
getSideSetParam<T>(std::string const& ss_name, std::string const& param_name, T def_val); \
template std::vector<T> \
Albany::MaterialDatabase:: \
getAllMatchingParams<T>(std::string const& param_name);

ALBANY_INST(double)
ALBANY_INST(int)
ALBANY_INST(bool)
ALBANY_INST(std::string)

#undef ALBANY_INST
