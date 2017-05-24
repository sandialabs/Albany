//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_MATERIALDATABASE_HPP
#define ALBANY_MATERIALDATABASE_HPP

#include "Teuchos_ParameterList.hpp"
#include "Albany_Utils.hpp"

namespace Albany {

/*!
 * \brief Centralized collection of material parameters
 */
class MaterialDatabase
{
public:

  //! Default constructor
  MaterialDatabase(
      std::string const& input_file,
      Teuchos::RCP<Teuchos::Comm<int> const>& tcomm);

  //! Destructor
  ~MaterialDatabase();

  //! No copying
  MaterialDatabase(MaterialDatabase const &) = delete;

  //! No copying
  MaterialDatabase & operator=(MaterialDatabase const &) = delete;

  //! Get a parameter / check existence
  bool isParam(std::string const& param_name);

  template<typename T>
  T getParam(std::string const& param_name);

  template<typename T>
  T getParam(std::string const& param_name, T def_val);

  //! Get a parameter for a particular material
  bool isMaterialParam(std::string const& material_name, std::string const& param_name);

  template<typename T>
  T getMaterialParam(std::string const& material_name, std::string const& param_name);

  template<typename T>
  T getMaterialParam(std::string const& material_name, std::string const& param_name, T def_val);

  //! Get a parameter for a particular element block (or assoc. material if param_name is not in element bloc)
  bool isElementBlockParam(std::string const& eb_name, std::string const& param_name);

  template<typename T>
  T getElementBlockParam(std::string const& eb_name, std::string const& param_name);

  template<typename T>
  T getElementBlockParam(std::string const& eb_name, std::string const& param_name, T def_val);

  //! Get a sublist from a particular element block
  bool isElementBlockSublist(std::string const& eb_name, std::string const& sublist_name);

  Teuchos::ParameterList&
  getElementBlockSublist(std::string const& eb_name, std::string const& sublist_name);

  //! Get a vector of the value of all parameters in the entire list with name == param_name
  template<typename T>
  std::vector<T> getAllMatchingParams(std::string const& param_name);

  //! Get a parameter for a particular node set
  bool isNodeSetParam(std::string const& material_name, std::string const& param_name);

  template<typename T>
  T getNodeSetParam(std::string const& material_name, std::string const& param_name);

  template<typename T>
  T getNodeSetParam(std::string const& material_name, std::string const& param_name, T def_val);

  //! Get a parameter for a particular side set
  bool isSideSetParam(std::string const& material_name, std::string const& param_name);

  template<typename T>
  T getSideSetParam(std::string const& material_name, std::string const& param_name);

  template<typename T>
  T getSideSetParam(std::string const& material_name, std::string const& param_name, T def_val);

private:
  template<typename T>
  void getAllMatchingParams_helper(std::string const& param_name,
           std::vector<T>& results, Teuchos::ParameterList& param_list);

  std::string translateDBSublistName(Teuchos::ParameterList* param_list, std::string const& list_name);

private:

  //! Encapsulated parameter list which holds all the data
  Teuchos::ParameterList data_{"Material Parameters"};
  Teuchos::ParameterList* p_materials_list_{nullptr};
  Teuchos::ParameterList* p_eb_list_{nullptr};
  Teuchos::ParameterList* p_ns_list_{nullptr};
  Teuchos::ParameterList* p_ss_list_{nullptr};
};

Teuchos::RCP<Albany::MaterialDatabase>
createMaterialDatabase(
    Teuchos::RCP<Teuchos::ParameterList> const& params,
    Teuchos::RCP<Teuchos_Comm const>& commT);

}  // namespace Albany

// Explicit instantiation declarations of functions above
#define ALBANY_INST_DECL(T) \
extern template T \
Albany::MaterialDatabase:: \
getParam<T>(std::string const& param_name); \
extern template T \
Albany::MaterialDatabase:: \
getParam<T>(std::string const& param_name, T def_val); \
extern template T \
Albany::MaterialDatabase:: \
getMaterialParam<T>(std::string const& material_name, std::string const& param_name); \
extern template T \
Albany::MaterialDatabase:: \
getMaterialParam<T>(std::string const& material_name, std::string const& param_name, T def_val); \
extern template T \
Albany::MaterialDatabase:: \
getElementBlockParam<T>(std::string const& material_name, std::string const& param_name); \
extern template T \
Albany::MaterialDatabase:: \
getElementBlockParam<T>(std::string const& material_name, std::string const& param_name, T def_val); \
extern template T \
Albany::MaterialDatabase:: \
getNodeSetParam<T>(std::string const& ns_name, std::string const& param_name); \
extern template T \
Albany::MaterialDatabase:: \
getNodeSetParam<T>(std::string const& ns_name, std::string const& param_name, T def_val); \
extern template T \
Albany::MaterialDatabase:: \
getSideSetParam<T>(std::string const& ss_name, std::string const& param_name); \
extern template T \
Albany::MaterialDatabase:: \
getSideSetParam<T>(std::string const& ss_name, std::string const& param_name, T def_val); \
extern template std::vector<T> \
Albany::MaterialDatabase:: \
getAllMatchingParams<T>(std::string const& param_name);

ALBANY_INST_DECL(double)
ALBANY_INST_DECL(int)
ALBANY_INST_DECL(bool)
ALBANY_INST_DECL(std::string)

#undef ALBANY_INST_DECL

#endif
