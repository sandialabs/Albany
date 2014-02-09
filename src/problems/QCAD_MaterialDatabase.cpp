//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "QCAD_MaterialDatabase.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_TestForException.hpp"
#include "Albany_Utils.hpp"

QCAD::MaterialDatabase::
MaterialDatabase( const std::string& inputFile,
		  const Teuchos::RCP<const Epetra_Comm>& ecomm)
  : data_("Material Parameters")
{
  if(inputFile.length() == 0) {
    pMaterialsList_ = NULL;
    pEBList_        = NULL;
    pNSList_        = NULL;
    pSSList_        = NULL;
    return;
  }

  const Albany_MPI_Comm& mcomm = Albany::getMpiCommFromEpetraComm(*ecomm);
  Teuchos::RCP<Teuchos::Comm<int> > tcomm = Albany::createTeuchosCommFromMpiComm(mcomm);

  if(ecomm->MyPID() == 0)
    std::cout << "Initializing material database from " << inputFile << std::endl;

  Teuchos::updateParametersFromXmlFileAndBroadcast(inputFile, Teuchos::ptrFromRef(data_), *tcomm);
//  Teuchos::updateParametersFromXmlFileAndBroadcast(inputFile, &data_, *tcomm);
  //Teuchos::updateParametersFromXmlFile(inputFile, &data_);
  
  //Check for and Set element block and materials sublists
  TEUCHOS_TEST_FOR_EXCEPTION(!data_.isSublist("Materials"), Teuchos::Exceptions::InvalidParameter,
	  std::endl << "Material Database Error: Materials sublist required" << std::endl);
  TEUCHOS_TEST_FOR_EXCEPTION(!data_.isSublist("ElementBlocks"), Teuchos::Exceptions::InvalidParameter,
	  std::endl << "Material Database Error: ElementBlocks sublist required" << std::endl);
  // Make the presence of NodeSet info in the material database optional
  //TEUCHOS_TEST_FOR_EXCEPTION(!data_.isSublist("NodeSets"), Teuchos::Exceptions::InvalidParameter,
	//  std::endl << "Material Database Error: NodeSets sublist required" << std::endl);

  pMaterialsList_ = &(data_.sublist("Materials"));
  pEBList_        = &(data_.sublist("ElementBlocks"));

  if(data_.isSublist("NodeSets"))
    pNSList_        = &(data_.sublist("NodeSets"));

  if(data_.isSublist("SideSets"))
    pSSList_        = &(data_.sublist("SideSets"));
}

QCAD::MaterialDatabase::
~MaterialDatabase()
{
}


template<typename T> T 
QCAD::MaterialDatabase:: 
getParam(const std::string& paramName)
{
  return data_.get<T>(paramName);
}

template<typename T> T 
QCAD::MaterialDatabase:: 
getParam(const std::string& paramName, T def_value)
{
  return data_.get<T>(paramName, def_value);
}

bool QCAD::MaterialDatabase:: 
isParam(const std::string& paramName)
{
  return data_.isParameter(paramName);
}



template<typename T> T 
QCAD::MaterialDatabase:: 
getMaterialParam(const std::string& materialName, const std::string& paramName)
{
  TEUCHOS_TEST_FOR_EXCEPTION(pMaterialsList_ == NULL, Teuchos::Exceptions::InvalidParameter,
		     std::endl << "MaterialDB Error! param required but no DB." << std::endl);

  TEUCHOS_TEST_FOR_EXCEPTION(materialName.length() == 0, Teuchos::Exceptions::InvalidParameter,
		     std::endl << "MaterialDB Error! Empty material name" << std::endl);

  TEUCHOS_TEST_FOR_EXCEPTION(!pMaterialsList_->isSublist(materialName), 
		     Teuchos::Exceptions::InvalidParameter, std::endl 
		     << "MaterialDB Error! Invalid material name " 
		     << materialName <<  std::endl);

  Teuchos::ParameterList& subList = pMaterialsList_->sublist(materialName);
  return subList.get<T>(paramName);
}

template<typename T> T 
QCAD::MaterialDatabase:: 
getMaterialParam(const std::string& materialName, const std::string& paramName, T def_value)
{
  if(pMaterialsList_ == NULL) return def_value;

  TEUCHOS_TEST_FOR_EXCEPTION(materialName.length() == 0, Teuchos::Exceptions::InvalidParameter,
		     std::endl << "MaterialDB Error! Empty material name" << std::endl);

  TEUCHOS_TEST_FOR_EXCEPTION(!pMaterialsList_->isSublist(materialName), 
		     Teuchos::Exceptions::InvalidParameter, std::endl 
		     << "MaterialDB Error! Invalid material name " 
		     << materialName <<  std::endl);

  Teuchos::ParameterList& subList = pMaterialsList_->sublist(materialName);
  return subList.get<T>(paramName, def_value);
}

bool QCAD::MaterialDatabase:: 
isMaterialParam(const std::string& materialName, const std::string& paramName)
{
  TEUCHOS_TEST_FOR_EXCEPTION(pMaterialsList_ == NULL, Teuchos::Exceptions::InvalidParameter,
		     std::endl << "MaterialDB Error! param required but no DB." << std::endl);
  if(!pMaterialsList_->isSublist(materialName)) return false;
  Teuchos::ParameterList& subList = pMaterialsList_->sublist(materialName);
  return subList.isParameter(paramName);
}



template<typename T> T 
QCAD::MaterialDatabase:: 
getNodeSetParam(const std::string& nsName, const std::string& paramName)
{
  TEUCHOS_TEST_FOR_EXCEPTION(pNSList_ == NULL, Teuchos::Exceptions::InvalidParameter,
		     std::endl << "MaterialDB Error! param required but no DB." << std::endl);

  TEUCHOS_TEST_FOR_EXCEPTION(nsName.length() == 0, Teuchos::Exceptions::InvalidParameter,
		     std::endl << "MaterialDB Error! Empty nodeset name" << std::endl);

  TEUCHOS_TEST_FOR_EXCEPTION(!pNSList_->isSublist(nsName), 
		     Teuchos::Exceptions::InvalidParameter, std::endl 
		     << "MaterialDB Error! Invalid nodeset name " 
		     << nsName <<  std::endl);

  Teuchos::ParameterList& subList = pNSList_->sublist(nsName);
  return subList.get<T>(paramName);
}

template<typename T> T 
QCAD::MaterialDatabase:: 
getNodeSetParam(const std::string& nsName, const std::string& paramName, T def_value)
{
  if(pNSList_ == NULL) return def_value;

  TEUCHOS_TEST_FOR_EXCEPTION(nsName.length() == 0, Teuchos::Exceptions::InvalidParameter,
		     std::endl << "MaterialDB Error! Empty nodeset name" << std::endl);

  TEUCHOS_TEST_FOR_EXCEPTION(!pNSList_->isSublist(nsName), 
		     Teuchos::Exceptions::InvalidParameter, std::endl 
		     << "MaterialDB Error! Invalid nodeset name " 
		     << nsName <<  std::endl);
  
  Teuchos::ParameterList& subList = pNSList_->sublist(nsName);
  return subList.get<T>(paramName, def_value);
}

bool QCAD::MaterialDatabase:: 
isNodeSetParam(const std::string& nsName, const std::string& paramName)
{
  TEUCHOS_TEST_FOR_EXCEPTION(pNSList_ == NULL, Teuchos::Exceptions::InvalidParameter,
		     std::endl << "MaterialDB Error! param required but no DB." << std::endl);

  if(!pNSList_->isSublist(nsName)) return false;
  Teuchos::ParameterList& subList = pNSList_->sublist(nsName);
  return subList.isParameter(paramName);
}



template<typename T> T 
QCAD::MaterialDatabase:: 
getSideSetParam(const std::string& ssName, const std::string& paramName)
{
  TEUCHOS_TEST_FOR_EXCEPTION(pSSList_ == NULL, Teuchos::Exceptions::InvalidParameter,
		     std::endl << "MaterialDB Error! param required but no DB." << std::endl);

  TEUCHOS_TEST_FOR_EXCEPTION(ssName.length() == 0, Teuchos::Exceptions::InvalidParameter,
		     std::endl << "MaterialDB Error! Empty sideset name" << std::endl);

  TEUCHOS_TEST_FOR_EXCEPTION(!pSSList_->isSublist(ssName), 
		     Teuchos::Exceptions::InvalidParameter, std::endl 
		     << "MaterialDB Error! Invalid sideset name " 
		     << ssName <<  std::endl);

  Teuchos::ParameterList& subList = pSSList_->sublist(ssName);
  return subList.get<T>(paramName);
}

template<typename T> T 
QCAD::MaterialDatabase:: 
getSideSetParam(const std::string& ssName, const std::string& paramName, T def_value)
{
  if(pSSList_ == NULL) return def_value;

  TEUCHOS_TEST_FOR_EXCEPTION(ssName.length() == 0, Teuchos::Exceptions::InvalidParameter,
		     std::endl << "MaterialDB Error! Empty sideset name" << std::endl);

  TEUCHOS_TEST_FOR_EXCEPTION(!pSSList_->isSublist(ssName), 
		     Teuchos::Exceptions::InvalidParameter, std::endl 
		     << "MaterialDB Error! Invalid sideset name " 
		     << ssName <<  std::endl);
  
  Teuchos::ParameterList& subList = pSSList_->sublist(ssName);
  return subList.get<T>(paramName, def_value);
}

bool QCAD::MaterialDatabase:: 
isSideSetParam(const std::string& ssName, const std::string& paramName)
{
  TEUCHOS_TEST_FOR_EXCEPTION(pSSList_ == NULL, Teuchos::Exceptions::InvalidParameter,
		     std::endl << "MaterialDB Error! param required but no DB." << std::endl);

  if(!pSSList_->isSublist(ssName)) return false;
  Teuchos::ParameterList& subList = pSSList_->sublist(ssName);
  return subList.isParameter(paramName);
}



template<typename T> T 
QCAD::MaterialDatabase:: 
getElementBlockParam(const std::string& ebName, const std::string& paramName)
{
  TEUCHOS_TEST_FOR_EXCEPTION(pEBList_ == NULL, Teuchos::Exceptions::InvalidParameter,
		     std::endl << "MaterialDB Error! param required but no DB." << std::endl);

  TEUCHOS_TEST_FOR_EXCEPTION(ebName.length() == 0, Teuchos::Exceptions::InvalidParameter,
		     std::endl << "MaterialDB Error! Empty element block name" << std::endl);

  std::string newname = translateDBSublistName(pEBList_, ebName);

  TEUCHOS_TEST_FOR_EXCEPTION(newname.length() == 0, Teuchos::Exceptions::InvalidParameter,
                    std::endl << "MaterialDB Error! Invalid element block name \"" 
                    << ebName << "\"."<< std::endl);

  // This call returns the sublist for the particular EB within the "ElementBlocks" list
  Teuchos::ParameterList& subList = pEBList_->sublist(newname);

  if( subList.isParameter(paramName) )
    return subList.get<T>(paramName);

  //check if related material exists (it always should)
  TEUCHOS_TEST_FOR_EXCEPTION(!subList.isParameter("material"), 
		     Teuchos::Exceptions::InvalidParameter, std::endl 
		     << "MaterialDB Error! Param " << paramName
		     << " not found in " << ebName << " list and there"
		     << " is no related material." << std::endl);

  //Parameter not directly in element block sublist, so try related material
  std::string materialName = subList.get<std::string>("material");
  TEUCHOS_TEST_FOR_EXCEPTION(!pMaterialsList_->isSublist(materialName), 
		     Teuchos::Exceptions::InvalidParameter, std::endl 
		     << "MaterialDB Error! Param " << paramName
		     << " not found in " << ebName << " list, and related"
		     << " material " << materialName << "is invalid." << std::endl);

  Teuchos::ParameterList& matSubList = pMaterialsList_->sublist(materialName);
  TEUCHOS_TEST_FOR_EXCEPTION(!matSubList.isParameter(paramName), 
		     Teuchos::Exceptions::InvalidParameter, std::endl 
		     << "MaterialDB Error! Param " << paramName
		     << " not found in " << ebName << " list or related"
		     << " material " << materialName << "list." << std::endl);
  return matSubList.get<T>(paramName);
}

std::string
QCAD::MaterialDatabase:: 
translateDBSublistName(Teuchos::ParameterList *list, const std::string& listname){
  /* 
    NOTE: STK Ioss lowercases all names in the Exodus file, including element block names. Lets
    lowercase the names used for the search so users are not confounded when they name the materials
    using mixed case when they enter mixed case names in as element blocks in CUbit.
  */

  std::string newname;

  for( Teuchos::ParameterList::ConstIterator i = list->begin(); i != list->end(); ++i ) {
    std::string name_i = list->name(i);
    const Teuchos::ParameterEntry &entry_i = list->entry(i);

    if(listname == name_i && entry_i.isList()){ // found it

      newname = list->name(i);

      return newname; 

    }

    // Try to lowercase the list entry

    std::transform(name_i.begin(), name_i.end(), name_i.begin(), (int (*)(int))std::tolower);

    if(listname == name_i && entry_i.isList()){ // found it

      newname = list->name(i);

      return newname; 

    }

  }

  return newname; // return string of length zero

}

Teuchos::ParameterList&
QCAD::MaterialDatabase:: 
getElementBlockSublist(const std::string& ebName, const std::string& subListName)
{
  TEUCHOS_TEST_FOR_EXCEPTION(pEBList_ == NULL, Teuchos::Exceptions::InvalidParameter,
                    std::endl << "MaterialDB Error! material subList requested but no DB." << std::endl);

  TEUCHOS_TEST_FOR_EXCEPTION(ebName.length() == 0, Teuchos::Exceptions::InvalidParameter,
                    std::endl << "MaterialDB Error! Empty element block name." << std::endl);

  std::string newname = translateDBSublistName(pEBList_, ebName);

  TEUCHOS_TEST_FOR_EXCEPTION(newname.length() == 0, Teuchos::Exceptions::InvalidParameter,
                    std::endl << "MaterialDB Error! Invalid element block name \"" 
                    << ebName << "\"."<< std::endl);

  // This call returns the sublist for the particular EB within the "ElementBlocks" list
  Teuchos::ParameterList& subList = pEBList_->sublist(newname);

  if( subList.isSublist(subListName) )
    return subList.sublist(subListName);

  // Didn't find the requested sublist directly in the EB sublist. Drill down to the material next.

  //check if related material exists (it always should)
  TEUCHOS_TEST_FOR_EXCEPTION(!subList.isParameter("material"), 
                    Teuchos::Exceptions::InvalidParameter, std::endl 
                    << "MaterialDB Error! Param " << subListName
                    << " not found in " << ebName << " list and there"
                    << " is no related material." << std::endl);

  //Parameter not directly in element block sublist, so try related material
  std::string materialName = subList.get<std::string>("material");

  TEUCHOS_TEST_FOR_EXCEPTION(!pMaterialsList_->isSublist(materialName), 
                    Teuchos::Exceptions::InvalidParameter, std::endl 
                    << "MaterialDB Error! Param " << subListName
                    << " not found in " << ebName << " list, and related"
                    << " material " << materialName << "is invalid." << std::endl);

  Teuchos::ParameterList& matSubList = pMaterialsList_->sublist(materialName);

  // In case the entire material subList is desired
  if (materialName == subListName) {
    return matSubList;
  }

  // Does the requested sublist appear in the material sublist?

  TEUCHOS_TEST_FOR_EXCEPTION(!matSubList.isParameter(subListName), 
                    Teuchos::Exceptions::InvalidParameter, std::endl 
                    << "MaterialDB Error! Param " << subListName
                    << " not found in " << ebName << " list or related"
                    << " material " << materialName << "list." << std::endl);

  // If so, return the requested sublist

  return matSubList.sublist(subListName);

}

template<typename T> T 
QCAD::MaterialDatabase:: 
getElementBlockParam(const std::string& ebName, const std::string& paramName, T def_value)
{
  if(pEBList_ == NULL) return def_value;

  TEUCHOS_TEST_FOR_EXCEPTION(ebName.length() == 0, Teuchos::Exceptions::InvalidParameter,
		     std::endl << "MaterialDB Error! Empty element block name" << std::endl);

  std::string newname = translateDBSublistName(pEBList_, ebName);

  //check if element block exists - if not return default
  if(newname.length() == 0) return def_value;

  Teuchos::ParameterList& subList = pEBList_->sublist(newname);

  if( subList.isParameter(paramName) )
    return subList.get<T>(paramName);

  //check if related material exists - if not return default
  if(!subList.isParameter("material")) return def_value;

  //Parameter not directly in element block sublist, so try related material
  std::string materialName = subList.get<std::string>("material");
  TEUCHOS_TEST_FOR_EXCEPTION(!pMaterialsList_->isSublist(materialName), 
		     Teuchos::Exceptions::InvalidParameter, std::endl 
		     << "MaterialDB Error! Param " << paramName
		     << " not found in " << ebName << " list, and related"
		     << " material " << materialName << "is invalid." << std::endl);

  Teuchos::ParameterList& matSubList = pMaterialsList_->sublist(materialName);
  return matSubList.get<T>(paramName, def_value);
}

bool QCAD::MaterialDatabase:: 
isElementBlockParam(const std::string& ebName, const std::string& paramName)
{
  TEUCHOS_TEST_FOR_EXCEPTION(pEBList_ == NULL, Teuchos::Exceptions::InvalidParameter,
		     std::endl << "MaterialDB Error! param required but no DB." << std::endl);

  std::string newname = translateDBSublistName(pEBList_, ebName);

  if(newname.length() == 0) return false;
  Teuchos::ParameterList& subList = pEBList_->sublist(newname);

  if(subList.isParameter(paramName)) return true;

  //check if related material exists (it always should)
  if(!subList.isParameter("material")) return false;

  //Parameter not directly in element block sublist, so try related material
  std::string materialName = subList.get<std::string>("material");
  if(!pMaterialsList_->isSublist(materialName)) return false;

  Teuchos::ParameterList& matSubList = pMaterialsList_->sublist(materialName);
  return matSubList.isParameter(paramName);
}

bool QCAD::MaterialDatabase:: 
isElementBlockSublist(const std::string& ebName, const std::string& subListName)
{
  TEUCHOS_TEST_FOR_EXCEPTION(pEBList_ == NULL, Teuchos::Exceptions::InvalidParameter,
		     std::endl << "MaterialDB Error! param required but no DB." << std::endl);

  std::string newname = translateDBSublistName(pEBList_, ebName);

  if(newname.length() == 0) return false;
  Teuchos::ParameterList& subList = pEBList_->sublist(newname);

  if(subList.isParameter(subListName)) return true;

  //check if related material exists (it always should)
  if(!subList.isParameter("material")) return false;

  //Parameter not directly in element block sublist, so try related material
  std::string materialName = subList.get<std::string>("material");
  if(!pMaterialsList_->isSublist(materialName)) return false;

  Teuchos::ParameterList& matSubList = pMaterialsList_->sublist(materialName);
  return matSubList.isSublist(subListName);
}



template<typename T> std::vector<T>
QCAD::MaterialDatabase:: 
getAllMatchingParams(const std::string& paramName)
{
  std::vector<T> results;
  getAllMatchingParams_helper(paramName, results, data_);
  return results;
}

template<typename T> void
QCAD::MaterialDatabase:: 
getAllMatchingParams_helper(const std::string& paramName, std::vector<T>& results, Teuchos::ParameterList& list)
{
  Teuchos::ParameterList::ConstIterator it;
  Teuchos::ParameterList* list_type = NULL;
  T* param_type = NULL;

  for(it = list.begin(); it != list.end(); it++) {
    if( it->second.isList() ) {
      Teuchos::ParameterList& subList = it->second.getValue(list_type);
      getAllMatchingParams_helper(paramName, results, subList);
      continue;
    }

    if( it->second.isType<T>() && it->first == paramName )
      results.push_back( it->second.getValue(param_type) );
  }
}



//explicit instantiation of functions above; otherwise they never
// seems to get intantiated by compiler

//double
template double QCAD::MaterialDatabase:: 
getParam<double>(const std::string& paramName);
template double QCAD::MaterialDatabase:: 
getParam<double>(const std::string& paramName, double def_val);

template double QCAD::MaterialDatabase:: 
getMaterialParam<double>(const std::string& materialName, const std::string& paramName);
template double QCAD::MaterialDatabase:: 
getMaterialParam<double>(const std::string& materialName, const std::string& paramName, double def_val);

template double QCAD::MaterialDatabase:: 
getNodeSetParam<double>(const std::string& nsName, const std::string& paramName);
template double QCAD::MaterialDatabase:: 
getNodeSetParam<double>(const std::string& nsName, const std::string& paramName, double def_val);

template double QCAD::MaterialDatabase:: 
getElementBlockParam<double>(const std::string& materialName, const std::string& paramName);
template double QCAD::MaterialDatabase:: 
getElementBlockParam<double>(const std::string& materialName, const std::string& paramName, double def_val);

//int
template int QCAD::MaterialDatabase:: 
getParam<int>(const std::string& paramName);
template int QCAD::MaterialDatabase:: 
getParam<int>(const std::string& paramName, int def_val);

template int QCAD::MaterialDatabase:: 
getMaterialParam<int>(const std::string& materialName, const std::string& paramName);
template int QCAD::MaterialDatabase:: 
getMaterialParam<int>(const std::string& materialName, const std::string& paramName, int def_val);

template int QCAD::MaterialDatabase:: 
getNodeSetParam<int>(const std::string& nsName, const std::string& paramName);
template int QCAD::MaterialDatabase:: 
getNodeSetParam<int>(const std::string& nsName, const std::string& paramName, int def_val);

template int QCAD::MaterialDatabase:: 
getElementBlockParam<int>(const std::string& materialName, const std::string& paramName);
template int QCAD::MaterialDatabase:: 
getElementBlockParam<int>(const std::string& materialName, const std::string& paramName, int def_val);


//bool
template bool QCAD::MaterialDatabase:: 
getParam<bool>(const std::string& paramName);
template bool QCAD::MaterialDatabase:: 
getParam<bool>(const std::string& paramName, bool def_val);

template bool QCAD::MaterialDatabase:: 
getMaterialParam<bool>(const std::string& materialName, const std::string& paramName);
template bool QCAD::MaterialDatabase:: 
getMaterialParam<bool>(const std::string& materialName, const std::string& paramName, bool def_val);

template bool QCAD::MaterialDatabase:: 
getNodeSetParam<bool>(const std::string& nsName, const std::string& paramName);
template bool QCAD::MaterialDatabase:: 
getNodeSetParam<bool>(const std::string& nsName, const std::string& paramName, bool def_val);

template bool QCAD::MaterialDatabase:: 
getSideSetParam<bool>(const std::string& ssName, const std::string& paramName);
template bool QCAD::MaterialDatabase:: 
getSideSetParam<bool>(const std::string& ssName, const std::string& paramName, bool def_val);

template bool QCAD::MaterialDatabase:: 
getElementBlockParam<bool>(const std::string& materialName, const std::string& paramName);
template bool QCAD::MaterialDatabase:: 
getElementBlockParam<bool>(const std::string& materialName, const std::string& paramName, bool def_val);

//string
template std::string QCAD::MaterialDatabase:: 
getParam<std::string>(const std::string& paramName);
template std::string QCAD::MaterialDatabase:: 
getParam<std::string>(const std::string& paramName, std::string def_val);

template std::string QCAD::MaterialDatabase:: 
getMaterialParam<std::string>(const std::string& materialName, const std::string& paramName);
template std::string QCAD::MaterialDatabase:: 
getMaterialParam<std::string>(const std::string& materialName, const std::string& paramName, std::string def_val);

template std::string QCAD::MaterialDatabase:: 
getNodeSetParam<std::string>(const std::string& nsName, const std::string& paramName);
template std::string QCAD::MaterialDatabase:: 
getNodeSetParam<std::string>(const std::string& nsName, const std::string& paramName, std::string def_val);

template std::string QCAD::MaterialDatabase:: 
getSideSetParam<std::string>(const std::string& ssName, const std::string& paramName);
template std::string QCAD::MaterialDatabase:: 
getSideSetParam<std::string>(const std::string& ssName, const std::string& paramName, std::string def_val);

template std::string QCAD::MaterialDatabase:: 
getElementBlockParam<std::string>(const std::string& materialName, const std::string& paramName);
template std::string QCAD::MaterialDatabase:: 
getElementBlockParam<std::string>(const std::string& materialName, const std::string& paramName, std::string def_val);

template std::vector<std::string> QCAD::MaterialDatabase:: 
getAllMatchingParams(const std::string& paramName);

