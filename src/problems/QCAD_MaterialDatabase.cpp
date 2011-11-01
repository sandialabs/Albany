/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


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
    return;
  }

  const Albany_MPI_Comm& mcomm = Albany::getMpiCommFromEpetraComm(*ecomm);
  Teuchos::RCP<Teuchos::Comm<int> > tcomm = Albany::createTeuchosCommFromMpiComm(mcomm);

  std::cout << "Initializing material database from " << inputFile << std::endl;
  Teuchos::updateParametersFromXmlFileAndBroadcast(inputFile, &data_, *tcomm);
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
getElementBlockParam(const std::string& ebName, const std::string& paramName)
{
  TEUCHOS_TEST_FOR_EXCEPTION(pEBList_ == NULL, Teuchos::Exceptions::InvalidParameter,
		     std::endl << "MaterialDB Error! param required but no DB." << std::endl);

  TEUCHOS_TEST_FOR_EXCEPTION(ebName.length() == 0, Teuchos::Exceptions::InvalidParameter,
		     std::endl << "MaterialDB Error! Empty element block name" << std::endl);

  TEUCHOS_TEST_FOR_EXCEPTION(!pEBList_->isSublist(ebName), Teuchos::Exceptions::InvalidParameter,
		     std::endl << "MaterialDB Error! Invalid element block name " 
		     << ebName << std::endl);

  Teuchos::ParameterList& subList = pEBList_->sublist(ebName);

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

Teuchos::ParameterList&
QCAD::MaterialDatabase:: 
getElementBlockSublist(const std::string& ebName, const std::string& subListName)
{
  TEUCHOS_TEST_FOR_EXCEPTION(pEBList_ == NULL, Teuchos::Exceptions::InvalidParameter,
                    std::endl << "MaterialDB Error! material subList requested but no DB." << std::endl);

  TEUCHOS_TEST_FOR_EXCEPTION(ebName.length() == 0, Teuchos::Exceptions::InvalidParameter,
                    std::endl << "MaterialDB Error! Empty element block name" << std::endl);

  TEUCHOS_TEST_FOR_EXCEPTION(!pEBList_->isSublist(ebName), Teuchos::Exceptions::InvalidParameter,
                    std::endl << "MaterialDB Error! Invalid element block name " 
                    << ebName << std::endl);

  // This call returns the sublist for the particular EB within the "ElementBlocks" list
  Teuchos::ParameterList& subList = pEBList_->sublist(ebName);

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

  //check if element block exists - if not return default
  if(!pEBList_->isSublist(ebName)) return def_value;

  Teuchos::ParameterList& subList = pEBList_->sublist(ebName);

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

  if(!pEBList_->isSublist(ebName)) return false;
  Teuchos::ParameterList& subList = pEBList_->sublist(ebName);

  if(subList.isParameter(paramName)) return true;

  //check if related material exists (it always should)
  if(!subList.isParameter("material")) return false;

  //Parameter not directly in element block sublist, so try related material
  std::string materialName = subList.get<std::string>("material");
  if(!pMaterialsList_->isSublist(materialName)) return false;

  Teuchos::ParameterList& matSubList = pMaterialsList_->sublist(materialName);
  return matSubList.isParameter(paramName);
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
getElementBlockParam<std::string>(const std::string& materialName, const std::string& paramName);
template std::string QCAD::MaterialDatabase:: 
getElementBlockParam<std::string>(const std::string& materialName, const std::string& paramName, std::string def_val);

template std::vector<std::string> QCAD::MaterialDatabase:: 
getAllMatchingParams(const std::string& paramName);

