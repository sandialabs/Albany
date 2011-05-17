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


QCAD::MaterialDatabase::
MaterialDatabase( const std::string& inputFile)
  : data_("Material Parameters")
{
  pCurrentSublist_ = NULL;
  currentSublistName_ = "";

  std::cout << "Initializing material database from " << inputFile << std::endl;

  if(inputFile.length() > 0)
    Teuchos::updateParametersFromXmlFile(inputFile, &data_);
}

QCAD::MaterialDatabase::
~MaterialDatabase()
{
}

template<typename T> T 
QCAD::MaterialDatabase:: 
getMaterialParam(const std::string& materialName, const std::string& paramName)
{
  if(materialName.length() == 0) {
    TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
		       std::endl << "Error! Empty material name supplied to material db" 
		       << std::endl);
  }

  if(materialName != currentSublistName_) {
    if(!data_.isSublist(materialName)) {
      TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
			 std::endl << "Error! Invalid material name " << materialName << 
			 " supplied to material db"  << std::endl);
    }
    Teuchos::ParameterList& subList = data_.sublist(materialName);
    pCurrentSublist_ = &subList;
    currentSublistName_ = materialName;
  }

  return pCurrentSublist_->get<T>(paramName);
}


//explicit instantiation of function above -- otherwise it never seems to get intantiated by compiler
template double QCAD::MaterialDatabase:: 
getMaterialParam<double>(const std::string& materialName, const std::string& paramName);
