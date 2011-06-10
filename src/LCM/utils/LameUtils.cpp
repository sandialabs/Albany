/********************************************************************\
*                Copyright (2011) Sandia Corporation                 *
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
*    Questions to David Littlewood, djlittl@sandia.gov               *
\********************************************************************/

#include "LameUtils.hpp"
#include "Teuchos_TestForException.hpp"
#include <string>
#include <algorithm>

namespace LameUtils {

Teuchos::RCP<lame::Material> constructLameMaterialModel(const Teuchos::ParameterList& lameMaterialParameters){

  // load the material properties into a lame::MatProps container.
  // LAME material properties must be of type int, double, or string.

  lame::MatProps props;

  for(Teuchos::ParameterList::ConstIterator it = lameMaterialParameters.begin() ; it != lameMaterialParameters.end() ; ++it){

    // The name should be all upper case with spaces replaced with underscores
    std::string name = lameMaterialParameters.name(it);
    std::transform(name.begin(), name.end(), name.begin(), toupper);
    std::replace(name.begin(), name.end(), ' ', '_');

    const Teuchos::ParameterEntry entry = lameMaterialParameters.entry(it);
    if(entry.isType<int>()){
      std::vector<int> propertyVector;
      propertyVector.push_back(Teuchos::getValue<int>(entry));
      props.insert(name, propertyVector);
      //std::cout << "LameUtils processing material property int " << name << " with value " << propertyVector[0] << std::endl;
    }
    else if(entry.isType<double>()){
      std::vector<double> propertyVector;
      propertyVector.push_back(Teuchos::getValue<double>(entry));
      props.insert(name, propertyVector);
      //std::cout << "LameUtils processing material property double " << name << " with value " << propertyVector[0] << std::endl;
    }
    else if(entry.isType<std::string>()){
      std::vector<std::string> propertyVector;
      propertyVector.push_back(Teuchos::getValue<std::string>(entry));
      props.insert(name, propertyVector);
      //std::cout << "LameUtils processing material property string " << name << " with value " << propertyVector[0] << std::endl;
    }
    else{
      TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter, " parameters for LAME material models must be of type double, int, or string.");
    }
  }

  // \todo Pass in the material model name and instantiate the right material model (with props)

  return Teuchos::rcp(new lame::Elastic(props));
}

}
