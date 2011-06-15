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

// LAME material models
#include "models/Elastic.h"
#include "models/NeoHookean.h"
#include "models/ElasticFracture.h"
#include "models/ElasticPlastic.h"
#include "models/EPPowerLaw.h"
#include "models/DuctileFracture.h"
#include "models/JohnsonCook.h"
#include "models/PowerLawCreep.h"
#include "models/FoamPlasticity.h"
#include "models/LowDensityFoam.h"
#include "models/StiffElastic.h"
#include "models/FeFp.h"
#include "models/CrystalPlasticity.h"

namespace LameUtils {

Teuchos::RCP<lame::Material> constructLameMaterialModel(const std::string& lameMaterialModelName,
                                                        const Teuchos::ParameterList& lameMaterialParameters){

  // load the material properties into a lame::MatProps container.
  // LAME material properties must be of type int, double, or string.

  // Strings should be all upper case with spaces replaced with underscores
  std::string materialModelName = lameMaterialModelName;
  std::transform(materialModelName.begin(), materialModelName.end(), materialModelName.begin(), toupper);
  std::replace(materialModelName.begin(), materialModelName.end(), ' ', '_');

  lame::MatProps props;

  for(Teuchos::ParameterList::ConstIterator it = lameMaterialParameters.begin() ; it != lameMaterialParameters.end() ; ++it){

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
      TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter, " parameters for LAME material models must be of type double, int, or string.\n");
    }
  }

  //std::cout << "Creating material model " << materialModelName << std::endl;

  Teuchos::RCP<lame::Material> materialModel;

  if(materialModelName == "ELASTIC")
    materialModel = Teuchos::rcp(new lame::Elastic(props));
  else if(materialModelName == "NEO_HOOKEAN")
    materialModel = Teuchos::rcp(new lame::NeoHookean(props));
  else if(materialModelName == "ELASTIC_FRACTURE")
    materialModel = Teuchos::rcp(new lame::ElasticFracture(props));
  else if(materialModelName == "ELASTIC_PLASTIC")
    materialModel = Teuchos::rcp(new lame::ElasticPlastic(props));
  else if(materialModelName == "EP_POWER_HARD")
    materialModel = Teuchos::rcp(new lame::EPPowerLaw(props));
  else if(materialModelName == "DUCTILE_FRACTURE")
    materialModel = Teuchos::rcp(new lame::DuctileFracture(props));
  else if(materialModelName == "JOHNSON_COOK")
    materialModel = Teuchos::rcp(new lame::JohnsonCook(props));
  else if(materialModelName == "POWER_LAW_CREEP")
    materialModel = Teuchos::rcp(new lame::PowerLawCreep(props));
  else if(materialModelName == "FOAM_PLASTICITY")
    materialModel = Teuchos::rcp(new lame::FoamPlasticity(props));
  else if(materialModelName == "LOW_DENSITY_FOAM")
    materialModel = Teuchos::rcp(new lame::LowDensityFoam(props));
  else if(materialModelName == "STIFF_ELASTIC")
    materialModel = Teuchos::rcp(new lame::StiffElastic(props));
  else if(materialModelName == "FEFP")
    materialModel = Teuchos::rcp(new lame::FeFp(props));
  else if(materialModelName == "CRYSTAL_PLASTICITY")
    materialModel = Teuchos::rcp(new lame::CrystalPlasticity(props));
  else
    TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter, " unsupported LAME material model: " + lameMaterialModelName + " (" + materialModelName + ")\n");

  return materialModel;
}

}
