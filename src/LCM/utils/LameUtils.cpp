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

namespace LameUtils {

Teuchos::RCP<lame::Material> constructLameMaterialModel(const Teuchos::ParameterList& lameMaterialParameters){

  Teuchos::RCP<lame::MatProps> props = Teuchos::rcp(new lame::MatProps());
  std::vector<double> elasticModulusVector;
  double elasticModulusValue = lameMaterialParameters.get<double>("Elastic Modulus");
  elasticModulusVector.push_back(elasticModulusValue);
  props->insert(std::string("YOUNGS_MODULUS"), elasticModulusVector);
  std::vector<double> poissonsRatioVector;
  double poissonsRatioValue = lameMaterialParameters.get<double>("Poissons Ratio");
  poissonsRatioVector.push_back(poissonsRatioValue);
  props->insert(std::string("POISSONS_RATIO"), poissonsRatioVector);
  return Teuchos::rcp(new lame::Elastic(*props));

}

}
