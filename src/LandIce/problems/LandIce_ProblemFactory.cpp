//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "LandIce_ProblemFactory.hpp"

#include "LandIce_Stokes.hpp"
#include "LandIce_StokesFO.hpp"
#include "LandIce_Hydrology.hpp"
#include "LandIce_Enthalpy.hpp"
#include "LandIce_StokesFOThermoCoupled.hpp"
#include "LandIce_LaplacianSampling.hpp"
#include "LandIce_StokesFOThickness.hpp"

namespace LandIce
{

bool LandIceProblemFactory::provides (const std::string& key) const
{
  return key == "LandIce Stokes" ||
         key == "LandIce Stokes 3D" ||
         key == "LandIce Stokes 2D" ||
         key == "LandIce Stokes First Order 2D" ||
         key == "LandIce Stokes FO 2D" ||
         key == "LandIce Stokes First Order 2D XZ" ||
         key == "LandIce Stokes FO 2D XZ" ||
         key == "LandIce Stokes First Order 3D" ||
         key == "LandIce Stokes FO 3D" ||
         key == "LandIce Coupled FO H 3D" ||
         key == "LandIce Hydrology 2D" ||
         key == "LandIce Enthalpy 3D" ||
         key == "LandIce Stokes FO Thermo Coupled 3D" ||
         key == "LandIce Laplacian Sampling";
}

Albany::ProblemFactory::obj_ptr_type
LandIceProblemFactory::
create (const std::string& key,
        const Teuchos::RCP<const Teuchos_Comm>&     /* comm */,
        const Teuchos::RCP<Teuchos::ParameterList>& topLevelParams,
        const Teuchos::RCP<ParamLib>&               paramLib) const
{
  obj_ptr_type problem;

  auto problemParams = Teuchos::sublist(topLevelParams, "Problem", true);
  auto discParams = Teuchos::sublist(topLevelParams, "Discretization");


  if (key == "LandIce Stokes" || key == "LandIce Stokes 3D" ) {
    problem = Teuchos::rcp(new LandIce::Stokes(problemParams, paramLib, 3));
  } else if (key == "LandIce Stokes 2D" ) {
    problem = Teuchos::rcp(new LandIce::Stokes(problemParams, paramLib, 2));
  } else if (key == "LandIce Stokes First Order 2D" || key == "LandIce Stokes FO 2D" ||
             key == "LandIce Stokes First Order 2D XZ" || key == "LandIce Stokes FO 2D XZ") {
    problem = Teuchos::rcp(new LandIce::StokesFO(problemParams, discParams, paramLib, 2));
  } else if (key == "LandIce Stokes First Order 3D" || key == "LandIce Stokes FO 3D" ) {
    problem = Teuchos::rcp(new LandIce::StokesFO(problemParams, discParams, paramLib, 3));
  } else if (key == "LandIce Coupled FO H 3D" ) {
    problem = Teuchos::rcp(new LandIce::StokesFOThickness(problemParams, discParams, paramLib, 3));
  } else if (key == "LandIce Hydrology 2D") {
    problem = Teuchos::rcp(new LandIce::Hydrology(problemParams, discParams, paramLib, 2));
  } else if (key == "LandIce Enthalpy 3D") {
    problem = Teuchos::rcp(new LandIce::Enthalpy(problemParams, discParams, paramLib, 3));
  } else if (key == "LandIce Stokes FO Thermo Coupled 3D") {
    problem = Teuchos::rcp(new LandIce::StokesFOThermoCoupled(problemParams, discParams, paramLib, 3));
  } else if (key == "LandIce Laplacian Sampling") {
    problem = Teuchos::rcp(new LandIce::LaplacianSampling(problemParams, discParams, paramLib, 2));
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error,
      "Error! Unrecognized key '" + key + "' in LandIceProblemFactory.\n"
      "       Did you forget to check with 'provides(key)' first?\n");
  }

  return problem;
}

} // Namespace LandIce
