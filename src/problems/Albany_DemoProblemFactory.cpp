//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_DemoProblemFactory.hpp"

#include "Albany_CahnHillProblem.hpp"
#include "Albany_Helmholtz2DProblem.hpp"
#include "Albany_NavierStokes.hpp"
#include "Albany_LinComprNSProblem.hpp"
#include "Albany_AdvDiffProblem.hpp"
#include "Albany_ReactDiffSystem.hpp"
#include "Albany_ComprNSProblem.hpp"
#include "Albany_ODEProblem.hpp"
#include "Albany_PNPProblem.hpp"
#include "Albany_ThermoElectrostaticsProblem.hpp"
#include "Albany_ThermalProblem.hpp"

namespace Albany
{

std::string getName(std::string const& key)
{
  if (key.size() < 3) return key;
  return key.substr(0, key.size() - 3);
}
// In "Thermal 3D", extract 3.
int getNumDim(std::string const& key)
{
  if (key.size() < 3) return -1;
  return static_cast<int>(key[key.size() - 2] - '0');
} 


bool DemoProblemFactory::provides (const std::string& key) const
{
  return key == "CahnHill 2D" ||
         key == "ODE" ||
         key == "Helmholtz 2D" ||
         key == "NavierStokes 1D" ||
         key == "NavierStokes 2D" ||
         key == "NavierStokes 3D" ||
         key == "LinComprNS 1D" ||
         key == "AdvDiff 1D" ||
         key == "AdvDiff 2D" ||
         key=="Reaction-Diffusion System 3D" ||
         key == "Reaction-Diffusion System" ||
         key == "LinComprNS 2D" ||
         key == "LinComprNS 3D" ||
         key == "ComprNS 2D" ||
         key == "ComprNS 3D" ||
         key == "PNP 1D" ||
         key == "PNP 2D" ||
         key == "PNP 3D" ||
         key == "Thermal 1D" ||
         key == "Thermal 2D" ||
         key == "Thermal 3D" ||
         key == "ThermoElectrostatics 1D" ||
         key == "ThermoElectrostatics 2D" ||
         key == "ThermoElectrostatics 3D";
}

DemoProblemFactory::obj_ptr_type
DemoProblemFactory::
create (const std::string& key,
        const Teuchos::RCP<const Teuchos_Comm>& comm,
        const Teuchos::RCP<Teuchos::ParameterList>& topLevelParams,
        const Teuchos::RCP<ParamLib>& paramLib) const
{
  obj_ptr_type problem;

  auto problemParams = Teuchos::sublist(topLevelParams, "Problem", true);
  auto discretizationParams = Teuchos::sublist(topLevelParams, "Discretization");

  if (key == "CahnHill 2D") {
    problem = Teuchos::rcp(new CahnHillProblem(problemParams, paramLib, 2, comm));
  } else if (key == "ODE") {
    problem = Teuchos::rcp(new ODEProblem(problemParams, paramLib, 0));
  } else if (key == "Helmholtz 2D") {
    problem = Teuchos::rcp(new Helmholtz2DProblem(problemParams, paramLib));
  } else if (key == "NavierStokes 1D") {
    problem = Teuchos::rcp(new NavierStokes(problemParams, paramLib, 1));
  } else if (key == "NavierStokes 2D") {
    problem = Teuchos::rcp(new NavierStokes(problemParams, paramLib, 2));
  } else if (key == "NavierStokes 3D") {
    problem = Teuchos::rcp(new NavierStokes(problemParams, paramLib, 3));
  } else if (key == "LinComprNS 1D") {
    problem = Teuchos::rcp(new LinComprNSProblem(problemParams, paramLib, 1));
  } else if (key == "AdvDiff 1D") {
    problem = Teuchos::rcp(new AdvDiffProblem(problemParams, paramLib, 1));
  } else if (key == "AdvDiff 2D") {
    problem = Teuchos::rcp(new AdvDiffProblem(problemParams, paramLib, 2));
  } else if (key=="Reaction-Diffusion System 3D" ||
             key == "Reaction-Diffusion System") {
    problem = Teuchos::rcp(new ReactDiffSystem(problemParams, paramLib, 3));
  } else if (key == "LinComprNS 2D") {
    problem = Teuchos::rcp(new LinComprNSProblem(problemParams, paramLib, 2));
  } else if (key == "LinComprNS 3D") {
    problem = Teuchos::rcp(new LinComprNSProblem(problemParams, paramLib, 3));
  } else if (key == "ComprNS 2D") {
    problem = Teuchos::rcp(new ComprNSProblem(problemParams, paramLib, 2));
  } else if (key == "ComprNS 3D") {
    problem = Teuchos::rcp(new ComprNSProblem(problemParams, paramLib, 3));
  } else if (key == "PNP 1D") {
    problem = Teuchos::rcp(new PNPProblem(problemParams, paramLib, 1));
  } else if (key == "PNP 2D") {
    problem = Teuchos::rcp(new PNPProblem(problemParams, paramLib, 2));
  } else if (key == "PNP 3D") {
    problem = Teuchos::rcp(new PNPProblem(problemParams, paramLib, 3));
  } else if (key == "ThermoElectrostatics 1D") {
    problem = Teuchos::rcp(new ThermoElectrostaticsProblem(problemParams, paramLib, 1));
  } else if (key == "ThermoElectrostatics 2D") {
    problem = Teuchos::rcp(new ThermoElectrostaticsProblem(problemParams, paramLib, 2));
  } else if (key == "ThermoElectrostatics 3D") {
    problem = Teuchos::rcp(new ThermoElectrostaticsProblem(problemParams, paramLib, 3));
  } else if (getName(key) == "Thermal") {
    problem =
        Teuchos::rcp(new ThermalProblem(problemParams, paramLib, getNumDim(key), comm));
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error,
      "Error! Unrecognized key in DemoProblemFactory. Did you forget to check with 'provides(key)' first?\n");
  }

  return problem;
}

} // namespace Albany
