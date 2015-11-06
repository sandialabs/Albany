//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_NavierStokes.hpp"
#include "AAdapt_InitialCondition.hpp"

#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "PHAL_FactoryTraits.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"

void
Albany::NavierStokes::
getVariableType(Teuchos::ParameterList& paramList,
		const std::string& defaultType,
		Albany::NavierStokes::NS_VAR_TYPE& variableType,
		bool& haveVariable,
		bool& haveEquation)
{
  std::string type = paramList.get("Variable Type", defaultType);
  if (type == "None")
    variableType = NS_VAR_TYPE_NONE;
  else if (type == "Constant")
    variableType = NS_VAR_TYPE_CONSTANT;
  else if (type == "DOF")
    variableType = NS_VAR_TYPE_DOF;
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
		       "Unknown variable type " << type << std::endl);
  haveVariable = (variableType != NS_VAR_TYPE_NONE);
  haveEquation = (variableType == NS_VAR_TYPE_DOF);
}

std::string
Albany::NavierStokes::
variableTypeToString(Albany::NavierStokes::NS_VAR_TYPE variableType)
{
  if (variableType == NS_VAR_TYPE_NONE)
    return "None";
  else if (variableType == NS_VAR_TYPE_CONSTANT)
    return "Constant";
  return "DOF";
}

Albany::NavierStokes::
NavierStokes( const Teuchos::RCP<Teuchos::ParameterList>& params_,
             const Teuchos::RCP<ParamLib>& paramLib_,
             const int numDim_) :
  Albany::AbstractProblem(params_, paramLib_),
  haveFlow(false),
  haveHeat(false),
  haveNeut(false),
  haveFlowEq(false),
  haveHeatEq(false),
  haveNeutEq(false),
  haveSource(false),
  haveNeutSource(false),
  havePSPG(false),
  haveSUPG(false),
  porousMedia(false),
  numDim(numDim_)
{
  if (numDim==1) periodic = params->get("Periodic BC", false);
  else           periodic = false;
  if (periodic) *out <<" Periodic Boundary Conditions being used." <<std::endl;

  getVariableType(params->sublist("Flow"), "DOF", flowType, 
		  haveFlow, haveFlowEq);
  getVariableType(params->sublist("Heat"), "None", heatType, 
		  haveHeat, haveHeatEq);
  getVariableType(params->sublist("Neutronics"), "None", neutType, 
		  haveNeut, haveNeutEq);

  if (haveFlowEq) {
    havePSPG = params->get("Have Pressure Stabilization", true);
    porousMedia = params->get("Porous Media",false);
  }

  if (haveFlow && (haveFlowEq || haveHeatEq))
    haveSUPG = params->get("Have SUPG Stabilization", true);

  if (haveHeatEq)
    haveSource =  params->isSublist("Source Functions");

  if (haveNeutEq)
    haveNeutSource =  params->isSublist("Neutron Source");

  // Compute number of equations
  int num_eq = 0;
  if (haveFlowEq) num_eq += numDim+1;
  if (haveHeatEq) num_eq += 1;
  if (haveNeutEq) num_eq += 1;
  this->setNumEquations(num_eq);

  // Print out a summary of the problem
  *out << "Navier-Stokes problem:" << std::endl
       << "\tSpatial dimension:      " << numDim << std::endl
       << "\tFlow variables:         " << variableTypeToString(flowType) 
       << std::endl
       << "\tHeat variables:         " << variableTypeToString(heatType) 
       << std::endl
       << "\tNeutronics variables:   " << variableTypeToString(neutType) 
       << std::endl
       << "\tPressure stabilization: " << havePSPG << std::endl
       << "\tUpwind stabilization:   " << haveSUPG << std::endl
       << "\tPorous media:           " << porousMedia << std::endl;
}

Albany::NavierStokes::
~NavierStokes()
{
}

void
Albany::NavierStokes::
buildProblem(
  Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
  Albany::StateManager& stateMgr)
{
  using Teuchos::rcp;

 /* Construct All Phalanx Evaluators */
  TEUCHOS_TEST_FOR_EXCEPTION(meshSpecs.size()!=1,std::logic_error,"Problem supports one Material Block");

  fm.resize(1);
  fm[0]  = rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
  buildEvaluators(*fm[0], *meshSpecs[0], stateMgr, BUILD_RESID_FM, 
		  Teuchos::null);

  if(meshSpecs[0]->nsNames.size() > 0) // Build a nodeset evaluator if nodesets are present

     constructDirichletEvaluators(meshSpecs[0]->nsNames);

  if(meshSpecs[0]->ssNames.size() > 0) // Build a sideset evaluator if sidesets are present

     constructNeumannEvaluators(meshSpecs[0]);

}

Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
Albany::NavierStokes::
buildEvaluators(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fmchoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructeEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  ConstructEvaluatorsOp<NavierStokes> op(
    *this, fm0, meshSpecs, stateMgr, fmchoice, responseList);
  Sacado::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes> fe(op);
  return *op.tags;
}

void
Albany::NavierStokes::constructDirichletEvaluators(
        const std::vector<std::string>& nodeSetIDs)
{
   // Construct Dirichlet evaluators for all nodesets and names
   std::vector<std::string> dirichletNames(neq);
   int index = 0;
   if (haveFlowEq) {
     dirichletNames[index++] = "ux";
     if (numDim>=2) dirichletNames[index++] = "uy";
     if (numDim==3) dirichletNames[index++] = "uz";
     dirichletNames[index++] = "p";
   }
   if (haveHeatEq) dirichletNames[index++] = "T";
   if (haveNeutEq) dirichletNames[index++] = "phi";
   Albany::BCUtils<Albany::DirichletTraits> dirUtils;
   dfm = dirUtils.constructBCEvaluators(nodeSetIDs, dirichletNames,
                                          this->params, this->paramLib);
}

// Neumann BCs
void
Albany::NavierStokes::constructNeumannEvaluators(const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs)
{

   // Note: we only enter this function if sidesets are defined in the mesh file
   // i.e. meshSpecs.ssNames.size() > 0

   Albany::BCUtils<Albany::NeumannTraits> nbcUtils;

   // Check to make sure that Neumann BCs are given in the input file

   if(!nbcUtils.haveBCSpecified(this->params)) {
      return;
   }


   // Construct BC evaluators for all side sets and names
   // Note that the string index sets up the equation offset, so ordering is important

   // Currently we aren't exactly doing this right.  I think to do this
   // correctly we need different neumann evaluators for each DOF (velocity,
   // pressure, temperature, flux) since velocity is a vector and the 
   // others are scalars.  The dof_names stuff is only used
   // for robin conditions, so at this point, as long as we don't enable
   // robin conditions, this should work.

   std::vector<std::string> nbcNames;
   Teuchos::RCP< Teuchos::Array<std::string> > dof_names = 
     Teuchos::rcp(new Teuchos::Array<std::string>);
   Teuchos::Array<Teuchos::Array<int> > offsets;
   int idx = 0;
   if (haveFlowEq) {
     nbcNames.push_back("ux");
     offsets.push_back(Teuchos::Array<int>(1,idx++));
     if (numDim>=2) {
       nbcNames.push_back("uy");
       offsets.push_back(Teuchos::Array<int>(1,idx++));
     }
     if (numDim==3) {
       nbcNames.push_back("uz");
       offsets.push_back(Teuchos::Array<int>(1,idx++));
     }
     nbcNames.push_back("p");
     offsets.push_back(Teuchos::Array<int>(1,idx++));
     dof_names->push_back("Velocity");
     dof_names->push_back("Pressure");
   }
   if (haveHeatEq) {
     nbcNames.push_back("T");
     offsets.push_back(Teuchos::Array<int>(1,idx++));
     dof_names->push_back("Temperature");
   }
   if (haveNeutEq) {
     nbcNames.push_back("phi");
     offsets.push_back(Teuchos::Array<int>(1,idx++));
     dof_names->push_back("Neutron Flux");
   }
   
   // Construct BC evaluators for all possible names of conditions
   // Should only specify flux vector components (dudx, dudy, dudz), or dudn, not both
   std::vector<std::string> condNames(2); //dudx, dudy, dudz, dudn

   // Note that sidesets are only supported for two and 3D currently
   if(numDim == 2) 
    condNames[0] = "(dudx, dudy)";
   else if(numDim == 3)
    condNames[0] = "(dudx, dudy, dudz)";
   else
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
       std::endl << "Error: Sidesets only supported in 2 and 3D." << std::endl);

   condNames[1] = "dudn";

   nfm.resize(1);

   nfm[0] = nbcUtils.constructBCEvaluators(meshSpecs, nbcNames, 
					   Teuchos::arcp(dof_names), 
					   false, 0, condNames, offsets, dl,
					   this->params, this->paramLib);
}


Teuchos::RCP<const Teuchos::ParameterList>
Albany::NavierStokes::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidNavierStokesParams");

  if (numDim==1)
    validPL->set<bool>("Periodic BC", false, "Flag to indicate periodic BC for 1D problems");
  validPL->set<bool>("Have Pressure Stabilization", true);
  validPL->set<bool>("Have SUPG Stabilization", true);
  validPL->set<bool>("Porous Media", false, "Flag to use porous media equations");
  validPL->sublist("Flow", false, "");
  validPL->sublist("Heat", false, "");
  validPL->sublist("Neutronics", false, "");
  validPL->sublist("Thermal Conductivity", false, "");
  validPL->sublist("Density", false, "");
  validPL->sublist("Viscosity", false, "");
  validPL->sublist("Volumetric Expansion Coefficient", false, "");
  validPL->sublist("Specific Heat", false, "");
  validPL->sublist("Body Force", false, "");
  validPL->sublist("Porosity", false, "");
  validPL->sublist("Permeability", false, "");
  validPL->sublist("Forchheimer", false, "");
  
  validPL->sublist("Neutron Source", false, "");
  validPL->sublist("Neutron Diffusion Coefficient", false, "");
  validPL->sublist("Absorption Cross Section", false, "");
  validPL->sublist("Fission Cross Section", false, "");
  validPL->sublist("Neutrons per Fission", false, "");
  validPL->sublist("Scattering Cross Section", false, "");
  validPL->sublist("Average Scattering Angle", false, "");
  validPL->sublist("Energy Released per Fission", false, "");

  return validPL;
}

