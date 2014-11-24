//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "FELIX_Stokes.hpp"

#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "PHAL_FactoryTraits.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"

void
FELIX::Stokes::
getVariableType(Teuchos::ParameterList& paramList,
		const std::string& defaultType,
		FELIX::Stokes::NS_VAR_TYPE& variableType,
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
FELIX::Stokes::
variableTypeToString(FELIX::Stokes::NS_VAR_TYPE variableType)
{
  if (variableType == NS_VAR_TYPE_NONE)
    return "None";
  else if (variableType == NS_VAR_TYPE_CONSTANT)
    return "Constant";
  return "DOF";
}

FELIX::Stokes::
Stokes( const Teuchos::RCP<Teuchos::ParameterList>& params_,
             const Teuchos::RCP<ParamLib>& paramLib_,
             const int numDim_) :
  Albany::AbstractProblem(params_, paramLib_),
  haveFlow(false),
  haveFlowEq(false),
  haveSource(false),
  havePSPG(false),
  numDim(numDim_)
{

  getVariableType(params->sublist("Flow"), "DOF", flowType, 
		  haveFlow, haveFlowEq);

  if (haveFlowEq) {
    havePSPG = params->get("Have Pressure Stabilization", true);
  }

  haveSource = true; 



  // Compute number of equations
  int num_eq = 0;
  if (haveFlowEq) num_eq += numDim+1;
  this->setNumEquations(num_eq);

  // Need to allocate a surface height and temperature fields in mesh database
  this->requirements.push_back("surface_height");
  this->requirements.push_back("temperature");
  this->requirements.push_back("basal_friction");
  this->requirements.push_back("thickness");
  this->requirements.push_back("flow_factor");

  // Print out a summary of the problem
  *out << "Stokes problem:" << std::endl
       << "\tSpatial dimension:      " << numDim << std::endl
       << "\tFlow variables:         " << variableTypeToString(flowType) 
       << std::endl
       << "\tPressure stabilization: " << havePSPG << std::endl;
}

FELIX::Stokes::
~Stokes()
{
}

void
FELIX::Stokes::
buildProblem(
  Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
  Albany::StateManager& stateMgr)
{
  using Teuchos::rcp;

 /* Construct All Phalanx Evaluators */
  TEUCHOS_TEST_FOR_EXCEPTION(meshSpecs.size()!=1,std::logic_error,"Problem supports one Material Block");
  fm.resize(1);
  fm[0]  = rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
  buildEvaluators(*fm[0], *meshSpecs[0], stateMgr, Albany::BUILD_RESID_FM, 
		  Teuchos::null);
  constructDirichletEvaluators(*meshSpecs[0]);
  //construct Neumann evaluators
  constructNeumannEvaluators(meshSpecs[0]);
}

Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
FELIX::Stokes::
buildEvaluators(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fmchoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructeEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  Albany::ConstructEvaluatorsOp<Stokes> op(
    *this, fm0, meshSpecs, stateMgr, fmchoice, responseList);
  boost::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes>(op);
  return *op.tags;
}

void
FELIX::Stokes::constructDirichletEvaluators(
        const Albany::MeshSpecsStruct& meshSpecs)
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
   Albany::BCUtils<Albany::DirichletTraits> dirUtils;
   dfm = dirUtils.constructBCEvaluators(meshSpecs.nsNames, dirichletNames,
                                          this->params, this->paramLib);
}

//Neumann BCs
void
FELIX::Stokes::constructNeumannEvaluators(const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs)
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
   //
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

   // Construct BC evaluators for all possible names of conditions
   // Should only specify flux vector components (dudx, dudy, dudz), or dudn, not both
   std::vector<std::string> condNames(3); //dudx, dudy, dudz, dudn, basal 

   // Note that sidesets are only supported for two and 3D currently
   //
   if(numDim == 2)
    condNames[0] = "(dudx, dudy)";
   else if(numDim == 3)
    condNames[0] = "(dudx, dudy, dudz)";
   else
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
       std::endl << "Error: Sidesets only supported in 2 and 3D." << std::endl);

   condNames[1] = "dudn";
   
   condNames[2] = "basal";

   nfm.resize(1);


   nfm[0] = nbcUtils.constructBCEvaluators(meshSpecs, nbcNames,
                                           Teuchos::arcp(dof_names),
                                           true, 0, condNames, offsets, dl,
                                           this->params, this->paramLib);
}



Teuchos::RCP<const Teuchos::ParameterList>
FELIX::Stokes::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidStokesParams");

  validPL->set<bool>("Have Pressure Stabilization", true);
  validPL->sublist("Flow", false, "");
  validPL->sublist("Density", false, "");
  validPL->sublist("Viscosity", false, "");
  validPL->sublist("FELIX Viscosity", false, "");
  validPL->sublist("Tau M", false, "");
  validPL->sublist("Body Force", false, "");

  return validPL;
}

