//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "MechanicsProblem.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "PHAL_AlbanyTraits.hpp"

void
Albany::MechanicsProblem::
getVariableType(Teuchos::ParameterList& paramList,
		const std::string& defaultType,
		Albany::MechanicsProblem::MECH_VAR_TYPE& variableType,
		bool& haveVariable,
		bool& haveEquation)
{
  std::string type = paramList.get("Variable Type", defaultType);
  if (type == "None")
    variableType = MECH_VAR_TYPE_NONE;
  else if (type == "Constant")
    variableType = MECH_VAR_TYPE_CONSTANT;
  else if (type == "DOF")
    variableType = MECH_VAR_TYPE_DOF;
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                               "Unknown variable type " << type << std::endl);
  haveVariable = (variableType != MECH_VAR_TYPE_NONE);
  haveEquation = (variableType == MECH_VAR_TYPE_DOF);
}
//------------------------------------------------------------------------------
std::string
Albany::MechanicsProblem::
variableTypeToString(Albany::MechanicsProblem::MECH_VAR_TYPE variableType)
{
  if (variableType == MECH_VAR_TYPE_NONE)
    return "None";
  else if (variableType == MECH_VAR_TYPE_CONSTANT)
    return "Constant";
  return "DOF";
}

//------------------------------------------------------------------------------
Albany::MechanicsProblem::
MechanicsProblem(const Teuchos::RCP<Teuchos::ParameterList>& params_,
                 const Teuchos::RCP<ParamLib>& paramLib_,
                 const int numDim_,
                 const Teuchos::RCP<const Epetra_Comm>& comm) :
  Albany::AbstractProblem(params_, paramLib_),
  haveSource(false),
  numDim(numDim_),
  haveMechEq(false),
  haveHeatEq(false),
  havePressureEq(false),
  haveTransportEq(false),
  haveHydroStressEq(false),
  haveMatDB(false)
{
 
  std::string& method = params->get("Name", "Mechanics ");
  *out << "Problem Name = " << method << std::endl;
  
  haveSource =  params->isSublist("Source Functions");

  getVariableType(params->sublist("Displacement"), "DOF", mechType, 
		  haveMech, haveMechEq);
  getVariableType(params->sublist("Heat"), "None", heatType, 
		  haveHeat, haveHeatEq);
  getVariableType(params->sublist("Pore Pressure"), "None", pressureType,
		  havePressure, havePressureEq);
  getVariableType(params->sublist("Transport"), "None", transportType,
		  haveTransport, haveTransportEq);
  getVariableType(params->sublist("HydroStress"), "None", hydrostressType,
  		  haveHydroStress, haveHydroStressEq);

  if (haveHeatEq)
    haveSource =  params->isSublist("Source Functions");

  // Compute number of equations
  int num_eq = 0;
  if (haveMechEq) num_eq += numDim;
  if (haveHeatEq) num_eq += 1;
  if (havePressureEq) num_eq += 1;
  if (haveTransportEq) num_eq += 1;
  if (haveHydroStressEq) num_eq +=1;
  this->setNumEquations(num_eq);

  // Print out a summary of the problem
  *out << "Mechanics problem:" << std::endl
       << "\tSpatial dimension:       " << numDim << std::endl
       << "\tMechanics variables:     " << variableTypeToString(mechType) 
       << std::endl
       << "\tHeat variables:          " << variableTypeToString(heatType) 
       << std::endl
       << "\tPore Pressure variables: " << variableTypeToString(pressureType)
       << std::endl
       << "\tTransport variables:     " << variableTypeToString(transportType)
       << std::endl
  << "\tHydroStress variables: " << variableTypeToString(hydrostressType)
         << std::endl;

  if(params->isType<string>("MaterialDB Filename")){
    haveMatDB = true;
    mtrlDbFilename = params->get<string>("MaterialDB Filename");
    materialDB = Teuchos::rcp(new QCAD::MaterialDatabase(mtrlDbFilename, comm));
  }
  TEUCHOS_TEST_FOR_EXCEPTION(!haveMatDB, std::logic_error,
                             "Mechanics Problem Requires a Material Database");

}
//------------------------------------------------------------------------------
Albany::MechanicsProblem::
~MechanicsProblem()
{
}
//------------------------------------------------------------------------------
//the following function returns the problem information required for setting the rigid body modes (RBMs) for elasticity problems (in src/Albany_SolverFactory.cpp)
//written by IK, Feb. 2012 
void
Albany::MechanicsProblem::getRBMInfoForML(
   int& numPDEs, int& numElasticityDim, int& numScalar, int& nullSpaceDim)
{
  // Need numPDEs should be numDim + nDOF for other governing equations  -SS

  numPDEs = neq;
  numElasticityDim = 0;
  if (haveMechEq) numElasticityDim = numDim;
  numScalar = neq - numElasticityDim;
  if (haveMechEq) {
    if (numDim == 1) {nullSpaceDim = 0; }
    if (numDim == 2) {nullSpaceDim = 3; }
    if (numDim == 3) {nullSpaceDim = 6; }
  }
}
//------------------------------------------------------------------------------
void
Albany::MechanicsProblem::
buildProblem(
  Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
  Albany::StateManager& stateMgr)
{
  /* Construct All Phalanx Evaluators */
  int physSets = meshSpecs.size();
  cout << "Num MeshSpecs: " << physSets << endl;
  fm.resize(physSets);

  cout << "Calling MechanicsProblem::buildEvaluators" << endl;
  for (int ps=0; ps<physSets; ps++) {
    fm[ps]  = Teuchos::rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
    buildEvaluators(*fm[ps], *meshSpecs[ps], stateMgr, BUILD_RESID_FM, 
		    Teuchos::null);
  }
  constructDirichletEvaluators(*meshSpecs[0]);
}
//------------------------------------------------------------------------------
Teuchos::Array<Teuchos::RCP<const PHX::FieldTag> >
Albany::MechanicsProblem::
buildEvaluators(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fmchoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructeEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  ConstructEvaluatorsOp<MechanicsProblem> op(
    *this, fm0, meshSpecs, stateMgr, fmchoice, responseList);
  boost::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes>(op);
  return *op.tags;
}
//------------------------------------------------------------------------------
void
Albany::MechanicsProblem::constructDirichletEvaluators(
        const Albany::MeshSpecsStruct& meshSpecs)
{

  // Construct Dirichlet evaluators for all nodesets and names
  std::vector<string> dirichletNames(neq);
  int index = 0;
  if (haveMechEq) {
    dirichletNames[index++] = "X";
    if (neq>1) dirichletNames[index++] = "Y";
    if (neq>2) dirichletNames[index++] = "Z";
  }

  if (haveHeatEq) dirichletNames[index++] = "T";
  if (havePressureEq) dirichletNames[index++] = "P";
  // Note: for hydrogen transport problem, L2 projection is need to derive the
  // source term/flux induced by volumetric deformation
  if (haveTransportEq) dirichletNames[index++] = "C"; // Lattice Concentration
  if (haveHydroStressEq) dirichletNames[index++] = "TAU"; // Projected Hydrostatic Stress

  Albany::BCUtils<Albany::DirichletTraits> dirUtils;
  dfm = dirUtils.constructBCEvaluators(meshSpecs.nsNames, dirichletNames,
                                       this->params, this->paramLib);
}
//------------------------------------------------------------------------------
Teuchos::RCP<const Teuchos::ParameterList>
Albany::MechanicsProblem::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidMechanicsProblemParams");

  validPL->set<string>("MaterialDB Filename",
                       "materials.xml",
                       "Filename of material database xml file");
  validPL->sublist("Displacement", false, "");
  validPL->sublist("Heat", false, "");
  validPL->sublist("Pore Pressure", false, "");
  validPL->sublist("Transport", false, "");
  validPL->sublist("HydroStress", false, "");
  

  return validPL;
}

void
Albany::MechanicsProblem::getAllocatedStates(
   Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid::FieldContainer<RealType> > > > oldState_,
   Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid::FieldContainer<RealType> > > > newState_
   ) const
{
  oldState_ = oldState;
  newState_ = newState;
}
//------------------------------------------------------------------------------
std::string 
Albany::MechanicsProblem::stateString(std::string name, bool surfaceFlag)
{
  std::string outputName(name);
  if (surfaceFlag) outputName = "Surface_"+name;
  return outputName;
}


