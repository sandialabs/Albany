//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "QCAD_SchrodingerProblem.hpp"
#include "QCAD_MaterialDatabase.hpp"
#include "Albany_InitialCondition.hpp"

#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "PHAL_FactoryTraits.hpp"

#include "Albany_Utils.hpp"


QCAD::SchrodingerProblem::
SchrodingerProblem( const Teuchos::RCP<Teuchos::ParameterList>& params_,
		    const Teuchos::RCP<ParamLib>& paramLib_,
		    const int numDim_,
		    const Teuchos::RCP<const Epetra_Comm>& comm_) :
  Albany::AbstractProblem(params_, paramLib_, 1),
  comm(comm_),
  havePotential(false), haveMaterial(false),
  numDim(numDim_)
{
  havePotential = params->isSublist("Potential");
  haveMaterial = params->isSublist("Material");

  //Note: can't use get("ParamName" , <default>) form b/c non-const
  energy_unit_in_eV = 1e-3; //default meV
  if(params->isType<double>("EnergyUnitInElectronVolts"))
    energy_unit_in_eV = params->get<double>("EnergyUnitInElectronVolts");
  *out << "Energy unit = " << energy_unit_in_eV << " eV" << endl;

  length_unit_in_m = 1e-9; //default to nm
  if(params->isType<double>("LengthUnitInMeters"))
    length_unit_in_m = params->get<double>("LengthUnitInMeters");
  *out << "Length unit = " << length_unit_in_m << " meters" << endl;

  mtrlDbFilename = "materials.xml";
  if(params->isType<string>("MaterialDB Filename"))
    mtrlDbFilename = params->get<string>("MaterialDB Filename");


  potentialStateName = "V"; //default name for potential at QPs field
  //nEigenvectorsToOuputAsStates = 0;
  bOnlySolveInQuantumBlocks = false;

  //Poisson coupling
  if(params->isSublist("Poisson Coupling")) {
    Teuchos::ParameterList& cList = params->sublist("Poisson Coupling");
    if(cList.isType<bool>("Only solve in quantum blocks"))
      bOnlySolveInQuantumBlocks = cList.get<bool>("Only solve in quantum blocks");
    if(cList.isType<string>("Potential State Name"))
    potentialStateName = cList.get<string>("Potential State Name");

    //if(cList.isType<int>("Save Eigenvectors as States"))
    //  nEigenvectorsToOuputAsStates = cList.get<int>("Save Eigenvectors as States");
  }

  //Check LOCA params to see if eigenvectors will be output to states.
  //Teuchos::ParameterList& locaStepperList = params->sublist("LOCA").sublist("Stepper");
  //if( locaStepperList.get("Compute Eigenvalues", false) > 0) {
  //  int nSave = locaStepperList.sublist("Eigensolver").get("Save Eigenvectors",0);
  //  int nSaveAsStates = locaStepperList.sublist("Eigensolver").get("Save Eigenvectors as States", 0);
  //  nEigenvectorsToOuputAsStates = (nSave < nSaveAsStates)? nSave : nSaveAsStates;
  //}

  TEUCHOS_TEST_FOR_EXCEPTION(params->isSublist("Source Functions"), Teuchos::Exceptions::InvalidParameter,
		     "\nError! Schrodinger problem does not parse Source Functions sublist\n" 
                     << "\tjust Potential sublist " << std::endl);
}

QCAD::SchrodingerProblem::
~SchrodingerProblem()
{
}

void
QCAD::SchrodingerProblem::
buildProblem(
  Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
  Albany::StateManager& stateMgr)
{
  /* Construct All Phalanx Evaluators */
  TEUCHOS_TEST_FOR_EXCEPTION(meshSpecs.size()!=1,std::logic_error,"Problem supports one Material Block");
  fm.resize(1);
  fm[0]  = Teuchos::rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
  buildEvaluators(*fm[0], *meshSpecs[0], stateMgr, Albany::BUILD_RESID_FM, 
		  Teuchos::null);
  constructDirichletEvaluators(*meshSpecs[0]);
}

Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
QCAD::SchrodingerProblem::
buildEvaluators(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fmchoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructeEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  Albany::ConstructEvaluatorsOp<SchrodingerProblem> op(
    *this, fm0, meshSpecs, stateMgr, fmchoice, responseList);
  boost::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes>(op);
  return *op.tags;
}

void
QCAD::SchrodingerProblem::constructDirichletEvaluators(
        const Albany::MeshSpecsStruct& meshSpecs)
{
   // Construct Dirichlet evaluators for all nodesets and names
   std::vector<string> dirichletNames(neq);
   dirichletNames[0] = "psi";
   Albany::BCUtils<Albany::DirichletTraits> dirUtils;
   dfm = dirUtils.constructBCEvaluators(meshSpecs.nsNames, dirichletNames,
                                          this->params, this->paramLib);
}

Teuchos::RCP<const Teuchos::ParameterList>
QCAD::SchrodingerProblem::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidSchrodingerProblemParams");

  if (numDim==1)
    validPL->set<bool>("Periodic BC", false, "Flag to indicate periodic BC for 1D problems");
  validPL->sublist("Material", false, "");
  validPL->sublist("Potential", false, "");
  validPL->set<double>("EnergyUnitInElectronVolts",1e-3,"Energy unit in electron volts");
  validPL->set<double>("LengthUnitInMeters",1e-9,"Length unit in meters");
  validPL->set<string>("MaterialDB Filename","materials.xml","Filename of material database xml file");

  validPL->sublist("Poisson Coupling", false, "");

  validPL->sublist("Poisson Coupling").set<bool>("Only solve in quantum blocks", false,"Only perform Schrodinger solve in element blocks marked as quatum regions.");
  validPL->sublist("Poisson Coupling").set<string>("Potential State Name", "","Name of State to use as potential");
  validPL->sublist("Poisson Coupling").set<int>("Save Eigenvectors as States", 0,"Number of eigenstates to save as states");
  return validPL;
}

