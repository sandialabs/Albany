//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "QCAD_SchrodingerProblem.hpp"
#include "QCAD_MaterialDatabase.hpp"

#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "PHAL_FactoryTraits.hpp"

#include "Albany_Utils.hpp"


QCAD::SchrodingerProblem::SchrodingerProblem( const Teuchos::RCP<Teuchos::ParameterList>& params_,
		    const Teuchos::RCP<ParamLib>& paramLib_,
		    const int numDim_,
                    Teuchos::RCP<const Teuchos::Comm<int> >& commT_):
  Albany::AbstractProblem(params_, paramLib_, 1),
  commT(commT_), havePotential(false), 
  numDim(numDim_)
{
  havePotential = params->isSublist("Potential");

  //Note: can't use get("ParamName" , <default>) form b/c non-const
  energy_unit_in_eV = 1.0; //default eV
  if(params->isType<double>("Energy Unit In Electron Volts"))
    energy_unit_in_eV = params->get<double>("Energy Unit In Electron Volts");
  *out << "Energy unit = " << energy_unit_in_eV << " eV" << std::endl;

  length_unit_in_m = 1e-6; //default to um
  if(params->isType<double>("Length Unit In Meters"))
    length_unit_in_m = params->get<double>("Length Unit In Meters");
  *out << "Length unit = " << length_unit_in_m << " meters" << std::endl;

  mtrlDbFilename = "materials.xml";
  if(params->isType<std::string>("MaterialDB Filename"))
    mtrlDbFilename = params->get<std::string>("MaterialDB Filename");

  bOnlySolveInQuantumBlocks = false;
  if(params->isType<bool>("Only solve in quantum blocks"))
    bOnlySolveInQuantumBlocks = params->get<bool>("Only solve in quantum blocks");


  potentialFieldName = "V"; // name for potential at QPs field
  potentialAuxIndex = -1; // if >= 0, index within workset's auxData multivector to import potential from

  //Extract Aux index if necessary:
  // Note: we can't do this from within SchrodingerPotential evaluator
  // since it isn't created in the case of importing from an aux vector.
  if(params->isSublist("Potential")) {
    Teuchos::ParameterList& pList = params->sublist("Potential");
    if(pList.isType<std::string>("Type")) {
      if(pList.get<std::string>("Type") == "From Aux Data Vector") {

	//do Potential list validation since SchrodingerPotential evaluator doesn't
	// ** maybe we should allow other values from SchrodingerPotential as well? **
	Teuchos::ParameterList validPL("Valid Schrodinger Potential Params");
	validPL.set<std::string>("Type", "defaultType", "Switch between different potential types");
	validPL.set<int>("Aux Index", 0, "Index of aux vector to import potential from.");
	pList.validateParameters(validPL, 0);

	potentialAuxIndex = pList.get<int>("Aux Index"); //error if doesn't exist
      }
    }
  }

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
  Sacado::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes> fe(op);
  return *op.tags;
}

void
QCAD::SchrodingerProblem::constructDirichletEvaluators(
        const Albany::MeshSpecsStruct& meshSpecs)
{
   // Construct Dirichlet evaluators for all nodesets and names
   std::vector<std::string> dirichletNames(neq);
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

  validPL->set<double>("Energy Unit In Electron Volts",1.0,"Energy unit in electron volts");
  validPL->set<double>("Length Unit In Meters",1e-6,"Length unit in meters");
  validPL->set<std::string>("MaterialDB Filename","materials.xml","Filename of material database xml file");
  validPL->set<bool>("Only solve in quantum blocks", false,"Only perform Schrodinger solve in element blocks marked as quatum regions.");

  validPL->sublist("Potential", false, "");

  return validPL;
}

