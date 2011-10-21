/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
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
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


#include "Albany_MultiHeatProblem.hpp"
#include "Albany_SolutionAverageResponseFunction.hpp"
#include "Albany_SolutionTwoNormResponseFunction.hpp"
#include "Albany_SolutionMaxValueResponseFunction.hpp"
#include "Albany_InitialCondition.hpp"

#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "PHAL_FactoryTraits.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"


Albany::MultiHeatProblem::
MultiHeatProblem( const Teuchos::RCP<Teuchos::ParameterList>& params_,
             const Teuchos::RCP<ParamLib>& paramLib_,
             const int numDim_,
             const Teuchos::RCP<const Epetra_Comm>& comm_) :
  Albany::AbstractProblem(params_, paramLib_),
  haveSource(false),
  haveAbsorption(false),
  haveMatDB(false),
  numDim(numDim_),
  comm(comm_)
{
  this->setNumEquations(1);

  if (numDim==1) periodic = params->get("Periodic BC", false);
  else           periodic = false;
  if (periodic) *out <<" Periodic Boundary Conditions being used." <<std::endl;

  haveSource =  params->isSublist("Source Functions");
  haveAbsorption =  params->isSublist("Absorption");

  if(params->isType<string>("MaterialDB Filename")){
	haveMatDB = true;
    mtrlDbFilename = params->get<string>("MaterialDB Filename");
 // Create Material Database
    materialDB = Teuchos::rcp(new QCAD::MaterialDatabase(mtrlDbFilename, comm));
  }

}

Albany::MultiHeatProblem::
~MultiHeatProblem()
{
}

void
Albany::MultiHeatProblem::
buildProblem(
    Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
    Albany::StateManager& stateMgr,
    Teuchos::ArrayRCP< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses)
{
  /* Construct All Phalanx Evaluators */
  int physSets = meshSpecs.size();
  cout << "MULTIHeat Num MeshSpecs: " << physSets << endl;
  fm.resize(physSets); rfm.resize(physSets);
  for (int ps=0; ps<physSets; ps++) {
    fm[ps]  = Teuchos::rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
    rfm[ps] = Teuchos::rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);

    constructResidEvaluators<PHAL::AlbanyTraits::Residual  >(*fm[ps], *meshSpecs[ps], stateMgr);
    constructResidEvaluators<PHAL::AlbanyTraits::Jacobian  >(*fm[ps], *meshSpecs[ps], stateMgr);
    constructResidEvaluators<PHAL::AlbanyTraits::Tangent   >(*fm[ps], *meshSpecs[ps], stateMgr);
    constructResidEvaluators<PHAL::AlbanyTraits::SGResidual>(*fm[ps], *meshSpecs[ps], stateMgr);
    constructResidEvaluators<PHAL::AlbanyTraits::SGJacobian>(*fm[ps], *meshSpecs[ps], stateMgr);
    constructResidEvaluators<PHAL::AlbanyTraits::SGTangent >(*fm[ps], *meshSpecs[ps], stateMgr);
    constructResidEvaluators<PHAL::AlbanyTraits::MPResidual>(*fm[ps], *meshSpecs[ps], stateMgr);
    constructResidEvaluators<PHAL::AlbanyTraits::MPJacobian>(*fm[ps], *meshSpecs[ps], stateMgr);
    constructResidEvaluators<PHAL::AlbanyTraits::MPTangent >(*fm[ps], *meshSpecs[ps], stateMgr);
    constructResponseEvaluators<PHAL::AlbanyTraits::Residual  >(*rfm[ps], *meshSpecs[ps], stateMgr, responses);
    constructResponseEvaluators<PHAL::AlbanyTraits::Jacobian  >(*rfm[ps], *meshSpecs[ps], stateMgr);
    constructResponseEvaluators<PHAL::AlbanyTraits::Tangent   >(*rfm[ps], *meshSpecs[ps], stateMgr);
    constructResponseEvaluators<PHAL::AlbanyTraits::SGResidual>(*rfm[ps], *meshSpecs[ps], stateMgr);
    constructResponseEvaluators<PHAL::AlbanyTraits::SGJacobian>(*rfm[ps], *meshSpecs[ps], stateMgr);
    constructResponseEvaluators<PHAL::AlbanyTraits::SGTangent >(*rfm[ps], *meshSpecs[ps], stateMgr);
    constructResponseEvaluators<PHAL::AlbanyTraits::MPResidual>(*rfm[ps], *meshSpecs[ps], stateMgr);
    constructResponseEvaluators<PHAL::AlbanyTraits::MPJacobian>(*rfm[ps], *meshSpecs[ps], stateMgr);
    constructResponseEvaluators<PHAL::AlbanyTraits::MPTangent >(*rfm[ps], *meshSpecs[ps], stateMgr);
  }
  constructDirichletEvaluators(*meshSpecs[0]);
}

void
Albany::MultiHeatProblem::constructDirichletEvaluators(
        const Albany::MeshSpecsStruct& meshSpecs)
{
   // Construct Dirichlet evaluators for all nodesets and names
   vector<string> dirichletNames(neq);
   dirichletNames[0] = "T";
   Albany::DirichletUtils dirUtils;
   dfm = dirUtils.constructDirichletEvaluators(meshSpecs.nsNames, dirichletNames,
                                          this->params, this->paramLib);
}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::MultiHeatProblem::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidMultiHeatProblemParams");

  if (numDim==1)
    validPL->set<bool>("Periodic BC", false, "Flag to indicate periodic BC for 1D problems");
  validPL->sublist("Thermal Conductivity", false, "");
  validPL->set("Convection Velocity", "{0,0,0}", "");
  validPL->set<bool>("Have Rho Cp", false, "Flag to indicate if rhoCp is used");
  validPL->set<string>("MaterialDB Filename","materials.xml","Filename of material database xml file");

  return validPL;
}

