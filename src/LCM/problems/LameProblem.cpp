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

#include "LameProblem.hpp"
#include "Albany_SolutionAverageResponseFunction.hpp"
#include "Albany_SolutionTwoNormResponseFunction.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"

Albany::LameProblem::
LameProblem(const Teuchos::RCP<Teuchos::ParameterList>& params_,
            const Teuchos::RCP<ParamLib>& paramLib_,
            const int numDim_,
            const Teuchos::RCP<const Epetra_Comm>& comm) :
  Albany::AbstractProblem(params_, paramLib_, numDim_),
  haveSource(false), haveMatDB(false)
{
 
  std::string& method = params->get("Name", "Library of Advanced Materials for Engineering (LAME) ");
  *out << "Problem Name = " << method << std::endl;
  
  haveSource =  params->isSublist("Source Functions");

  if(params->isType<string>("MaterialDB Filename")){
        haveMatDB = true;
    mtrlDbFilename = params->get<string>("MaterialDB Filename");
    materialDB = Teuchos::rcp(new QCAD::MaterialDatabase(mtrlDbFilename, comm));
  }

  // currently only support 3D analyses
  TEUCHOS_TEST_FOR_EXCEPTION(neq != 3,
                     Teuchos::Exceptions::InvalidParameter,
                     "\nOnly three-dimensional analyses are suppored when using the Library of Advanced Materials for Engineering (LAME)\n");
}

Albany::LameProblem::
~LameProblem()
{
}

void
Albany::LameProblem::
buildProblem(
    Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
    Albany::StateManager& stateMgr,
    Teuchos::ArrayRCP< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses)
{
  /* Construct All Phalanx Evaluators */
  int physSets = meshSpecs.size();
  cout << "Lame Problem: Num MeshSpecs: " << physSets << endl;
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
Albany::LameProblem::constructDirichletEvaluators(
        const Albany::MeshSpecsStruct& meshSpecs)
{
   // Construct Dirichlet evaluators for all nodesets and names
   std::vector<std::string> dirichletNames(neq);
   dirichletNames[0] = "X";
   if (neq>1) dirichletNames[1] = "Y";
   if (neq>2) dirichletNames[2] = "Z";
   Albany::BCUtils<Albany::DirichletTraits> dirUtils;
   dfm = dirUtils.constructBCEvaluators(meshSpecs.nsNames, dirichletNames,
                                          this->params, this->paramLib);
}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::LameProblem::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidLameProblemParams");

  validPL->set<string>("Lame Material Model", "", "The name of the LAME material model.");
  validPL->sublist("Lame Material Parameters", false, "");
  validPL->sublist("aveJ", false, "If true, the determinate of the deformation gradient for each integration point is replaced with the average value over all integration points in the element (produces constant volumetric response).");
  validPL->sublist("volaveJ", false, "If true, the determinate of the deformation gradient for each integration point is replaced with the volume-averaged value over all integration points in the element (produces constant volumetric response).");
  validPL->set<string>("MaterialDB Filename","materials.xml","Filename of material database xml file");

  return validPL;
}

