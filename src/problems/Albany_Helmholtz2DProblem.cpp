//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_Helmholtz2DProblem.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"

Albany::Helmholtz2DProblem::
Helmholtz2DProblem(
                         const Teuchos::RCP<Teuchos::ParameterList>& params_,
                         const Teuchos::RCP<ParamLib>& paramLib_) :
  Albany::AbstractProblem(params_, paramLib_, 2)
{

  std::string& method = params->get("Name", "Helmholtz 2D Problem");
  *out << "Problem Name = " << method << std::endl;
  
  ksqr = params->get<double>("Ksqr",1.0);

  haveSource =  params->isSublist("Source Functions");
}

Albany::Helmholtz2DProblem::
~Helmholtz2DProblem()
{
}

void
Albany::Helmholtz2DProblem::
buildProblem(
   Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
   Albany::StateManager& stateMgr)
{
  /* Construct All Phalanx Evaluators */
  TEUCHOS_TEST_FOR_EXCEPTION(meshSpecs.size()!=1,std::logic_error,"Problem supports one Material Block");
  fm.resize(1);
  fm[0]  = Teuchos::rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
  buildEvaluators(*fm[0], *meshSpecs[0], stateMgr, BUILD_RESID_FM, 
		  Teuchos::null);
  constructDirichletEvaluators(*meshSpecs[0]);
}

Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
Albany::Helmholtz2DProblem::
buildEvaluators(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fmchoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructeEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  ConstructEvaluatorsOp<Helmholtz2DProblem> op(
    *this, fm0, meshSpecs, stateMgr, fmchoice, responseList);
  Sacado::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes> fe(op);
  return *op.tags;
}

void
Albany::Helmholtz2DProblem::constructDirichletEvaluators(
        const Albany::MeshSpecsStruct& meshSpecs)
{
   // Construct Dirichlet evaluators for all nodesets and names
   std::vector<std::string> dirichletNames(neq);
   dirichletNames[0] = "U";
   dirichletNames[1] = "V";
   Albany::BCUtils<Albany::DirichletTraits> dirUtils;
   dfm = dirUtils.constructBCEvaluators(meshSpecs.nsNames, dirichletNames,
                                          this->params, this->paramLib);
   offsets_ = dirUtils.getOffsets(); 
}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::Helmholtz2DProblem::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidHelmhotz2DProblemParams");
  validPL->set<double>("Left BC", 0.0, "Value of Left BC [required]");
  validPL->set<double>("Right BC", 0.0, "Value to Right BC [required]");
  validPL->set<double>("Top BC", 0.0, "Value of Top BC [required]");
  validPL->set<double>("Bottom BC", 0.0, "Value to Bottom BC [required]");
  validPL->set<double>("Ksqr", 1.0, "Value of wavelength-squared [required]");

  return validPL;
}
