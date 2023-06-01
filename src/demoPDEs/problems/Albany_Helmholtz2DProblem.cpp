//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_Helmholtz2DProblem.hpp"
#include "Albany_Utils.hpp"
#include "Albany_BCUtils.hpp"
#include "Albany_ProblemUtils.hpp"

namespace Albany {

Helmholtz2DProblem::
Helmholtz2DProblem(const Teuchos::RCP<Teuchos::ParameterList>& params_,
                   const Teuchos::RCP<ParamLib>& paramLib_)
 : AbstractProblem(params_, paramLib_, 2)
 , use_sdbcs_(false)
{
  std::string& method = params->get("Name", "Helmholtz 2D Problem");
  *out << "Problem Name = " << method << std::endl;
  
  ksqr = params->get<double>("Ksqr",1.0);

  haveSource =  params->isSublist("Source Functions");
}

void
Helmholtz2DProblem::
buildProblem (Teuchos::ArrayRCP<Teuchos::RCP<MeshSpecs>> meshSpecs,
              StateManager& stateMgr)
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
Helmholtz2DProblem::
buildEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                 const MeshSpecs& meshSpecs,
                 StateManager& stateMgr,
                 FieldManagerChoice fmchoice,
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
Helmholtz2DProblem::
constructDirichletEvaluators(const MeshSpecs& meshSpecs)
{
  // Construct Dirichlet evaluators for all nodesets and names
  std::vector<std::string> dirichletNames(neq);
  dirichletNames[0] = "U";
  dirichletNames[1] = "V";
  BCUtils<DirichletTraits> dirUtils;
  dfm = dirUtils.constructBCEvaluators(meshSpecs.nsNames, dirichletNames,
                                         this->params, this->paramLib);
  use_sdbcs_ = dirUtils.useSDBCs(); 
  offsets_ = dirUtils.getOffsets(); 
  nodeSetIDs_ = dirUtils.getNodeSetIDs();
}

Teuchos::RCP<const Teuchos::ParameterList>
Helmholtz2DProblem::getValidProblemParameters() const
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

} // namespace Albany
