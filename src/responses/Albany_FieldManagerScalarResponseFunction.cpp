//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include "Albany_FieldManagerScalarResponseFunction.hpp"
#include <algorithm>
#include "PHAL_Utilities.hpp"

#include "Albany_TpetraThyraUtils.hpp"

Albany::FieldManagerScalarResponseFunction::
FieldManagerScalarResponseFunction(
  const Teuchos::RCP<Albany::Application>& application_,
  const Teuchos::RCP<Albany::AbstractProblem>& problem_,
  const Teuchos::RCP<Albany::MeshSpecsStruct>&  meshSpecs_,
  const Teuchos::RCP<Albany::StateManager>& stateMgr_,
  Teuchos::ParameterList& responseParams) :
  ScalarResponseFunction(application_->getComm()),
  application(application_),
  problem(problem_),
  meshSpecs(meshSpecs_),
  stateMgr(stateMgr_),
  performedPostRegSetup(false)
{
  setup(responseParams);
}

Albany::FieldManagerScalarResponseFunction::
FieldManagerScalarResponseFunction(
  const Teuchos::RCP<Albany::Application>& application_,
  const Teuchos::RCP<Albany::AbstractProblem>& problem_,
  const Teuchos::RCP<Albany::MeshSpecsStruct>&  meshSpecs_,
  const Teuchos::RCP<Albany::StateManager>& stateMgr_) :
  ScalarResponseFunction(application_->getComm()),
  application(application_),
  problem(problem_),
  meshSpecs(meshSpecs_),
  stateMgr(stateMgr_),
  performedPostRegSetup(false)
{
}

void
Albany::FieldManagerScalarResponseFunction::
setup(Teuchos::ParameterList& responseParams)
{
  Teuchos::RCP<const Teuchos_Comm> commT = application->getComm();

  // FIXME: The adding of the Phalanx Graph Viz parameter
  // below causes problems if this function is called with
  // the same responseParams more than once. This happens
  // when the meshSpecs is but one entry in an array
  // of meshSpecs, which happens in meshes with multiple
  // blocks. In addition, if the building of evaluators
  // below does not recognize the Phalanx Graph Viz parameter,
  // then an exception will be thrown. Quick and dirty fix:
  // Remove the option if it already exists before building
  // the evaluators, it will be added again below anyhow.
  responseParams.remove("Phalanx Graph Visualization Detail", false);

  // Restrict to the element block?
  const char* reb_parm = "Restrict to Element Block";
  const bool
    reb_parm_present = responseParams.isType<bool>(reb_parm),
    reb = reb_parm_present && responseParams.get<bool>(reb_parm, false);
  element_block_index = reb ? meshSpecs->ebNameToIndex[meshSpecs->ebName] : -1;
  if (reb_parm_present) responseParams.remove(reb_parm, false);

  // Create field manager
  rfm = Teuchos::rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
    
  // Create evaluators for field manager
  Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> > tags = 
    problem->buildEvaluators(*rfm, *meshSpecs, *stateMgr, 
                             BUILD_RESPONSE_FM,
                             Teuchos::rcp(&responseParams,false));
  int rank = tags[0]->dataLayout().rank();
  num_responses = tags[0]->dataLayout().dimension(rank-1);
  if (num_responses == 0)
    num_responses = 1;
  
  // MPerego: In order to do post-registration setup, need to call postRegSetup function,
  // which is now called in AlbanyApplications (at this point the derivative dimensions cannot be
  // computed correctly because the discretization has not been created yet). 

  // Visualize rfm graph -- get file name from name of response function
  // (with spaces replaced by _ and lower case)
  vis_response_graph = 
    responseParams.get("Phalanx Graph Visualization Detail", 0);
  vis_response_name = responseParams.get<std::string>("Name");
  std::replace(vis_response_name.begin(), vis_response_name.end(), ' ', '_');
  std::transform(vis_response_name.begin(), vis_response_name.end(), 
		 vis_response_name.begin(), ::tolower);

  if (reb_parm_present) responseParams.set<bool>(reb_parm, reb);
}

Albany::FieldManagerScalarResponseFunction::
~FieldManagerScalarResponseFunction()
{
}

unsigned int
Albany::FieldManagerScalarResponseFunction::
numResponses() const 
{
  return num_responses;
}

//amb This is not right because rfm doesn't account for multiple element
// blocks. Make do for now. Also, rewrite this code to get rid of all this
// redundancy.
void Albany::FieldManagerScalarResponseFunction::
postRegSetup()
{
  { std::vector<PHX::index_size_type> derivative_dimensions;
    derivative_dimensions.push_back(
      PHAL::getDerivativeDimensions<PHAL::AlbanyTraits::Jacobian>(
        application.get(), meshSpecs.get()));
    rfm->setKokkosExtendedDataTypeDimensions<PHAL::AlbanyTraits::Jacobian>(
      derivative_dimensions); }
  { std::vector<PHX::index_size_type> derivative_dimensions;
    derivative_dimensions.push_back(
      PHAL::getDerivativeDimensions<PHAL::AlbanyTraits::Tangent>(
        application.get(), meshSpecs.get()));
    rfm->setKokkosExtendedDataTypeDimensions<PHAL::AlbanyTraits::Tangent>(
      derivative_dimensions); }
  // MP implementation gets deriv info from the regular evaluation types
  { std::vector<PHX::index_size_type> derivative_dimensions;
    derivative_dimensions.push_back(
      PHAL::getDerivativeDimensions<PHAL::AlbanyTraits::DistParamDeriv>(
        application.get(), meshSpecs.get()));
    rfm->setKokkosExtendedDataTypeDimensions<PHAL::AlbanyTraits::DistParamDeriv>(
      derivative_dimensions); }
  rfm->postRegistrationSetup("");
  performedPostRegSetup = true;
}

template<typename EvalT>
void Albany::FieldManagerScalarResponseFunction::
evaluate (PHAL::Workset& workset) {
  const WorksetArray<int>::type&
    wsPhysIndex = application->getDiscretization()->getWsPhysIndex();
  rfm->preEvaluate<EvalT>(workset);
  for (int ws = 0, numWorksets = application->getNumWorksets();
       ws < numWorksets; ws++) {
    if (element_block_index >= 0 && element_block_index != wsPhysIndex[ws])
      continue;
    application->loadWorksetBucketInfo<EvalT>(workset, ws);
    rfm->evaluateFields<EvalT>(workset);
  }
  rfm->postEvaluate<EvalT>(workset);
}

void
Albany::FieldManagerScalarResponseFunction::
evaluateResponse(const double current_time,
     const Teuchos::RCP<const Thyra_Vector>& x,
     const Teuchos::RCP<const Thyra_Vector>& xdot,
     const Teuchos::RCP<const Thyra_Vector>& xdotdot,
		 const Teuchos::Array<ParamVec>& p,
		 Tpetra_Vector& gT)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
      !performedPostRegSetup, Teuchos::Exceptions::InvalidParameter,
      std::endl << "Post registration setup not performed in field manager " <<
      std::endl << "Forgot to call \"postRegSetup\"? ");

  visResponseGraph<PHAL::AlbanyTraits::Residual>("");

  // Set data in Workset struct
  PHAL::Workset workset;
  application->setupBasicWorksetInfo(workset, current_time, x, xdot, xdotdot, p);
  workset.gT = Teuchos::rcp(&gT,false);

  // Perform fill via field manager
  evaluate<PHAL::AlbanyTraits::Residual>(workset);
}

void
Albany::FieldManagerScalarResponseFunction::
evaluateTangent(const double alpha, 
		const double beta,
		const double omega,
		const double current_time,
		bool sum_derivs,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
		const Teuchos::Array<ParamVec>& p,
		ParamVec* deriv_p,
    const Teuchos::RCP<const Thyra_MultiVector>& Vx,
    const Teuchos::RCP<const Thyra_MultiVector>& Vxdot,
    const Teuchos::RCP<const Thyra_MultiVector>& Vxdotdot,
    const Teuchos::RCP<const Thyra_MultiVector>& Vp,
		Tpetra_Vector* gT,
		Tpetra_MultiVector* gxT,
		Tpetra_MultiVector* gpT)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
      !performedPostRegSetup, Teuchos::Exceptions::InvalidParameter,
      std::endl << "Post registration setup not performed in field manager " <<
      std::endl << "Forgot to call \"postRegSetup\"? ");

  visResponseGraph<PHAL::AlbanyTraits::Tangent>("_tangent");
  
  // Set data in Workset struct
  PHAL::Workset workset;
  application->setupTangentWorksetInfo(workset, current_time, sum_derivs, 
                x, xdot, xdotdot, p, deriv_p, Vx, Vxdot, Vxdotdot, Vp);
  workset.gT = Teuchos::rcp(gT, false);
  workset.dgdxT = Teuchos::rcp(gxT, false);
  workset.dgdpT = Teuchos::rcp(gpT, false);
  
  // Perform fill via field manager
  evaluate<PHAL::AlbanyTraits::Tangent>(workset);
}

void
Albany::FieldManagerScalarResponseFunction::
evaluateGradient(const double current_time,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
		const Teuchos::Array<ParamVec>& p,
		ParamVec* deriv_p,
		Tpetra_Vector* gT,
		Tpetra_MultiVector* dg_dxT,
		Tpetra_MultiVector* dg_dxdotT,
		Tpetra_MultiVector* dg_dxdotdotT,
		Tpetra_MultiVector* dg_dpT)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
      !performedPostRegSetup, Teuchos::Exceptions::InvalidParameter,
      std::endl << "Post registration setup not performed in field manager " <<
      std::endl << "Forgot to call \"postRegSetup\"? ");

  visResponseGraph<PHAL::AlbanyTraits::Jacobian>("_gradient");

  // Set data in Workset struct
  PHAL::Workset workset;
  application->setupBasicWorksetInfo(workset, current_time, x, xdot, xdotdot, p);
  
  workset.gT = Teuchos::rcp(gT, false);
  
  // Perform fill via field manager (dg/dx)
  if (dg_dxT != NULL) {
    workset.m_coeff = 0.0;
    workset.j_coeff = 1.0;
    workset.n_coeff = 0.0;
    workset.dgdxT = Teuchos::rcp(dg_dxT, false);
    workset.overlapped_dgdxT = 
      Teuchos::rcp(new Tpetra_MultiVector(workset.x_importerT->getTargetMap(),
					  dg_dxT->getNumVectors()));
    evaluate<PHAL::AlbanyTraits::Jacobian>(workset);
  }

  // Perform fill via field manager (dg/dxdot)
  if (dg_dxdotT != NULL) {
    workset.m_coeff = 1.0;
    workset.j_coeff = 0.0;
    workset.n_coeff = 0.0;
    workset.dgdxT = Teuchos::null;
    workset.dgdxdotT = Teuchos::rcp(dg_dxdotT, false);
    workset.overlapped_dgdxdotT = 
      Teuchos::rcp(new Tpetra_MultiVector(workset.x_importerT->getTargetMap(),
					  dg_dxdotT->getNumVectors()));
    evaluate<PHAL::AlbanyTraits::Jacobian>(workset);
  }  
  // Perform fill via field manager (dg/dxdotdot)
  if (dg_dxdotdotT != NULL) {
    workset.m_coeff = 0.0;
    workset.j_coeff = 0.0;
    workset.n_coeff = 1.0;
    workset.dgdxT = Teuchos::null;
    workset.dgdxdotdotT = Teuchos::rcp(dg_dxdotdotT, false);
    workset.overlapped_dgdxdotdotT = 
      Teuchos::rcp(new Tpetra_MultiVector(workset.x_importerT->getTargetMap(),
					  dg_dxdotdotT->getNumVectors()));
    evaluate<PHAL::AlbanyTraits::Jacobian>(workset);
  }  
}

void
Albany::FieldManagerScalarResponseFunction::
evaluateDistParamDeriv(
    const double current_time,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec>& param_array,
    const std::string& dist_param_name,
    Tpetra_MultiVector* dg_dpT)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
      !performedPostRegSetup, Teuchos::Exceptions::InvalidParameter,
      std::endl << "Post registration setup not performed in field manager " <<
      std::endl << "Forgot to call \"postRegSetup\"? ");

  // Set data in Workset struct
  PHAL::Workset workset;

  application->setupBasicWorksetInfo(workset, current_time, x, xdot, xdotdot, param_array);

  // Perform fill via field manager (dg/dx)
  if(dg_dpT != NULL) {
    workset.dist_param_deriv_name = dist_param_name;
    workset.dgdpT = Teuchos::rcp(dg_dpT, false);
    {
      Teuchos::RCP<Tpetra_MultiVector> overlapped_dgdpT = Teuchos::rcp(
        new Tpetra_MultiVector(
          workset.distParamLib->get(dist_param_name)->overlap_map(),
          dg_dpT->getNumVectors()));
      workset.overlapped_dgdpT = overlapped_dgdpT;
    }
    evaluate<PHAL::AlbanyTraits::DistParamDeriv>(workset);
  }
}
