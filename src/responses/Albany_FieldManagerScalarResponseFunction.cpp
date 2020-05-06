//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_FieldManagerScalarResponseFunction.hpp"
#include "Albany_DistributedParameterLibrary.hpp"
#include "Albany_MeshSpecs.hpp"
#include "Albany_Application.hpp"
#include "Albany_AbstractProblem.hpp"
#include "Albany_StateManager.hpp"

#include "PHAL_Utilities.hpp"

namespace Albany
{

FieldManagerScalarResponseFunction::
FieldManagerScalarResponseFunction(
  const Teuchos::RCP<Application>& application_,
  const Teuchos::RCP<AbstractProblem>& problem_,
  const Teuchos::RCP<MeshSpecsStruct>&  meshSpecs_,
  const Teuchos::RCP<StateManager>& stateMgr_,
  Teuchos::ParameterList& responseParams) 
 : ScalarResponseFunction(application_->getComm())
 , application(application_)
 , problem(problem_)
 , meshSpecs(meshSpecs_)
 , stateMgr(stateMgr_)
 , vis_response_graph(0)
 , performedPostRegSetup(false)
{
  setup(responseParams);
}

FieldManagerScalarResponseFunction::
FieldManagerScalarResponseFunction(
  const Teuchos::RCP<Application>& application_,
  const Teuchos::RCP<AbstractProblem>& problem_,
  const Teuchos::RCP<MeshSpecsStruct>&  meshSpecs_,
  const Teuchos::RCP<StateManager>& stateMgr_)
 : ScalarResponseFunction(application_->getComm())
 , application(application_)
 , problem(problem_)
 , meshSpecs(meshSpecs_)
 , stateMgr(stateMgr_)
 , num_responses(0)
 , vis_response_graph(0)
 , element_block_index(0)
 , performedPostRegSetup(false)
{
  // Nothing to be done here
}

void FieldManagerScalarResponseFunction::
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
  const char* phx_graph_parm = "Phalanx Graph Visualization Detail";
  const bool phx_graph_parm_present = responseParams.isType<int>(phx_graph_parm);
  if (phx_graph_parm_present) {
    vis_response_graph = responseParams.get(phx_graph_parm, 0);
    responseParams.remove("Phalanx Graph Visualization Detail", false);
  }

  // Visualize rfm graph -- get file name from name of response function
  // (with spaces replaced by _ and lower case)
  vis_response_name = responseParams.get<std::string>("Name");
  std::replace(vis_response_name.begin(), vis_response_name.end(), ' ', '_');
  std::transform(vis_response_name.begin(), vis_response_name.end(),
		 vis_response_name.begin(), ::tolower);

  // Restrict to the element block?
  const char* reb_parm = "Restrict to Element Block";
  const bool
    reb_parm_present = responseParams.isType<bool>(reb_parm),
    reb = reb_parm_present && responseParams.get<bool>(reb_parm, false);
  element_block_index = reb ? meshSpecs->ebNameToIndex[meshSpecs->ebName] : -1;
  if (reb_parm_present) {
    responseParams.remove(reb_parm, false);
  }
  // Create field manager
  rfm = Teuchos::rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
    
  // Create evaluators for field manager
  Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> > tags = 
    problem->buildEvaluators(*rfm, *meshSpecs, *stateMgr, 
                             BUILD_RESPONSE_FM,
                             Teuchos::rcp(&responseParams,false));
  int rank = tags[0]->dataLayout().rank();
  num_responses = tags[0]->dataLayout().extent(rank-1);
  if (num_responses == 0) {
    num_responses = 1;
  }
  // MPerego: In order to do post-registration setup, need to call postRegSetup function,
  // which is now called in AlbanyApplications (at this point the derivative dimensions cannot be
  // computed correctly because the discretization has not been created yet). 

  if (phx_graph_parm_present) responseParams.set<int>(phx_graph_parm, vis_response_graph);
  if (reb_parm_present) responseParams.set<bool>(reb_parm, reb);
}

template <typename EvalT>
void FieldManagerScalarResponseFunction::
postRegDerivImpl()
{
  const auto phxSetup = application->getPhxSetup();
  std::vector<PHX::index_size_type> derivative_dimensions;
  derivative_dimensions.push_back(
      PHAL::getDerivativeDimensions<EvalT>(application.get(), meshSpecs.get(), true));
  rfm->setKokkosExtendedDataTypeDimensions<EvalT>(derivative_dimensions);
  rfm->postRegistrationSetupForType<EvalT>(*phxSetup);
}

template <>
void FieldManagerScalarResponseFunction::
postRegImpl<PHAL::AlbanyTraits::Residual>()
{
  using EvalT = PHAL::AlbanyTraits::Residual;
  const auto phxSetup = application->getPhxSetup();
  rfm->postRegistrationSetupForType<EvalT>(*phxSetup);
}

template <>
void FieldManagerScalarResponseFunction::
postRegImpl<PHAL::AlbanyTraits::Jacobian>()
{
  postRegDerivImpl<PHAL::AlbanyTraits::Jacobian>();
}

template <>
void FieldManagerScalarResponseFunction::
postRegImpl<PHAL::AlbanyTraits::Tangent>()
{
  postRegDerivImpl<PHAL::AlbanyTraits::Tangent>();
}

template <>
void FieldManagerScalarResponseFunction::
postRegImpl<PHAL::AlbanyTraits::DistParamDeriv>()
{
  postRegDerivImpl<PHAL::AlbanyTraits::DistParamDeriv>();
}

template <typename EvalT>
void FieldManagerScalarResponseFunction::
postReg()
{
  const auto phxSetup = application->getPhxSetup();

  const std::string evalName = PHAL::evalName<EvalT>("RFM",0) + "_" + vis_response_name;
  phxSetup->insert_eval(evalName);

  postRegImpl<EvalT>();

  // Update phalanx saved/unsaved fields based on field dependencies
  phxSetup->check_fields(rfm->getFieldTagsForSizing<EvalT>());
  phxSetup->update_fields();

  writePhalanxGraph<EvalT>(evalName);
}

template <typename EvalT>
void FieldManagerScalarResponseFunction::
writePhalanxGraph(const std::string& evalName)
{
  if (vis_response_graph > 0) {
    const bool detail = (vis_response_graph > 1) ? true : false;
    Teuchos::RCP<Teuchos::FancyOStream> out =
      Teuchos::VerboseObjectBase::getDefaultOStream();
    *out << "Phalanx writing graphviz file for graph of " << evalName <<
        " (detail = " << vis_response_graph << ")" << std::endl;
    const std::string graphName = "phalanxGraph" + evalName;
    *out << "Process using 'dot -Tpng -O " << graphName << std::endl;
    rfm->writeGraphvizFile<EvalT>(graphName, detail, detail);

    // Print phalanx setup info
    const auto phxSetup = application->getPhxSetup();
    phxSetup->print(*out);
  }
}

//amb This is not right because rfm doesn't account for multiple element
// blocks. Make do for now. Also, rewrite this code to get rid of all this
// redundancy.
void FieldManagerScalarResponseFunction::
postRegSetup()
{
  postReg<PHAL::AlbanyTraits::Residual>();
  postReg<PHAL::AlbanyTraits::Jacobian>();
  postReg<PHAL::AlbanyTraits::Tangent>();
  postReg<PHAL::AlbanyTraits::DistParamDeriv>();
  performedPostRegSetup = true;
}

template<typename EvalT>
void FieldManagerScalarResponseFunction::
evaluate(PHAL::Workset& workset)
{
  const WorksetArray<int>::type&
    wsPhysIndex = application->getDiscretization()->getWsPhysIndex();
  rfm->preEvaluate<EvalT>(workset);
  for (int ws = 0, numWorksets = application->getNumWorksets();
       ws < numWorksets; ws++) {
    if (element_block_index >= 0 && element_block_index != wsPhysIndex[ws])
      continue;
    const std::string evalName = PHAL::evalName<EvalT>("RFM", wsPhysIndex[ws]) + "_" + vis_response_name;
    application->loadWorksetBucketInfo<EvalT>(workset, ws, evalName);
    rfm->evaluateFields<EvalT>(workset);
  }
  rfm->postEvaluate<EvalT>(workset);
}

void FieldManagerScalarResponseFunction::
evaluateResponse(const double current_time,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
		const Teuchos::Array<ParamVec>& p,
    const Teuchos::RCP<Thyra_Vector>& g)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
      !performedPostRegSetup, Teuchos::Exceptions::InvalidParameter,
      std::endl << "Post registration setup not performed in field manager " <<
      std::endl << "Forgot to call \"postRegSetup\"? ");

  // Set data in Workset struct
  PHAL::Workset workset;
  application->setupBasicWorksetInfo(workset, current_time, x, xdot, xdotdot, p);
  workset.g = g;

  // Perform fill via field manager
  evaluate<PHAL::AlbanyTraits::Residual>(workset);
}

void FieldManagerScalarResponseFunction::
evaluateTangent(const double /* alpha */, 
		const double /* beta */,
		const double /* omega */,
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
    const Teuchos::RCP<Thyra_Vector>& g,
    const Teuchos::RCP<Thyra_MultiVector>& gx,
    const Teuchos::RCP<Thyra_MultiVector>& gp)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
      !performedPostRegSetup, Teuchos::Exceptions::InvalidParameter,
      std::endl << "Post registration setup not performed in field manager " <<
      std::endl << "Forgot to call \"postRegSetup\"? ");

  // Set data in Workset struct
  PHAL::Workset workset;
  application->setupTangentWorksetInfo(workset, current_time, sum_derivs, 
                x, xdot, xdotdot, p, deriv_p, Vx, Vxdot, Vxdotdot, Vp);
  workset.g = g;
  workset.dgdx = gx;
  workset.dgdp = gp;
  
  // Perform fill via field manager
  evaluate<PHAL::AlbanyTraits::Tangent>(workset);
}

void FieldManagerScalarResponseFunction::
evaluateGradient(const double current_time,
  const Teuchos::RCP<const Thyra_Vector>& x,
  const Teuchos::RCP<const Thyra_Vector>& xdot,
  const Teuchos::RCP<const Thyra_Vector>& xdotdot,
	const Teuchos::Array<ParamVec>& p,
	ParamVec* /* deriv_p */,
  const Teuchos::RCP<Thyra_Vector>& g,
  const Teuchos::RCP<Thyra_MultiVector>& dg_dx,
  const Teuchos::RCP<Thyra_MultiVector>& dg_dxdot,
  const Teuchos::RCP<Thyra_MultiVector>& dg_dxdotdot,
  const Teuchos::RCP<Thyra_MultiVector>& /* dg_dp */)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
      !performedPostRegSetup, Teuchos::Exceptions::InvalidParameter,
      std::endl << "Post registration setup not performed in field manager " <<
      std::endl << "Forgot to call \"postRegSetup\"? ");

  // Set data in Workset struct
  PHAL::Workset workset;
  application->setupBasicWorksetInfo(workset, current_time, x, xdot, xdotdot, p);
  
  workset.g = g;
  
  // Perform fill via field manager (dg/dx)
  if (!dg_dx.is_null()) {
    workset.m_coeff = 0.0;
    workset.j_coeff = 1.0;
    workset.n_coeff = 0.0;
    workset.dgdx = dg_dx;
    workset.overlapped_dgdx = Thyra::createMembers(workset.x_cas_manager->getOverlappedVectorSpace(),dg_dx->domain()->dim());
    evaluate<PHAL::AlbanyTraits::Jacobian>(workset);
  }

  // Perform fill via field manager (dg/dxdot)
  if (!dg_dxdot.is_null()) {
    workset.m_coeff = 1.0;
    workset.j_coeff = 0.0;
    workset.n_coeff = 0.0;
    // LB: WHY?!?! 
    workset.dgdx = Teuchos::null;
    workset.dgdxdot = dg_dxdot;
    workset.overlapped_dgdxdot = Thyra::createMembers(workset.x_cas_manager->getOverlappedVectorSpace(),dg_dxdot->domain()->dim());
    evaluate<PHAL::AlbanyTraits::Jacobian>(workset);
  }  
  // Perform fill via field manager (dg/dxdotdot)
  if (!dg_dxdotdot.is_null()) {
    workset.m_coeff = 0.0;
    workset.j_coeff = 0.0;
    workset.n_coeff = 1.0;
    // LB: WHY?!?! 
    workset.dgdx = Teuchos::null;
    workset.dgdxdot = Teuchos::null;
    workset.dgdxdotdot = dg_dxdotdot;
    workset.overlapped_dgdxdotdot = Thyra::createMembers(workset.x_cas_manager->getOverlappedVectorSpace(),dg_dxdotdot->domain()->dim());
    evaluate<PHAL::AlbanyTraits::Jacobian>(workset);
  }  
}

void FieldManagerScalarResponseFunction::
evaluateDistParamDeriv(
    const double current_time,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec>& param_array,
    const std::string& dist_param_name,
    const Teuchos::RCP<Thyra_MultiVector>& dg_dp)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
      !performedPostRegSetup, Teuchos::Exceptions::InvalidParameter,
      std::endl << "Post registration setup not performed in field manager " <<
      std::endl << "Forgot to call \"postRegSetup\"? ");

  // Set data in Workset struct
  PHAL::Workset workset;

  application->setupBasicWorksetInfo(workset, current_time, x, xdot, xdotdot, param_array);

  // Perform fill via field manager (dg/dx)
  if(!dg_dp.is_null()) {
    workset.dist_param_deriv_name = dist_param_name;
    workset.p_cas_manager = workset.distParamLib->get(dist_param_name)->get_cas_manager();
    workset.dgdp = dg_dp;
    workset.overlapped_dgdp = Thyra::createMembers(workset.p_cas_manager->getOverlappedVectorSpace(),dg_dp->domain()->dim());
    evaluate<PHAL::AlbanyTraits::DistParamDeriv>(workset);
  }
}

} // namespace Albany
