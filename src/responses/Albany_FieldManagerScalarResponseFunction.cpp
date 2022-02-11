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

#include "Albany_Hessian.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Thyra_VectorStdOps.hpp"

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
  application->setDynamicLayoutSizes<EvalT>(rfm);
  rfm->postRegistrationSetupForType<EvalT>(*phxSetup);
}

template <>
void FieldManagerScalarResponseFunction::
postRegImpl<PHAL::AlbanyTraits::Residual>()
{
  using EvalT = PHAL::AlbanyTraits::Residual;
  const auto phxSetup = application->getPhxSetup();
  application->setDynamicLayoutSizes<EvalT>(rfm);
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

template <>
void FieldManagerScalarResponseFunction::
postRegImpl<PHAL::AlbanyTraits::HessianVec>()
{
  postRegDerivImpl<PHAL::AlbanyTraits::HessianVec>();
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
  postReg<PHAL::AlbanyTraits::HessianVec>();
  performedPostRegSetup = true;
}

template<typename EvalT>
void FieldManagerScalarResponseFunction::
evaluate(PHAL::Workset& workset)
{
  const auto& wsPhysIndex = application->getDiscretization()->getWsPhysIndex();
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

  if (g_.is_null())
    g_ = Thyra::createMember(g->space());

  g_->assign(*g);
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
		Teuchos::Array<ParamVec>& p,
    const int parameter_index,
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
                x, xdot, xdotdot, p, parameter_index, Vx, Vxdot, Vxdotdot, Vp);
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
	const int  /*parameter_index*/,
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

void FieldManagerScalarResponseFunction::
evaluate_HessVecProd_xx(
    const double current_time,
    const Teuchos::RCP<const Thyra_MultiVector>& v,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec>& param_array,
    const Teuchos::RCP<Thyra_MultiVector>& Hv_dp)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
      !performedPostRegSetup, Teuchos::Exceptions::InvalidParameter,
      std::endl << "Post registration setup not performed in field manager " <<
      std::endl << "Forgot to call \"postRegSetup\"? ");

  // Set data in Workset struct
  PHAL::Workset workset;

  application->setupBasicWorksetInfo(workset, current_time, x, xdot, xdotdot, param_array);

  if(!v.is_null()) {
    workset.hessianWorkset.direction_x = Thyra::createMembers(workset.x_cas_manager->getOverlappedVectorSpace(),v->domain()->dim());
    workset.x_cas_manager->scatter(v->clone_mv(), workset.hessianWorkset.direction_x, Albany::CombineMode::INSERT);
  }

  if(!Hv_dp.is_null()) {
    workset.j_coeff = 1.0;
    workset.hessianWorkset.hess_vec_prod_g_xx = Hv_dp;
    workset.hessianWorkset.overlapped_hess_vec_prod_g_xx = Thyra::createMembers(workset.x_cas_manager->getOverlappedVectorSpace(),Hv_dp->domain()->dim());
    evaluate<PHAL::AlbanyTraits::HessianVec>(workset);
  }
}

void FieldManagerScalarResponseFunction::
evaluate_HessVecProd_xp(
    const double current_time,
    const Teuchos::RCP<const Thyra_MultiVector>& v,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec>& param_array,
    const std::string& param_direction_name,
    const Teuchos::RCP<Thyra_MultiVector>& Hv_dp)
{
  // First, the function checks whether the parameter associated to param_direction_name
  // is a distributed parameter (l2_is_distributed==true) or a parameter vector
  // (l2_is_distributed==false).
  int l2;
  bool l2_is_distributed;
  Albany::getParameterVectorID(l2, l2_is_distributed, param_direction_name);

  TEUCHOS_TEST_FOR_EXCEPTION(
      !performedPostRegSetup, Teuchos::Exceptions::InvalidParameter,
      std::endl << "Post registration setup not performed in field manager " <<
      std::endl << "Forgot to call \"postRegSetup\"? ");

  // Set data in Workset struct
  PHAL::Workset workset;

  application->setupBasicWorksetInfo(workset, current_time, x, xdot, xdotdot, param_array);

  // If the parameter associated to param_direction_name is a parameter vector, 
  // the initialization of the second derivatives must be performed now:
  if (!l2_is_distributed) {
    ParamVec params_l2 = param_array[l2];
    unsigned int num_cols_p_l2 = params_l2.size();

    Teuchos::ArrayRCP<const ST> v_constView;
    if(!v.is_null()) {
      v_constView = Albany::getLocalData(v->col(0));
    }

    HessianVecFad p_val;
    for (unsigned int i = 0; i < num_cols_p_l2; i++) {
      p_val = params_l2[i].family->getValue<PHAL::AlbanyTraits::HessianVec>();
      p_val.val().fastAccessDx(0) = v_constView[i];
      params_l2[i].family->setValue<PHAL::AlbanyTraits::HessianVec>(p_val);
    }
  }

  if(!v.is_null() && !Hv_dp.is_null()) {
    workset.j_coeff = 1.0;
    workset.hessianWorkset.dist_param_deriv_direction_name = param_direction_name;
    // If the parameter associated to param_direction_name is a distributed parameter,
    // the direction vectors should be scattered to have overlapped directions:
    if (l2_is_distributed) {
      workset.hessianWorkset.p_direction_cas_manager = workset.distParamLib->get(param_direction_name)->get_cas_manager();
      workset.hessianWorkset.direction_p = Thyra::createMembers(workset.hessianWorkset.p_direction_cas_manager->getOverlappedVectorSpace(),v->domain()->dim());
      workset.hessianWorkset.p_direction_cas_manager->scatter(v->clone_mv(), workset.hessianWorkset.direction_p, Albany::CombineMode::INSERT);
    }
    workset.hessianWorkset.hess_vec_prod_g_xp = Hv_dp;
    workset.hessianWorkset.overlapped_hess_vec_prod_g_xp = Thyra::createMembers(workset.x_cas_manager->getOverlappedVectorSpace(),Hv_dp->domain()->dim());
    evaluate<PHAL::AlbanyTraits::HessianVec>(workset);
  }
}

void FieldManagerScalarResponseFunction::
evaluate_HessVecProd_px(
    const double current_time,
    const Teuchos::RCP<const Thyra_MultiVector>& v,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec>& param_array,
    const std::string& param_name,
    const Teuchos::RCP<Thyra_MultiVector>& Hv_dp)
{
  // First, the function checks whether the parameter associated to param_name
  // is a distributed parameter (l1_is_distributed==true) or a parameter vector
  // (l1_is_distributed==false).
  int l1;
  bool l1_is_distributed;
  Albany::getParameterVectorID(l1, l1_is_distributed, param_name);

  TEUCHOS_TEST_FOR_EXCEPTION(
      !performedPostRegSetup, Teuchos::Exceptions::InvalidParameter,
      std::endl << "Post registration setup not performed in field manager " <<
      std::endl << "Forgot to call \"postRegSetup\"? ");

  // Set data in Workset struct
  PHAL::Workset workset;

  application->setupBasicWorksetInfo(workset, current_time, x, xdot, xdotdot, param_array);

  // If the parameter associated to param_name is a parameter vector, 
  // the initialization of the first derivatives must be performed now:
  if (!l1_is_distributed) {
    ParamVec params_l1 = param_array[l1];
    unsigned int num_cols_p_l1 = params_l1.size();

    HessianVecFad p_val;
    for (unsigned int i = 0; i < num_cols_p_l1; i++) {
      p_val = HessianVecFad(num_cols_p_l1, params_l1[i].baseValue);
      p_val.fastAccessDx(i).val() = 1.0;
      params_l1[i].family->setValue<PHAL::AlbanyTraits::HessianVec>(p_val);
    }
  }

  if(!v.is_null()) {
    workset.hessianWorkset.direction_x = Thyra::createMembers(workset.x_cas_manager->getOverlappedVectorSpace(),v->domain()->dim());
    workset.x_cas_manager->scatter(v->clone_mv(), workset.hessianWorkset.direction_x, Albany::CombineMode::INSERT);
  }

  if(!Hv_dp.is_null()) {
    workset.j_coeff = 1.0;
    workset.dist_param_deriv_name = param_name;
    if (l1_is_distributed) {
      workset.p_cas_manager = workset.distParamLib->get(param_name)->get_cas_manager();
      workset.hessianWorkset.overlapped_hess_vec_prod_g_px = Thyra::createMembers(workset.p_cas_manager->getOverlappedVectorSpace(),Hv_dp->domain()->dim());
    }
    else {
      auto overlapped = Hv_dp->col(0)->space();

      int n_local_params = 0;
      int n_total_params = overlapped->dim();

      if (workset.comm->getRank()==0)
        n_local_params = n_total_params;
      std::vector<GO> my_gids;
      for (int i=0; i<n_local_params; ++i)
        my_gids.push_back(i);
      Teuchos::ArrayView<GO> gids(my_gids);

      auto owned = Albany::createVectorSpace(workset.comm, gids, n_total_params);
      workset.p_cas_manager = createCombineAndScatterManager(owned, overlapped);
      workset.hessianWorkset.overlapped_hess_vec_prod_g_px =
        Thyra::createMembers(workset.p_cas_manager->getOverlappedVectorSpace(), Hv_dp->domain()->dim());
    }
    workset.hessianWorkset.hess_vec_prod_g_px = Hv_dp;
    evaluate<PHAL::AlbanyTraits::HessianVec>(workset);
  }
}

void FieldManagerScalarResponseFunction::
evaluate_HessVecProd_pp(
    const double current_time,
    const Teuchos::RCP<const Thyra_MultiVector>& v,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
    const Teuchos::Array<ParamVec>& param_array,
    const std::string& param_name,
    const std::string& param_direction_name,
    const Teuchos::RCP<Thyra_MultiVector>& Hv_dp)
{
  // First, the function checks whether the parameter associated to param_name
  // is a distributed parameter (l1_is_distributed==true) or a parameter vector
  // (l1_is_distributed==false).
  int l1;
  bool l1_is_distributed;
  Albany::getParameterVectorID(l1, l1_is_distributed, param_name);

  // Then the function checks whether the parameter associated to param_direction_name
  // is a distributed parameter (l2_is_distributed==true) or a parameter vector
  // (l2_is_distributed==false).
  int l2;
  bool l2_is_distributed;
  Albany::getParameterVectorID(l2, l2_is_distributed, param_direction_name);

  TEUCHOS_TEST_FOR_EXCEPTION(
      !performedPostRegSetup, Teuchos::Exceptions::InvalidParameter,
      std::endl << "Post registration setup not performed in field manager " <<
      std::endl << "Forgot to call \"postRegSetup\"? ");

  // Set data in Workset struct
  PHAL::Workset workset;

  application->setupBasicWorksetInfo(workset, current_time, x, xdot, xdotdot, param_array);

  // If the parameter associated to param_name is a parameter vector, 
  // the initialization of the first derivatives must be performed now:
  if (!l1_is_distributed) {
    ParamVec params_l1 = param_array[l1];
    unsigned int num_cols_p_l1 = params_l1.size();

    HessianVecFad p_val;
    for (unsigned int i = 0; i < num_cols_p_l1; i++) {
      p_val = HessianVecFad(num_cols_p_l1, params_l1[i].baseValue);
      p_val.fastAccessDx(i).val() = 1.0;
      params_l1[i].family->setValue<PHAL::AlbanyTraits::HessianVec>(p_val);
    }
  }

  // If the parameter associated to param_direction_name is a parameter vector, 
  // the initialization of the second derivatives must be performed now:
  if (!l2_is_distributed) {
    ParamVec params_l2 = param_array[l2];
    unsigned int num_cols_p_l2 = params_l2.size();

    Teuchos::ArrayRCP<const ST> v_constView;
    if(!v.is_null()) {
      v_constView = Albany::getLocalData(v->col(0));
    }

    HessianVecFad p_val;
    for (unsigned int i = 0; i < num_cols_p_l2; i++) {
      p_val = params_l2[i].family->getValue<PHAL::AlbanyTraits::HessianVec>();
      p_val.val().fastAccessDx(0) = v_constView[i];
      params_l2[i].family->setValue<PHAL::AlbanyTraits::HessianVec>(p_val);
    }
  }

  if(!v.is_null() && !Hv_dp.is_null()) {
    workset.dist_param_deriv_name = param_name;
    workset.hessianWorkset.dist_param_deriv_direction_name = param_direction_name;
    if (l1_is_distributed) {
      workset.p_cas_manager = workset.distParamLib->get(param_name)->get_cas_manager();
      workset.hessianWorkset.overlapped_hess_vec_prod_g_pp = Thyra::createMembers(workset.p_cas_manager->getOverlappedVectorSpace(),Hv_dp->domain()->dim());
    }
    else {
      auto overlapped = Hv_dp->col(0)->space();

      int n_local_params = 0;
      int n_total_params = overlapped->dim();

      if (workset.comm->getRank()==0)
        n_local_params = n_total_params;
      std::vector<GO> my_gids;
      for (int i=0; i<n_local_params; ++i)
        my_gids.push_back(i);
      Teuchos::ArrayView<GO> gids(my_gids);

      auto owned = Albany::createVectorSpace(workset.comm, gids, n_total_params);
      workset.p_cas_manager = createCombineAndScatterManager(owned, overlapped);
      workset.hessianWorkset.overlapped_hess_vec_prod_g_pp =
        Thyra::createMembers(workset.p_cas_manager->getOverlappedVectorSpace(), Hv_dp->domain()->dim());
    }
    // If the parameter associated to param_direction_name is a distributed parameter,
    // the direction vectors should be scattered to have overlapped directions:
    if (l2_is_distributed) {
      workset.hessianWorkset.p_direction_cas_manager = workset.distParamLib->get(param_direction_name)->get_cas_manager();
      workset.hessianWorkset.direction_p = Thyra::createMembers(workset.hessianWorkset.p_direction_cas_manager->getOverlappedVectorSpace(),v->domain()->dim());
      workset.hessianWorkset.p_direction_cas_manager->scatter(v->clone_mv(), workset.hessianWorkset.direction_p, Albany::CombineMode::INSERT);
    }
    workset.hessianWorkset.hess_vec_prod_g_pp = Hv_dp;
    evaluate<PHAL::AlbanyTraits::HessianVec>(workset);
  }
}

void
FieldManagerScalarResponseFunction::
printResponse(Teuchos::RCP<Teuchos::FancyOStream> out)
{
  if (g_.is_null()) {
    *out << " the response has not been evaluated yet!";
    return;
  }

  std::size_t precision = 8;
  std::size_t value_width = precision + 4;
  *out << std::setw(value_width) << Thyra::get_ele(*g_,0);
}
} // namespace Albany
