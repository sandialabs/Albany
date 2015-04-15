//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include "Albany_FieldManagerScalarResponseFunction.hpp"
#if defined(ALBANY_EPETRA)
#include "Petra_Converters.hpp"
#endif
#include <algorithm>
#include "PHAL_Utilities.hpp"

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
  stateMgr(stateMgr_)
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
  stateMgr(stateMgr_)
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
  
  // Do post-registration setup
  
  //amb This is not right because rfm doesn't account for multiple element
  // blocks. Make do for now. Also, rewrite this code to get rid of all this
  // redundancy.
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
  { std::vector<PHX::index_size_type> derivative_dimensions;
    derivative_dimensions.push_back(
      PHAL::getDerivativeDimensions<PHAL::AlbanyTraits::DistParamDeriv>(
        application.get(), meshSpecs.get()));
    rfm->setKokkosExtendedDataTypeDimensions<PHAL::AlbanyTraits::DistParamDeriv>(
      derivative_dimensions); }
  rfm->postRegistrationSetup("");

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
evaluateResponseT(const double current_time,
		 const Tpetra_Vector* xdotT,
		 const Tpetra_Vector* xdotdotT,
		 const Tpetra_Vector& xT,
		 const Teuchos::Array<ParamVec>& p,
		 Tpetra_Vector& gT)
{
  visResponseGraph<PHAL::AlbanyTraits::Residual>("");

  // Set data in Workset struct
  PHAL::Workset workset;
 
  application->setupBasicWorksetInfoT(workset, current_time, rcp(xdotT, false), rcp(xdotdotT, false), rcpFromRef(xT), p);
  workset.gT = Teuchos::rcp(&gT,false);

  // Perform fill via field manager
  evaluate<PHAL::AlbanyTraits::Residual>(workset);
}


void
Albany::FieldManagerScalarResponseFunction::
evaluateTangentT(const double alpha, 
		const double beta,
		const double omega,
		const double current_time,
		bool sum_derivs,
		const Tpetra_Vector* xdotT,
		const Tpetra_Vector* xdotdotT,
		const Tpetra_Vector& xT,
		const Teuchos::Array<ParamVec>& p,
		ParamVec* deriv_p,
		const Tpetra_MultiVector* VxdotT,
		const Tpetra_MultiVector* VxdotdotT,
		const Tpetra_MultiVector* VxT,
		const Tpetra_MultiVector* VpT,
		Tpetra_Vector* gT,
		Tpetra_MultiVector* gxT,
		Tpetra_MultiVector* gpT)
{
  visResponseGraph<PHAL::AlbanyTraits::Tangent>("_tangent");
  
  // Set data in Workset struct
  PHAL::Workset workset;
  application->setupTangentWorksetInfoT(workset, current_time, sum_derivs, 
				       rcp(xdotT), rcp(xdotdotT), rcpFromRef(xT), p, 
				       deriv_p, rcp(VxdotT), rcp(VxdotdotT), rcp(VxT), rcp(VpT));
  workset.gT = Teuchos::rcp(gT, false);
  workset.dgdxT = Teuchos::rcp(gxT, false);
  workset.dgdpT = Teuchos::rcp(gpT, false);
  
  // Perform fill via field manager
  evaluate<PHAL::AlbanyTraits::Tangent>(workset);
}

#if defined(ALBANY_EPETRA)
void
Albany::FieldManagerScalarResponseFunction::
evaluateGradient(const double current_time,
		 const Epetra_Vector* xdot,
		 const Epetra_Vector* xdotdot,
		 const Epetra_Vector& x,
		 const Teuchos::Array<ParamVec>& p,
		 ParamVec* deriv_p,
		 Epetra_Vector* g,
		 Epetra_MultiVector* dg_dx,
		 Epetra_MultiVector* dg_dxdot,
		 Epetra_MultiVector* dg_dxdotdot,
		 Epetra_MultiVector* dg_dp)
{
  visResponseGraph<PHAL::AlbanyTraits::Jacobian>("_gradient");
  Teuchos::RCP<const Teuchos_Comm> commT = application->getComm();
  //Create Tpetra copy of x, called xT
  Teuchos::RCP<const Tpetra_Vector> xT = Petra::EpetraVector_To_TpetraVectorConst(x, commT);
  //Create Tpetra copy of xdot, called xdotT
  Teuchos::RCP<const Tpetra_Vector> xdotT;
  if (xdot != NULL) {
    xdotT = Petra::EpetraVector_To_TpetraVectorConst(*xdot, commT);
   }
  //Create Tpetra copy of xdotdot, called xdotdotT
  Teuchos::RCP<const Tpetra_Vector> xdotdotT;
  if (xdotdot != NULL) {
    xdotdotT = Petra::EpetraVector_To_TpetraVectorConst(*xdotdot, commT);
   }

  // Set data in Workset struct
  PHAL::Workset workset;
  application->setupBasicWorksetInfoT(workset, current_time, xdotT, xdotdotT, xT, p);

  { //amb All the evaluator-based response functions that are called from here
    // use the T versions of the workset fields. So set those here, call
    // evaluateGradientT, copy to Epetra, and then return.
    //   If this is not done, there is a seg fault on dgT in
    //     SeparableScatterScalarResponse<PHAL::AlbanyTraits::
    //       Jacobian, Traits>::evaluateFields
    // because it is expecting Tpetra input rather than Epetra.
    //   Once it is confirmed this works, we can remove the remainder of this
    // method's impl.
    Teuchos::RCP<Tpetra_Vector> gT = g ?
      Petra::EpetraVector_To_TpetraVectorNonConst(*g, commT) :
      Teuchos::null;
    Teuchos::RCP<Tpetra_MultiVector> dg_dxT = dg_dx ?
      Petra::EpetraMultiVector_To_TpetraMultiVector(*dg_dx, commT):
      Teuchos::null;
    Teuchos::RCP<Tpetra_MultiVector> dg_dxdotT = dg_dxdot ?
      Petra::EpetraMultiVector_To_TpetraMultiVector(*dg_dxdot, commT):
      Teuchos::null;
    Teuchos::RCP<Tpetra_MultiVector> dg_dxdotdotT = dg_dxdotdot ?
      Petra::EpetraMultiVector_To_TpetraMultiVector(*dg_dxdotdot, commT):
      Teuchos::null;
    Teuchos::RCP<Tpetra_MultiVector> dg_dpT = dg_dp ?
      Petra::EpetraMultiVector_To_TpetraMultiVector(*dg_dp, commT):
      Teuchos::null;
    evaluateGradientT(current_time, xdotT.get(), xdotdotT.get(), *xT,
                      p, deriv_p, gT.get(), dg_dxT.get(), dg_dxdotT.get(),
                      dg_dxdotdotT.get(), dg_dpT.get());
    const Teuchos::RCP<const Epetra_Comm>
      comm = createEpetraCommFromTeuchosComm(commT);
    if (g)
      Petra::TpetraVector_To_EpetraVector(gT, *g, comm);
    if (dg_dx)
      Petra::TpetraMultiVector_To_EpetraMultiVector(dg_dxT, *dg_dx, comm);
    if (dg_dxdot)
      Petra::TpetraMultiVector_To_EpetraMultiVector(dg_dxdotT, *dg_dxdot, comm);
    if (dg_dxdotdot)
      Petra::TpetraMultiVector_To_EpetraMultiVector(
        dg_dxdotdotT, *dg_dxdotdot, comm);
    if (dg_dp)
      Petra::TpetraMultiVector_To_EpetraMultiVector(dg_dpT, *dg_dp, comm);
    return; }

  workset.g = Teuchos::rcp(g, false);
  
  // Perform fill via field manager (dg/dx)
  if (dg_dx != NULL) {
    Teuchos::RCP<Epetra_Comm> comm = Albany::createEpetraCommFromTeuchosComm(commT);
    workset.m_coeff = 0.0;
    workset.j_coeff = 1.0;
    workset.n_coeff = 0.0;
    workset.dgdx = Teuchos::rcp(dg_dx, false);
    Teuchos::RCP<const Tpetra_Map> tgt_map = workset.x_importerT->getTargetMap();
    workset.overlapped_dgdx = 
      Teuchos::rcp(new Epetra_MultiVector(*Petra::TpetraMap_To_EpetraMap(tgt_map, comm),
					  dg_dx->NumVectors()));
    evaluate<PHAL::AlbanyTraits::Jacobian>(workset);
  }

  // Perform fill via field manager (dg/dxdot)
  if (dg_dxdot != NULL) {
    workset.m_coeff = 1.0;
    workset.j_coeff = 0.0;
    workset.n_coeff = 0.0;
    workset.dgdx = Teuchos::null;
    workset.dgdxdot = Teuchos::rcp(dg_dxdot, false);
    workset.overlapped_dgdxdot = 
      Teuchos::rcp(new Epetra_MultiVector(workset.x_importer->TargetMap(),
					  dg_dxdot->NumVectors()));
    evaluate<PHAL::AlbanyTraits::Jacobian>(workset);
  }  

  // Perform fill via field manager (dg/dxdotdot)
  if (dg_dxdotdot != NULL) {
    workset.m_coeff = 0.0;
    workset.j_coeff = 0.0;
    workset.n_coeff = 1.0;
    workset.dgdx = Teuchos::null;
    workset.dgdxdotdot = Teuchos::rcp(dg_dxdotdot, false);
    workset.overlapped_dgdxdotdot = 
      Teuchos::rcp(new Epetra_MultiVector(workset.x_importer->TargetMap(),
					  dg_dxdotdot->NumVectors()));
    evaluate<PHAL::AlbanyTraits::Jacobian>(workset);
  }  
}
#endif

void
Albany::FieldManagerScalarResponseFunction::
evaluateGradientT(const double current_time,
		 const Tpetra_Vector* xdotT,
		 const Tpetra_Vector* xdotdotT,
		 const Tpetra_Vector& xT,
		 const Teuchos::Array<ParamVec>& p,
		 ParamVec* deriv_p,
		 Tpetra_Vector* gT,
		 Tpetra_MultiVector* dg_dxT,
		 Tpetra_MultiVector* dg_dxdotT,
		 Tpetra_MultiVector* dg_dxdotdotT,
		 Tpetra_MultiVector* dg_dpT)
{
  visResponseGraph<PHAL::AlbanyTraits::Jacobian>("_gradient");

  // Set data in Workset struct
  PHAL::Workset workset;
  application->setupBasicWorksetInfoT(workset, current_time, rcp(xdotT, false), rcp(xdotdotT, false), rcpFromRef(xT), p);
  
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

#if defined(ALBANY_EPETRA)
void
Albany::FieldManagerScalarResponseFunction::
evaluateDistParamDeriv(
      const double current_time,
      const Epetra_Vector* xdot,
      const Epetra_Vector* xdotdot,
      const Epetra_Vector& x,
      const Teuchos::Array<ParamVec>& param_array,
      const std::string& dist_param_name,
      Epetra_MultiVector* dg_dp)
{
  // Set data in Workset struct
  PHAL::Workset workset;
  Teuchos::RCP<const Tpetra_Vector> xdotT;
   if (xdot != NULL) {
      xdotT = Petra::EpetraVector_To_TpetraVectorConst(*xdot, application->getComm());
   }

   Teuchos::RCP<const Tpetra_Vector> xdotdotT;
   if (xdotdot != NULL) {
      xdotdotT = Petra::EpetraVector_To_TpetraVectorConst(*xdotdot, application->getComm());
   }

  Teuchos::RCP<const Tpetra_Vector> xT = Petra::EpetraVector_To_TpetraVectorConst(x,application->getComm());

  application->setupBasicWorksetInfoT(workset, current_time, xdotT, xdotdotT, xT, param_array);

  // Perform fill via field manager (dg/dx)
  if(dg_dp != NULL) {
    workset.dist_param_deriv_name = dist_param_name;
    workset.dgdp = Teuchos::rcp(dg_dp, false);
    { // workset.overlapped_dgdp = Teuchos::rcp(new Epetra_MultiVector(*workset.distParamLib->get(dist_param_name)->overlap_map(), dg_dp->NumVectors()));
      Teuchos::RCP<Tpetra_MultiVector> overlapped_dgdpT = Teuchos::rcp(
        new Tpetra_MultiVector(
          workset.distParamLib->get(dist_param_name)->overlap_map(),
          dg_dp->NumVectors()));
      const Teuchos::RCP<const Epetra_Comm>
        comm = createEpetraCommFromTeuchosComm(application->getComm());   

      Teuchos::RCP<Epetra_Map> emap = Petra::TpetraMap_To_EpetraMap(overlapped_dgdpT->getMap(), comm);
      workset.overlapped_dgdp = Teuchos::rcp(new Epetra_MultiVector(*emap, overlapped_dgdpT->getNumVectors()));
      Petra::TpetraMultiVector_To_EpetraMultiVector(
        overlapped_dgdpT, *workset.overlapped_dgdp, comm);
    }
    evaluate<PHAL::AlbanyTraits::DistParamDeriv>(workset);
  }
}
#endif

#ifdef ALBANY_SG_MP
void
Albany::FieldManagerScalarResponseFunction::
evaluateSGResponse(
  const double current_time,
  const Stokhos::EpetraVectorOrthogPoly* sg_xdot,
  const Stokhos::EpetraVectorOrthogPoly* sg_xdotdot,
  const Stokhos::EpetraVectorOrthogPoly& sg_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& sg_p_index,
  const Teuchos::Array< Teuchos::Array<SGType> >& sg_p_vals,
  Stokhos::EpetraVectorOrthogPoly& sg_g)
{
  visResponseGraph<PHAL::AlbanyTraits::SGResidual>("_sg");

  // Set data in Workset struct
  PHAL::Workset workset;
  application->setupBasicWorksetInfo(workset, current_time, sg_xdot, sg_xdotdot, &sg_x, p,
				     sg_p_index, sg_p_vals);
  workset.sg_g = Teuchos::rcp(&sg_g,false);

  // Perform fill via field manager
  evaluate<PHAL::AlbanyTraits::SGResidual>(workset);
}

void
Albany::FieldManagerScalarResponseFunction::
evaluateSGTangent(
  const double alpha, 
  const double beta, 
  const double omega, 
  const double current_time,
  bool sum_derivs,
  const Stokhos::EpetraVectorOrthogPoly* sg_xdot,
  const Stokhos::EpetraVectorOrthogPoly* sg_xdotdot,
  const Stokhos::EpetraVectorOrthogPoly& sg_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& sg_p_index,
  const Teuchos::Array< Teuchos::Array<SGType> >& sg_p_vals,
  ParamVec* deriv_p,
  const Epetra_MultiVector* Vx,
  const Epetra_MultiVector* Vxdot,
  const Epetra_MultiVector* Vxdotdot,
  const Epetra_MultiVector* Vp,
  Stokhos::EpetraVectorOrthogPoly* sg_g,
  Stokhos::EpetraMultiVectorOrthogPoly* sg_JV,
  Stokhos::EpetraMultiVectorOrthogPoly* sg_gp)
{
  visResponseGraph<PHAL::AlbanyTraits::SGTangent>("_sg_tangent");

  // Set data in Workset struct
  PHAL::Workset workset;
  application->setupTangentWorksetInfo(workset, current_time, sum_derivs, 
				       sg_xdot, sg_xdotdot, &sg_x, p, deriv_p, 
				       sg_p_index, sg_p_vals,
				       Vxdot, Vxdotdot, Vx, Vp);
  workset.sg_g = Teuchos::rcp(sg_g, false);
  workset.sg_dgdx = Teuchos::rcp(sg_JV, false);
  workset.sg_dgdp = Teuchos::rcp(sg_gp, false);
  
  // Perform fill via field manager
  evaluate<PHAL::AlbanyTraits::SGTangent>(workset);
}

void
Albany::FieldManagerScalarResponseFunction::
evaluateSGGradient(
  const double current_time,
  const Stokhos::EpetraVectorOrthogPoly* sg_xdot,
  const Stokhos::EpetraVectorOrthogPoly* sg_xdotdot,
  const Stokhos::EpetraVectorOrthogPoly& sg_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& sg_p_index,
  const Teuchos::Array< Teuchos::Array<SGType> >& sg_p_vals,
  ParamVec* deriv_p,
  Stokhos::EpetraVectorOrthogPoly* sg_g,
  Stokhos::EpetraMultiVectorOrthogPoly* sg_dg_dx,
  Stokhos::EpetraMultiVectorOrthogPoly* sg_dg_dxdot,
  Stokhos::EpetraMultiVectorOrthogPoly* sg_dg_dxdotdot,
  Stokhos::EpetraMultiVectorOrthogPoly* sg_dg_dp)
{
  visResponseGraph<PHAL::AlbanyTraits::SGJacobian>("_sg_gradient");

  // Set data in Workset struct
  PHAL::Workset workset;
  application->setupBasicWorksetInfo(workset, current_time, sg_xdot, sg_xdotdot, &sg_x, p,
				     sg_p_index, sg_p_vals);
  workset.sg_g = Teuchos::rcp(sg_g, false);
  
  // Perform fill via field manager (dg/dx)
  if (sg_dg_dx != NULL) {
    workset.m_coeff = 0.0;
    workset.j_coeff = 1.0;
    workset.n_coeff = 0.0;
    workset.sg_dgdx = Teuchos::rcp(sg_dg_dx, false);
    workset.overlapped_sg_dgdx = 
      Teuchos::rcp(new Stokhos::EpetraMultiVectorOrthogPoly(
		     sg_dg_dx->basis(),
		     sg_dg_dx->map(),
		     Teuchos::rcp(&(workset.x_importer->TargetMap()),false),
		     sg_dg_dx->productComm(),
		     sg_dg_dx->numVectors()));
    evaluate<PHAL::AlbanyTraits::SGJacobian>(workset);
  }

  // Perform fill via field manager (dg/dxdot)
  if (sg_dg_dxdot != NULL) {
    workset.m_coeff = 1.0;
    workset.j_coeff = 0.0;
    workset.n_coeff = 0.0;
    workset.sg_dgdx = Teuchos::null;
    workset.sg_dgdxdot = Teuchos::rcp(sg_dg_dxdot, false);
    workset.overlapped_sg_dgdx = Teuchos::null;
    workset.overlapped_sg_dgdxdot = 
      Teuchos::rcp(new Stokhos::EpetraMultiVectorOrthogPoly(
		     sg_dg_dxdot->basis(),
		     sg_dg_dxdot->map(),
		     Teuchos::rcp(&(workset.x_importer->TargetMap()),false),
		     sg_dg_dxdot->productComm(),
		     sg_dg_dxdot->numVectors()));
    evaluate<PHAL::AlbanyTraits::SGJacobian>(workset);
  }  

  // Perform fill via field manager (dg/dxdotdot)
  if (sg_dg_dxdotdot != NULL) {
    workset.m_coeff = 0.0;
    workset.j_coeff = 0.0;
    workset.n_coeff = 1.0;
    workset.sg_dgdx = Teuchos::null;
    workset.sg_dgdxdotdot = Teuchos::rcp(sg_dg_dxdotdot, false);
    workset.overlapped_sg_dgdx = Teuchos::null;
    workset.overlapped_sg_dgdxdotdot = 
      Teuchos::rcp(new Stokhos::EpetraMultiVectorOrthogPoly(
		     sg_dg_dxdotdot->basis(),
		     sg_dg_dxdotdot->map(),
		     Teuchos::rcp(&(workset.x_importer->TargetMap()),false),
		     sg_dg_dxdotdot->productComm(),
		     sg_dg_dxdotdot->numVectors()));
    evaluate<PHAL::AlbanyTraits::SGJacobian>(workset);
  }  
}

void
Albany::FieldManagerScalarResponseFunction::
evaluateMPResponse(
  const double current_time,
  const Stokhos::ProductEpetraVector* mp_xdot,
  const Stokhos::ProductEpetraVector* mp_xdotdot,
  const Stokhos::ProductEpetraVector& mp_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& mp_p_index,
  const Teuchos::Array< Teuchos::Array<MPType> >& mp_p_vals,
  Stokhos::ProductEpetraVector& mp_g)
{
  visResponseGraph<PHAL::AlbanyTraits::MPResidual>("_mp");

  // Set data in Workset struct
  PHAL::Workset workset;
  application->setupBasicWorksetInfo(workset, current_time, mp_xdot, mp_xdotdot, &mp_x, p,
				     mp_p_index, mp_p_vals);
  workset.mp_g = Teuchos::rcp(&mp_g,false);

  // Perform fill via field manager
  evaluate<PHAL::AlbanyTraits::MPResidual>(workset);
}

void
Albany::FieldManagerScalarResponseFunction::
evaluateMPTangent(
  const double alpha, 
  const double beta, 
  const double omega, 
  const double current_time,
  bool sum_derivs,
  const Stokhos::ProductEpetraVector* mp_xdot,
  const Stokhos::ProductEpetraVector* mp_xdotdot,
  const Stokhos::ProductEpetraVector& mp_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& mp_p_index,
  const Teuchos::Array< Teuchos::Array<MPType> >& mp_p_vals,
  ParamVec* deriv_p,
  const Epetra_MultiVector* Vx,
  const Epetra_MultiVector* Vxdot,
  const Epetra_MultiVector* Vxdotdot,
  const Epetra_MultiVector* Vp,
  Stokhos::ProductEpetraVector* mp_g,
  Stokhos::ProductEpetraMultiVector* mp_JV,
  Stokhos::ProductEpetraMultiVector* mp_gp)
{
  visResponseGraph<PHAL::AlbanyTraits::MPTangent>("_mp_tangent");

  // Set data in Workset struct
  PHAL::Workset workset;
  application->setupTangentWorksetInfo(workset, current_time, sum_derivs, 
				       mp_xdot, mp_xdotdot, &mp_x, p, deriv_p, 
				       mp_p_index, mp_p_vals,
				       Vxdot, Vxdotdot, Vx, Vp);
  workset.mp_g = Teuchos::rcp(mp_g, false);
  workset.mp_dgdx = Teuchos::rcp(mp_JV, false);
  workset.mp_dgdp = Teuchos::rcp(mp_gp, false);
  
  // Perform fill via field manager
  evaluate<PHAL::AlbanyTraits::MPTangent>(workset);
}

void
Albany::FieldManagerScalarResponseFunction::
evaluateMPGradient(
  const double current_time,
  const Stokhos::ProductEpetraVector* mp_xdot,
  const Stokhos::ProductEpetraVector* mp_xdotdot,
  const Stokhos::ProductEpetraVector& mp_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& mp_p_index,
  const Teuchos::Array< Teuchos::Array<MPType> >& mp_p_vals,
  ParamVec* deriv_p,
  Stokhos::ProductEpetraVector* mp_g,
  Stokhos::ProductEpetraMultiVector* mp_dg_dx,
  Stokhos::ProductEpetraMultiVector* mp_dg_dxdot,
  Stokhos::ProductEpetraMultiVector* mp_dg_dxdotdot,
  Stokhos::ProductEpetraMultiVector* mp_dg_dp)
{
  visResponseGraph<PHAL::AlbanyTraits::MPJacobian>("_mp_gradient");

  // Set data in Workset struct
  PHAL::Workset workset;
  application->setupBasicWorksetInfo(workset, current_time, mp_xdot, mp_xdotdot, &mp_x, p,
				     mp_p_index, mp_p_vals);
  workset.mp_g = Teuchos::rcp(mp_g, false);
  
  // Perform fill via field manager (dg/dx)
  if (mp_dg_dx != NULL) {
    workset.m_coeff = 0.0;
    workset.j_coeff = 1.0;
    workset.n_coeff = 0.0;
    workset.mp_dgdx = Teuchos::rcp(mp_dg_dx, false);
    workset.overlapped_mp_dgdx = 
      Teuchos::rcp(new Stokhos::ProductEpetraMultiVector(
		     mp_dg_dx->map(),
		     Teuchos::rcp(&(workset.x_importer->TargetMap()),false),
		     mp_dg_dx->productComm(),
		     mp_dg_dx->numVectors()));
    evaluate<PHAL::AlbanyTraits::MPJacobian>(workset);
  }

  // Perform fill via field manager (dg/dxdot)
  if (mp_dg_dxdot != NULL) {
    workset.m_coeff = 1.0;
    workset.j_coeff = 0.0;
    workset.n_coeff = 0.0;
    workset.mp_dgdx = Teuchos::null;
    workset.mp_dgdxdot = Teuchos::rcp(mp_dg_dxdot, false);
    workset.overlapped_mp_dgdx = Teuchos::null;
    workset.overlapped_mp_dgdxdot = 
      Teuchos::rcp(new Stokhos::ProductEpetraMultiVector(
		     mp_dg_dxdot->map(),
		     Teuchos::rcp(&(workset.x_importer->TargetMap()),false),
		     mp_dg_dxdot->productComm(),
		     mp_dg_dxdot->numVectors()));
    evaluate<PHAL::AlbanyTraits::MPJacobian>(workset);
  }  

  // Perform fill via field manager (dg/dxdotdot)
  if (mp_dg_dxdotdot != NULL) {
    workset.m_coeff = 0.0;
    workset.j_coeff = 0.0;
    workset.n_coeff = 1.0;
    workset.mp_dgdx = Teuchos::null;
    workset.mp_dgdxdotdot = Teuchos::rcp(mp_dg_dxdotdot, false);
    workset.overlapped_mp_dgdx = Teuchos::null;
    workset.overlapped_mp_dgdxdotdot = 
      Teuchos::rcp(new Stokhos::ProductEpetraMultiVector(
		     mp_dg_dxdotdot->map(),
		     Teuchos::rcp(&(workset.x_importer->TargetMap()),false),
		     mp_dg_dxdotdot->productComm(),
		     mp_dg_dxdotdot->numVectors()));
    evaluate<PHAL::AlbanyTraits::MPJacobian>(workset);
  }  
}
#endif //ALBANY_SG_MP
