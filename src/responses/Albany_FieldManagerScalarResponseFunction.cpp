//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_FieldManagerScalarResponseFunction.hpp"
#include <algorithm>

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
  Teuchos::RCP<const Epetra_Comm> comm = application->getComm();

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
  rfm->postRegistrationSetup("");

  // Visualize rfm graph -- get file name from name of response function
  // (with spaces replaced by _ and lower case)
  vis_response_graph = 
    responseParams.get("Phalanx Graph Visualization Detail", 0);
  vis_response_name = responseParams.get<std::string>("Name");
  std::replace(vis_response_name.begin(), vis_response_name.end(), ' ', '_');
  std::transform(vis_response_name.begin(), vis_response_name.end(), 
		 vis_response_name.begin(), ::tolower);
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

void
Albany::FieldManagerScalarResponseFunction::
evaluateResponse(const double current_time,
		 const Epetra_Vector* xdot,
		 const Epetra_Vector* xdotdot,
		 const Epetra_Vector& x,
		 const Teuchos::Array<ParamVec>& p,
		 Epetra_Vector& g)
{
  visResponseGraph<PHAL::AlbanyTraits::Residual>("");

  // Set data in Workset struct
  PHAL::Workset workset;
  application->setupBasicWorksetInfo(workset, current_time, xdot, xdotdot, &x, p);
  workset.g = Teuchos::rcp(&g,false);

  // Perform fill via field manager
  int numWorksets = application->getNumWorksets();
  rfm->preEvaluate<PHAL::AlbanyTraits::Residual>(workset);
  for (int ws=0; ws < numWorksets; ws++) {
    application->loadWorksetBucketInfo<PHAL::AlbanyTraits::Residual>(
      workset, ws);
    rfm->evaluateFields<PHAL::AlbanyTraits::Residual>(workset);
  }
  rfm->postEvaluate<PHAL::AlbanyTraits::Residual>(workset);
}

void
Albany::FieldManagerScalarResponseFunction::
evaluateTangent(const double alpha, 
		const double beta,
		const double omega,
		const double current_time,
		bool sum_derivs,
		const Epetra_Vector* xdot,
		const Epetra_Vector* xdotdot,
		const Epetra_Vector& x,
		const Teuchos::Array<ParamVec>& p,
		ParamVec* deriv_p,
		const Epetra_MultiVector* Vxdot,
		const Epetra_MultiVector* Vxdotdot,
		const Epetra_MultiVector* Vx,
		const Epetra_MultiVector* Vp,
		Epetra_Vector* g,
		Epetra_MultiVector* gx,
		Epetra_MultiVector* gp)
{
  visResponseGraph<PHAL::AlbanyTraits::Tangent>("_tangent");

  // Set data in Workset struct
  PHAL::Workset workset;
  application->setupTangentWorksetInfo(workset, current_time, 
                       sum_derivs, xdot, xdotdot, &x, p, 
				       deriv_p, Vxdot, Vxdotdot, Vx, Vp);
  workset.g = Teuchos::rcp(g, false);
  workset.dgdx = Teuchos::rcp(gx, false);
  workset.dgdp = Teuchos::rcp(gp, false);
  
  // Perform fill via field manager
  int numWorksets = application->getNumWorksets();
  rfm->preEvaluate<PHAL::AlbanyTraits::Tangent>(workset);
  for (int ws=0; ws < numWorksets; ws++) {
    application->loadWorksetBucketInfo<PHAL::AlbanyTraits::Tangent>(
      workset, ws);
    rfm->evaluateFields<PHAL::AlbanyTraits::Tangent>(workset);
  }
  rfm->postEvaluate<PHAL::AlbanyTraits::Tangent>(workset);
}

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

  // Set data in Workset struct
  PHAL::Workset workset;
  application->setupBasicWorksetInfo(workset, current_time, xdot, xdotdot, &x, p);
  workset.g = Teuchos::rcp(g, false);
  
  // Perform fill via field manager (dg/dx)
  int numWorksets = application->getNumWorksets();
  if (dg_dx != NULL) {
    workset.m_coeff = 0.0;
    workset.j_coeff = 1.0;
    workset.n_coeff = 0.0;
    workset.dgdx = Teuchos::rcp(dg_dx, false);
    workset.overlapped_dgdx = 
      Teuchos::rcp(new Epetra_MultiVector(workset.x_importer->TargetMap(),
					  dg_dx->NumVectors()));
    rfm->preEvaluate<PHAL::AlbanyTraits::Jacobian>(workset);
    for (int ws=0; ws < numWorksets; ws++) {
      application->loadWorksetBucketInfo<PHAL::AlbanyTraits::Jacobian>(
	workset, ws);
      rfm->evaluateFields<PHAL::AlbanyTraits::Jacobian>(workset);
    }
    rfm->postEvaluate<PHAL::AlbanyTraits::Jacobian>(workset);
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
    rfm->preEvaluate<PHAL::AlbanyTraits::Jacobian>(workset);
    for (int ws=0; ws < numWorksets; ws++) {
      application->loadWorksetBucketInfo<PHAL::AlbanyTraits::Jacobian>(
	workset, ws);
      rfm->evaluateFields<PHAL::AlbanyTraits::Jacobian>(workset);
    }
    rfm->postEvaluate<PHAL::AlbanyTraits::Jacobian>(workset);
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
    rfm->preEvaluate<PHAL::AlbanyTraits::Jacobian>(workset);
    for (int ws=0; ws < numWorksets; ws++) {
      application->loadWorksetBucketInfo<PHAL::AlbanyTraits::Jacobian>(
	workset, ws);
      rfm->evaluateFields<PHAL::AlbanyTraits::Jacobian>(workset);
    }
    rfm->postEvaluate<PHAL::AlbanyTraits::Jacobian>(workset);
  }  
}
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
  int numWorksets = application->getNumWorksets();
  rfm->preEvaluate<PHAL::AlbanyTraits::SGResidual>(workset);
  for (int ws=0; ws < numWorksets; ws++) {
    application->loadWorksetBucketInfo<PHAL::AlbanyTraits::SGResidual>(
      workset, ws);
    rfm->evaluateFields<PHAL::AlbanyTraits::SGResidual>(workset);
  }
  rfm->postEvaluate<PHAL::AlbanyTraits::SGResidual>(workset);
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
  int numWorksets = application->getNumWorksets();
  rfm->preEvaluate<PHAL::AlbanyTraits::SGTangent>(workset);
  for (int ws=0; ws < numWorksets; ws++) {
    application->loadWorksetBucketInfo<PHAL::AlbanyTraits::SGTangent>(
      workset, ws);
    rfm->evaluateFields<PHAL::AlbanyTraits::SGTangent>(workset);
  }
  rfm->postEvaluate<PHAL::AlbanyTraits::SGTangent>(workset);
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
  int numWorksets = application->getNumWorksets();
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
    rfm->preEvaluate<PHAL::AlbanyTraits::SGJacobian>(workset);
    for (int ws=0; ws < numWorksets; ws++) {
      application->loadWorksetBucketInfo<PHAL::AlbanyTraits::SGJacobian>(
	workset, ws);
      rfm->evaluateFields<PHAL::AlbanyTraits::SGJacobian>(workset);
    }
    rfm->postEvaluate<PHAL::AlbanyTraits::SGJacobian>(workset);
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
    rfm->preEvaluate<PHAL::AlbanyTraits::SGJacobian>(workset);
    for (int ws=0; ws < numWorksets; ws++) {
      application->loadWorksetBucketInfo<PHAL::AlbanyTraits::SGJacobian>(
	workset, ws);
      rfm->evaluateFields<PHAL::AlbanyTraits::SGJacobian>(workset);
    }
    rfm->postEvaluate<PHAL::AlbanyTraits::SGJacobian>(workset);
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
    rfm->preEvaluate<PHAL::AlbanyTraits::SGJacobian>(workset);
    for (int ws=0; ws < numWorksets; ws++) {
      application->loadWorksetBucketInfo<PHAL::AlbanyTraits::SGJacobian>(
	workset, ws);
      rfm->evaluateFields<PHAL::AlbanyTraits::SGJacobian>(workset);
    }
    rfm->postEvaluate<PHAL::AlbanyTraits::SGJacobian>(workset);
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
  int numWorksets = application->getNumWorksets();
  rfm->preEvaluate<PHAL::AlbanyTraits::MPResidual>(workset);
  for (int ws=0; ws < numWorksets; ws++) {
    application->loadWorksetBucketInfo<PHAL::AlbanyTraits::MPResidual>(
      workset, ws);
    rfm->evaluateFields<PHAL::AlbanyTraits::MPResidual>(workset);
  }
  rfm->postEvaluate<PHAL::AlbanyTraits::MPResidual>(workset);
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
  int numWorksets = application->getNumWorksets();
  rfm->preEvaluate<PHAL::AlbanyTraits::MPTangent>(workset);
  for (int ws=0; ws < numWorksets; ws++) {
    application->loadWorksetBucketInfo<PHAL::AlbanyTraits::MPTangent>(
      workset, ws);
    rfm->evaluateFields<PHAL::AlbanyTraits::MPTangent>(workset);
  }
  rfm->postEvaluate<PHAL::AlbanyTraits::MPTangent>(workset);
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
  int numWorksets = application->getNumWorksets();
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
    rfm->preEvaluate<PHAL::AlbanyTraits::MPJacobian>(workset);
    for (int ws=0; ws < numWorksets; ws++) {
      application->loadWorksetBucketInfo<PHAL::AlbanyTraits::MPJacobian>(
	workset, ws);
      rfm->evaluateFields<PHAL::AlbanyTraits::MPJacobian>(workset);
    }
    rfm->postEvaluate<PHAL::AlbanyTraits::MPJacobian>(workset);
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
    rfm->preEvaluate<PHAL::AlbanyTraits::MPJacobian>(workset);
    for (int ws=0; ws < numWorksets; ws++) {
      application->loadWorksetBucketInfo<PHAL::AlbanyTraits::MPJacobian>(
	workset, ws);
      rfm->evaluateFields<PHAL::AlbanyTraits::MPJacobian>(workset);
    }
    rfm->postEvaluate<PHAL::AlbanyTraits::MPJacobian>(workset);
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
    rfm->preEvaluate<PHAL::AlbanyTraits::MPJacobian>(workset);
    for (int ws=0; ws < numWorksets; ws++) {
      application->loadWorksetBucketInfo<PHAL::AlbanyTraits::MPJacobian>(
	workset, ws);
      rfm->evaluateFields<PHAL::AlbanyTraits::MPJacobian>(workset);
    }
    rfm->postEvaluate<PHAL::AlbanyTraits::MPJacobian>(workset);
  }  
}
#endif //ALBANY_SG_MP
