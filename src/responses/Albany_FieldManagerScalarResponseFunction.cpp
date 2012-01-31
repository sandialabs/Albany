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


#include "Albany_FieldManagerScalarResponseFunction.hpp"
#include <algorithm>

Albany::FieldManagerScalarResponseFunction::
FieldManagerScalarResponseFunction(
  const Teuchos::RCP<Albany::Application>& application_,
  const Teuchos::RCP<Albany::AbstractProblem>& problem_,
  const Teuchos::RCP<Albany::MeshSpecsStruct>&  meshSpecs_,
  const Teuchos::RCP<Albany::StateManager>& stateMgr_,
  Teuchos::ParameterList& responseParams) :
  SamplingBasedScalarResponseFunction(application_->getComm()),
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
  SamplingBasedScalarResponseFunction(application_->getComm()),
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
		 const Epetra_Vector& x,
		 const Teuchos::Array<ParamVec>& p,
		 Epetra_Vector& g)
{
  visResponseGraph<PHAL::AlbanyTraits::Residual>("");

  // Set data in Workset struct
  PHAL::Workset workset;
  application->setupBasicWorksetInfo(workset, current_time, xdot, &x, p);
  workset.g = Teuchos::rcp(&g,false);

  // Perform fill via field manager
  int numWorksets = application->getNumWorksets();
  rfm->preEvaluate<PHAL::AlbanyTraits::Residual>(workset);
  for (int ws=0; ws < numWorksets; ws++) {
    application->loadWorksetBucketInfo(workset, ws);
    rfm->evaluateFields<PHAL::AlbanyTraits::Residual>(workset);
  }
  rfm->postEvaluate<PHAL::AlbanyTraits::Residual>(workset);
}

void
Albany::FieldManagerScalarResponseFunction::
evaluateTangent(const double alpha, 
		const double beta,
		const double current_time,
		bool sum_derivs,
		const Epetra_Vector* xdot,
		const Epetra_Vector& x,
		const Teuchos::Array<ParamVec>& p,
		ParamVec* deriv_p,
		const Epetra_MultiVector* Vxdot,
		const Epetra_MultiVector* Vx,
		const Epetra_MultiVector* Vp,
		Epetra_Vector* g,
		Epetra_MultiVector* gx,
		Epetra_MultiVector* gp)
{
  visResponseGraph<PHAL::AlbanyTraits::Tangent>("_tangent");

  // Set data in Workset struct
  PHAL::Workset workset;
  application->setupTangentWorksetInfo(workset, sum_derivs, 
				       current_time, xdot, &x, p, 
				       deriv_p, Vxdot, Vx, Vp);
  workset.g = Teuchos::rcp(g, false);
  workset.dgdx = Teuchos::rcp(gx, false);
  workset.dgdp = Teuchos::rcp(gp, false);
  
  // Perform fill via field manager
  int numWorksets = application->getNumWorksets();
  rfm->preEvaluate<PHAL::AlbanyTraits::Tangent>(workset);
  for (int ws=0; ws < numWorksets; ws++) {
    application->loadWorksetBucketInfo(workset, ws);
    rfm->evaluateFields<PHAL::AlbanyTraits::Tangent>(workset);
  }
  rfm->postEvaluate<PHAL::AlbanyTraits::Tangent>(workset);
}

void
Albany::FieldManagerScalarResponseFunction::
evaluateGradient(const double current_time,
		 const Epetra_Vector* xdot,
		 const Epetra_Vector& x,
		 const Teuchos::Array<ParamVec>& p,
		 ParamVec* deriv_p,
		 Epetra_Vector* g,
		 Epetra_MultiVector* dg_dx,
		 Epetra_MultiVector* dg_dxdot,
		 Epetra_MultiVector* dg_dp)
{
  visResponseGraph<PHAL::AlbanyTraits::Jacobian>("_gradient");

  // Set data in Workset struct
  PHAL::Workset workset;
  application->setupBasicWorksetInfo(workset, current_time, xdot, &x, p);
  workset.g = Teuchos::rcp(g, false);
  
  // Perform fill via field manager (dg/dx)
  int numWorksets = application->getNumWorksets();
  if (dg_dx != NULL) {
    workset.m_coeff = 0.0;
    workset.j_coeff = 1.0;
    workset.dgdx = Teuchos::rcp(dg_dx, false);
    workset.overlapped_dgdx = 
      Teuchos::rcp(new Epetra_MultiVector(workset.x_importer->TargetMap(),
					  dg_dx->NumVectors()));
    rfm->preEvaluate<PHAL::AlbanyTraits::Jacobian>(workset);
    for (int ws=0; ws < numWorksets; ws++) {
      application->loadWorksetBucketInfo(workset, ws);
      rfm->evaluateFields<PHAL::AlbanyTraits::Jacobian>(workset);
    }
    rfm->postEvaluate<PHAL::AlbanyTraits::Jacobian>(workset);
  }

  // Perform fill via field manager (dg/dxdot)
  if (dg_dxdot != NULL) {
    workset.m_coeff = 1.0;
    workset.j_coeff = 0.0;
    workset.dgdx = Teuchos::null;
    workset.dgdxdot = Teuchos::rcp(dg_dxdot, false);
    workset.overlapped_dgdxdot = 
      Teuchos::rcp(new Epetra_MultiVector(workset.x_importer->TargetMap(),
					  dg_dxdot->NumVectors()));
    rfm->preEvaluate<PHAL::AlbanyTraits::Jacobian>(workset);
    for (int ws=0; ws < numWorksets; ws++) {
      application->loadWorksetBucketInfo(workset, ws);
      rfm->evaluateFields<PHAL::AlbanyTraits::Jacobian>(workset);
    }
    rfm->postEvaluate<PHAL::AlbanyTraits::Jacobian>(workset);
  }  
}
