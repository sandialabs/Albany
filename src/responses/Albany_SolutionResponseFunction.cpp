//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_SolutionResponseFunction.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_Application.hpp"
#include "Albany_GlobalLocalIndexer.hpp"

namespace Albany {

SolutionResponseFunction::
SolutionResponseFunction(const Teuchos::RCP<Albany::Application>& application,
                         const Teuchos::ParameterList& responseParams)
 : solution_vs(getSpmdVectorSpace(application->getVectorSpace()))
{
  // Build list of DOFs we want to keep
  // This should be replaced by DOF names eventually
  int numDOF = application->getProblem()->numEquations();
  if (responseParams.isType< Teuchos::Array<int> >("Keep DOF Indices")) {
    Teuchos::Array<int> dofs = responseParams.get< Teuchos::Array<int> >("Keep DOF Indices");
    keepDOF.resize(numDOF,false);
    numKeepDOF = 0;
    for (int i=0; i<dofs.size(); i++) {
      keepDOF[dofs[i]] = true;
      ++numKeepDOF;
    }
  } else {
    keepDOF.resize(numDOF,true);
    numKeepDOF = numDOF;
  }
}

void Albany::SolutionResponseFunction::setup()
{
  // Build culled vs
  int Neqns = keepDOF.size();
  int N = solution_vs->localSubDim();

  TEUCHOS_ASSERT( !(N % Neqns) ); // Assume that all the equations for
                                  // a given node are on the assigned
                                  // processor. I.e. need to ensure
                                  // that N is exactly Neqns-divisible

  int nnodes = N / Neqns;          // number of fem nodes
  int N_new = nnodes * numKeepDOF; // length of local x_new

  Teuchos::Array<LO> subspace_components(N_new);
  for (int ieqn=0, idx=0; ieqn<Neqns; ++ieqn) {
    if (keepDOF[ieqn]) {
      for (int inode=0; inode<nnodes; ++inode, ++idx) {
        subspace_components[idx] = inode*Neqns+ieqn;
      }
    }
  }
  culled_vs = getSpmdVectorSpace(createSubspace(solution_vs,subspace_components));


  // Create graph for gradient operator -- diagonal matrix
  cull_op_factory = Teuchos::rcp(new ThyraCrsMatrixFactory(solution_vs,culled_vs));
  auto culled_vs_indexer = createGlobalLocalIndexer(culled_vs);
  for (int i=0; i<culled_vs->localSubDim(); i++) {
    const GO row = culled_vs_indexer->getGlobalElement(i);
    cull_op_factory->insertGlobalIndices(row,Teuchos::arrayView(&row,1));
  }
  cull_op_factory->fillComplete();

  // Create the culling operator
  cull_op = cull_op_factory->createOp();
  assign(cull_op,1.0);
  fillComplete(cull_op);
}

Teuchos::RCP<Thyra_LinearOp>
SolutionResponseFunction::createGradientOp() const
{
  auto gradOp = cull_op_factory->createOp();
  fillComplete(gradOp);
  return gradOp;
}

void SolutionResponseFunction::
evaluateResponse(
    const double /*current_time*/,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
		const Teuchos::Array<ParamVec>& /*p*/,
    const Teuchos::RCP<Thyra_Vector>& g)
{
  cullSolution(x, g);
}

void SolutionResponseFunction::
evaluateTangent(const double /*alpha*/,
		const double beta,
		const double /*omega*/,
		const double /*current_time*/,
		bool /*sum_derivs*/,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
		const Teuchos::Array<ParamVec>& /*p*/,
		ParamVec* /*deriv_p*/,
    const Teuchos::RCP<const Thyra_MultiVector>& Vx,
    const Teuchos::RCP<const Thyra_MultiVector>& /*Vxdot*/,
    const Teuchos::RCP<const Thyra_MultiVector>& /*Vxdotdot*/,
    const Teuchos::RCP<const Thyra_MultiVector>& /*Vp*/,
    const Teuchos::RCP<Thyra_Vector>& g,
    const Teuchos::RCP<Thyra_MultiVector>& gx,
    const Teuchos::RCP<Thyra_MultiVector>& gp)
{
  if (!g.is_null()) {
    cullSolution(x, g);
  }

  if (!gx.is_null()) {
    gx->assign(0.0);
    if (!Vx.is_null()) {
      cullSolution(Vx, gx);
      gx->scale(beta);
    }
  }

  if (!gp.is_null()) {
    gp->assign(0.0);
  }
}

void SolutionResponseFunction::
evaluateGradient(const double /*current_time*/,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
		const Teuchos::Array<ParamVec>& /*p*/,
		ParamVec* /*deriv_p*/,
    const Teuchos::RCP<Thyra_Vector>& g,
    const Teuchos::RCP<Thyra_LinearOp>& dg_dx,
    const Teuchos::RCP<Thyra_LinearOp>& dg_dxdot,
    const Teuchos::RCP<Thyra_LinearOp>& dg_dxdotdot,
    const Teuchos::RCP<Thyra_MultiVector>& dg_dp)
{
  if (!g.is_null()) {
    cullSolution(x, g);
  }

  if (!dg_dx.is_null()) {
    assign(dg_dx, 1.0); // matrix only stores the diagonal
  }

  if (!dg_dxdot.is_null()) {
    assign(dg_dxdot,0.0); // matrix only stores the diagonal
  }

  if (!dg_dxdotdot.is_null()) {
    assign(dg_dxdotdot, 0.0); // matrix only stores the diagonal
  }

  if (!dg_dp.is_null()) {
    dg_dp->assign(0.0);
  }
}

void SolutionResponseFunction::
evaluateDistParamDeriv(
    const double /*current_time*/,
    const Teuchos::RCP<const Thyra_Vector>& /*x*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
    const Teuchos::Array<ParamVec>& /*param_array*/,
    const std::string& /*dist_param_name*/,
    const Teuchos::RCP<Thyra_MultiVector>& dg_dp)
{
  if (!dg_dp.is_null()) {
    dg_dp->assign(0.0);
  }
}

void SolutionResponseFunction::
cullSolution(const Teuchos::RCP<const Thyra_MultiVector>& x,
             const Teuchos::RCP<      Thyra_MultiVector>& x_culled) const
{
  cull_op->apply(Thyra::EOpTransp::NOTRANS,*x,x_culled.ptr(),1.0,0.0);
}

} // namespace Albany
