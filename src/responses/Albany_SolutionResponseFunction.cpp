//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_SolutionResponseFunction.hpp"
#include "Albany_ThyraUtils.hpp"
#include <algorithm>

// TODO: remove this include once you figured out how to abstract away from Tpetra
#include "Albany_TpetraThyraUtils.hpp"

Albany::SolutionResponseFunction::
SolutionResponseFunction(
  const Teuchos::RCP<Albany::Application>& application_,
  Teuchos::ParameterList& responseParams) :
  application(application_)
{
  // Build list of DOFs we want to keep
  // This should be replaced by DOF names eventually
  int numDOF = application->getProblem()->numEquations();
  if (responseParams.isType< Teuchos::Array<int> >("Keep DOF Indices")) {
    Teuchos::Array<int> dofs =
      responseParams.get< Teuchos::Array<int> >("Keep DOF Indices");
    keepDOF = Teuchos::Array<int>(numDOF, 0);
    for (int i=0; i<dofs.size(); i++)
      keepDOF[dofs[i]] = 1;
  }
  else {
    keepDOF = Teuchos::Array<int>(numDOF, 1);
  }
}


void
Albany::SolutionResponseFunction::
setup()
{
  // Build culled map and importer - Tpetra
  Teuchos::RCP<const Tpetra_Map> x_mapT = application->getMapT();
  Teuchos::RCP<const Teuchos::Comm<int> > commT = application->getComm(); 
  //Tpetra version of culled_map
  culled_mapT = buildCulledMapT(*x_mapT, keepDOF);

  importerT = Teuchos::rcp(new Tpetra_Import(x_mapT, culled_mapT));

  // Create graph for gradient operator -- diagonal matrix: Tpetra version
  Teuchos::ArrayView<Tpetra_GO> rowAV;
  gradient_graphT =
    Teuchos::rcp(new Tpetra_CrsGraph(culled_mapT, 1));
  for (int i=0; i<culled_mapT->getNodeNumElements(); i++) {
    Tpetra_GO row = culled_mapT->getGlobalElement(i);
    rowAV = Teuchos::arrayView(&row, 1);
    gradient_graphT->insertGlobalIndices(row, rowAV);
  }
  gradient_graphT->fillComplete();
  //IK, 8/22/13: Tpetra_CrsGraph does not appear to have optimizeStorage() function...
  //gradient_graphT->optimizeStorage();
}

Albany::SolutionResponseFunction::
~SolutionResponseFunction()
{
}

Teuchos::RCP<const Tpetra_Map>
Albany::SolutionResponseFunction::
responseMapT() const
{
  return culled_mapT;
}

Teuchos::RCP<Tpetra_Operator>
Albany::SolutionResponseFunction::
createGradientOpT() const
{
  Teuchos::RCP<Tpetra_CrsMatrix> GT =
    Teuchos::rcp(new Tpetra_CrsMatrix(gradient_graphT));
  GT->fillComplete();
  return GT;
}


void
Albany::SolutionResponseFunction::
evaluateResponse(const double /*current_time*/,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
		const Teuchos::Array<ParamVec>& /*p*/,
    const Teuchos::RCP<Thyra_Vector>& g)
{
  cullSolution(x, g);
}

void
Albany::SolutionResponseFunction::
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

void
Albany::SolutionResponseFunction::
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
  bool callFillComplete = false;

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

void
Albany::SolutionResponseFunction::
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

Teuchos::RCP<const Tpetra_Map>
Albany::SolutionResponseFunction::
buildCulledMapT(const Tpetra_Map& x_mapT,
	       const Teuchos::Array<int>& keepDOF) const
{
  int numKeepDOF = std::accumulate(keepDOF.begin(), keepDOF.end(), 0);
  int Neqns = keepDOF.size();
  int N = x_mapT.getNodeNumElements(); // x_mapT is map for solution vector

  TEUCHOS_ASSERT( !(N % Neqns) ); // Assume that all the equations for
                                  // a given node are on the assigned
                                  // processor. I.e. need to ensure
                                  // that N is exactly Neqns-divisible

  int nnodes = N / Neqns;          // number of fem nodes
  int N_new = nnodes * numKeepDOF; // length of local x_new

  Teuchos::ArrayView<const Tpetra_GO> gidsT = x_mapT.getNodeElementList();
  Teuchos::Array<Tpetra_GO> gids_new(N_new);
  int idx = 0;
  for ( int inode = 0; inode < N/Neqns ; ++inode) // For every node
    for ( int ieqn = 0; ieqn < Neqns; ++ieqn )  // Check every dof on the node
      if ( keepDOF[ieqn] == 1 )  // then want to keep this dof
	gids_new[idx++] = gidsT[(inode*Neqns)+ieqn];
  // end cull

  Teuchos::RCP<const Tpetra_Map> x_new_mapT = Tpetra::createNonContigMapWithNode<LO, Tpetra_GO, KokkosNode> (gids_new, x_mapT.getComm(), x_mapT.getNode());

  return x_new_mapT;

}

void
Albany::SolutionResponseFunction::
cullSolution(const Teuchos::RCP<const Thyra_MultiVector>& x, const Teuchos::RCP<Thyra_MultiVector>& x_culled) const
{
  auto xT = Albany::getConstTpetraMultiVector(x);
  auto x_culledT = Albany::getTpetraMultiVector(x_culled);
  x_culledT->doImport(*xT, *importerT, Tpetra::INSERT);
}
