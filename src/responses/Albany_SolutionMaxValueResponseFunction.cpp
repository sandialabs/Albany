//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include "Albany_SolutionMaxValueResponseFunction.hpp"
#include "Teuchos_CommHelpers.hpp"
#include "Tpetra_DistObject.hpp"

Albany::SolutionMaxValueResponseFunction::
SolutionMaxValueResponseFunction(const Teuchos::RCP<const Teuchos_Comm>& commT,
				 int neq_, int eq_, bool interleavedOrdering_) :
  SamplingBasedScalarResponseFunction(commT),
  commT_(commT), 
  neq(neq_), eq(eq_), interleavedOrdering(interleavedOrdering_)
{
}

Albany::SolutionMaxValueResponseFunction::
~SolutionMaxValueResponseFunction()
{
}

unsigned int
Albany::SolutionMaxValueResponseFunction::
numResponses() const 
{
  return 1;
}


void
Albany::SolutionMaxValueResponseFunction::
evaluateResponseT(const double current_time,
		 const Tpetra_Vector* xdotT,
		 const Tpetra_Vector* xdotdotT,
		 const Tpetra_Vector& xT,
		 const Teuchos::Array<ParamVec>& p,
		 Tpetra_Vector& gT)
{
  int index;
  Teuchos::ArrayRCP<ST> gT_nonconstView = gT.get1dViewNonConst();
  computeMaxValueT(xT, gT_nonconstView[0], index);
}


void
Albany::SolutionMaxValueResponseFunction::
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

  if (gxT != NULL || gpT != NULL)
    evaluateGradientT(current_time, xdotT, xdotdotT, xT, p, deriv_p, gT, gxT, NULL, NULL, gpT);

  if (gxT != NULL && VxT != NULL) {
    Teuchos::RCP<Tpetra_MultiVector> dgdxT = Teuchos::rcp(new Tpetra_MultiVector(*gxT)); //is this needed? 
    Teuchos::ETransp T = Teuchos::TRANS; 
    Teuchos::ETransp N = Teuchos::NO_TRANS; 
    gxT->multiply(T, N, alpha, *dgdxT, *VxT, 0.0);
  }
}

void
Albany::SolutionMaxValueResponseFunction::
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
  int global_index;
  double mxv;
  computeMaxValueT(xT, mxv, global_index);
  
  // Evaluate response g
  if (gT != NULL) {
    Teuchos::ArrayRCP<ST> gT_nonconstView = gT->get1dViewNonConst();
    gT_nonconstView[0] = mxv;
  }

  Teuchos::ArrayRCP<const ST> xT_constView = xT.get1dView();
  
  // Evaluate dg/dx
  if (dg_dxT != NULL) {
    Teuchos::ArrayRCP<ST> dg_dxT_nonconstView;
    int im = -1;
    for (int i=0; i<xT.getMap()->getNodeNumElements(); i++) {
       dg_dxT_nonconstView = dg_dxT->getDataNonConst(0); 
       if (xT_constView[i] == mxv) { dg_dxT_nonconstView[i] = 1.0; im = i; }
       else                          dg_dxT_nonconstView[i] = 0.0;
    }

  }

  // Evaluate dg/dxdot
  if (dg_dxdotT != NULL)
    dg_dxdotT->putScalar(0.0);
  if (dg_dxdotdotT != NULL)
    dg_dxdotdotT->putScalar(0.0);

  // Evaluate dg/dp
  if (dg_dpT != NULL)
    dg_dpT->putScalar(0.0);

}

//! Evaluate distributed parameter derivative dg/dp
void
Albany::SolutionMaxValueResponseFunction::
evaluateDistParamDerivT(
    const double current_time,
    const Tpetra_Vector* xdotT,
    const Tpetra_Vector* xdotdotT,
    const Tpetra_Vector& xT,
    const Teuchos::Array<ParamVec>& param_array,
    const std::string& dist_param_name,
    Tpetra_MultiVector* dg_dpT)
{
  if (dg_dpT) {
      dg_dpT->putScalar(0.0);
  }
}

void
Albany::SolutionMaxValueResponseFunction::
computeMaxValueT(const Tpetra_Vector& xT, double& global_max, int& global_index)
{
  //The following is needed b/c Epetra_MaxDouble comes from Trilinos Epetra package.
  double Tpetra_MaxDouble = 1.0E+100; 
  double my_max = -Tpetra_MaxDouble;
  int my_index = -1, index;
  
  Teuchos::ArrayRCP<const ST> xT_constView = xT.get1dView();
  
  // Loop over nodes to find max value for equation eq
  int num_my_nodes = xT.getLocalLength() / neq;
  for (int node=0; node<num_my_nodes; node++) {
    if (interleavedOrdering)  index = node*neq+eq;
    else                      index = node + eq*num_my_nodes;
    if (xT_constView[index] > my_max) {
      my_max = xT_constView[index];
      my_index = index;
    }
  }

  // Check remainder (AGS: NOT SURE HOW THIS CODE GETS CALLED?)
  if (num_my_nodes*neq+eq < xT.getLocalLength()) {
    if (interleavedOrdering)  index = num_my_nodes*neq+eq;
    else                      index = num_my_nodes + eq*num_my_nodes;
    if (xT_constView[index] > my_max) {
      my_max = xT_constView[index];
      my_index = index;
    }
  }

  Teuchos::RCP<const Teuchos::Comm<int> > commT = xT.getMap()->getComm(); 
  // Get max value across all proc's
  Teuchos::reduceAll(*commT, Teuchos::REDUCE_MAX, my_max, Teuchos::ptr(&global_max)); 

  // Compute min of all global indices equal to max value
  if (my_max == global_max)
    my_index = xT.getMap()->getGlobalElement(my_index);
  else
    my_index = xT.getGlobalLength();
  Teuchos::reduceAll(*commT, Teuchos::REDUCE_MIN, my_index, Teuchos::ptr(&global_index)); 
}
