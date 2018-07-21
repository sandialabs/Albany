//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_SolutionMaxValueResponseFunction.hpp"
#include "Albany_TpetraThyraUtils.hpp"

#include "Teuchos_CommHelpers.hpp"
#include "Tpetra_DistObject.hpp"
#include "Thyra_SpmdVectorBase.hpp"

#include <limits>

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
evaluateResponse(const double /*current_time*/,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
		const Teuchos::Array<ParamVec>& /*p*/,
		Tpetra_Vector& gT)
{
  Teuchos::ArrayRCP<ST> gT_nonconstView = gT.get1dViewNonConst();
  computeMaxValue(x, gT_nonconstView[0]);
}

void
Albany::SolutionMaxValueResponseFunction::
evaluateTangent(const double alpha, 
		const double /*beta*/,
		const double /*omega*/,
		const double current_time,
		bool /*sum_derivs*/,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& xdot,
    const Teuchos::RCP<const Thyra_Vector>& xdotdot,
		const Teuchos::Array<ParamVec>& p,
		ParamVec* deriv_p,
    const Teuchos::RCP<const Thyra_MultiVector>& Vx,
    const Teuchos::RCP<const Thyra_MultiVector>& /*Vxdot*/,
    const Teuchos::RCP<const Thyra_MultiVector>& /*Vxdotdot*/,
    const Teuchos::RCP<const Thyra_MultiVector>& /*Vp*/,
		Tpetra_Vector* gT,
		Tpetra_MultiVector* gxT,
		Tpetra_MultiVector* gpT)
{

  if (gxT != NULL || gpT != NULL) {
    evaluateGradient(current_time, x, xdot, xdotdot, p, deriv_p, gT, gxT, NULL, NULL, gpT);
  }

  if (gxT != NULL && !Vx.is_null()) {
    Teuchos::RCP<Tpetra_MultiVector> dgdxT = Teuchos::rcp(new Tpetra_MultiVector(*gxT)); //is this needed? 
    Teuchos::ETransp T = Teuchos::TRANS; 
    Teuchos::ETransp N = Teuchos::NO_TRANS; 
    auto VxT = Albany::getConstTpetraMultiVector(Vx);
    gxT->multiply(T, N, alpha, *dgdxT, *VxT, 0.0);
  }
}

void
Albany::SolutionMaxValueResponseFunction::
evaluateGradient(const double /*current_time*/,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
		const Teuchos::Array<ParamVec>& /*p*/,
		ParamVec* /*deriv_p*/,
		Tpetra_Vector* gT,
		Tpetra_MultiVector* dg_dxT,
		Tpetra_MultiVector* dg_dxdotT,
		Tpetra_MultiVector* dg_dxdotdotT,
		Tpetra_MultiVector* dg_dpT)
{
  ST max_val;
  computeMaxValue(x, max_val);
  
  // Evaluate response g
  if (gT != NULL) {
    Teuchos::ArrayRCP<ST> gT_nonconstView = gT->get1dViewNonConst();
    gT_nonconstView[0] = max_val;
  }

  // Evaluate dg/dx
  if (dg_dxT != NULL) {
    // In order to loop throught the vector entries, we must assume
    // the thyra vector concrete type inherits from Thyra::SpmdMultiVectorBase,
    // which is the interface for distributed memory thyra vectors.
    using SpmdVector = Thyra::SpmdVectorBase<ST>;
    auto xspmd = Teuchos::rcp_dynamic_cast<const SpmdVector>(x);
    TEUCHOS_TEST_FOR_EXCEPTION (xspmd.is_null(), std::runtime_error, "Error! Could not cast to Spmd vector.\n");

    auto x_local = xspmd->getLocalSubVector();
    Teuchos::ArrayRCP<ST> dg_dxT_nonconstView;
    for (int i=0; i<xspmd->spmdSpace()->localSubDim(); ++i) {
      dg_dxT_nonconstView = dg_dxT->getDataNonConst(0); 
      if (x_local[i] == max_val) {
        dg_dxT_nonconstView[i] = 1.0;
      } else {
        dg_dxT_nonconstView[i] = 0.0;
      }
    }
  }

  // Evaluate dg/dxdot
  if (dg_dxdotT != NULL) {
    dg_dxdotT->putScalar(0.0);
  }
  if (dg_dxdotdotT != NULL) {
    dg_dxdotdotT->putScalar(0.0);
  }

  // Evaluate dg/dp
  if (dg_dpT != NULL) {
    dg_dpT->putScalar(0.0);
  }
}

//! Evaluate distributed parameter derivative dg/dp
void
Albany::SolutionMaxValueResponseFunction::
evaluateDistParamDeriv(
    const double /*current_time*/,
    const Teuchos::RCP<const Thyra_Vector>& /*x*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
    const Teuchos::Array<ParamVec>& /*param_array*/,
    const std::string& /*dist_param_name*/,
    Tpetra_MultiVector* dg_dpT)
{
  if (dg_dpT) {
    dg_dpT->putScalar(0.0);
  }
}

void
Albany::SolutionMaxValueResponseFunction::
computeMaxValue(const Teuchos::RCP<const Thyra_Vector>& x, ST& global_max)
{
  // In order to loop throught the vector entries, we must assume
  // the thyra vector concrete type inherits from Thyra::SpmdMultiVectorBase,
  // which is the interface for distributed memory thyra vectors.
  using SpmdVector = Thyra::SpmdVectorBase<ST>;
  auto xspmd = Teuchos::rcp_dynamic_cast<const SpmdVector>(x);
  TEUCHOS_TEST_FOR_EXCEPTION (xspmd.is_null(), std::runtime_error, "Error! Could not cast to Spmd vector.\n");
  auto x_local = xspmd->getLocalSubVector();
  
  // Loop over nodes to find max value for equation eq
  int num_my_nodes = xspmd->spmdSpace()->localSubDim() / neq;
  int index;
  ST my_max = std::numeric_limits<ST>::lowest();
  for (int node=0; node<num_my_nodes; node++) {
    if (interleavedOrdering) {
      index = node*neq+eq;
    } else {
      index = node + eq*num_my_nodes;
    }
    if (x_local[index] > my_max) {
      my_max = x_local[index];
    }
  }

  // Check remainder (AGS: NOT SURE HOW THIS CODE GETS CALLED?)
  // LB: I believe this code would get called if equations at a given node are not
  //     forced to be on the same process, in which case neq may not divide the local
  //     dimension. I also believe Albany makes sure this does not happen, so I *think*
  //     these lines *should* be safe to remove...
  if (num_my_nodes*neq+eq < xspmd->spmdSpace()->localSubDim()) {
    if (interleavedOrdering) {
      index = num_my_nodes*neq+eq;
    } else {
      index = num_my_nodes + eq*num_my_nodes;
    }
    if (x_local[index] > my_max) {
      my_max = x_local[index];
    }
  }

  // Get max value across all proc's
  Teuchos::reduceAll(*commT_, Teuchos::REDUCE_MAX, 1, &my_max, &global_max); 
}
