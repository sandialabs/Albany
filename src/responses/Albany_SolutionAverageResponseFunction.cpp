//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include "Albany_SolutionAverageResponseFunction.hpp"

#include "Albany_TpetraThyraUtils.hpp"

Albany::SolutionAverageResponseFunction::
SolutionAverageResponseFunction(const Teuchos::RCP<const Teuchos_Comm>& commT) :
  ScalarResponseFunction(commT)
{
}

Albany::SolutionAverageResponseFunction::
~SolutionAverageResponseFunction()
{
}

unsigned int
Albany::SolutionAverageResponseFunction::
numResponses() const 
{
  return 1;
}

void
Albany::SolutionAverageResponseFunction::
evaluateResponse(const double /*current_time*/,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
		const Teuchos::Array<ParamVec>& /*p*/,
		const Teuchos::RCP<Thyra_Vector>& g)
{
  evaluateResponseImpl(*x,*g);
}

void
Albany::SolutionAverageResponseFunction::
evaluateTangent(const double alpha, 
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
  // Evaluate response g
  if (!g.is_null()) {
    evaluateResponseImpl(*x,*g);
  }

  // Evaluate tangent of g: dg/dx*Vx + dg/dxdot*Vxdot + dg/dp*Vp
  //                      =    gx    +       0        +    gp
  // If Vx is null, Vx is the identity
  if (!gx.is_null()) {
    if (!Vx.is_null()) {
      if (ones.is_null() || ones->domain()->dim()!=Vx->domain()->dim()) {
        ones = Thyra::createMembers(Vx->range(), Vx->domain()->dim());
        ones->assign(1.0);
      }
      Teuchos::Array<ST> means; 
      means.resize(Vx->domain()->dim());
      Vx->dots(*ones,means());
      for (auto& mean : means) {
        mean /= Vx->domain()->dim();
      }
      for (int j=0; j<Vx->domain()->dim(); j++) {  
        gx->col(j)->assign(means[j]);
      }
    }
    else {
      gx->assign(1.0/x->space()->dim());
    }
    gx->scale(alpha);
  }
  
  if (!gp.is_null()) {
    gp->assign(0.0);
  }
}

void
Albany::SolutionAverageResponseFunction::
evaluateGradient(const double current_time,
    const Teuchos::RCP<const Thyra_Vector>& x,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
		const Teuchos::Array<ParamVec>& p,
		ParamVec* deriv_p,
		const Teuchos::RCP<Thyra_Vector>& g,
		const Teuchos::RCP<Thyra_MultiVector>& dg_dx,
		const Teuchos::RCP<Thyra_MultiVector>& dg_dxdot,
		const Teuchos::RCP<Thyra_MultiVector>& dg_dxdotdot,
		const Teuchos::RCP<Thyra_MultiVector>& dg_dp)
{
  // Evaluate response g
  if (!g.is_null()) {
    evaluateResponseImpl(*x,*g);
  }

  // Evaluate dg/dx
  if (!dg_dx.is_null()) {
    dg_dx->assign(1.0 / x->space()->dim());
  }

  // Evaluate dg/dxdot
  if (!dg_dxdot.is_null()) {
    dg_dxdot->assign(0.0);
  }

  // Evaluate dg/dxdotdot
  if (!dg_dxdotdot.is_null()) {
    dg_dxdotdot->assign(0.0);
  }

  // Evaluate dg/dp
  if (!dg_dp.is_null()) {
    dg_dp->assign(0.0);
  }
}

void
Albany::SolutionAverageResponseFunction::
evaluateDistParamDeriv(
    const double /*current_time*/,
    const Teuchos::RCP<const Thyra_Vector>& /*x*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
    const Teuchos::Array<ParamVec>& /*param_array*/,
    const std::string& /*dist_param_name*/,
    const Teuchos::RCP<Thyra_MultiVector>& dg_dp)
{
  // Evaluate response derivative dg_dp
  if (!dg_dp.is_null()) {
    dg_dp->assign(0.0);
  }
}

void 
Albany::SolutionAverageResponseFunction::
evaluateResponseImpl (
    const Thyra_Vector& x,
		Thyra_Vector& g)
{
  //IKT, 12/11/19: I had to add these conversions to TpetraVectors b/c I do 
  //not believe there is a method equivalent to getLocalLength in Thyra.
  //The checks done here with the local length are needed for problems
  //where the mesh can adapt. 
  int one_ll = 0; 
  if (!one.is_null()) {
    auto oneTpetraVector = ConverterT::getConstTpetraVector(one); 
    one_ll = oneTpetraVector->getLocalLength(); 
  }
  auto xTpetraVector = ConverterT::getConstTpetraVector(Teuchos::rcpFromRef(x)); 
  int x_ll = xTpetraVector->getLocalLength(); 
  if (one.is_null() || (x_ll != one_ll)) {
    one = Thyra::createMember(x.space());
    one->assign(1.0);
  }
  const ST mean = one->dot(x) / x.space()->dim();
  g.assign(mean);
}
