//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_QuadraticLinearOperatorBasedResponseFunction.hpp"
#include "Albany_Application.hpp"
#include "Albany_DistributedParameterLibrary.hpp"
#include "Thyra_VectorStdOps.hpp"
#include "Albany_TpetraTypes.hpp"
#include "Albany_TpetraThyraUtils.hpp"
#include "Tpetra_Core.hpp"
#include "MatrixMarket_Tpetra.hpp"

Albany::QuadraticLinearOperatorBasedResponseFunction::
QuadraticLinearOperatorBasedResponseFunction(const Teuchos::RCP<const Albany::Application> &app,
    Teuchos::ParameterList &responseParams) :
  SamplingBasedScalarResponseFunction(app->getComm()),
  app_(app)
{
  coeff_ = responseParams.get<double>("Scaling Coefficient");
  field_name_ = responseParams.get<std::string>("Field Name");
  file_name_A_ = responseParams.get<std::string>("Linear Operator File Name");
  file_name_D_ = responseParams.get<std::string>("Diagonal Scaling File Name");
}

Albany::QuadraticLinearOperatorBasedResponseFunction::
~QuadraticLinearOperatorBasedResponseFunction()
{
}

unsigned int
Albany::QuadraticLinearOperatorBasedResponseFunction::
numResponses() const 
{
  return 1;
}

void
Albany::QuadraticLinearOperatorBasedResponseFunction::
loadLinearOperator() {
  Teuchos::RCP<const Thyra_Vector> field = app_->getDistributedParameterLibrary()->get(field_name_)->vector();
  Teuchos::RCP<const Tpetra_Map> rowMap = Albany::getTpetraMap(field->space());

  Teuchos::RCP<const Tpetra_Map> colMap;
  Teuchos::RCP<const Tpetra_Map> domainMap = rowMap;
  Teuchos::RCP<const Tpetra_Map> rangeMap = rowMap;
  typedef Tpetra::MatrixMarket::Reader<Tpetra_CrsMatrix> reader_type;

  bool mapIsContiguous =
      (static_cast<Tpetra_GO>(rowMap->getMaxAllGlobalIndex()+1-rowMap->getMinAllGlobalIndex()) ==
       static_cast<Tpetra_GO>(rowMap->getGlobalNumElements()));

  TEUCHOS_TEST_FOR_EXCEPTION (!mapIsContiguous, std::runtime_error,
                              "Error! Row Map needs to be contiguous for the Matrix reader to work.\n");

  auto tpetra_mat =
      reader_type::readSparseFile (file_name_A_, rowMap, colMap, domainMap, rangeMap);

  auto tpetra_diag_mat =
      reader_type::readSparseFile (file_name_D_, rowMap, colMap, domainMap, rangeMap);
  Teuchos::RCP<Tpetra_Vector> tpetra_diag_vec = Teuchos::rcp(new Tpetra_Vector(rowMap));
  tpetra_diag_mat->getLocalDiagCopy (*tpetra_diag_vec);

  A_ = Albany::createThyraLinearOp(tpetra_mat);
  D_ = Albany::createThyraVector(tpetra_diag_vec);
  vec1_ = Thyra::createMember(A_->range());
  vec2_ = Thyra::createMember(A_->range());
}

void
Albany::QuadraticLinearOperatorBasedResponseFunction::
evaluateResponse(const double /*current_time*/,
    const Teuchos::RCP<const Thyra_Vector>& /*x*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
		const Teuchos::Array<ParamVec>& /*p*/,
		const Teuchos::RCP<Thyra_Vector>& g)
{

  if(A_.is_null())
    loadLinearOperator();

  Teuchos::RCP<const Thyra_Vector> field = app_->getDistributedParameterLibrary()->get(field_name_)->vector();

  // A p
  A_->apply(Thyra::EOpTransp::NOTRANS, *field, vec1_.ptr(), 1.0, 0.0);

  // coeff inv(D) A p
  vec2_->assign(0.0);
  Thyra::ele_wise_divide( coeff_, *vec1_, *D_, vec2_.ptr() );

  //  coeff p' A' inv(D) A p
  g->assign(Thyra::dot(*vec1_,*vec2_));

  if (g_.is_null())
    g_ = Thyra::createMember(g->space());
  g_->assign(*g);
}

void
Albany::QuadraticLinearOperatorBasedResponseFunction::
evaluateTangent(const double alpha, 
		const double /*beta*/,
		const double /*omega*/,
		const double /*current_time*/,
		bool /*sum_derivs*/,
    const Teuchos::RCP<const Thyra_Vector>& /*x*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
		Teuchos::Array<ParamVec>& /*p*/,
    const int  /*parameter_index*/,
    const Teuchos::RCP<const Thyra_MultiVector>& /*Vx*/,
    const Teuchos::RCP<const Thyra_MultiVector>& /*Vxdot*/,
    const Teuchos::RCP<const Thyra_MultiVector>& /*Vxdotdot*/,
    const Teuchos::RCP<const Thyra_MultiVector>& /*Vp*/,
    const Teuchos::RCP<Thyra_Vector>& g,
    const Teuchos::RCP<Thyra_MultiVector>& gx,
    const Teuchos::RCP<Thyra_MultiVector>& gp)
{
  if (!g.is_null()) {
    if(A_.is_null())
      loadLinearOperator();

    Teuchos::RCP<const Thyra_Vector> field = app_->getDistributedParameterLibrary()->get(field_name_)->vector();

    // A p
    A_->apply(Thyra::EOpTransp::NOTRANS, *field, vec1_.ptr(), 1.0, 0.0);

    // coeff inv(D) A p
    vec2_->assign(0.0);
    Thyra::ele_wise_divide( coeff_, *vec1_, *D_, vec2_.ptr() );

    //  coeff p' A' inv(D) A p
    g->assign(Thyra::dot(*vec1_,*vec2_));

    if (g_.is_null())
      g_ = Thyra::createMember(g->space());
    g_->assign(*g);
  }

  if (!gx.is_null()) {
    gx->assign(0);
  }

  if (!gp.is_null()) {
    gp->assign(0.0);
  }
}

void
Albany::QuadraticLinearOperatorBasedResponseFunction::
evaluateGradient(const double /*current_time*/,
    const Teuchos::RCP<const Thyra_Vector>& /*x*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
		const Teuchos::Array<ParamVec>& /*p*/,
		const int  /*parameter_index*/,
		const Teuchos::RCP<Thyra_Vector>& g,
		const Teuchos::RCP<Thyra_MultiVector>& dg_dx,
		const Teuchos::RCP<Thyra_MultiVector>& dg_dxdot,
		const Teuchos::RCP<Thyra_MultiVector>& dg_dxdotdot,
		const Teuchos::RCP<Thyra_MultiVector>& dg_dp)
{
  if (!g.is_null()) {

    if(A_.is_null())
      loadLinearOperator();
    Teuchos::RCP<const Thyra_Vector> field = app_->getDistributedParameterLibrary()->get(field_name_)->vector();

    // A p
    A_->apply(Thyra::EOpTransp::NOTRANS, *field, vec1_.ptr(), 1.0, 0.0);

    // coeff inv(D) A p
    vec2_->assign(0.0);
    Thyra::ele_wise_divide( coeff_, *vec1_, *D_, vec2_.ptr() );

    //  coeff p' A' inv(D) A p
    g->assign(Thyra::dot(*vec1_,*vec2_));

    if (g_.is_null())
      g_ = Thyra::createMember(g->space());
    g_->assign(*g);
  }
  
  // Evaluate dg/dx
  if (!dg_dx.is_null()) {
    // V_StV stands for V_out = Scalar * V_in
    dg_dx->assign(0.0);
  }

  // Evaluate dg/dxdot
  if (!dg_dxdot.is_null()) {
    dg_dxdot->assign(0.0);
  }

  // Evaluate dg/dxdot
  if (!dg_dxdotdot.is_null()) {
    dg_dxdotdot->assign(0.0);
  }

  // Evaluate dg/dp
  if (!dg_dp.is_null()) {
    dg_dp->assign(0.0);
  }
}

//! Evaluate distributed parameter derivative dg/dp
void
Albany::QuadraticLinearOperatorBasedResponseFunction::
evaluateDistParamDeriv(
    const double /*current_time*/,
    const Teuchos::RCP<const Thyra_Vector>& /*x*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
    const Teuchos::Array<ParamVec>& /*param_array*/,
    const std::string& dist_param_name,
    const Teuchos::RCP<Thyra_MultiVector>& dg_dp)
{
  if (!dg_dp.is_null()) {
    if(dist_param_name == field_name_) {
      if(A_.is_null())
        loadLinearOperator();
      Teuchos::RCP<const Thyra_Vector> field = app_->getDistributedParameterLibrary()->get(field_name_)->vector();

      //A p
      A_->apply(Thyra::EOpTransp::NOTRANS, *field, vec1_.ptr(), 1.0, 0.0);

      // 2 coeff inv(D) A p
      vec2_->assign(0.0);
      Thyra::ele_wise_divide( 2.0*coeff_, *vec1_, *D_, vec2_.ptr() );

      // 2 coeff A' inv(D) A p
      A_->apply(Thyra::EOpTransp::TRANS, *vec2_, dg_dp.ptr(), 1.0, 0.0);
    } else
      dg_dp->assign(0.0);
  }
}

void
Albany::QuadraticLinearOperatorBasedResponseFunction::
evaluate_HessVecProd_xx(
    const double current_time,
    const Teuchos::RCP<const Thyra_MultiVector>& v,
    const Teuchos::RCP<const Thyra_Vector>& /*x*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
    const Teuchos::Array<ParamVec>& param_array,
    const Teuchos::RCP<Thyra_MultiVector>& Hv_dxdx)
{
  if (!Hv_dxdx.is_null()) {
    Hv_dxdx->assign(0.0);
  }
}

void
Albany::QuadraticLinearOperatorBasedResponseFunction::
evaluate_HessVecProd_xp(
    const double current_time,
    const Teuchos::RCP<const Thyra_MultiVector>& /*v*/,
    const Teuchos::RCP<const Thyra_Vector>& /*x*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
    const Teuchos::Array<ParamVec>& /*param_array*/,
    const std::string& /*dist_param_direction_name*/,
    const Teuchos::RCP<Thyra_MultiVector>& Hv_dp)
{
  if (!Hv_dp.is_null()) {
    Hv_dp->assign(0.0);
  }
}

void
Albany::QuadraticLinearOperatorBasedResponseFunction::
evaluate_HessVecProd_px(
    const double current_time,
    const Teuchos::RCP<const Thyra_MultiVector>& /*v*/,
    const Teuchos::RCP<const Thyra_Vector>& /*x*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
    const Teuchos::Array<ParamVec>& /*param_array*/,
    const std::string& /*dist_param_name*/,
    const Teuchos::RCP<Thyra_MultiVector>& Hv_dp)
{
  if (!Hv_dp.is_null()) {
    Hv_dp->assign(0.0);
  }
}

void
Albany::QuadraticLinearOperatorBasedResponseFunction::
evaluate_HessVecProd_pp(
    const double current_time,
    const Teuchos::RCP<const Thyra_MultiVector>& v,
    const Teuchos::RCP<const Thyra_Vector>& /*x*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdot*/,
    const Teuchos::RCP<const Thyra_Vector>& /*xdotdot*/,
    const Teuchos::Array<ParamVec>& /*param_array*/,
    const std::string& dist_param_name,
    const std::string& dist_param_direction_name,
    const Teuchos::RCP<Thyra_MultiVector>& Hv_dp)
{
  if (!Hv_dp.is_null()) {
    if((dist_param_name == field_name_) && (dist_param_direction_name == field_name_)) {
      if(A_.is_null())
        loadLinearOperator();

      // A v
      A_->apply(Thyra::EOpTransp::NOTRANS, *v, vec1_.ptr(), 1.0, 0.0);

      // 2 coeff inv(D) A v
      vec2_->assign(0.0);
      Thyra::ele_wise_divide( 2.0*coeff_, *vec1_, *D_, vec2_.ptr() );

      // 2 coeff A' inv(D) A v
      A_->apply(Thyra::EOpTransp::TRANS, *vec2_, Hv_dp.ptr(), 1.0, 0.0);
    }
    else
      Hv_dp->assign(0.0);
  }
}

void
Albany::QuadraticLinearOperatorBasedResponseFunction::
printResponse(Teuchos::RCP<Teuchos::FancyOStream> out)
{
  if (g_.is_null()) {
    *out << " the response has not been evaluated yet!";
    return;
  }

  std::size_t precision = 8;
  std::size_t value_width = precision + 4;
  int gsize = g_->space()->dim();

  for (int j = 0; j < gsize; j++) {
    *out << std::setw(value_width) << Thyra::get_ele(*g_,j);
    if (j < gsize-1)
      *out << ", ";
  }
}
