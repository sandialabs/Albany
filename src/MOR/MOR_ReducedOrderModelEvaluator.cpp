//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "MOR_ReducedOrderModelEvaluator.hpp"

#include "MOR_ReducedSpace.hpp"
#include "MOR_ReducedOperatorFactory.hpp"

#include "Epetra_Vector.h"
#include "Epetra_CrsMatrix.h"

#include "Teuchos_Tuple.hpp"
#include "Teuchos_TestForException.hpp"

namespace MOR {

using Teuchos::RCP;
using Teuchos::rcp;
using Teuchos::rcp_dynamic_cast;
using Teuchos::null;
using Teuchos::is_null;
using Teuchos::nonnull;
using Teuchos::Tuple;
using Teuchos::tuple;

ReducedOrderModelEvaluator::ReducedOrderModelEvaluator(const RCP<EpetraExt::ModelEvaluator> &fullOrderModel,
                                                       const RCP<const ReducedSpace> &solutionSpace,
                                                       const RCP<ReducedOperatorFactory> &reducedOpFactory) :
  fullOrderModel_(fullOrderModel),
  solutionSpace_(solutionSpace),
  reducedOpFactory_(reducedOpFactory),
  x_init_(null),
  x_dot_init_(null)
{
  reset_x_and_x_dot_init();
}

void ReducedOrderModelEvaluator::reset_x_and_x_dot_init()
{
  reset_x_init();
  reset_x_dot_init();
}

void ReducedOrderModelEvaluator::reset_x_init()
{
  if (nonnull(fullOrderModel_->get_x_init())) {
    x_init_ = solutionSpace_->reduction(*fullOrderModel_->get_x_init());
  } else {
    x_init_ = null;
  }
}

void ReducedOrderModelEvaluator::reset_x_dot_init()
{
  if (nonnull(fullOrderModel_->get_x_dot_init())) {
    x_dot_init_ = solutionSpace_->linearReduction(*fullOrderModel_->get_x_dot_init());
  } else {
    x_dot_init_ = null;
  }
}

Teuchos::RCP<const EpetraExt::ModelEvaluator> ReducedOrderModelEvaluator::getFullOrderModel() const
{
  return fullOrderModel_;
}

Teuchos::RCP<const ReducedSpace> ReducedOrderModelEvaluator::getSolutionSpace() const
{
  return solutionSpace_;
}

const Epetra_Map &ReducedOrderModelEvaluator::componentMap() const
{
  return solutionSpace_->componentMap();
}

RCP<const Epetra_Map> ReducedOrderModelEvaluator::componentMapRCP() const
{
  // TODO more efficient
  return rcp(new Epetra_Map(this->componentMap()));
}

RCP<const Epetra_Map> ReducedOrderModelEvaluator::get_x_map() const
{
  return componentMapRCP();
}

RCP<const Epetra_Map> ReducedOrderModelEvaluator::get_f_map() const
{
  return componentMapRCP();
}

RCP<const Epetra_Map> ReducedOrderModelEvaluator::get_p_map(int l) const
{
  return fullOrderModel_->get_p_map(l);
}

RCP<const Teuchos::Array<std::string> > ReducedOrderModelEvaluator::get_p_names(int l) const
{
  return fullOrderModel_->get_p_names(l);
}

RCP<const Epetra_Map> ReducedOrderModelEvaluator::get_g_map(int j) const
{
  return fullOrderModel_->get_g_map(j);
}

RCP<const Epetra_Vector> ReducedOrderModelEvaluator::get_x_init() const
{
  return x_init_;
}

RCP<const Epetra_Vector> ReducedOrderModelEvaluator::get_x_dot_init() const
{
  return x_dot_init_;
}

RCP<const Epetra_Vector> ReducedOrderModelEvaluator::get_p_init(int l) const
{
  return fullOrderModel_->get_p_init(l);
}

double ReducedOrderModelEvaluator::get_t_init() const
{
  return fullOrderModel_->get_t_init();
}

double ReducedOrderModelEvaluator::getInfBound() const
{
  return fullOrderModel_->getInfBound();
}

RCP<const Epetra_Vector> ReducedOrderModelEvaluator::get_p_lower_bounds(int l) const
{
  return fullOrderModel_->get_p_lower_bounds(l);
}

RCP<const Epetra_Vector> ReducedOrderModelEvaluator::get_p_upper_bounds(int l) const
{
  return fullOrderModel_->get_p_upper_bounds(l);
}

double ReducedOrderModelEvaluator::get_t_lower_bound() const
{
  return fullOrderModel_->get_t_lower_bound();
}

double ReducedOrderModelEvaluator::get_t_upper_bound() const
{
  return fullOrderModel_->get_t_upper_bound();
}

RCP<Epetra_Operator> ReducedOrderModelEvaluator::create_W() const
{
  const RCP<Epetra_Operator> fullOrderOperator = fullOrderModel_->create_W();
  if (is_null(fullOrderOperator)) {
    return null;
  }

  return reducedOpFactory_->reducedJacobianNew();
}

RCP<Epetra_Operator> ReducedOrderModelEvaluator::create_DgDp_op(int j, int l) const
{
  return fullOrderModel_->create_DgDp_op(j, l);
}

EpetraExt::ModelEvaluator::InArgs ReducedOrderModelEvaluator::createInArgs() const
{
  const InArgs fullInArgs = fullOrderModel_->createInArgs();

  InArgsSetup result;

  result.setModelEvalDescription("MOR applied to " + fullInArgs.modelEvalDescription());

  result.set_Np(fullInArgs.Np());

  // Requires underlying full order model to accept a state input
  TEUCHOS_TEST_FOR_EXCEPT(!fullInArgs.supports(IN_ARG_x));
  const Tuple<EInArgsMembers, 5> optionalMembers = tuple(IN_ARG_x, IN_ARG_x_dot, IN_ARG_t, IN_ARG_alpha, IN_ARG_beta);
  for (Tuple<EInArgsMembers, 5>::const_iterator it = optionalMembers.begin(); it != optionalMembers.end(); ++it) {
    const EInArgsMembers member = *it;
    result.setSupports(member, fullInArgs.supports(member));
  }

  return result;
}

EpetraExt::ModelEvaluator::OutArgs ReducedOrderModelEvaluator::createOutArgs() const
{
  const OutArgs fullOutArgs = fullOrderModel_->createOutArgs();

  OutArgsSetup result;

  result.setModelEvalDescription("MOR applied to " + fullOutArgs.modelEvalDescription());

  result.set_Np_Ng(fullOutArgs.Np(), fullOutArgs.Ng());

  for (int j = 0; j < fullOutArgs.Ng(); ++j) {
    if (fullOutArgs.supports(OUT_ARG_DgDx, j).supports(DERIV_TRANS_MV_BY_ROW)) {
      result.setSupports(OUT_ARG_DgDx, j, DERIV_TRANS_MV_BY_ROW);
      result.set_DgDx_properties(j, fullOutArgs.get_DgDx_properties(j));
    }
  }

  for (int j = 0; j < fullOutArgs.Ng(); ++j) {
    if (fullOutArgs.supports(OUT_ARG_DgDx_dot, j).supports(DERIV_TRANS_MV_BY_ROW)) {
      result.setSupports(OUT_ARG_DgDx_dot, j, DERIV_TRANS_MV_BY_ROW);
      result.set_DgDx_dot_properties(j, fullOutArgs.get_DgDx_dot_properties(j));
    }
  }

  for (int l = 0; l < fullOutArgs.Np(); ++l) {
    if (fullOutArgs.supports(OUT_ARG_DfDp, l).supports(DERIV_MV_BY_COL)) {
      result.setSupports(OUT_ARG_DfDp, l, DERIV_MV_BY_COL);
      result.set_DfDp_properties(l, fullOutArgs.get_DfDp_properties(l));
    }
  }

  for (int j = 0; j < fullOutArgs.Ng(); ++j) {
    for (int l = 0; l < fullOutArgs.Np(); ++l) {
      result.setSupports(OUT_ARG_DgDp, j, l, fullOutArgs.supports(OUT_ARG_DgDp, j, l));
      result.set_DgDp_properties(j, l, fullOutArgs.get_DgDp_properties(j, l));
    }
  }

  const Tuple<EOutArgsMembers, 2> optionalMembers = tuple(OUT_ARG_f, OUT_ARG_W);
  for (Tuple<EOutArgsMembers, 2>::const_iterator it = optionalMembers.begin(); it != optionalMembers.end(); ++it) {
    const EOutArgsMembers member = *it;
    result.setSupports(member, fullOutArgs.supports(member));
  }

  result.set_W_properties(fullOutArgs.get_W_properties());

  return result;
}

void ReducedOrderModelEvaluator::evalModel(const InArgs &inArgs, const OutArgs &outArgs) const
{
  // Copy arguments to be able to modify x and x_dot
  InArgs fullInArgs = fullOrderModel_->createInArgs();
  {
    // Copy untouched supported inArgs content
    if (fullInArgs.supports(IN_ARG_t))     fullInArgs.set_t(inArgs.get_t());
    if (fullInArgs.supports(IN_ARG_alpha)) fullInArgs.set_alpha(inArgs.get_alpha());
    if (fullInArgs.supports(IN_ARG_beta))  fullInArgs.set_beta(inArgs.get_beta());
    for (int l = 0; l < fullInArgs.Np(); ++l) {
      fullInArgs.set_p(l, inArgs.get_p(l));
    }

    // x <- basis * x_r + x_origin
    TEUCHOS_TEST_FOR_EXCEPT(is_null(inArgs.get_x()));
    fullInArgs.set_x(solutionSpace_->expansion(*inArgs.get_x()));

    // x_dot <- basis * x_dot_r
    if (inArgs.supports(IN_ARG_x_dot) && nonnull(inArgs.get_x_dot())) {
      fullInArgs.set_x_dot(solutionSpace_->linearExpansion(*inArgs.get_x_dot()));
    }
  }

  // Copy arguments to be able to modify f and W
  OutArgs fullOutArgs = fullOrderModel_->createOutArgs();

  const bool supportsResidual = fullOutArgs.supports(OUT_ARG_f);
  const bool requestedResidual = supportsResidual && nonnull(outArgs.get_f());

  const bool supportsJacobian = fullOutArgs.supports(OUT_ARG_W);
  const bool requestedJacobian = supportsJacobian && nonnull(outArgs.get_W());

  bool requestedAnyDfDp = false;
  for (int l = 0; l < outArgs.Np(); ++l) {
    const bool supportsDfDp = !outArgs.supports(OUT_ARG_DfDp, l).none();
    const bool requestedDfDp = supportsDfDp && (!outArgs.get_DfDp(l).isEmpty());
    if (requestedDfDp) {
      requestedAnyDfDp = true;
      break;
    }
  }
  const bool requestedProjection = requestedResidual || requestedAnyDfDp;

  const bool fullJacobianRequired =
    reducedOpFactory_->fullJacobianRequired(requestedProjection, requestedJacobian);

  {
    // Prepare forwarded outArgs content (g and DgDp)
    for (int j = 0; j < outArgs.Ng(); ++j) {
      fullOutArgs.set_g(j, outArgs.get_g(j));
      for (int l = 0; l < inArgs.Np(); ++l) {
        if (!outArgs.supports(OUT_ARG_DgDp, j, l).none()) {
          fullOutArgs.set_DgDp(j, l, outArgs.get_DgDp(j, l));
        }
      }
    }

    // Prepare reduced residual (f_r)
    if (requestedResidual) {
      const Evaluation<Epetra_Vector> f_r = outArgs.get_f();
      const Evaluation<Epetra_Vector> f(
          rcp(new Epetra_Vector(*fullOrderModel_->get_f_map(), false)),
          f_r.getType());
      fullOutArgs.set_f(f);
    }

    if (fullJacobianRequired) {
      fullOutArgs.set_W(fullOrderModel_->create_W());
    }

    // Prepare reduced sensitivities DgDx_r (Only mv with gradient orientation is supported)
    for (int j = 0; j < outArgs.Ng(); ++j) {
      if (!outArgs.supports(OUT_ARG_DgDx, j).none()) {
        TEUCHOS_ASSERT(outArgs.supports(OUT_ARG_DgDx, j).supports(DERIV_TRANS_MV_BY_ROW));
        if (!outArgs.get_DgDx(j).isEmpty()) {
          TEUCHOS_ASSERT(
              nonnull(outArgs.get_DgDx(j).getMultiVector()) &&
              outArgs.get_DgDx(j).getMultiVectorOrientation() == DERIV_TRANS_MV_BY_ROW);
          const int g_size = fullOrderModel_->get_g_map(j)->NumGlobalElements();
          const RCP<Epetra_MultiVector> full_dgdx_mv = rcp(
              new Epetra_MultiVector(*fullOrderModel_->get_x_map(), g_size, false));
          const Derivative full_dgdx_deriv(full_dgdx_mv, DERIV_TRANS_MV_BY_ROW);
          fullOutArgs.set_DgDx(j, full_dgdx_deriv);
        }
      }
    }

    // Prepare reduced sensitivities DgDx_dot_r (Only mv with gradient orientation is supported)
    for (int j = 0; j < outArgs.Ng(); ++j) {
      if (!outArgs.supports(OUT_ARG_DgDx_dot, j).none()) {
        TEUCHOS_ASSERT(outArgs.supports(OUT_ARG_DgDx_dot, j).supports(DERIV_TRANS_MV_BY_ROW));
        if (!outArgs.get_DgDx_dot(j).isEmpty()) {
          TEUCHOS_ASSERT(
              nonnull(outArgs.get_DgDx_dot(j).getMultiVector()) &&
              outArgs.get_DgDx_dot(j).getMultiVectorOrientation() == DERIV_TRANS_MV_BY_ROW);
          const int g_size = fullOrderModel_->get_g_map(j)->NumGlobalElements();
          const RCP<Epetra_MultiVector> full_dgdx_dot_mv = rcp(
              new Epetra_MultiVector(*fullOrderModel_->get_x_map(), g_size, false));
          const Derivative full_dgdx_dot_deriv(full_dgdx_dot_mv, DERIV_TRANS_MV_BY_ROW);
          fullOutArgs.set_DgDx_dot(j, full_dgdx_dot_deriv);
        }
      }
    }

    // Prepare reduced sensitivities DfDp_r (Only mv with jacobian orientation is supported)
    for (int l = 0; l < outArgs.Np(); ++l) {
      if (!outArgs.supports(OUT_ARG_DfDp, l).none()) {
        TEUCHOS_ASSERT(outArgs.supports(OUT_ARG_DfDp, l).supports(DERIV_MV_BY_COL));
        if (!outArgs.get_DfDp(l).isEmpty()) {
          TEUCHOS_ASSERT(
              nonnull(outArgs.get_DfDp(l).getMultiVector()) &&
              outArgs.get_DfDp(l).getMultiVectorOrientation() == DERIV_MV_BY_COL);
          const int p_size = fullOrderModel_->get_p_map(l)->NumGlobalElements();
          const RCP<Epetra_MultiVector> full_dfdp_mv = rcp(
              new Epetra_MultiVector(*fullOrderModel_->get_f_map(), p_size, false));
          const Derivative full_dfdp_deriv(full_dfdp_mv, DERIV_MV_BY_COL);
          fullOutArgs.set_DfDp(l, full_dfdp_deriv);
        }
      }
    }
  }

  // (f, W) <- fullOrderModel(x, x_dot, ...)
  fullOrderModel_->evalModel(fullInArgs, fullOutArgs);

  // (W * basis, W_r) <- W
  if (fullJacobianRequired) {
    reducedOpFactory_->fullJacobianIs(*fullOutArgs.get_W());
  }

  // f_r <- leftBasis^T * f
  if (requestedResidual) {
    reducedOpFactory_->leftProjection(*fullOutArgs.get_f(), *outArgs.get_f());
  }

  // Wr <- leftBasis^T * W * basis
  if (requestedJacobian) {
    const RCP<Epetra_CrsMatrix> W_r = rcp_dynamic_cast<Epetra_CrsMatrix>(outArgs.get_W());
    TEUCHOS_TEST_FOR_EXCEPT(is_null((W_r)));
    reducedOpFactory_->reducedJacobian(*W_r);
  }

  // (DgDx_r)^T <- basis^T * (DgDx)^T
  for (int j = 0; j < outArgs.Ng(); ++j) {
    if (!outArgs.supports(OUT_ARG_DgDx, j).none()) {
      const RCP<Epetra_MultiVector> dgdx_r_mv = outArgs.get_DgDx(j).getMultiVector();
      if (nonnull(dgdx_r_mv)) {
        const RCP<const Epetra_MultiVector> full_dgdx_mv = fullOutArgs.get_DgDx(j).getMultiVector();
        solutionSpace_->linearReduction(*full_dgdx_mv, *dgdx_r_mv);
      }
    }
  }

  // (DgDx_dot_r)^T <- basis^T * (DgDx_dot)^T
  for (int j = 0; j < outArgs.Ng(); ++j) {
    if (!outArgs.supports(OUT_ARG_DgDx_dot, j).none()) {
      const RCP<Epetra_MultiVector> dgdx_dot_r_mv = outArgs.get_DgDx_dot(j).getMultiVector();
      if (nonnull(dgdx_dot_r_mv)) {
        const RCP<const Epetra_MultiVector> full_dgdx_dot_mv = fullOutArgs.get_DgDx_dot(j).getMultiVector();
        solutionSpace_->linearReduction(*full_dgdx_dot_mv, *dgdx_dot_r_mv);
      }
    }
  }

  // DfDp_r <- leftBasis^T * DfDp
  for (int l = 0; l < outArgs.Np(); ++l) {
    if (!outArgs.supports(OUT_ARG_DfDp, l).none()) {
      const RCP<Epetra_MultiVector> dfdp_r_mv = outArgs.get_DfDp(l).getMultiVector();
      if (nonnull(dfdp_r_mv)) {
        const RCP<const Epetra_MultiVector> full_dfdp_mv = fullOutArgs.get_DfDp(l).getMultiVector();
        reducedOpFactory_->leftProjection(*full_dfdp_mv, *dfdp_r_mv);
      }
    }
  }
}

} // namespace MOR
