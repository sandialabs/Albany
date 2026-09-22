//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_AlgebraicConstraintModelEvaluator.hpp"

#include "Albany_ThyraUtils.hpp"
#include "Albany_CombineAndScatterManager.hpp"

#include "Thyra_VectorStdOps.hpp"
#include "Teuchos_TestForException.hpp"

namespace Albany {

// ============================ Mask utilities ============================ //

Teuchos::RCP<Thyra_Vector>
binaryMask (const Thyra_Vector& x, const bool complement)
{
  auto y = Thyra::createMember(x.space());
  auto x_data = getLocalData(x);
  auto y_data = getNonconstLocalData(*y);
  for (LO i=0; i<static_cast<LO>(y_data.size()); ++i) {
    const bool on = (x_data[i]!=0.0);
    y_data[i] = (on != complement) ? 1.0 : 0.0;
  }
  return y;
}

// ======================== MaskedDofsEliminator ======================== //

MaskedDofsEliminator::
MaskedDofsEliminator (const Teuchos::RCP<const Thyra_Vector>& mask)
{
  setMask(mask);
}

void MaskedDofsEliminator::
setMask (const Teuchos::RCP<const Thyra_Vector>& mask)
{
  TEUCHOS_TEST_FOR_EXCEPTION (mask.is_null(), std::invalid_argument,
      "Error! MaskedDofsEliminator: the mask vector is null.\n");
  mask_ = mask;

  // Drop the cached column-space data: it is rebuilt at the next apply
  col_vs_   = Teuchos::null;
  col_mask_ = Teuchos::null;
  row_gids_.clear();
  col_gids_.clear();
}

void MaskedDofsEliminator::
apply (const Teuchos::RCP<Thyra_LinearOp>& A) const
{
  // (Re)compute the column-space mask and the row/col gids, if needed
  const auto col_vs = getColumnSpace(A);
  if (col_vs_.is_null() || !sameAs(col_vs_,col_vs)) {
    col_vs_   = col_vs;
    col_mask_ = Thyra::createMember(col_vs_);
    col_mask_->assign(0.0);
    auto cas = createCombineAndScatterManager(A->domain(),col_vs_);
    cas->scatter(*mask_,*col_mask_,CombineMode::INSERT);

    row_gids_ = getGlobalElements(A->range());
    col_gids_ = getGlobalElements(col_vs_);
  }

  auto row_mask = getLocalData(*mask_);
  auto col_mask = getLocalData(*col_mask_);

  Teuchos::Array<LO> indices;
  Teuchos::Array<ST> values;
  const LO num_rows = row_gids_.size();

  beginModify(A);
  for (LO row=0; row<num_rows; ++row) {
    getLocalRowValues(A,row,indices,values);
    const LO nnz = indices.size();
    if (row_mask[row]!=0.0) {
      // Masked row: identity
      bool found_diag = false;
      for (LO k=0; k<nnz; ++k) {
        if (col_gids_[indices[k]]==row_gids_[row]) {
          values[k] = 1.0;
          found_diag = true;
        } else {
          values[k] = 0.0;
        }
      }
      TEUCHOS_TEST_FOR_EXCEPTION (!found_diag, std::runtime_error,
          "Error! MaskedDofsEliminator: the operator graph has no diagonal entry for masked row "
          << row_gids_[row] << ".\n");
      setLocalRowValues(A,row,indices(),values());
    } else {
      // Unmasked row: drop the coupling with the masked dofs
      bool modified = false;
      for (LO k=0; k<nnz; ++k) {
        if (col_mask[indices[k]]!=0.0 && values[k]!=0.0) {
          values[k] = 0.0;
          modified = true;
        }
      }
      if (modified) {
        setLocalRowValues(A,row,indices(),values());
      }
    }
  }
  endModify(A);
}

// ================== AlgebraicConstraintModelEvaluator ================== //

AlgebraicConstraintModelEvaluator::
AlgebraicConstraintModelEvaluator (const Teuchos::RCP<Thyra_ModelEvaluator>& model,
                                   const Teuchos::RCP<const Thyra_Vector>&   solve_mask)
 : Thyra::ModelEvaluatorDelegatorBase<ST>(model)
{
  setMask(solve_mask);
}

void AlgebraicConstraintModelEvaluator::
setMask (const Teuchos::RCP<const Thyra_Vector>& solve_mask)
{
  const auto model = this->getUnderlyingModel();
  TEUCHOS_TEST_FOR_EXCEPTION (solve_mask.is_null(), std::invalid_argument,
      "Error! AlgebraicConstraintModelEvaluator: the mask vector is null.\n");
  TEUCHOS_TEST_FOR_EXCEPTION (!solve_mask->space()->isCompatible(*model->get_x_space()), std::invalid_argument,
      "Error! AlgebraicConstraintModelEvaluator: the mask vector is not in the solution vector space.\n");

  solve_mask_ = solve_mask;
  fixed_mask_ = binaryMask(*solve_mask,/* complement = */ true);

  if (Teuchos::nonnull(eliminator_)) {
    eliminator_->setMask(fixed_mask_);
  } else {
    eliminator_ = Teuchos::rcp(new MaskedDofsEliminator(fixed_mask_));
  }

  x_fixed_    = Thyra::createMember(model->get_x_space());
  x_before_   = Thyra::createMember(model->get_x_space());
  x_dot_zero_ = Thyra::createMember(model->get_x_space());
  x_fixed_->assign(0.0);
  x_dot_zero_->assign(0.0);
}

void AlgebraicConstraintModelEvaluator::setFixedValues (const Thyra_Vector& x)
{
  Thyra::copy(x,x_fixed_.ptr());
}

Thyra_InArgs AlgebraicConstraintModelEvaluator::createInArgs () const
{
  const auto model = this->getUnderlyingModel();
  Thyra::ModelEvaluatorBase::InArgsSetup<ST> result;
  result.setModelEvalDescription(this->description());
  result.set_Np_Ng(model->Np(),model->Ng());
  result.setSupports(Thyra::ModelEvaluatorBase::IN_ARG_x, true);
  return static_cast<Thyra_InArgs>(result);
}

Thyra_InArgs AlgebraicConstraintModelEvaluator::getNominalValues () const
{
  auto nominal = createInArgs();
  nominal.setArgs(this->getUnderlyingModel()->getNominalValues(), /* ignoreUnsupported = */ true);
  return nominal;
}

Thyra_InArgs AlgebraicConstraintModelEvaluator::getLowerBounds () const
{
  auto bounds = createInArgs();
  bounds.setArgs(this->getUnderlyingModel()->getLowerBounds(), /* ignoreUnsupported = */ true);
  return bounds;
}

Thyra_InArgs AlgebraicConstraintModelEvaluator::getUpperBounds () const
{
  auto bounds = createInArgs();
  bounds.setArgs(this->getUnderlyingModel()->getUpperBounds(), /* ignoreUnsupported = */ true);
  return bounds;
}

Thyra_OutArgs AlgebraicConstraintModelEvaluator::createOutArgsImpl () const
{
  using MEB = Thyra::ModelEvaluatorBase;
  const auto model_outArgs = this->getUnderlyingModel()->createOutArgs();

  MEB::OutArgsSetup<ST> result;
  result.setModelEvalDescription(this->description());
  result.set_Np_Ng(model_outArgs.Np(),model_outArgs.Ng());
  result.setSupports(MEB::OUT_ARG_f,    model_outArgs.supports(MEB::OUT_ARG_f));
  result.setSupports(MEB::OUT_ARG_W_op, model_outArgs.supports(MEB::OUT_ARG_W_op));
  if (model_outArgs.supports(MEB::OUT_ARG_W_op)) {
    result.set_W_properties(model_outArgs.get_W_properties());
  }
  return static_cast<Thyra_OutArgs>(result);
}

void AlgebraicConstraintModelEvaluator::
evalModelImpl (const Thyra_InArgs&  inArgs,
               const Thyra_OutArgs& outArgs) const
{
  using MEB = Thyra::ModelEvaluatorBase;

  const auto model = this->getUnderlyingModel();
  const auto x = inArgs.get_x();
  TEUCHOS_TEST_FOR_EXCEPTION (x.is_null(), std::logic_error,
      "Error! AlgebraicConstraintModelEvaluator::evalModel called with a null x.\n");

  // ---- Underlying model inputs ---- //
  auto modelInArgs = model->createInArgs();
  modelInArgs.set_x(x);
  for (int l=0; l<modelInArgs.Np(); ++l) {
    modelInArgs.set_p(l,inArgs.get_p(l));
  }
  if (modelInArgs.supports(MEB::IN_ARG_x_dot)) {
    // Non-null zero x_dot, so that Albany uses t, dt, alpha and beta from the InArgs
    modelInArgs.set_x_dot(x_dot_zero_);
  }
  if (modelInArgs.supports(MEB::IN_ARG_t))         { modelInArgs.set_t(t_); }
  if (modelInArgs.supports(MEB::IN_ARG_step_size)) { modelInArgs.set_step_size(dt_); }
  if (modelInArgs.supports(MEB::IN_ARG_alpha))     { modelInArgs.set_alpha(0.0); }
  if (modelInArgs.supports(MEB::IN_ARG_beta))      { modelInArgs.set_beta(1.0); }
  if (modelInArgs.supports(MEB::IN_ARG_W_x_dot_dot_coeff)) { modelInArgs.set_W_x_dot_dot_coeff(0.0); }

  // ---- Underlying model outputs ---- //
  const auto f = outArgs.get_f();
  const auto W = outArgs.get_W_op();

  auto modelOutArgs = model->createOutArgs();
  modelOutArgs.set_f(f);
  modelOutArgs.set_W_op(W);

  // Strong Dirichlet BCs may overwrite entries of x in place (only when f is requested).
  // Keep a copy, so we can detect it on the fixed dofs.
  if (Teuchos::nonnull(f)) {
    Thyra::copy(*x,x_before_.ptr());
  }

  model->evalModel(modelInArgs,modelOutArgs);

  // ---- Fixed rows of the residual: x_i - x_fixed_i ---- //
  if (Teuchos::nonnull(f)) {
    auto f_data  = getNonconstLocalData(*f);
    auto xf_data = getNonconstLocalData(*x_fixed_);
    auto x_data  = getLocalData(*x);
    auto xb_data = getLocalData(*x_before_);
    auto m_data  = getLocalData(*fixed_mask_);
    const LO n = f_data.size();
    for (LO i=0; i<n; ++i) {
      if (m_data[i]==0.0) continue;
      if (x_data[i]!=xb_data[i]) {
        // The model prescribed this dof (e.g., SDBC). Keep its residual, adopt its value.
        xf_data[i] = x_data[i];
        ++num_inplace_overrides_;
      } else {
        f_data[i] = x_data[i] - xf_data[i];
      }
    }
  }

  // ---- Fixed rows/cols of the Jacobian: identity ---- //
  if (Teuchos::nonnull(W)) {
    eliminator_->apply(W);
  }
}

} // namespace Albany
