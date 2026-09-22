//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_ALGEBRAIC_CONSTRAINT_MODEL_EVALUATOR_HPP
#define ALBANY_ALGEBRAIC_CONSTRAINT_MODEL_EVALUATOR_HPP

#include "Albany_ThyraTypes.hpp"

#include "Thyra_ModelEvaluatorDelegatorBase.hpp"
#include "Teuchos_Array.hpp"

namespace Albany {

//! Binary copy of a mask: y_i = (x_i != 0) ? 1 : 0, or its complement
Teuchos::RCP<Thyra_Vector>
binaryMask (const Thyra_Vector& x, const bool complement = false);

/*
 * Replaces the rows and the columns of the masked dofs of a sparse linear
 * operator with the identity (same elimination as PHAL::SDirichlet).
 * The mask lives in the domain (= range) space of the operator; entries != 0
 * flag the dofs to eliminate. The column-space data is cached, and recomputed
 * only if the column space of the operator changes (e.g., after adaptation).
 */
class MaskedDofsEliminator
{
public:
  explicit MaskedDofsEliminator (const Teuchos::RCP<const Thyra_Vector>& mask);

  //! Change the dofs to eliminate (e.g., after mesh adaptation)
  void setMask (const Teuchos::RCP<const Thyra_Vector>& mask);

  void apply (const Teuchos::RCP<Thyra_LinearOp>& A) const;

private:
  Teuchos::RCP<const Thyra_Vector> mask_;

  mutable Teuchos::RCP<const Thyra_VectorSpace> col_vs_;
  mutable Teuchos::RCP<Thyra_Vector>            col_mask_;
  mutable Teuchos::Array<GO>                    row_gids_;
  mutable Teuchos::Array<GO>                    col_gids_;
};

/*
 * The algebraic constraint of a semi-explicit index-1 DAE, exposed as a *steady*
 * nonlinear problem in the algebraic dofs alone.
 *
 * Given a (possibly transient) model f(x_dot,x,t) = 0, and a mask flagging the
 * algebraic dofs x_a (the dofs this model solves for; the other dofs, x_r, are held
 * fixed at prescribed values), this model evaluator exposes
 *
 *   f_c(x)_i = f(x_dot=0, x, t)_i     if i is an algebraic dof
 *   f_c(x)_i = x_i - x_fixed_i        otherwise
 *
 *   W_c = df/dx, with the rows and the columns of the fixed dofs replaced by
 *         the identity.
 *
 * Since the fixed rows are identity rows with zero residual (once the initial guess
 * matches x_fixed), a Newton solve on f_c only moves the algebraic dofs, i.e., it
 * solves the algebraic constraint  f(0, [x_a, x_r], t)_a = 0  for x_a.
 * In LandIce::StokesFOThickness this is the diagnostic solve: the FO velocity for a
 * given ice thickness. It is used by Albany::ExplicitODEModelEvaluator to eliminate
 * the constraint from the time integration.
 *
 * Notes:
 *  - The underlying model is always evaluated with a *non-null* zero x_dot, and
 *    with alpha=0, beta=1 (Albany::ModelEvaluator takes alpha/beta from the InArgs
 *    whenever x_dot is non-null, and Thyra defaults both to 0), plus the time and
 *    step size set via setTime/setStepSize. With a null x_dot, Albany would use
 *    t=0 and dt=0, and some gather evaluators would skip x_dot altogether.
 *  - Albany strong Dirichlet BCs (SDBCs) write the BC value into the input x in
 *    place and zero the corresponding residual. If this happens on a fixed dof,
 *    the BC wins: the residual is left untouched and the fixed value is updated
 *    to the BC value, so the Newton solver does not fight the BC.
 *  - The model is exposed as steady: InArgs support only x (and the parameters).
 *    Only f and W_op are supported as outputs (no W_prec: let Stratimikos build
 *    the preconditioner from the modified W_op).
 */
class AlgebraicConstraintModelEvaluator : public Thyra::ModelEvaluatorDelegatorBase<ST>
{
public:
  // solve_mask: vector in the x space; entries != 0 flag the algebraic dofs, that is,
  //             the dofs this model solves for. All the other dofs are held fixed.
  AlgebraicConstraintModelEvaluator (const Teuchos::RCP<Thyra_ModelEvaluator>& model,
                                     const Teuchos::RCP<const Thyra_Vector>&   solve_mask);

  //! Change the algebraic dofs, and rebuild everything tied to the solution space.
  //! Needed after mesh adaptation (see Albany::ExplicitODEModelEvaluator::reinitialize).
  void setMask (const Teuchos::RCP<const Thyra_Vector>& solve_mask);

  //! Set the values of the fixed dofs (only the non-algebraic entries of x are used)
  void setFixedValues (const Thyra_Vector& x);

  //! Time and step size passed to the underlying model (if it supports them)
  void setTime     (const ST t)  { t_  = t;  }
  void setStepSize (const ST dt) { dt_ = dt; }

  Teuchos::RCP<const Thyra_Vector> getMask () const { return solve_mask_; }
  Teuchos::RCP<const Thyra_Vector> getFixedValues () const { return x_fixed_; }

  //! Number of times a fixed dof was overwritten in place by the model (e.g., SDBCs)
  int numInPlaceOverrides () const { return num_inplace_overrides_; }

  /** \name Overridden from Thyra::ModelEvaluator */
  //@{
  Thyra_InArgs createInArgs () const override;
  Thyra_InArgs getNominalValues () const override;
  Thyra_InArgs getLowerBounds () const override;
  Thyra_InArgs getUpperBounds () const override;

  Teuchos::RCP<Thyra_Preconditioner> create_W_prec () const override { return Teuchos::null; }

  // The final point of an inner solve is not a meaningful final point for the
  // application (e.g., it could overwrite nominal values), so we do not report it.
  void reportFinalPoint (const Thyra_InArgs& /* finalPoint */, const bool /* wasSolved */) override {}
  //@}

private:

  /** \name Overridden from Thyra::ModelEvaluatorDefaultBase */
  //@{
  Thyra_OutArgs createOutArgsImpl () const override;

  void evalModelImpl (const Thyra_InArgs&  inArgs,
                      const Thyra_OutArgs& outArgs) const override;
  //@}

  Teuchos::RCP<const Thyra_Vector>  solve_mask_;   // algebraic dofs (solved for)
  Teuchos::RCP<Thyra_Vector>        fixed_mask_;   // 1 - solve_mask
  Teuchos::RCP<Thyra_Vector>        x_fixed_;
  Teuchos::RCP<Thyra_Vector>        x_dot_zero_;
  Teuchos::RCP<Thyra_Vector>        x_before_;

  Teuchos::RCP<MaskedDofsEliminator> eliminator_;

  ST t_  = 0.0;
  ST dt_ = 0.0;

  mutable int num_inplace_overrides_ = 0;
};

} // namespace Albany

#endif // ALBANY_ALGEBRAIC_CONSTRAINT_MODEL_EVALUATOR_HPP
