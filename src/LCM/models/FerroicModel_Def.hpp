//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include <MiniTensor.h>
#include "Albany_Utils.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Teuchos_TestForException.hpp"

#include <MiniNonlinearSolver.h>

namespace FM {

/******************************************************************************/
template <typename EvalT>
FerroicModel<EvalT>::FerroicModel()
{
}
/******************************************************************************/

/******************************************************************************/
template <typename EvalT>
void
FerroicModel<EvalT>::PostParseInitialize()
/******************************************************************************/
{
  // create transitions
  //
  int nVariants = crystalVariants.size();
  transitions.resize(nVariants * nVariants);
  int transIndex = 0;
  for (int I = 0; I < nVariants; I++)
    for (int J = 0; J < nVariants; J++) {
      FM::Transition& t = transitions[transIndex];
      //      t.fromVariant = crystalVariants[I];
      //      t.toVariant = crystalVariants[J];
      CrystalVariant& fromVariant = crystalVariants[I];
      CrystalVariant& toVariant   = crystalVariants[J];
      t.transStrain = toVariant.spontStrain - fromVariant.spontStrain;
      t.transEDisp  = toVariant.spontEDisp - fromVariant.spontEDisp;
      transIndex++;
    }

  // create/initialize transition matrix
  //
  int nTransitions = transitions.size();
  aMatrix = Kokkos::DynRankView<RealType>("aMatrix", nVariants, nTransitions);
  for (int I = 0; I < nVariants; I++) {
    for (int J = 0; J < nVariants; J++) {
      aMatrix(I, nVariants * I + J) = -1.0;
      aMatrix(J, nVariants * I + J) = 1.0;
    }
    aMatrix(I, nVariants * I + I) = 0.0;
  }
}

/******************************************************************************/
template <typename EvalT>
void
FerroicModel<EvalT>::computeState(
    const minitensor::Tensor<ScalarT, FM::THREE_D>& x,
    const minitensor::Vector<ScalarT, FM::THREE_D>& E,
    const Teuchos::Array<RealType>&                 oldfractions,
    minitensor::Tensor<ScalarT, FM::THREE_D>&       X,
    minitensor::Vector<ScalarT, FM::THREE_D>&       D,
    Teuchos::Array<ScalarT>&                        newfractions)
/******************************************************************************/
{
  // create non-linear system
  //
  using NLS = FM::DomainSwitching<EvalT>;
  NLS domainSwitching(
      crystalVariants,
      transitions,
      tBarriers,
      oldfractions,
      aMatrix,
      x,
      E,
      /* dt= */ 1.0);

  // solution variable
  //
  minitensor::Vector<ScalarT, FM::MAX_TRNS> xi;

  // solve for xi
  //
  switch (m_integrationType) {
    default: break;

    case FM::IntegrationType::EXPLICIT: {
      switch (m_explicitMethod) {
        default: break;

        case FM::ExplicitMethod::SCALED_DESCENT: {
          FM::ScaledDescent(domainSwitching, xi);
        }
        case FM::ExplicitMethod::DESCENT_NORM: {
          FM::DescentNorm(domainSwitching, xi);
        }
      }
    }

    case FM::IntegrationType::IMPLICIT: {
      // create minimizer
      using ValueT = typename Sacado::ValueType<ScalarT>::type;
      using MIN    = minitensor::Minimizer<ValueT, MAX_TRNS>;
      MIN minimizer;

      // create stepper
      using STEP = minitensor::StepBase<NLS, ValueT, MAX_TRNS>;
      std::unique_ptr<STEP> pstep =
          minitensor::stepFactory<NLS, ValueT, MAX_TRNS>(m_step_type);
      STEP& step = *pstep;

      // create solution vector with initial guess
      int ntrans = domainSwitching.getNumStates();
      xi.set_dimension(ntrans);
      for (int itrans = 0; itrans < ntrans; itrans++)
        xi(itrans) = Sacado::ScalarValue<ScalarT>::eval(0.0);

      // solve
      LCM::MiniSolver<MIN, STEP, NLS, EvalT, MAX_TRNS> mini_solver(
          minimizer, step, domainSwitching, xi);
    }
  }

  // update based on new xi values
  //
}

}  // namespace FM
