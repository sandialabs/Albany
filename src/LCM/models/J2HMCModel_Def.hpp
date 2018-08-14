//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <MiniTensor.h>
#include "Albany_Utils.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Teuchos_TestForException.hpp"

#include "LocalNonlinearSolver.hpp"

namespace LCM {

/******************************************************************************/
template <typename EvalT, typename Traits>
J2HMCModel<EvalT, Traits>::J2HMCModel(
    Teuchos::ParameterList*              p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : LCM::ConstitutiveModel<EvalT, Traits>(p, dl),
      numMicroScales(p->get<int>("Additional Scales")),
      macroKinematicModulus(p->get<RealType>("Kinematic Hardening Modulus")),
      macroIsotropicModulus(p->get<RealType>("Isotropic Hardening Modulus")),
      macroYieldStress0(p->get<RealType>("Initial Yield Stress")),
      C11(p->get<RealType>("C11")),
      C33(p->get<RealType>("C33")),
      C12(p->get<RealType>("C12")),
      C23(p->get<RealType>("C23")),
      C44(p->get<RealType>("C44")),
      C66(p->get<RealType>("C66"))
/******************************************************************************/
{
  lengthScale.resize(numMicroScales);
  betaParameter.resize(numMicroScales);
  microYieldStress0.resize(numMicroScales);
  microKinematicModulus.resize(numMicroScales);
  microIsotropicModulus.resize(numMicroScales);
  doubleYieldStress0.resize(numMicroScales);
  doubleKinematicModulus.resize(numMicroScales);
  doubleIsotropicModulus.resize(numMicroScales);
  for (int i = 0; i < numMicroScales; i++) {
    std::string mySublist                 = Albany::strint("Microscale", i + 1);
    const Teuchos::ParameterList& msModel = p->sublist(mySublist);
    lengthScale[i]       = msModel.get<RealType>("Length Scale");
    betaParameter[i]     = msModel.get<RealType>("Beta Constant");
    microYieldStress0[i] = msModel.get<RealType>("Initial Micro Yield Stress");
    microKinematicModulus[i] =
        msModel.get<RealType>("Micro Kinematic Hardening Modulus");
    microIsotropicModulus[i] =
        msModel.get<RealType>("Micro Isotropic Hardening Modulus");
    doubleYieldStress0[i] =
        msModel.get<RealType>("Initial Double Yield Stress");
    doubleKinematicModulus[i] =
        msModel.get<RealType>("Double Kinematic Hardening Modulus");
    doubleIsotropicModulus[i] =
        msModel.get<RealType>("Double Isotropic Hardening Modulus");
  }

  initializeElasticConstants();

  // DEFINE THE EVALUATED FIELDS
  updated_macroStressName = "Stress";
  current_macroStressName = updated_macroStressName + "_old";
  this->eval_field_map_.insert(
      std::make_pair(updated_macroStressName, dl->qp_tensor));

  for (int i = 0; i < numMicroScales; i++) {
    updated_microStressName.push_back(Albany::strint("Micro Stress", i));
    current_microStressName.push_back(updated_microStressName[i] + "_old");
    this->eval_field_map_.insert(
        std::make_pair(updated_microStressName[i], dl->qp_tensor));

    updated_doubleStressName.push_back(Albany::strint("Double Stress", i));
    current_doubleStressName.push_back(updated_doubleStressName[i] + "_old");
    this->eval_field_map_.insert(
        std::make_pair(updated_doubleStressName[i], dl->qp_tensor3));
  }

  // DEFINE THE DEPENDENT FIELDS
  delta_macroStrainName = "Strain Increment";
  this->dep_field_map_.insert(
      std::make_pair(delta_macroStrainName, dl->qp_tensor));

  for (int i = 0; i < numMicroScales; i++) {
    // increment of strain difference
    delta_strainDifferenceName.push_back(
        Albany::strint("Strain Difference", i) + " Increment");
    this->dep_field_map_.insert(
        std::make_pair(delta_strainDifferenceName[i], dl->qp_tensor));

    // increment of microstrain gradient
    delta_microStrainGradientName.push_back(
        Albany::strint("DeltaMicrostrain", i) + " Gradient");
    this->dep_field_map_.insert(
        std::make_pair(delta_microStrainGradientName[i], dl->qp_tensor3));
  }

  // DEFINE STATE VARIABLES

  // macro alpha
  updated_macroAlphaName = "MacroAlpha";
  current_macroAlphaName = updated_macroAlphaName + "_old";
  this->eval_field_map_.insert(
      std::make_pair(updated_macroAlphaName, dl->qp_scalar));

  // macro back stress
  updated_macroBackStressName = "MacroBackStress";
  current_macroBackStressName = updated_macroBackStressName + "_old";
  this->eval_field_map_.insert(
      std::make_pair(updated_macroBackStressName, dl->qp_tensor));

  for (int i = 0; i < numMicroScales; i++) {
    // micro alpha
    updated_microAlphaName.push_back(Albany::strint("MicroAlpha", i));
    current_microAlphaName.push_back(updated_microAlphaName[i] + "_old");
    this->eval_field_map_.insert(
        std::make_pair(updated_microAlphaName[i], dl->qp_scalar));
    // micro back stress
    updated_microBackStressName.push_back(Albany::strint("MicroBackStress", i));
    current_microBackStressName.push_back(
        updated_microBackStressName[i] + "_old");
    this->eval_field_map_.insert(
        std::make_pair(updated_microBackStressName[i], dl->qp_tensor));

    // double alpha
    updated_doubleAlphaName.push_back(Albany::strint("DoubleAlpha", i));
    current_doubleAlphaName.push_back(updated_doubleAlphaName[i] + "_old");
    this->eval_field_map_.insert(
        std::make_pair(updated_doubleAlphaName[i], dl->qp_scalar));
    // double back stress
    updated_doubleBackStressName.push_back(
        Albany::strint("DoubleBackStress", i));
    current_doubleBackStressName.push_back(
        updated_doubleBackStressName[i] + "_old");
    this->eval_field_map_.insert(
        std::make_pair(updated_doubleBackStressName[i], dl->qp_tensor3));
  }

  // macro stress
  this->num_state_variables_++;
  this->state_var_names_.push_back(updated_macroStressName);
  this->state_var_layouts_.push_back(dl->qp_tensor);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(true);

  // macro alpha
  this->num_state_variables_++;
  this->state_var_names_.push_back(updated_macroAlphaName);
  this->state_var_layouts_.push_back(dl->qp_scalar);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(true);

  // macro back stress
  this->num_state_variables_++;
  this->state_var_names_.push_back(updated_macroBackStressName);
  this->state_var_layouts_.push_back(dl->qp_tensor);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(true);

  for (int i = 0; i < numMicroScales; i++) {
    // micro stresses
    this->num_state_variables_++;
    this->state_var_names_.push_back(updated_microStressName[i]);
    this->state_var_layouts_.push_back(dl->qp_tensor);
    this->state_var_init_types_.push_back("scalar");
    this->state_var_init_values_.push_back(0.0);
    this->state_var_old_state_flags_.push_back(true);
    this->state_var_output_flags_.push_back(true);

    // micro alpha
    this->num_state_variables_++;
    this->state_var_names_.push_back(updated_microAlphaName[i]);
    this->state_var_layouts_.push_back(dl->qp_scalar);
    this->state_var_init_types_.push_back("scalar");
    this->state_var_init_values_.push_back(0.0);
    this->state_var_old_state_flags_.push_back(true);
    this->state_var_output_flags_.push_back(true);

    // micro back stress
    this->num_state_variables_++;
    this->state_var_names_.push_back(updated_microBackStressName[i]);
    this->state_var_layouts_.push_back(dl->qp_tensor);
    this->state_var_init_types_.push_back("scalar");
    this->state_var_init_values_.push_back(0.0);
    this->state_var_old_state_flags_.push_back(true);
    this->state_var_output_flags_.push_back(true);

    // double stresses
    this->num_state_variables_++;
    this->state_var_names_.push_back(updated_doubleStressName[i]);
    this->state_var_layouts_.push_back(dl->qp_tensor3);
    this->state_var_init_types_.push_back("scalar");
    this->state_var_init_values_.push_back(0.0);
    this->state_var_old_state_flags_.push_back(true);
    this->state_var_output_flags_.push_back(true);

    // double alpha
    this->num_state_variables_++;
    this->state_var_names_.push_back(updated_doubleAlphaName[i]);
    this->state_var_layouts_.push_back(dl->qp_scalar);
    this->state_var_init_types_.push_back("scalar");
    this->state_var_init_values_.push_back(0.0);
    this->state_var_old_state_flags_.push_back(true);
    this->state_var_output_flags_.push_back(true);

    // double back stress
    this->num_state_variables_++;
    this->state_var_names_.push_back(updated_doubleBackStressName[i]);
    this->state_var_layouts_.push_back(dl->qp_tensor3);
    this->state_var_init_types_.push_back("scalar");
    this->state_var_init_values_.push_back(0.0);
    this->state_var_old_state_flags_.push_back(true);
    this->state_var_output_flags_.push_back(true);
  }
}

/******************************************************************************/
template <typename EvalT, typename Traits>
void
J2HMCModel<EvalT, Traits>::computeState(
    typename Traits::EvalData workset,
    DepFieldMap               dep_fields,
    FieldMap                  eval_fields)
/******************************************************************************/
{
  // extract increments
  std::vector<PHX::MDField<const ScalarT>> delta_strainDifference(
      numMicroScales);
  std::vector<PHX::MDField<const ScalarT>> delta_microStrainGradient(
      numMicroScales);
  auto delta_macroStrain = *dep_fields[delta_macroStrainName];
  for (int i = 0; i < numMicroScales; i++) {
    delta_strainDifference[i] = *dep_fields[delta_strainDifferenceName[i]];
    delta_microStrainGradient[i] =
        *dep_fields[delta_microStrainGradientName[i]];
  }

  // extract states to be updated (i.e., state at N+1):
  auto updated_macroStress = *eval_fields[updated_macroStressName];
  std::vector<PHX::MDField<ScalarT>> updated_microStress(numMicroScales);
  std::vector<PHX::MDField<ScalarT>> updated_doubleStress(numMicroScales);
  for (int i = 0; i < numMicroScales; i++) {
    updated_microStress[i]  = *eval_fields[updated_microStressName[i]];
    updated_doubleStress[i] = *eval_fields[updated_doubleStressName[i]];
  }

  computeTrialState(
      workset,
      delta_macroStrain,
      delta_strainDifference,
      delta_microStrainGradient,
      updated_macroStress,
      updated_microStress,
      updated_doubleStress);

  // extract states to be updated (i.e., state at N+1):
  PHX::MDField<ScalarT> updated_macroBackStress =
      *eval_fields[updated_macroBackStressName];
  PHX::MDField<ScalarT> updated_macroAlpha =
      *eval_fields[updated_macroAlphaName];
  std::vector<PHX::MDField<ScalarT>> updated_microBackStress(numMicroScales);
  std::vector<PHX::MDField<ScalarT>> updated_doubleBackStress(numMicroScales);
  std::vector<PHX::MDField<ScalarT>> updated_microAlpha(numMicroScales);
  std::vector<PHX::MDField<ScalarT>> updated_doubleAlpha(numMicroScales);
  for (int i = 0; i < numMicroScales; i++) {
    updated_microBackStress[i]  = *eval_fields[updated_microBackStressName[i]];
    updated_doubleBackStress[i] = *eval_fields[updated_doubleBackStressName[i]];
    updated_microAlpha[i]       = *eval_fields[updated_microAlphaName[i]];
    updated_doubleAlpha[i]      = *eval_fields[updated_doubleAlphaName[i]];
  }

  radialReturn(
      workset,
      updated_macroStress,
      updated_macroBackStress,
      updated_macroAlpha,
      updated_microStress,
      updated_microBackStress,
      updated_microAlpha,
      updated_doubleStress,
      updated_doubleBackStress,
      updated_doubleAlpha);
}

/******************************************************************************/
template <typename EvalT, typename Traits>
void
J2HMCModel<EvalT, Traits>::computeStateParallel(
    typename Traits::EvalData workset,
    DepFieldMap               dep_fields,
    FieldMap                  eval_fields)
/******************************************************************************/
{
  TEUCHOS_TEST_FOR_EXCEPTION(
      true,
      std::invalid_argument,
      ">>> ERROR (J2HMCModel): computeStateParallel not implemented");
}

/******************************************************************************/
template <typename EvalT, typename Traits>
void
J2HMCModel<EvalT, Traits>::radialReturn(
    typename Traits::EvalData workset,
    // macro
    PHX::MDField<ScalarT>& trial_macroStress,
    PHX::MDField<ScalarT>& new_macroBackStress,
    PHX::MDField<ScalarT>& new_macroAlpha,
    // micro
    std::vector<PHX::MDField<ScalarT>>& trial_microStress,
    std::vector<PHX::MDField<ScalarT>>& new_microBackStress,
    std::vector<PHX::MDField<ScalarT>>& new_microAlpha,
    // double
    std::vector<PHX::MDField<ScalarT>>& trial_doubleStress,
    std::vector<PHX::MDField<ScalarT>>& new_doubleBackStress,
    std::vector<PHX::MDField<ScalarT>>& new_doubleAlpha)
/******************************************************************************/
{
  // macro
  ScalarT                     macroAlpha;
  minitensor::Tensor<ScalarT> elTrialMacroStress(num_dims_);
  minitensor::Tensor<ScalarT> macroBackStress(num_dims_);

  std::vector<ScalarT>                     microAlpha(numMicroScales);
  std::vector<minitensor::Tensor<ScalarT>> elTrialMicroStress(numMicroScales);
  std::vector<minitensor::Tensor<ScalarT>> microBackStress(numMicroScales);

  std::vector<ScalarT>                      doubleAlpha(numMicroScales);
  std::vector<minitensor::Tensor3<ScalarT>> elTrialDoubleStress(numMicroScales);
  std::vector<minitensor::Tensor3<ScalarT>> doubleBackStress(numMicroScales);

  for (int i = 0; i < numMicroScales; i++) {
    elTrialMicroStress[i].set_dimension(num_dims_);
    microBackStress[i].set_dimension(num_dims_);
    elTrialDoubleStress[i].set_dimension(num_dims_);
    doubleBackStress[i].set_dimension(num_dims_);
  }

  // current state (i.e., state at N) is stored in the state manager:
  Albany::MDArray current_macroAlpha =
      (*workset.stateArrayPtr)[current_macroAlphaName];
  Albany::MDArray current_macroBackStress =
      (*workset.stateArrayPtr)[current_macroBackStressName];
  std::vector<Albany::MDArray> current_microAlpha(numMicroScales);
  std::vector<Albany::MDArray> current_microBackStress(numMicroScales);
  std::vector<Albany::MDArray> current_doubleAlpha(numMicroScales);
  std::vector<Albany::MDArray> current_doubleBackStress(numMicroScales);

  for (int i = 0; i < numMicroScales; i++) {
    current_microAlpha[i] = (*workset.stateArrayPtr)[current_microAlphaName[i]];
    current_microBackStress[i] =
        (*workset.stateArrayPtr)[current_microBackStressName[i]];
    current_doubleAlpha[i] =
        (*workset.stateArrayPtr)[current_doubleAlphaName[i]];
    current_doubleBackStress[i] =
        (*workset.stateArrayPtr)[current_doubleBackStressName[i]];
  }

  ScalarT Hval, Kval;
  ScalarT sq32(std::sqrt(3.0 / 2.0));

  std::size_t zero(0);

  int                  nvars = 1 + 2 * numMicroScales;
  std::vector<ScalarT> Fvals(nvars);

  int numCells = workset.numCells;
  for (std::size_t cell = 0; cell < numCells; ++cell) {
    for (std::size_t qp = 0; qp < num_pts_; ++qp) {
      macroAlpha = current_macroAlpha(cell, qp);
      elTrialMacroStress.fill(trial_macroStress, (int)cell, (int)qp, 0, 0);
      for (std::size_t i = 0; i < num_dims_; i++)
        for (std::size_t j = 0; j < num_dims_; j++)
          macroBackStress(i, j) = current_macroBackStress(cell, qp, i, j);

      for (int m = 0; m < numMicroScales; m++) {
        microAlpha[m]  = current_microAlpha[m](cell, qp);
        doubleAlpha[m] = current_doubleAlpha[m](cell, qp);
        for (std::size_t i = 0; i < num_dims_; i++)
          for (std::size_t j = 0; j < num_dims_; j++) {
            elTrialMicroStress[m](i, j) = trial_microStress[m](cell, qp, i, j);
            microBackStress[m](i, j) =
                current_microBackStress[m](cell, qp, i, j);
            for (std::size_t k = 0; k < num_dims_; k++) {
              elTrialDoubleStress[m](i, j, k) =
                  trial_doubleStress[m](cell, qp, i, j, k);
              doubleBackStress[m](i, j, k) =
                  current_doubleBackStress[m](cell, qp, i, j, k);
            }
          }
      }

      yieldFunction(
          Fvals,
          elTrialMacroStress,
          macroAlpha,
          macroBackStress,
          elTrialMicroStress,
          microAlpha,
          microBackStress,
          elTrialDoubleStress,
          doubleAlpha,
          doubleBackStress);

      bool yielding = false;
      for (int i = 0; i < Fvals.size(); i++)
        if (Fvals[i] > 1e-12) {
          yielding = true;
          break;
        }

      if (yielding) {
        std::vector<int> yieldMask(nvars), yieldMap(nvars);
        int              nyield = 0;
        for (int i = 0; i < Fvals.size(); i++) {
          if (Fvals[i] > 0.0) {
            yieldMap[i]  = nyield;
            yieldMask[i] = 1;
            nyield++;
          } else {
            yieldMask[i] = 0;
          }
        }
        std::vector<ScalarT> X(nyield);
        std::vector<ScalarT> R(nyield);
        std::vector<ScalarT> dRdX(nyield * nyield);

        for (int i = 0; i < X.size(); i++) X[i] = 0.0;

        LocalNonlinearSolver<EvalT, Traits> solver;

        int     iter     = 0;
        ScalarT initNorm = 0.0;

        while (true) {
          computeResidualandJacobian(
              yieldMask,
              yieldMap,
              X,
              R,
              dRdX,
              elTrialMacroStress,
              macroBackStress,
              macroAlpha,
              elTrialMicroStress,
              microBackStress,
              microAlpha,
              elTrialDoubleStress,
              doubleBackStress,
              doubleAlpha);

          if (converged(R, iter, initNorm)) break;

          solver.solve(dRdX, X, R);

          iter++;
        }

        solver.computeFadInfo(dRdX, X, R);

        // update macro scale
        ScalarT dGamma(0.0);
        if (yieldMask[0]) dGamma = X[0];
        minitensor::Tensor<ScalarT> returnDir =
            devNorm(elTrialMacroStress - macroBackStress);
        macroAlpha += dGamma;
        // JR: todo: create a plasticity submodel object to encapsulate this.
        // Hval = macroKinematicHardening->H(macroAlpha);
        Hval = macroKinematicModulus * macroAlpha;
        macroBackStress += sq32 * dGamma * Hval * returnDir;
        elTrialMacroStress -=
            dGamma * minitensor::dotdot(macroCelastic, returnDir);
        new_macroAlpha(cell, qp) = macroAlpha;
        for (int i = 0; i < num_dims_; i++)
          for (int j = 0; j < num_dims_; j++) {
            new_macroBackStress(cell, qp, i, j) = macroBackStress(i, j);
            trial_macroStress(cell, qp, i, j)   = elTrialMacroStress(i, j);
          }

        // Micro scales
        for (int ims = 0; ims < numMicroScales; ims++) {
          // micro
          int xIndex = 2 * ims + 1;
          if (yieldMask[xIndex]) {
            ScalarT                     dGamma = X[yieldMap[xIndex]];
            minitensor::Tensor<ScalarT> returnDir =
                devNorm(elTrialMicroStress[ims] - microBackStress[ims]);
            microAlpha[ims] += dGamma;
            // JR: todo: create a plasticity submodel object to encapsulate
            // this. Hval = microKinematicHardening->H(microAlpha[ims]);
            Hval = microKinematicModulus[ims] * microAlpha[ims];
            microBackStress[ims] += sq32 * dGamma * Hval * returnDir;
            elTrialMicroStress[ims] -=
                dGamma * minitensor::dotdot(microCelastic[ims], returnDir);
          }
          new_microAlpha[ims](cell, qp) = microAlpha[ims];
          for (int i = 0; i < num_dims_; i++)
            for (int j = 0; j < num_dims_; j++) {
              new_microBackStress[ims](cell, qp, i, j) =
                  microBackStress[ims](i, j);
              trial_microStress[ims](cell, qp, i, j) =
                  elTrialMicroStress[ims](i, j);
            }

          // double
          xIndex = 2 * ims + 2;
          if (yieldMask[xIndex]) {
            ScalarT                      dGamma = X[yieldMap[xIndex]];
            minitensor::Tensor3<ScalarT> returnDir =
                devNorm(elTrialDoubleStress[ims] - doubleBackStress[ims]);
            doubleAlpha[ims] += dGamma;
            // JR: todo: create a plasticity submodel object to encapsulate
            // this. Hval = doubleKinematicHardening->H(doubleAlpha[ims]);
            Hval = doubleKinematicModulus[ims] * doubleAlpha[ims];
            doubleBackStress[ims] += sq32 * dGamma * Hval * returnDir;
            elTrialDoubleStress[ims] -=
                dGamma * dotdotdot(doubleCelastic[ims], returnDir);
          }
          new_doubleAlpha[ims](cell, qp) = doubleAlpha[ims];
          for (int i = 0; i < num_dims_; i++)
            for (int j = 0; j < num_dims_; j++)
              for (int k = 0; k < num_dims_; k++) {
                new_doubleBackStress[ims](cell, qp, i, j, k) =
                    doubleBackStress[ims](i, j, k);
                trial_doubleStress[ims](cell, qp, i, j, k) =
                    elTrialDoubleStress[ims](i, j, k);
              }
        }
      } else {
        // macro scale
        new_macroAlpha(cell, qp) = current_macroAlpha(cell, qp);
        for (std::size_t i = 0; i < num_dims_; i++)
          for (std::size_t j = 0; j < num_dims_; j++)
            new_macroBackStress(cell, qp, i, j) =
                current_macroBackStress(cell, qp, i, j);
        // Micro scales
        for (int ims = 0; ims < numMicroScales; ims++) {
          // micro
          {
            new_microAlpha[ims](cell, qp) = current_microAlpha[ims](cell, qp);
            for (std::size_t i = 0; i < num_dims_; i++)
              for (std::size_t j = 0; j < num_dims_; j++)
                new_microBackStress[ims](cell, qp, i, j) =
                    current_microBackStress[ims](cell, qp, i, j);
          }
          // double
          {
            new_doubleAlpha[ims](cell, qp) = current_doubleAlpha[ims](cell, qp);
            for (std::size_t i = 0; i < num_dims_; i++)
              for (std::size_t j = 0; j < num_dims_; j++)
                for (std::size_t k = 0; k < num_dims_; k++)
                  new_doubleBackStress[ims](cell, qp, i, j, k) =
                      current_doubleBackStress[ims](cell, qp, i, j, k);
          }
        }
      }
    }
  }
}

/******************************************************************************/
template <typename EvalT, typename Traits>
bool
J2HMCModel<EvalT, Traits>::converged(
    std::vector<ScalarT>& R,
    int                   iteration,
    ScalarT&              initNorm)
/******************************************************************************/
{
  bool converged = true;

  int     nvals = R.size();
  ScalarT norm  = 0.0;
  for (int ival = 0; ival < nvals; ival++) { norm += R[ival] * R[ival]; }
  if (norm > 0.0) norm = sqrt(norm);

  if (iteration == 0) initNorm = norm;

  if (initNorm == 0.0)
    converged = true;
  else
    converged = (norm / initNorm < 1.0e-11 || norm < 1.0e-11 || iteration > 20);

  return converged;
}

/******************************************************************************/
template <typename EvalT, typename Traits>
void
J2HMCModel<EvalT, Traits>::computeResidualandJacobian(
    std::vector<int>&                          yieldMask,
    std::vector<int>&                          yieldMap,
    std::vector<ScalarT>&                      X,
    std::vector<ScalarT>&                      R,
    std::vector<ScalarT>&                      dRdX,
    minitensor::Tensor<ScalarT>&               elTrialMacroStress,
    minitensor::Tensor<ScalarT>&               macroBackStress,
    ScalarT&                                   alpha,
    std::vector<minitensor::Tensor<ScalarT>>&  elTrialMicroStress,
    std::vector<minitensor::Tensor<ScalarT>>&  microBackStress,
    std::vector<ScalarT>&                      microAlpha,
    std::vector<minitensor::Tensor3<ScalarT>>& elTrialDoubleStress,
    std::vector<minitensor::Tensor3<ScalarT>>& doubleBackStress,
    std::vector<ScalarT>&                      doubleAlpha)
/******************************************************************************/
{
  int                   nvars = X.size();
  std::vector<DFadType> Rfad(nvars);
  std::vector<DFadType> Xfad(nvars);
  std::vector<ScalarT>  Xval(nvars);
  for (std::size_t i = 0; i < nvars; ++i) {
    Xval[i] = Sacado::ScalarValue<ScalarT>::eval(X[i]);
    Xfad[i] = DFadType(nvars, i, Xval[i]);
  }

  ScalarT sq32(std::sqrt(3.0 / 2.0));

  // Macro scale
  if (yieldMask[0] > 0) {
    minitensor::Tensor<ScalarT> returnDir =
        devNorm(elTrialMacroStress - macroBackStress);
    minitensor::Tensor<ScalarT> stressDir;
    stressDir = minitensor::dotdot(macroCelastic, returnDir);
    DFadType                     dGamma = Xfad[0];
    minitensor::Tensor<DFadType> stressNew(num_dims_);
    for (size_t i = 0; i < num_dims_; i++)
      for (size_t j = 0; j < num_dims_; j++)
        stressNew(i, j) = elTrialMacroStress(i, j) - dGamma * stressDir(i, j);
    DFadType alphaNew = alpha + dGamma;
    //  JR: todo: create a plasticity submodel object to encapsulate this.
    //  DFadType Hval = macroKinematicHardening->H(alphaNew);
    //  DFadType Kval = macroIsotropicHardening->K(alphaNew);
    DFadType Hval = macroKinematicModulus * alphaNew;
    DFadType Kval = macroYieldStress0 + macroIsotropicModulus * alphaNew;
    minitensor::Tensor<DFadType> backStressNew(num_dims_);
    for (size_t i = 0; i < num_dims_; i++)
      for (size_t j = 0; j < num_dims_; j++)
        backStressNew(i, j) =
            macroBackStress(i, j) + sq32 * dGamma * Hval * returnDir(i, j);
    Rfad[0] = sq32 * devMag(stressNew - backStressNew) - Kval;
  }

  // Micro scales
  for (int ims = 0; ims < numMicroScales; ims++) {
    int xIndex;
    // micro
    xIndex = 2 * ims + 1;
    if (yieldMask[xIndex] > 0) {
      xIndex = yieldMap[xIndex];
      minitensor::Tensor<ScalarT> returnDir =
          devNorm(elTrialMicroStress[ims] - microBackStress[ims]);
      minitensor::Tensor<ScalarT> stressDir =
          minitensor::dotdot(microCelastic[ims], returnDir);
      DFadType                     dGamma = Xfad[xIndex];
      minitensor::Tensor<DFadType> stressNew(num_dims_);
      for (size_t i = 0; i < num_dims_; i++)
        for (size_t j = 0; j < num_dims_; j++)
          stressNew(i, j) =
              elTrialMicroStress[ims](i, j) - dGamma * stressDir(i, j);
      DFadType alphaNew = microAlpha[ims] + dGamma;
      //  JR: todo: create a plasticity submodel object to encapsulate this.
      //  DFadType Hval = microKinematicHardening[ims]->(alphaNew);
      //  DFadType Kval = microIsotropicHardening[ims]->(alphaNew);
      DFadType Hval = microKinematicModulus[ims] * alphaNew;
      DFadType Kval =
          microYieldStress0[ims] + microIsotropicModulus[ims] * alphaNew;
      minitensor::Tensor<DFadType> microBackStressNew(num_dims_);
      for (size_t i = 0; i < num_dims_; i++)
        for (size_t j = 0; j < num_dims_; j++)
          microBackStressNew(i, j) = microBackStress[ims](i, j) +
                                     sq32 * dGamma * Hval * returnDir(i, j);
      Rfad[xIndex] = sq32 * devMag(stressNew - microBackStressNew) - Kval;
    }

    // double
    const ScalarT a = 18.0;
    const ScalarT b = 1.0 / a;
    xIndex          = 2 * ims + 2;
    if (yieldMask[xIndex] > 0) {
      xIndex = yieldMap[xIndex];
      minitensor::Tensor3<ScalarT> returnDir =
          devNorm(elTrialDoubleStress[ims] - doubleBackStress[ims]);
      returnDir *= a / lengthScale[ims];
      minitensor::Tensor3<ScalarT> stressDir;
      stressDir = dotdotdot(doubleCelastic[ims], returnDir);
      DFadType                      dGamma = Xfad[xIndex];
      minitensor::Tensor3<DFadType> stressNew(num_dims_);
      for (size_t i = 0; i < num_dims_; i++)
        for (size_t j = 0; j < num_dims_; j++)
          for (size_t k = 0; k < num_dims_; k++)
            stressNew(i, j, k) =
                elTrialDoubleStress[ims](i, j, k) - dGamma * stressDir(i, j, k);
      DFadType alphaNew = doubleAlpha[ims] + dGamma;
      //  JR: todo: create a plasticity submodel object to encapsulate this.
      //     DFadType Hval = doubleKinematicHardening[ims]->(alphaNew);
      //     DFadType Kval = doubleIsotropicHardening[ims]->(alphaNew);
      DFadType Hval = doubleKinematicModulus[ims] * alphaNew;
      DFadType Kval =
          doubleYieldStress0[ims] + doubleIsotropicModulus[ims] * alphaNew;
      minitensor::Tensor3<DFadType> doubleBackStressNew(num_dims_);
      for (size_t i = 0; i < num_dims_; i++)
        for (size_t j = 0; j < num_dims_; j++)
          for (size_t k = 0; k < num_dims_; k++)
            doubleBackStressNew(i, j, k) =
                doubleBackStress[ims](i, j, k) +
                sq32 * dGamma * Hval * returnDir(i, j, k);
      ScalarT aOverL = a / lengthScale[ims];
      Rfad[xIndex]   = aOverL * devMag(stressNew - doubleBackStressNew) - Kval;
    }
  }

  // Residual
  for (int i = 0; i < nvars; i++) R[i] = Rfad[i].val();

  // Jacobian
  for (int i = 0; i < nvars; i++)
    for (int j = 0; j < nvars; j++) dRdX[i + nvars * j] = Rfad[i].dx(j);
}

/******************************************************************************/
template <typename EvalT, typename Traits>
template <typename T>
T
J2HMCModel<EvalT, Traits>::devMag(const minitensor::Tensor<T>& t)
/******************************************************************************/
{
  // returns magnitude of the deviatoric part

  int dim = t.get_dimension();
  T   tr  = 0.0;
  for (int i = 0; i < dim; i++) tr += t(i, i);
  tr /= (double)dim;
  T mag = 0.0;
  for (int i = 0; i < dim; i++)
    for (int j = 0; j < dim; j++) {
      if (i == j)
        mag += (t(i, j) - tr) * (t(i, j) - tr);
      else
        mag += t(i, j) * t(i, j);
    }
  if (mag > 0.0)
    return sqrt(mag);
  else
    return 0.0;
}

/******************************************************************************/
template <typename EvalT, typename Traits>
template <typename T>
T
J2HMCModel<EvalT, Traits>::devMag(const minitensor::Tensor3<T>& t)
/******************************************************************************/
{
  // returns magnitude of the deviatoric part

  int dim = t.get_dimension();
  T   mag = 0.0;
  for (int k = 0; k < dim; k++) {
    T tr = 0.0;
    for (int i = 0; i < dim; i++) tr += t(i, i, k);
    tr /= (double)dim;

    for (int i = 0; i < dim; i++)
      for (int j = 0; j < dim; j++) {
        if (i == j)
          mag += (t(i, j, k) - tr) * (t(i, j, k) - tr);
        else
          mag += t(i, j, k) * t(i, j, k);
      }
  }
  if (mag > 0.0)
    return sqrt(mag);
  else
    return 0.0;
}

/******************************************************************************/
template <typename EvalT, typename Traits>
template <typename T>
minitensor::Tensor<T>
J2HMCModel<EvalT, Traits>::devNorm(const minitensor::Tensor<T>& t)
/******************************************************************************/
{
  // returns normalized deviatoric part

  minitensor::Tensor<T> n(t);

  int dim = t.get_dimension();
  T   tr  = 0.0;
  for (int i = 0; i < dim; i++) tr += n(i, i);
  for (int i = 0; i < dim; i++) n(i, i) -= tr / ((double)dim);
  T mag = 0.0;
  for (int i = 0; i < dim; i++)
    for (int j = 0; j < dim; j++) mag += n(i, j) * n(i, j);

  if (mag > 0.0) n /= std::sqrt(mag);

  return n;
}

/******************************************************************************/
template <typename EvalT, typename Traits>
template <typename T>
minitensor::Tensor3<T>
J2HMCModel<EvalT, Traits>::devNorm(const minitensor::Tensor3<T>& t)
/******************************************************************************/
{
  // returns normalized deviatoric part

  minitensor::Tensor3<T> n(t);

  int dim = t.get_dimension();
  T   tr;
  for (int j = 0; j < dim; j++) {
    tr = 0.0;
    for (int i = 0; i < dim; i++) tr += n(i, i, j);
    for (int i = 0; i < dim; i++) n(i, i, j) -= tr / ((double)dim);
  }
  T mag = 0.0;
  for (int i = 0; i < dim; i++)
    for (int j = 0; j < dim; j++)
      for (int k = 0; k < dim; k++) mag += n(i, j, k) * n(i, j, k);

  if (mag > 0.0) n /= std::sqrt(mag);

  return n;
}

/******************************************************************************/
template <typename EvalT, typename Traits>
minitensor::Tensor3<typename EvalT::ScalarT>
J2HMCModel<EvalT, Traits>::dotdotdot(
    minitensor::Tensor4<ScalarT>& C,
    minitensor::Tensor3<ScalarT>& dir)
/******************************************************************************/
{
  minitensor::Tensor3<ScalarT> retVal(num_dims_);

  for (int k = 0; k < num_dims_; k++)
    for (int i = 0; i < num_dims_; i++)
      for (int j = 0; j < num_dims_; j++)
        for (int m = 0; m < num_dims_; m++)
          for (int n = 0; n < num_dims_; n++)
            retVal(i, j, k) = C(i, j, m, n) * dir(m, n, k);

  return retVal;
}

/******************************************************************************/
template <typename EvalT, typename Traits>
void
J2HMCModel<EvalT, Traits>::yieldFunction(
    std::vector<typename EvalT::ScalarT>&      Fval,
    minitensor::Tensor<ScalarT>&               macStress,
    ScalarT&                                   macroAlpha,
    minitensor::Tensor<ScalarT>&               macroBackStress,
    std::vector<minitensor::Tensor<ScalarT>>&  micStress,
    std::vector<ScalarT>&                      microAlpha,
    std::vector<minitensor::Tensor<ScalarT>>&  microBackStress,
    std::vector<minitensor::Tensor3<ScalarT>>& doubleStress,
    std::vector<ScalarT>&                      doubleAlpha,
    std::vector<minitensor::Tensor3<ScalarT>>& doubleBackStress)
/******************************************************************************/
{
  ScalarT sq32(std::sqrt(3.0 / 2.0));

  // Macro scale
  ScalarT Kval = macroYieldStress0 + macroIsotropicModulus * macroAlpha;
  Fval[0]      = sq32 * devMag(macStress - macroBackStress) - Kval;

  // Micro scales
  for (int ims = 0; ims < numMicroScales; ims++) {
    // micro
    {
      ScalarT Kval =
          microYieldStress0[ims] + microIsotropicModulus[ims] * microAlpha[ims];
      Fval[1 + 2 * ims] =
          sq32 * devMag(micStress[ims] - microBackStress[ims]) - Kval;
    }

    // double
    const ScalarT a = 18.0;
    const ScalarT b = 1.0 / a;
    {
      ScalarT aOverL = a / lengthScale[ims];
      ScalarT Kval   = doubleYieldStress0[ims] +
                     doubleIsotropicModulus[ims] * doubleAlpha[ims];
      Fval[2 + 2 * ims] =
          aOverL * devMag(doubleStress[ims] - doubleBackStress[ims]) - Kval;
    }
  }
}

/******************************************************************************/
template <typename EvalT, typename Traits>
void
J2HMCModel<EvalT, Traits>::computeTrialState(
    typename Traits::EvalData workset,
    /* increments */
    PHX::MDField<const ScalarT>&              delta_macroStrain,
    std::vector<PHX::MDField<const ScalarT>>& delta_strainDifference,
    std::vector<PHX::MDField<const ScalarT>>& delta_microStrainGradient,
    /* updated state */
    PHX::MDField<ScalarT>&              updated_macroStress,
    std::vector<PHX::MDField<ScalarT>>& updated_microStress,
    std::vector<PHX::MDField<ScalarT>>& updated_doubleStress)
/******************************************************************************/
{
  // current state (i.e., state at N) is stored in the state manager:
  std::vector<Albany::MDArray> current_microStress(numMicroScales);
  std::vector<Albany::MDArray> current_doubleStress(numMicroScales);
  Albany::MDArray              current_macroStress =
      (*workset.stateArrayPtr)[current_macroStressName];

  for (int i = 0; i < numMicroScales; i++) {
    current_microStress[i] =
        (*workset.stateArrayPtr)[current_microStressName[i]];
    current_doubleStress[i] =
        (*workset.stateArrayPtr)[current_doubleStressName[i]];
  }

  const std::size_t zero(0);
  const std::size_t one(1);
  const std::size_t two(2);

  int numCells = workset.numCells;

  if (num_dims_ == 1) {
    // Compute Stress (uniaxial strain)
    for (std::size_t cell = 0; cell < numCells; ++cell)
      for (std::size_t qp = 0; qp < num_pts_; ++qp)
        updated_macroStress(cell, qp, 0, 0) =
            ScalarT(current_macroStress(cell, qp, zero, zero)) +
            C11 * delta_macroStrain(cell, qp, 0, 0);
  } else if (num_dims_ == 2) {
    // Compute Stress (plane strain)
    for (std::size_t cell = 0; cell < numCells; ++cell) {
      for (std::size_t qp = 0; qp < num_pts_; ++qp) {
        const ScalarT &de1 = delta_macroStrain(cell, qp, 0, 0),
                      &de2 = delta_macroStrain(cell, qp, 1, 1),
                      &de3 = delta_macroStrain(cell, qp, 0, 1);
        ScalarT s1(current_macroStress(cell, qp, zero, zero)),
            s2(current_macroStress(cell, qp, one, one)),
            s3(current_macroStress(cell, qp, zero, one));
        updated_macroStress(cell, qp, 0, 0) = s1 + C11 * de1 + C12 * de2;
        updated_macroStress(cell, qp, 1, 1) = s2 + C12 * de1 + C11 * de2;
        updated_macroStress(cell, qp, 0, 1) = s3 + C44 * de3;
        updated_macroStress(cell, qp, 1, 0) =
            updated_macroStress(cell, qp, 0, 1);
      }
    }
    // Compute Micro Stress
    for (int i = 0; i < numMicroScales; i++) {
      auto&   dsd  = delta_strainDifference[i];
      auto&   ms   = updated_microStress[i];
      ScalarT beta = betaParameter[i];
      for (std::size_t cell = 0; cell < numCells; ++cell) {
        for (std::size_t qp = 0; qp < num_pts_; ++qp) {
          const ScalarT &de1 = dsd(cell, qp, 0, 0), &de2 = dsd(cell, qp, 1, 1),
                        &de3 = dsd(cell, qp, 0, 1), &de4 = dsd(cell, qp, 1, 0);
          ScalarT s1(current_microStress[i](cell, qp, zero, zero)),
              s2(current_microStress[i](cell, qp, one, one)),
              s3(current_microStress[i](cell, qp, zero, one)),
              s4(current_microStress[i](cell, qp, one, zero));
          ms(cell, qp, 0, 0) = s1 + beta * (C11 * de1 + C12 * de2);
          ms(cell, qp, 1, 1) = s2 + beta * (C12 * de1 + C11 * de2);
          ms(cell, qp, 0, 1) = s3 + beta * (C44 * de3);
          ms(cell, qp, 1, 0) = s4 + beta * (C44 * de4);
        }
      }
    }
    // Compute Double Stress
    for (int i = 0; i < numMicroScales; i++) {
      auto&   dmsg = delta_microStrainGradient[i];
      auto&   ds   = updated_doubleStress[i];
      ScalarT beta = lengthScale[i] * lengthScale[i] * betaParameter[i];
      for (std::size_t cell = 0; cell < numCells; ++cell) {
        for (std::size_t qp = 0; qp < num_pts_; ++qp) {
          for (std::size_t k = 0; k < num_dims_; ++k) {
            const ScalarT &de1 = dmsg(cell, qp, 0, 0, k),
                          &de2 = dmsg(cell, qp, 1, 1, k),
                          &de3 = dmsg(cell, qp, 0, 1, k),
                          &de4 = dmsg(cell, qp, 1, 0, k);
            ScalarT ds1(current_doubleStress[i](cell, qp, zero, zero, k)),
                ds2(current_doubleStress[i](cell, qp, one, one, k)),
                ds3(current_doubleStress[i](cell, qp, zero, one, k)),
                ds4(current_doubleStress[i](cell, qp, one, zero, k));
            ds(cell, qp, 0, 0, k) = ds1 + beta * (C11 * de1 + C12 * de2);
            ds(cell, qp, 1, 1, k) = ds2 + beta * (C12 * de1 + C11 * de2);
            ds(cell, qp, 0, 1, k) = ds3 + beta * (C44 * de3);
            ds(cell, qp, 1, 0, k) = ds4 + beta * (C44 * de4);
          }
        }
      }
    }
  } else if (num_dims_ == 3) {
    // Compute Stress
    for (std::size_t cell = 0; cell < numCells; ++cell) {
      for (std::size_t qp = 0; qp < num_pts_; ++qp) {
        const ScalarT &de1 = delta_macroStrain(cell, qp, 0, 0),
                      &de2 = delta_macroStrain(cell, qp, 1, 1),
                      &de3 = delta_macroStrain(cell, qp, 2, 2),
                      &de4 = delta_macroStrain(cell, qp, 1, 2),
                      &de5 = delta_macroStrain(cell, qp, 0, 2),
                      &de6 = delta_macroStrain(cell, qp, 0, 1);
        ScalarT cms1(current_macroStress(cell, qp, zero, zero)),
            cms2(current_macroStress(cell, qp, one, one)),
            cms3(current_macroStress(cell, qp, two, two)),
            cms4(current_macroStress(cell, qp, one, two)),
            cms5(current_macroStress(cell, qp, zero, two)),
            cms6(current_macroStress(cell, qp, zero, one));
        updated_macroStress(cell, qp, 0, 0) =
            cms1 + C11 * de1 + C12 * de2 + C23 * de3;
        updated_macroStress(cell, qp, 1, 1) =
            cms2 + C12 * de1 + C11 * de2 + C23 * de3;
        updated_macroStress(cell, qp, 2, 2) =
            cms3 + C23 * de1 + C23 * de2 + C33 * de3;
        updated_macroStress(cell, qp, 1, 2) = cms4 + C44 * de4;
        updated_macroStress(cell, qp, 0, 2) = cms5 + C44 * de5;
        updated_macroStress(cell, qp, 0, 1) = cms6 + C66 * de6;
        updated_macroStress(cell, qp, 1, 0) =
            updated_macroStress(cell, qp, 0, 1);
        updated_macroStress(cell, qp, 2, 0) =
            updated_macroStress(cell, qp, 0, 2);
        updated_macroStress(cell, qp, 2, 1) =
            updated_macroStress(cell, qp, 1, 2);
      }
    }
    // Compute Micro Stress
    for (int i = 0; i < numMicroScales; i++) {
      auto&   dsd  = delta_strainDifference[i];
      auto&   ms   = updated_microStress[i];
      ScalarT beta = betaParameter[i];
      for (std::size_t cell = 0; cell < numCells; ++cell) {
        for (std::size_t qp = 0; qp < num_pts_; ++qp) {
          const ScalarT &de1 = dsd(cell, qp, 0, 0), &de2 = dsd(cell, qp, 1, 1),
                        &de3 = dsd(cell, qp, 2, 2);
          const ScalarT &de4 = dsd(cell, qp, 1, 2), &de5 = dsd(cell, qp, 0, 2),
                        &de6 = dsd(cell, qp, 0, 1);
          const ScalarT &de7 = dsd(cell, qp, 2, 1), &de8 = dsd(cell, qp, 2, 0),
                        &de9 = dsd(cell, qp, 1, 0);
          ScalarT cms1(current_microStress[i](cell, qp, zero, zero)),
              cms2(current_microStress[i](cell, qp, one, one)),
              cms3(current_microStress[i](cell, qp, two, two)),
              cms4(current_microStress[i](cell, qp, one, two)),
              cms5(current_microStress[i](cell, qp, zero, two)),
              cms6(current_microStress[i](cell, qp, zero, one)),
              cms7(current_microStress[i](cell, qp, two, one)),
              cms8(current_microStress[i](cell, qp, two, zero)),
              cms9(current_microStress[i](cell, qp, one, zero));
          ms(cell, qp, 0, 0) =
              cms1 + beta * (C11 * de1 + C12 * de2 + C23 * de3);
          ms(cell, qp, 1, 1) =
              cms2 + beta * (C12 * de1 + C11 * de2 + C23 * de3);
          ms(cell, qp, 2, 2) =
              cms3 + beta * (C23 * de1 + C23 * de2 + C33 * de3);
          ms(cell, qp, 1, 2) = cms4 + beta * (C44 * de4);
          ms(cell, qp, 0, 2) = cms5 + beta * (C44 * de5);
          ms(cell, qp, 0, 1) = cms6 + beta * (C66 * de6);
          ms(cell, qp, 1, 0) = cms7 + beta * (C44 * de9);
          ms(cell, qp, 2, 0) = cms8 + beta * (C44 * de8);
          ms(cell, qp, 2, 1) = cms9 + beta * (C66 * de7);
        }
      }
    }
    // Compute Double Stress
    for (int i = 0; i < numMicroScales; i++) {
      auto&   dmsg = delta_microStrainGradient[i];
      auto&   ds   = updated_doubleStress[i];
      ScalarT beta = lengthScale[i] * lengthScale[i] * betaParameter[i];
      for (std::size_t cell = 0; cell < numCells; ++cell) {
        for (std::size_t qp = 0; qp < num_pts_; ++qp) {
          for (std::size_t k = 0; k < num_dims_; ++k) {
            const ScalarT &de1 = dmsg(cell, qp, 0, 0, k),
                          &de2 = dmsg(cell, qp, 1, 1, k),
                          &de3 = dmsg(cell, qp, 2, 2, k);
            const ScalarT &de4 = dmsg(cell, qp, 1, 2, k),
                          &de5 = dmsg(cell, qp, 0, 2, k),
                          &de6 = dmsg(cell, qp, 0, 1, k);
            const ScalarT &de7 = dmsg(cell, qp, 2, 1, k),
                          &de8 = dmsg(cell, qp, 2, 0, k),
                          &de9 = dmsg(cell, qp, 1, 0, k);
            ScalarT cds1(current_doubleStress[i](cell, qp, zero, zero, k)),
                cds2(current_doubleStress[i](cell, qp, one, one, k)),
                cds3(current_doubleStress[i](cell, qp, two, two, k)),
                cds4(current_doubleStress[i](cell, qp, one, two, k)),
                cds5(current_doubleStress[i](cell, qp, zero, two, k)),
                cds6(current_doubleStress[i](cell, qp, zero, one, k)),
                cds7(current_doubleStress[i](cell, qp, two, one, k)),
                cds8(current_doubleStress[i](cell, qp, two, zero, k)),
                cds9(current_doubleStress[i](cell, qp, one, zero, k));
            ds(cell, qp, 0, 0, k) =
                cds1 + beta * (C11 * de1 + C12 * de2 + C23 * de3);
            ds(cell, qp, 1, 1, k) =
                cds1 + beta * (C12 * de1 + C11 * de2 + C23 * de3);
            ds(cell, qp, 2, 2, k) =
                cds1 + beta * (C23 * de1 + C23 * de2 + C33 * de3);
            ds(cell, qp, 1, 2, k) = cds1 + beta * (C44 * de4);
            ds(cell, qp, 0, 2, k) = cds1 + beta * (C44 * de5);
            ds(cell, qp, 0, 1, k) = cds1 + beta * (C66 * de6);
            ds(cell, qp, 1, 0, k) = cds1 + beta * (C44 * de9);
            ds(cell, qp, 2, 0, k) = cds1 + beta * (C44 * de8);
            ds(cell, qp, 2, 1, k) = cds1 + beta * (C66 * de7);
          }
        }
      }
    }
  }
}
//----------------------------------------------------------------------------

/******************************************************************************/
template <typename EvalT, typename Traits>
void
J2HMCModel<EvalT, Traits>::initializeElasticConstants()
/******************************************************************************/
{
  // JR:  revisit this.  factors from voigt to tensor form.
  if (num_dims_ == 2) {
    macroCelastic.set_dimension(num_dims_);
    macroCelastic.clear();
    macroCelastic(0, 0, 0, 0) = C11;
    macroCelastic(0, 0, 1, 1) = C12;
    macroCelastic(1, 1, 0, 0) = C12;
    macroCelastic(1, 1, 1, 1) = C11;
    macroCelastic(0, 1, 0, 1) = C44;
    macroCelastic(1, 0, 1, 0) = C44;
  } else if (num_dims_ == 3) {
    macroCelastic.set_dimension(num_dims_);
    macroCelastic.clear();
    macroCelastic(0, 0, 0, 0) = C11;
    macroCelastic(0, 0, 1, 1) = C12;
    macroCelastic(0, 0, 2, 2) = C23;
    macroCelastic(1, 1, 0, 0) = C12;
    macroCelastic(1, 1, 1, 1) = C11;
    macroCelastic(1, 1, 2, 2) = C23;
    macroCelastic(2, 2, 0, 0) = C23;
    macroCelastic(2, 2, 1, 1) = C23;
    macroCelastic(2, 2, 2, 2) = C33;
    macroCelastic(1, 2, 1, 2) = C44;
    macroCelastic(2, 1, 2, 1) = C44;
    macroCelastic(0, 2, 0, 2) = C44;
    macroCelastic(2, 0, 2, 0) = C44;
    macroCelastic(0, 1, 0, 1) = C66;
    macroCelastic(1, 0, 1, 0) = C66;
  }

  microCelastic.resize(numMicroScales);
  doubleCelastic.resize(numMicroScales);
  for (int i = 0; i < numMicroScales; i++) {
    microCelastic[i].set_dimension(num_dims_);
    microCelastic[i] = macroCelastic;
    microCelastic[i] *= betaParameter[i];
    doubleCelastic[i].set_dimension(num_dims_);
    doubleCelastic[i] = macroCelastic;
    doubleCelastic[i] *= lengthScale[i] * lengthScale[i] * betaParameter[i];
  }
}

/******************************************************************************/
template <typename EvalT, typename Traits>
void
J2HMCModel<EvalT, Traits>::computeVolumeAverage(
    typename Traits::EvalData workset,
    DepFieldMap               dep_fields,
    FieldMap                  eval_fields)
/******************************************************************************/
{
  minitensor::Tensor<ScalarT> sig(num_dims_);
  minitensor::Tensor<ScalarT> I(minitensor::eye<ScalarT>(num_dims_));

  std::string cauchy("Stress");
  auto        stress = *eval_fields[cauchy];

  ScalarT volume, pbar, p;

  for (int cell(0); cell < workset.numCells; ++cell) {
    volume = pbar = 0.0;
    for (int pt(0); pt < num_pts_; ++pt) {
      sig.fill(stress, cell, pt, 0, 0);
      pbar += weights_(cell, pt) * (1. / num_dims_) * minitensor::trace(sig);
      volume += weights_(cell, pt);
    }

    pbar /= volume;

    for (int pt(0); pt < num_pts_; ++pt) {
      sig.fill(stress, cell, pt, 0, 0);
      p = (1. / num_dims_) * minitensor::trace(sig);
      sig += (pbar - p) * I;

      for (int i = 0; i < num_dims_; ++i) {
        stress(cell, pt, i, i) = sig(i, i);
      }
    }
  }
}

}  // namespace LCM
