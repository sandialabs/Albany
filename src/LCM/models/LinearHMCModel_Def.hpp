//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <MiniTensor.h>
#include "Albany_Utils.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Teuchos_TestForException.hpp"

namespace LCM {

//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
LinearHMCModel<EvalT, Traits>::LinearHMCModel(
    Teuchos::ParameterList*              p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : LCM::ConstitutiveModel<EvalT, Traits>(p, dl),
      numMicroScales(p->get<int>("Additional Scales")),
      C11(p->get<RealType>("C11")),
      C33(p->get<RealType>("C33")),
      C12(p->get<RealType>("C12")),
      C23(p->get<RealType>("C23")),
      C44(p->get<RealType>("C44")),
      C66(p->get<RealType>("C66"))
{
  lengthScale.resize(numMicroScales);
  betaParameter.resize(numMicroScales);
  for (int i = 0; i < numMicroScales; i++) {
    std::string mySublist                 = Albany::strint("Microscale", i + 1);
    const Teuchos::ParameterList& msModel = p->sublist(mySublist);
    lengthScale[i]   = msModel.get<RealType>("Length Scale");
    betaParameter[i] = msModel.get<RealType>("Beta Constant");
  }

  // define the dependent fields
  macroStrainName = "Strain";
  this->dep_field_map_.insert(std::make_pair(macroStrainName, dl->qp_tensor));
  strainDifferenceName.resize(numMicroScales);
  microStrainGradientName.resize(numMicroScales);
  for (int i = 0; i < numMicroScales; i++) {
    std::stringstream sdname;
    sdname << "Strain Difference " << i;
    strainDifferenceName[i] = sdname.str();
    this->dep_field_map_.insert(std::make_pair(sdname.str(), dl->qp_tensor));
    std::stringstream sgradname;
    sgradname << "Microstrain " << i << " Gradient";
    microStrainGradientName[i] = sgradname.str();
    this->dep_field_map_.insert(
        std::make_pair(sgradname.str(), dl->qp_tensor3));
  }

  // define the evaluated fields
  macroStressName = "Stress";
  this->eval_field_map_.insert(std::make_pair(macroStressName, dl->qp_tensor));
  microStressName.resize(numMicroScales);
  doubleStressName.resize(numMicroScales);
  for (int i = 0; i < numMicroScales; i++) {
    std::stringstream msname;
    msname << "Micro Stress " << i;
    microStressName[i] = msname.str();
    this->eval_field_map_.insert(std::make_pair(msname.str(), dl->qp_tensor));
    std::stringstream dsname;
    dsname << "Double Stress " << i;
    doubleStressName[i] = dsname.str();
    this->eval_field_map_.insert(std::make_pair(dsname.str(), dl->qp_tensor3));
  }

  // define the state variables
  this->num_state_variables_++;
  this->state_var_names_.push_back(macroStressName);
  this->state_var_layouts_.push_back(dl->qp_tensor);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(false);
  this->state_var_output_flags_.push_back(true);
}
//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
void
LinearHMCModel<EvalT, Traits>::computeState(
    typename Traits::EvalData workset,
    DepFieldMap               dep_fields,
    FieldMap                  eval_fields)
{
  // extract independent MDFields
  auto macroStrain = *dep_fields[macroStrainName];
  std::vector<PHX::MDField<const ScalarT>> strainDifference(numMicroScales);
  std::vector<PHX::MDField<const ScalarT>> microStrainGradient(numMicroScales);
  for (int i = 0; i < numMicroScales; i++) {
    strainDifference[i]    = *dep_fields[strainDifferenceName[i]];
    microStrainGradient[i] = *dep_fields[microStrainGradientName[i]];
  }

  // extract evaluated MDFields
  auto macroStress = *eval_fields[macroStressName];
  std::vector<PHX::MDField<ScalarT>> microStress(numMicroScales);
  std::vector<PHX::MDField<ScalarT>> doubleStress(numMicroScales);
  for (int i = 0; i < numMicroScales; i++) {
    microStress[i]  = *eval_fields[microStressName[i]];
    doubleStress[i] = *eval_fields[doubleStressName[i]];
  }

  switch (num_dims_) {
    case 1:
      // Compute Stress (uniaxial strain)
      for (std::size_t cell = 0; cell < workset.numCells; ++cell)
        for (std::size_t qp = 0; qp < num_pts_; ++qp)
          macroStress(cell, qp, 0, 0) = C11 * macroStrain(cell, qp, 0, 0);
      break;
    case 2:
      // Compute Stress (plane strain)
      for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
        for (std::size_t qp = 0; qp < num_pts_; ++qp) {
          auto e1                     = macroStrain(cell, qp, 0, 0);
          auto e2                     = macroStrain(cell, qp, 1, 1);
          auto e3                     = macroStrain(cell, qp, 0, 1);
          macroStress(cell, qp, 0, 0) = C11 * e1 + C12 * e2;
          macroStress(cell, qp, 1, 1) = C12 * e1 + C11 * e2;
          macroStress(cell, qp, 0, 1) = C44 * e3;
          macroStress(cell, qp, 1, 0) = macroStress(cell, qp, 0, 1);
        }
      }
      // Compute Micro Stress
      for (int i = 0; i < numMicroScales; i++) {
        auto&   sd   = strainDifference[i];
        auto&   ms   = microStress[i];
        ScalarT beta = betaParameter[i];
        for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
          for (std::size_t qp = 0; qp < num_pts_; ++qp) {
            const ScalarT e1 = sd(cell, qp, 0, 0), e2 = sd(cell, qp, 1, 1),
                          e3 = sd(cell, qp, 0, 1), e4 = sd(cell, qp, 1, 0);
            ms(cell, qp, 0, 0) = beta * (C11 * e1 + C12 * e2);
            ms(cell, qp, 1, 1) = beta * (C12 * e1 + C11 * e2);
            ms(cell, qp, 0, 1) = beta * (C44 * e3);
            ms(cell, qp, 1, 0) = beta * (C44 * e4);
          }
        }
      }
      // Compute Double Stress
      for (int i = 0; i < numMicroScales; i++) {
        auto&   msg  = microStrainGradient[i];
        auto&   ds   = doubleStress[i];
        ScalarT beta = lengthScale[i] * lengthScale[i] * betaParameter[i];
        for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
          for (std::size_t qp = 0; qp < num_pts_; ++qp) {
            for (std::size_t k = 0; k < num_dims_; ++k) {
              auto e1               = msg(cell, qp, 0, 0, k);
              auto e2               = msg(cell, qp, 1, 1, k);
              auto e3               = msg(cell, qp, 0, 1, k);
              auto e4               = msg(cell, qp, 1, 0, k);
              ds(cell, qp, 0, 0, k) = beta * (C11 * e1 + C12 * e2);
              ds(cell, qp, 1, 1, k) = beta * (C12 * e1 + C11 * e2);
              ds(cell, qp, 0, 1, k) = beta * (C44 * e3);
              ds(cell, qp, 1, 0, k) = beta * (C44 * e4);
            }
          }
        }
      }
      break;
    case 3:
      // Compute Stress
      for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
        for (std::size_t qp = 0; qp < num_pts_; ++qp) {
          auto e1                     = macroStrain(cell, qp, 0, 0);
          auto e2                     = macroStrain(cell, qp, 1, 1);
          auto e3                     = macroStrain(cell, qp, 2, 2);
          auto e4                     = macroStrain(cell, qp, 1, 2);
          auto e5                     = macroStrain(cell, qp, 0, 2);
          auto e6                     = macroStrain(cell, qp, 0, 1);
          macroStress(cell, qp, 0, 0) = C11 * e1 + C12 * e2 + C23 * e3;
          macroStress(cell, qp, 1, 1) = C12 * e1 + C11 * e2 + C23 * e3;
          macroStress(cell, qp, 2, 2) = C23 * e1 + C23 * e2 + C33 * e3;
          macroStress(cell, qp, 1, 2) = C44 * e4;
          macroStress(cell, qp, 0, 2) = C44 * e5;
          macroStress(cell, qp, 0, 1) = C66 * e6;
          macroStress(cell, qp, 1, 0) = macroStress(cell, qp, 0, 1);
          macroStress(cell, qp, 2, 0) = macroStress(cell, qp, 0, 2);
          macroStress(cell, qp, 2, 1) = macroStress(cell, qp, 1, 2);
        }
      }
      // Compute Micro Stress
      for (int i = 0; i < numMicroScales; i++) {
        auto&   sd   = strainDifference[i];
        auto&   ms   = microStress[i];
        ScalarT beta = betaParameter[i];
        for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
          for (std::size_t qp = 0; qp < num_pts_; ++qp) {
            auto e1            = sd(cell, qp, 0, 0);
            auto e2            = sd(cell, qp, 1, 1);
            auto e3            = sd(cell, qp, 2, 2);
            auto e4            = sd(cell, qp, 1, 2);
            auto e5            = sd(cell, qp, 0, 2);
            auto e6            = sd(cell, qp, 0, 1);
            auto e7            = sd(cell, qp, 2, 1);
            auto e8            = sd(cell, qp, 2, 0);
            auto e9            = sd(cell, qp, 1, 0);
            ms(cell, qp, 0, 0) = beta * (C11 * e1 + C12 * e2 + C23 * e3);
            ms(cell, qp, 1, 1) = beta * (C12 * e1 + C11 * e2 + C23 * e3);
            ms(cell, qp, 2, 2) = beta * (C23 * e1 + C23 * e2 + C33 * e3);
            ms(cell, qp, 1, 2) = beta * (C44 * e4);
            ms(cell, qp, 0, 2) = beta * (C44 * e5);
            ms(cell, qp, 0, 1) = beta * (C66 * e6);
            ms(cell, qp, 1, 0) = beta * (C44 * e9);
            ms(cell, qp, 2, 0) = beta * (C44 * e8);
            ms(cell, qp, 2, 1) = beta * (C66 * e7);
          }
        }
      }
      // Compute Double Stress
      for (int i = 0; i < numMicroScales; i++) {
        auto&   msg  = microStrainGradient[i];
        auto&   ds   = doubleStress[i];
        ScalarT beta = lengthScale[i] * lengthScale[i] * betaParameter[i];
        for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
          for (std::size_t qp = 0; qp < num_pts_; ++qp) {
            for (std::size_t k = 0; k < num_dims_; ++k) {
              auto e1               = msg(cell, qp, 0, 0, k);
              auto e2               = msg(cell, qp, 1, 1, k);
              auto e3               = msg(cell, qp, 2, 2, k);
              auto e4               = msg(cell, qp, 1, 2, k);
              auto e5               = msg(cell, qp, 0, 2, k);
              auto e6               = msg(cell, qp, 0, 1, k);
              auto e7               = msg(cell, qp, 2, 1, k);
              auto e8               = msg(cell, qp, 2, 0, k);
              auto e9               = msg(cell, qp, 1, 0, k);
              ds(cell, qp, 0, 0, k) = beta * (C11 * e1 + C12 * e2 + C23 * e3);
              ds(cell, qp, 1, 1, k) = beta * (C12 * e1 + C11 * e2 + C23 * e3);
              ds(cell, qp, 2, 2, k) = beta * (C23 * e1 + C23 * e2 + C33 * e3);
              ds(cell, qp, 1, 2, k) = beta * (C44 * e4);
              ds(cell, qp, 0, 2, k) = beta * (C44 * e5);
              ds(cell, qp, 0, 1, k) = beta * (C66 * e6);
              ds(cell, qp, 1, 0, k) = beta * (C44 * e9);
              ds(cell, qp, 2, 0, k) = beta * (C44 * e8);
              ds(cell, qp, 2, 1, k) = beta * (C66 * e7);
            }
          }
        }
      }
      break;
  }
}
//----------------------------------------------------------------------------
}  // namespace LCM
