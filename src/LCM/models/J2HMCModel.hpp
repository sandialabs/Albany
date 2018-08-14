//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_J2HMCModel_hpp)
#define LCM_J2HMCModel_hpp

#include <MiniTensor.h>
#include "Albany_Layouts.hpp"
#include "LCM/models/ConstitutiveModel.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

#include "Sacado.hpp"

namespace LCM {

//! \brief J2 plasticity for HMC
template <typename EvalT, typename Traits>
class J2HMCModel : public LCM::ConstitutiveModel<EvalT, Traits>
{
 public:
  using Base        = LCM::ConstitutiveModel<EvalT, Traits>;
  using DepFieldMap = typename Base::DepFieldMap;
  using FieldMap    = typename Base::FieldMap;

  typedef typename EvalT::ScalarT                             ScalarT;
  typedef typename EvalT::MeshScalarT                         MeshScalarT;
  typedef typename Sacado::mpl::apply<FadType, ScalarT>::type DFadType;

  using ConstitutiveModel<EvalT, Traits>::num_dims_;
  using ConstitutiveModel<EvalT, Traits>::num_pts_;
  using ConstitutiveModel<EvalT, Traits>::field_name_map_;
  using ConstitutiveModel<EvalT, Traits>::weights_;

  ///
  /// Constructor
  ///
  J2HMCModel(
      Teuchos::ParameterList*              p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  ///
  /// Virtual Destructor
  ///
  virtual ~J2HMCModel(){};

  ///
  /// Method to compute the state (e.g. energy, stress, tangent)
  ///
  virtual void
  computeState(
      typename Traits::EvalData workset,
      DepFieldMap               dep_fields,
      FieldMap                  eval_fields);

  // Kokkos
  virtual void
  computeStateParallel(
      typename Traits::EvalData workset,
      DepFieldMap               dep_fields,
      FieldMap                  eval_fields);

  void
  computeVolumeAverage(
      typename Traits::EvalData workset,
      DepFieldMap               dep_fields,
      FieldMap                  eval_fields);

 private:
  ///
  /// Private methods
  ///
  void
  computeTrialState(
      typename Traits::EvalData workset,
      /* increments */
      PHX::MDField<const ScalarT>&              delta_macroStrain,
      std::vector<PHX::MDField<const ScalarT>>& delta_strainDifference,
      std::vector<PHX::MDField<const ScalarT>>& delta_microStrainGradient,
      /* updated state */
      PHX::MDField<ScalarT>&              updated_macroStress,
      std::vector<PHX::MDField<ScalarT>>& updated_microStress,
      std::vector<PHX::MDField<ScalarT>>& updated_doubleStress);

  void
  radialReturn(
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
      std::vector<PHX::MDField<ScalarT>>& new_doubleAlpha);

  void
  computeResidualandJacobian(
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
      std::vector<ScalarT>&                      doubleAlpha);

  template <typename T>
  T
  devMag(const minitensor::Tensor<T>&);
  template <typename T>
  T
  devMag(const minitensor::Tensor3<T>&);

  template <typename T>
  minitensor::Tensor<T>
  devNorm(const minitensor::Tensor<T>& t);
  template <typename T>
  minitensor::Tensor3<T>
  devNorm(const minitensor::Tensor3<T>& t);

  void
  yieldFunction(
      std::vector<typename EvalT::ScalarT>&      Fvals,
      minitensor::Tensor<ScalarT>&               macStress,
      ScalarT&                                   macroAlpha,
      minitensor::Tensor<ScalarT>&               macroBackStress,
      std::vector<minitensor::Tensor<ScalarT>>&  micStress,
      std::vector<ScalarT>&                      microAlpha,
      std::vector<minitensor::Tensor<ScalarT>>&  microBackStress,
      std::vector<minitensor::Tensor3<ScalarT>>& doubleStress,
      std::vector<ScalarT>&                      doubleAlpha,
      std::vector<minitensor::Tensor3<ScalarT>>& doubleBackStress);

  void
  initializeElasticConstants();

  bool
  converged(std::vector<ScalarT>& R, int iteration, ScalarT& initNorm);

  minitensor::Tensor3<typename EvalT::ScalarT>
  dotdotdot(
      minitensor::Tensor4<ScalarT>& doubleCelastic,
      minitensor::Tensor3<ScalarT>& elTrialDoubleStress);

  ///
  /// Private to prohibit copying
  ///
  J2HMCModel(const J2HMCModel&);

  ///
  /// Private to prohibit copying
  ///
  J2HMCModel&
  operator=(const J2HMCModel&);

  ///
  /// material parameters
  ///
  RealType              C11, C33, C12, C23, C44, C66;
  std::vector<RealType> lengthScale;
  std::vector<RealType> betaParameter;
  RealType              macroYieldStress0;
  RealType              macroKinematicModulus;
  RealType              macroIsotropicModulus;
  std::vector<RealType> microYieldStress0;
  std::vector<RealType> microKinematicModulus;
  std::vector<RealType> microIsotropicModulus;
  std::vector<RealType> doubleYieldStress0;
  std::vector<RealType> doubleKinematicModulus;
  std::vector<RealType> doubleIsotropicModulus;

  minitensor::Tensor4<ScalarT>              macroCelastic;
  std::vector<minitensor::Tensor4<ScalarT>> microCelastic;
  std::vector<minitensor::Tensor4<ScalarT>> doubleCelastic;
  ///
  /// model parameters
  ///
  int numMicroScales;

  ///
  /// INDEPENDENT FIELD NAMES
  ///

  //  increments
  std::string              delta_macroStrainName;
  std::vector<std::string> delta_strainDifferenceName;
  std::vector<std::string> delta_microStrainGradientName;

  // current stress state (i.e., state at N)
  std::string              current_macroStressName;
  std::vector<std::string> current_microStressName;
  std::vector<std::string> current_doubleStressName;

  ///
  /// EVALUATED FIELD NAMES
  ///

  // updated stress state (i.e., state at N+1)
  std::string              updated_macroStressName;
  std::vector<std::string> updated_microStressName;
  std::vector<std::string> updated_doubleStressName;

  ///
  /// STATE VARIABLE NAMES
  ///
  std::string              current_macroAlphaName;
  std::string              updated_macroAlphaName;
  std::string              current_macroBackStressName;
  std::string              updated_macroBackStressName;
  std::vector<std::string> current_microAlphaName;
  std::vector<std::string> updated_microAlphaName;
  std::vector<std::string> current_microBackStressName;
  std::vector<std::string> updated_microBackStressName;
  std::vector<std::string> current_doubleAlphaName;
  std::vector<std::string> updated_doubleAlphaName;
  std::vector<std::string> current_doubleBackStressName;
  std::vector<std::string> updated_doubleBackStressName;

  typedef PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim>      HMC2Tensor;
  typedef PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim, Dim> HMC3Tensor;
};
}  // namespace LCM

#endif
