//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_J2HMCModel_hpp)
#define LCM_J2HMCModel_hpp

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"
#include "LCM/models/ConstitutiveModel.hpp"
#include <Intrepid2_MiniTensor.h>

#include "Sacado.hpp"

namespace LCM
{

//! \brief J2 plasticity for HMC
template<typename EvalT, typename Traits>
class J2HMCModel: public LCM::ConstitutiveModel<EvalT, Traits>
{
public:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;
  typedef typename Sacado::mpl::apply<FadType,ScalarT>::type DFadType;

  using ConstitutiveModel<EvalT, Traits>::num_dims_;
  using ConstitutiveModel<EvalT, Traits>::num_pts_;
  using ConstitutiveModel<EvalT, Traits>::field_name_map_;
  using ConstitutiveModel<EvalT, Traits>::weights_;

  ///
  /// Constructor
  ///
  J2HMCModel(Teuchos::ParameterList* p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  ///
  /// Virtual Destructor
  ///
  virtual
  ~J2HMCModel()
  {};

  ///
  /// Method to compute the state (e.g. energy, stress, tangent)
  ///
  virtual
  void
  computeState(typename Traits::EvalData workset,
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>>> dep_fields,
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>>> eval_fields);

  //Kokkos
  virtual
  void
  computeStateParallel(typename Traits::EvalData workset,
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>>> dep_fields,
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>>> eval_fields);

  void
  computeVolumeAverage(
      typename Traits::EvalData workset,
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>>> dep_fields,
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>>> eval_fields);

private:
  
  ///
  /// Private methods
  ///
  void 
  computeTrialState( typename Traits::EvalData workset,
                   /* increments */
                   PHX::MDField<ScalarT> &                delta_macroStrain,
                   std::vector< PHX::MDField<ScalarT>> & delta_strainDifference,
                   std::vector< PHX::MDField<ScalarT>> & delta_microStrainGradient,
                   /* updated state */
                   PHX::MDField<ScalarT> &                updated_macroStress,
                   std::vector< PHX::MDField<ScalarT>> & updated_microStress,
                   std::vector< PHX::MDField<ScalarT>> & updated_doubleStress);


void 
radialReturn( typename Traits::EvalData workset,
// macro
             PHX::MDField<ScalarT> &                trial_macroStress, 
             PHX::MDField<ScalarT> &                new_macroBackStress, 
             PHX::MDField<ScalarT> &                new_macroAlpha, 
// micro
             std::vector< PHX::MDField<ScalarT>> & trial_microStress,
             std::vector< PHX::MDField<ScalarT>> & new_microBackStress,
             std::vector< PHX::MDField<ScalarT>> & new_microAlpha,
// double
             std::vector< PHX::MDField<ScalarT>> & trial_doubleStress,
             std::vector< PHX::MDField<ScalarT>> & new_doubleBackStress,
             std::vector< PHX::MDField<ScalarT>> & new_doubleAlpha);

void 
computeResidualandJacobian(
  std::vector<int> & yieldMask, std::vector<int> & yieldMap,
  std::vector<ScalarT> & X, std::vector<ScalarT> & R, std::vector<ScalarT> & dRdX,
  Intrepid2::Tensor<ScalarT> & elTrialMacroStress,
  Intrepid2::Tensor<ScalarT> & macroBackStress, ScalarT & alpha,
  std::vector<Intrepid2::Tensor<ScalarT>> & elTrialMicroStress,
  std::vector<Intrepid2::Tensor<ScalarT>> & microBackStress,
  std::vector<ScalarT> & microAlpha,
  std::vector<Intrepid2::Tensor3<ScalarT>> & elTrialDoubleStress,
  std::vector<Intrepid2::Tensor3<ScalarT>> & doubleBackStress,
  std::vector<ScalarT> & doubleAlpha);

template <typename T> T devMag( const Intrepid2::Tensor<T> & );
template <typename T> T devMag( const Intrepid2::Tensor3<T> & );

template <typename T> Intrepid2::Tensor<T> devNorm( const Intrepid2::Tensor<T> & t );
template <typename T> Intrepid2::Tensor3<T> devNorm( const Intrepid2::Tensor3<T> & t );

void
yieldFunction( std::vector<typename EvalT::ScalarT>&      Fvals,
               Intrepid2::Tensor<ScalarT>&                 macStress,
               ScalarT&                                   macroAlpha,
               Intrepid2::Tensor<ScalarT>&                 macroBackStress,
               std::vector< Intrepid2::Tensor<ScalarT>>&  micStress,
               std::vector<ScalarT>&                      microAlpha,
               std::vector< Intrepid2::Tensor<ScalarT>>&  microBackStress,
               std::vector< Intrepid2::Tensor3<ScalarT>>& doubleStress,
               std::vector<ScalarT>&                      doubleAlpha,
               std::vector< Intrepid2::Tensor3<ScalarT>>& doubleBackStress);

void initializeElasticConstants();


bool 
converged(std::vector<ScalarT> & R, int iteration, ScalarT& initNorm);

Intrepid2::Tensor3<typename EvalT::ScalarT> 
dotdotdot( 
  Intrepid2::Tensor4<ScalarT> & doubleCelastic, 
  Intrepid2::Tensor3<ScalarT> & elTrialDoubleStress);

  ///
  /// Private to prohibit copying
  ///
  J2HMCModel(const J2HMCModel&);

  ///
  /// Private to prohibit copying
  ///
  J2HMCModel& operator=(const J2HMCModel&);

  ///
  /// material parameters
  ///
  RealType C11, C33, C12, C23, C44, C66;
  std::vector<RealType> lengthScale;
  std::vector<RealType> betaParameter;
  RealType macroYieldStress0;
  RealType macroKinematicModulus;
  RealType macroIsotropicModulus;
  std::vector<RealType> microYieldStress0;
  std::vector<RealType> microKinematicModulus;
  std::vector<RealType> microIsotropicModulus;
  std::vector<RealType> doubleYieldStress0;
  std::vector<RealType> doubleKinematicModulus;
  std::vector<RealType> doubleIsotropicModulus;

  Intrepid2::Tensor4<ScalarT> macroCelastic;
  std::vector<Intrepid2::Tensor4<ScalarT>> microCelastic;
  std::vector<Intrepid2::Tensor4<ScalarT>> doubleCelastic;
  ///
  /// model parameters
  ///
  int numMicroScales;


  ///
  /// INDEPENDENT FIELD NAMES
  ///

  //  increments
  std::string delta_macroStrainName;
  std::vector<std::string> delta_strainDifferenceName;
  std::vector<std::string> delta_microStrainGradientName;
 
  // current stress state (i.e., state at N)
  std::string current_macroStressName;
  std::vector<std::string> current_microStressName;
  std::vector<std::string> current_doubleStressName;
  
  ///
  /// EVALUATED FIELD NAMES
  ///
  
  // updated stress state (i.e., state at N+1)
  std::string updated_macroStressName;
  std::vector<std::string> updated_microStressName;
  std::vector<std::string> updated_doubleStressName;

  ///
  /// STATE VARIABLE NAMES
  ///
  std::string current_macroAlphaName;
  std::string updated_macroAlphaName;
  std::string current_macroBackStressName;
  std::string updated_macroBackStressName;
  std::vector<std::string> current_microAlphaName;
  std::vector<std::string> updated_microAlphaName;
  std::vector<std::string> current_microBackStressName;
  std::vector<std::string> updated_microBackStressName;
  std::vector<std::string> current_doubleAlphaName;
  std::vector<std::string> updated_doubleAlphaName;
  std::vector<std::string> current_doubleBackStressName;
  std::vector<std::string> updated_doubleBackStressName;
  
  typedef PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> HMC2Tensor;
  typedef PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim,Dim> HMC3Tensor;

};
}

#endif
