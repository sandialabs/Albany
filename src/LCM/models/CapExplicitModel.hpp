//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#if !defined(CapExplicitModel_hpp)
#define CapExplicitModel_hpp

#include <MiniTensor.h>
#include "Albany_Layouts.hpp"
#include "ConstitutiveModel.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

namespace LCM {
/// \brief CapExplicit stress response
///
/// This evaluator computes stress based on a cap plasticity model.
///

template <typename EvalT, typename Traits>
class CapExplicitModel : public LCM::ConstitutiveModel<EvalT, Traits>
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

  ///
  /// Constructor
  ///
  CapExplicitModel(
      Teuchos::ParameterList*              p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  ///
  /// Virtual Destructor
  ///
  virtual ~CapExplicitModel(){};

  ///
  /// Implementation of physics
  ///
  virtual void
  computeState(
      typename Traits::EvalData workset,
      DepFieldMap               dep_fields,
      FieldMap                  eval_fields);

  virtual void
  computeStateParallel(
      typename Traits::EvalData workset,
      DepFieldMap               dep_fields,
      FieldMap                  eval_fields)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Not implemented.");
  }

 private:
  ///
  /// Private to prohibit copying
  ///
  CapExplicitModel(const CapExplicitModel&);

  ///
  /// Private to prohibit copying
  ///
  CapExplicitModel&
  operator=(const CapExplicitModel&);

  ///
  /// functions for integrating cap model stress
  ///
  ScalarT
  compute_f(
      minitensor::Tensor<ScalarT>& sigma,
      minitensor::Tensor<ScalarT>& alpha,
      ScalarT&                     kappa);

  minitensor::Tensor<ScalarT>
  compute_dfdsigma(
      minitensor::Tensor<ScalarT>& sigma,
      minitensor::Tensor<ScalarT>& alpha,
      ScalarT&                     kappa);

  minitensor::Tensor<ScalarT>
  compute_dgdsigma(
      minitensor::Tensor<ScalarT>& sigma,
      minitensor::Tensor<ScalarT>& alpha,
      ScalarT&                     kappa);

  ScalarT
  compute_dfdkappa(
      minitensor::Tensor<ScalarT>& sigma,
      minitensor::Tensor<ScalarT>& alpha,
      ScalarT&                     kappa);

  ScalarT
  compute_Galpha(ScalarT& J2_alpha);

  minitensor::Tensor<ScalarT>
  compute_halpha(minitensor::Tensor<ScalarT>& dgdsigma, ScalarT& J2_alpha);

  ScalarT
  compute_dedkappa(ScalarT& kappa);

  ///
  /// constant material parameters in Cap plasticity model
  ///
  RealType A;
  RealType B;
  RealType C;
  RealType theta;
  RealType R;
  RealType kappa0;
  RealType W;
  RealType D1;
  RealType D2;
  RealType calpha;
  RealType psi;
  RealType N;
  RealType L;
  RealType phi;
  RealType Q;

  std::string strainName, stressName;
  std::string backStressName, capParameterName, eqpsName, volPlasticStrainName;

  ///
  /// Tensors for local computations
  ///
  minitensor::Tensor4<ScalarT> Celastic, compliance, id1, id2, id3;
  minitensor::Tensor<ScalarT>  I;
  minitensor::Tensor<ScalarT>  depsilon, sigmaN, strainN, sigmaVal, alphaVal;
  minitensor::Tensor<ScalarT>  deps_plastic, sigmaTr, alphaTr;
  minitensor::Tensor<ScalarT>  dfdsigma, dgdsigma, dfdalpha, halpha;
  minitensor::Tensor<ScalarT>  dfdotCe, sigmaK, alphaK, dsigma, dev_plastic;
  minitensor::Tensor<ScalarT>  xi, sN, s, strainCurrent;
  minitensor::Tensor<ScalarT>  dJ3dsigma, eps_dev;
};
}  // namespace LCM

#endif
