//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#if !defined(CapImplicitModel_hpp)
#define CapImplicitModel_hpp

#include <MiniTensor.h>
#include "Albany_Layouts.hpp"
#include "LCM/models/ConstitutiveModel.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

#include "Sacado.hpp"

namespace LCM {
/// \brief CapImplicit stress response
///
/// This evaluator computes stress based on a cap plasticity model.
///

template <typename EvalT, typename Traits>
class CapImplicitModel : public LCM::ConstitutiveModel<EvalT, Traits>
{
 public:
  using Base        = LCM::ConstitutiveModel<EvalT, Traits>;
  using DepFieldMap = typename Base::DepFieldMap;
  using FieldMap    = typename Base::FieldMap;

  typedef typename EvalT::ScalarT                              ScalarT;
  typedef typename EvalT::MeshScalarT                          MeshScalarT;
  typedef typename Sacado::mpl::apply<FadType, ScalarT>::type  DFadType;
  typedef typename Sacado::mpl::apply<FadType, DFadType>::type D2FadType;

  using ConstitutiveModel<EvalT, Traits>::num_dims_;
  using ConstitutiveModel<EvalT, Traits>::num_pts_;
  using ConstitutiveModel<EvalT, Traits>::field_name_map_;

  // optional material tangent computation
  using ConstitutiveModel<EvalT, Traits>::compute_tangent_;

  ///
  /// Constructor
  ///
  CapImplicitModel(
      Teuchos::ParameterList*              p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  ///
  /// Virtual Destructor
  ///
  virtual ~CapImplicitModel(){};

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
  CapImplicitModel(const CapImplicitModel&);

  ///
  /// Private to prohibit copying
  ///
  CapImplicitModel&
  operator=(const CapImplicitModel&);

  // all local functions used in computing cap model stress:

  // yield function
  template <typename T>
  T
  compute_f(
      minitensor::Tensor<T>& sigma,
      minitensor::Tensor<T>& alpha,
      T&                     kappa);

  // unknow variable value list
  std::vector<ScalarT>
  initialize(
      minitensor::Tensor<ScalarT>& sigmaVal,
      minitensor::Tensor<ScalarT>& alphaVal,
      ScalarT&                     kappaVal,
      ScalarT&                     dgammaVal);

  // local iteration jacobian
  void
  compute_ResidJacobian(
      std::vector<ScalarT> const&         XXVal,
      std::vector<ScalarT>&               R,
      std::vector<ScalarT>&               dRdX,
      const minitensor::Tensor<ScalarT>&  sigmaVal,
      const minitensor::Tensor<ScalarT>&  alphaVal,
      const ScalarT&                      kappaVal,
      minitensor::Tensor4<ScalarT> const& Celastic,
      bool                                kappa_flag);

  // plastic potential
  template <typename T>
  T
  compute_g(
      minitensor::Tensor<T>& sigma,
      minitensor::Tensor<T>& alpha,
      T&                     kappa);

  // derivative
  minitensor::Tensor<ScalarT>
  compute_dfdsigma(std::vector<ScalarT> const& XX);

  ScalarT
  compute_dfdkappa(std::vector<ScalarT> const& XX);

  minitensor::Tensor<ScalarT>
  compute_dgdsigma(std::vector<ScalarT> const& XX);

  minitensor::Tensor<DFadType>
  compute_dgdsigma(std::vector<DFadType> const& XX);

  // hardening functions
  template <typename T>
  T
  compute_Galpha(T J2_alpha);

  template <typename T>
  minitensor::Tensor<T>
  compute_halpha(minitensor::Tensor<T> const& dgdsigma, T const J2_alpha);

  template <typename T>
  T
  compute_dedkappa(T const kappa);

  template <typename T>
  T
  compute_hkappa(T const I1_dgdsigma, T const dedkappa);

  // elasto-plastic tangent modulus
  minitensor::Tensor4<ScalarT>
  compute_Cep(
      minitensor::Tensor4<ScalarT>& Celastic,
      minitensor::Tensor<ScalarT>&  sigma,
      minitensor::Tensor<ScalarT>&  alpha,
      ScalarT&                      kappa,
      ScalarT&                      dgamma);
  minitensor::Tensor4<ScalarT>
  compute_Cepp(
      minitensor::Tensor4<ScalarT>& Celastic,
      minitensor::Tensor<ScalarT>&  sigma,
      minitensor::Tensor<ScalarT>&  alpha,
      ScalarT&                      kappa,
      ScalarT&                      dgamma);

  // local temp variables
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
};
}  // namespace LCM

#endif
