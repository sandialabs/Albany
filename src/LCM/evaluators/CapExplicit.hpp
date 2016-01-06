//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef CAPEXPLICIT_HPP
#define CAPEXPLICIT_HPP

#include <Intrepid2_MiniTensor.h>
#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace LCM {
  /// \brief CapExplicit stress response
  ///
  /// This evaluator computes stress based on a cap plasticity model.
  ///

  template<typename EvalT, typename Traits>
  class CapExplicit: public PHX::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalT, Traits> {

  public:

    ///
    /// Constructor
    ///
    CapExplicit(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Albany::Layouts>& dl);

    ///
    /// Phalanx method to allocate space
    ///
    void postRegistrationSetup(typename Traits::SetupData d,
        PHX::FieldManager<Traits>& vm);

    ///
    /// Implementation of physics
    ///
    void evaluateFields(typename Traits::EvalData d);

  private:

    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;

    ///
    /// functions for integrating cap model stress
    ///
    ScalarT
    compute_f(Intrepid2::Tensor<ScalarT> & sigma,
        Intrepid2::Tensor<ScalarT> & alpha, ScalarT & kappa);

    Intrepid2::Tensor<ScalarT>
    compute_dfdsigma(Intrepid2::Tensor<ScalarT> & sigma,
        Intrepid2::Tensor<ScalarT> & alpha, ScalarT & kappa);

    Intrepid2::Tensor<ScalarT>
    compute_dgdsigma(Intrepid2::Tensor<ScalarT> & sigma,
        Intrepid2::Tensor<ScalarT> & alpha, ScalarT & kappa);

    ScalarT
    compute_dfdkappa(Intrepid2::Tensor<ScalarT> & sigma,
        Intrepid2::Tensor<ScalarT> & alpha, ScalarT & kappa);

    ScalarT
    compute_Galpha(ScalarT & J2_alpha);

    Intrepid2::Tensor<ScalarT>
    compute_halpha(Intrepid2::Tensor<ScalarT> & dgdsigma, ScalarT & J2_alpha);

    ScalarT compute_dedkappa(ScalarT & kappa);

    ///
    /// number of integration points
    ///
    unsigned int numQPs;

    ///
    /// number of global dimensions
    ///
    unsigned int numDims;

    ///
    /// Input: small strain
    ///
    PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> strain;

    ///
    /// Input: Young's Modulus
    ///
    PHX::MDField<ScalarT, Cell, QuadPoint> elasticModulus;

    ///
    /// Input: Poisson's Ratio
    ///
    PHX::MDField<ScalarT, Cell, QuadPoint> poissonsRatio;

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
    std::string backStressName, capParameterName, eqpsName,volPlasticStrainName;

    ///
    /// Output: Cauchy stress
    ///
    PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> stress;

    ///
    /// Output: kinematic hardening backstress
    ///
    PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> backStress;

    ///
    /// Output: isotropic hardening cap size
    ///
    PHX::MDField<ScalarT, Cell, QuadPoint> capParameter;

    ///
    /// Output: friction coefficient
    ///
    //PHX::MDField<ScalarT, Cell, QuadPoint> friction;

    ///
    /// Output: dilatancy parameter
    ///
    //PHX::MDField<ScalarT, Cell, QuadPoint> dilatancy;

    ///
    /// Output: equivalent plastic strain
    ///
    PHX::MDField<ScalarT, Cell, QuadPoint> eqps;

    ///
    /// Output: volumetric plastic strain
    ///
    PHX::MDField<ScalarT, Cell, QuadPoint> volPlasticStrain;

    ///
    /// Output: generalized plastic hardening modulus
    ///
    //PHX::MDField<ScalarT, Cell, QuadPoint> hardeningModulus;

    ///
    /// Tensors for local computations
    ///
    Intrepid2::Tensor4<ScalarT> Celastic, compliance, id1, id2, id3;
    Intrepid2::Tensor<ScalarT> I;
    Intrepid2::Tensor<ScalarT> depsilon, sigmaN, strainN, sigmaVal, alphaVal;
    Intrepid2::Tensor<ScalarT> deps_plastic, sigmaTr, alphaTr;
    Intrepid2::Tensor<ScalarT> dfdsigma, dgdsigma, dfdalpha, halpha;
    Intrepid2::Tensor<ScalarT> dfdotCe, sigmaK, alphaK, dsigma, dev_plastic;
    Intrepid2::Tensor<ScalarT> xi, sN, s, strainCurrent;
    Intrepid2::Tensor<ScalarT> dJ3dsigma, eps_dev;


  };
}

#endif

