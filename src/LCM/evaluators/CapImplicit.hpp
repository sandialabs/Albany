//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef CAPIMPLICIT_HPP
#define CAPIMPLICIT_HPP

#include <Intrepid2_MiniTensor.h>
#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

#include "Sacado.hpp"

namespace LCM {
  /** \brief CapImplicit stress response

   This evaluator computes stress based on a cap plasticity model.

   */

  template<typename EvalT, typename Traits>
  class CapImplicit: public PHX::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalT, Traits> {

  public:

    CapImplicit(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Albany::Layouts>& dl);

    void postRegistrationSetup(typename Traits::SetupData d,
        PHX::FieldManager<Traits>& vm);

    void evaluateFields(typename Traits::EvalData d);

  private:

    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;
    // typedef typename Sacado::Fad::DFad<ScalarT> DFadType;
    // typedef typename Sacado::Fad::DFad<DFadType> D2FadType;
    typedef typename Sacado::mpl::apply<FadType,ScalarT>::type DFadType;
    typedef typename Sacado::mpl::apply<FadType,DFadType>::type D2FadType;

    // all local functions used in computing cap model stress:
    ScalarT
    compute_f(Intrepid2::Tensor<ScalarT> & sigma,
        Intrepid2::Tensor<ScalarT> & alpha,
        ScalarT & kappa);

    std::vector<ScalarT>
    initialize(Intrepid2::Tensor<ScalarT> & sigmaVal,
        Intrepid2::Tensor<ScalarT> & alphaVal, ScalarT & kappaVal,
        ScalarT & dgammaVal);

    void
    compute_ResidJacobian(std::vector<ScalarT> const & XXVal,
        std::vector<ScalarT> & R, std::vector<ScalarT> & dRdX,
        const Intrepid2::Tensor<ScalarT> & sigmaVal,
        const Intrepid2::Tensor<ScalarT> & alphaVal, const ScalarT & kappaVal,
        Intrepid2::Tensor4<ScalarT> const & Celastic, bool kappa_flag);

    DFadType
    compute_f(Intrepid2::Tensor<DFadType> & sigma,
        Intrepid2::Tensor<DFadType> & alpha, DFadType & kappa);

    D2FadType
    compute_g(Intrepid2::Tensor<D2FadType> & sigma,
        Intrepid2::Tensor<D2FadType> & alpha, D2FadType & kappa);

    Intrepid2::Tensor<DFadType>
    compute_dgdsigma(std::vector<DFadType> const & XX);

    DFadType
    compute_Galpha(DFadType J2_alpha);

    Intrepid2::Tensor<DFadType>
    compute_halpha(Intrepid2::Tensor<DFadType> const & dgdsigma,
        DFadType const J2_alpha);

    DFadType
    compute_dedkappa(DFadType const kappa);

    DFadType
    compute_hkappa(DFadType const I1_dgdsigma, DFadType const dedkappa);

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

  };
}

#endif

