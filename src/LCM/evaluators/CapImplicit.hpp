//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef CAPIMPLICIT_HPP
#define CAPIMPLICIT_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Tensor.h"
#include "Sacado.hpp"

namespace LCM {
  /** \brief CapImplicit stress response

   This evaluator computes stress based on a cap plasticity model.

   */

  template<typename EvalT, typename Traits>
  class CapImplicit: public PHX::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalT, Traits> {

  public:

    CapImplicit(const Teuchos::ParameterList& p);

    void postRegistrationSetup(typename Traits::SetupData d,
        PHX::FieldManager<Traits>& vm);

    void evaluateFields(typename Traits::EvalData d);

  private:

    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;
    typedef typename Sacado::Fad::DFad<ScalarT> DFadType;
    typedef typename Sacado::Fad::DFad<DFadType> D2FadType;

    // all local functions used in computing cap model stress:
    ScalarT
    compute_f(LCM::Tensor<ScalarT> & sigma, LCM::Tensor<ScalarT> & alpha,
        ScalarT & kappa);

    std::vector<ScalarT>
    initialize(LCM::Tensor<ScalarT> & sigmaVal,
        LCM::Tensor<ScalarT> & alphaVal, ScalarT & kappaVal,
        ScalarT & dgammaVal);

    void
    compute_ResidJacobian(std::vector<ScalarT> const & XXVal,
        std::vector<ScalarT> & R, std::vector<ScalarT> & dRdX,
        const LCM::Tensor<ScalarT> & sigmaVal,
        const LCM::Tensor<ScalarT> & alphaVal, const ScalarT & kappaVal,
        LCM::Tensor4<ScalarT> const & Celastic, bool kappa_flag);

    DFadType
    compute_f(LCM::Tensor<DFadType> & sigma,
        LCM::Tensor<DFadType> & alpha, DFadType & kappa);

    D2FadType
    compute_g(LCM::Tensor<D2FadType> & sigma,
        LCM::Tensor<D2FadType> & alpha, D2FadType & kappa);

    LCM::Tensor<DFadType>
    compute_dgdsigma(std::vector<DFadType> const & XX);

    DFadType
    compute_Galpha(DFadType J2_alpha);

    LCM::Tensor<DFadType>
    compute_halpha(LCM::Tensor<DFadType> const & dgdsigma,
        DFadType const J2_alpha);

    DFadType
    compute_dedkappa(DFadType const kappa);

    DFadType
    compute_hkappa(DFadType const I1_dgdsigma, DFadType const dedkappa);

    //Input
    PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> strain;
    PHX::MDField<ScalarT, Cell, QuadPoint> elasticModulus;
    PHX::MDField<ScalarT, Cell, QuadPoint> poissonsRatio;

    unsigned int numQPs;
    unsigned int numDims;

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
    std::string backStressName, capParameterName, eqpsName;

    //output
    PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> stress;
    PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> backStress;
    PHX::MDField<ScalarT, Cell, QuadPoint> capParameter;
    PHX::MDField<ScalarT, Cell, QuadPoint> friction;
    PHX::MDField<ScalarT, Cell, QuadPoint> dilatancy;
    PHX::MDField<ScalarT, Cell, QuadPoint> eqps;
    PHX::MDField<ScalarT, Cell, QuadPoint> evolps;
    PHX::MDField<ScalarT, Cell, QuadPoint> hardeningModulus;

  };
}

#endif

