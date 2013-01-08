//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef CAPEXPLICIT_HPP
#define CAPEXPLICIT_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "VectorTensorBase.h"

namespace LCM {
  /** \brief CapExplicit stress response

   This evaluator computes stress based on a cap plasticity model.

   */

  template<typename EvalT, typename Traits>
  class CapExplicit: public PHX::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalT, Traits> {

  public:

    CapExplicit(const Teuchos::ParameterList& p);

    void postRegistrationSetup(typename Traits::SetupData d,
        PHX::FieldManager<Traits>& vm);

    void evaluateFields(typename Traits::EvalData d);

  private:

    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;

    // all local functions used in computing cap model stress:
    ScalarT compute_f(LCM::Tensor<ScalarT> & sigma,
        LCM::Tensor<ScalarT> & alpha, ScalarT & kappa);

    LCM::Tensor<ScalarT> compute_dfdsigma(LCM::Tensor<ScalarT> & sigma,
        LCM::Tensor<ScalarT> & alpha, ScalarT & kappa);

    LCM::Tensor<ScalarT> compute_dgdsigma(LCM::Tensor<ScalarT> & sigma,
        LCM::Tensor<ScalarT> & alpha, ScalarT & kappa);

    ScalarT compute_dfdkappa(LCM::Tensor<ScalarT> & sigma,
        LCM::Tensor<ScalarT> & alpha, ScalarT & kappa);

    ScalarT compute_Galpha(ScalarT & J2_alpha);

    LCM::Tensor<ScalarT> compute_halpha(LCM::Tensor<ScalarT> & dgdsigma,
        ScalarT & J2_alpha);

    ScalarT compute_dedkappa(ScalarT & kappa);

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

