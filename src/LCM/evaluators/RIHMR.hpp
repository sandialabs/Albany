//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef RIHMR_HPP
#define RIHMR_HPP

#include <Intrepid_MiniTensor.h>
#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

namespace LCM {
  /** \brief Rate-Independent Hardening Minus Recovery (RIHMR) Stress evaluator

   This evaluator computes stress based on a uncoupled J2Stress
   potential

   */

  template<typename EvalT, typename Traits>
  class RIHMR: public PHX::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalT, Traits> {

  public:

    RIHMR(const Teuchos::ParameterList& p);

    void postRegistrationSetup(typename Traits::SetupData d,
        PHX::FieldManager<Traits>& vm);

    void evaluateFields(typename Traits::EvalData d);

  private:

    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;
    typedef typename Sacado::Fad::DFad<ScalarT> DFadType;

    void
    compute_ResidJacobian(std::vector<ScalarT> & X, std::vector<ScalarT> & R,
        std::vector<ScalarT> & dRdX, const ScalarT & es, const ScalarT & smag,
        const ScalarT & mubar, ScalarT & mu, ScalarT & kappa, ScalarT & K,
        ScalarT & Y, ScalarT & Rd);

    // Input:
    PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> defgrad;
    PHX::MDField<ScalarT, Cell, QuadPoint> J;
    PHX::MDField<ScalarT, Cell, QuadPoint> elasticModulus;
    PHX::MDField<ScalarT, Cell, QuadPoint> poissonsRatio;
    PHX::MDField<ScalarT, Cell, QuadPoint> yieldStrength;
    PHX::MDField<ScalarT, Cell, QuadPoint> hardeningModulus;
    PHX::MDField<ScalarT, Cell, QuadPoint> recoveryModulus;

    // Output:
    PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> stress;
    PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> logFp;
    PHX::MDField<ScalarT, Cell, QuadPoint> eqps;
    PHX::MDField<ScalarT, Cell, QuadPoint> isoHardening;

    std::string logFpName, eqpsName, isoHardeningName;
    unsigned int numQPs;
    unsigned int numDims;

    //Intrepid::FieldContainer<ScalarT> Fpinv;
    //Intrepid::FieldContainer<ScalarT> FpinvT;
    //Intrepid::FieldContainer<ScalarT> Cpinv;

  };
}

#endif
