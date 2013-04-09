//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef GURSONHMR_HPP
#define GURSONHMR_HPP

#include <Intrepid_MiniTensor.h>
#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Sacado.hpp"

namespace LCM {
  /** Large Deformation Gurson Model with Hardening Minus Recovery (HMR) law
   *
   * This evaluator compute stress based on Gurson damage model
   *
   */

  template<typename EvalT, typename Traits>
  class GursonHMR: public PHX::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalT, Traits> {

  public:

    GursonHMR(const Teuchos::ParameterList& p);

    void postRegistrationSetup(typename Traits::SetupData d,
        PHX::FieldManager<Traits>& vm);

    void evaluateFields(typename Traits::EvalData d);

  private:

    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;
    typedef typename Sacado::mpl::apply<FadType,ScalarT>::type DFadType;

    // all local functions used in computing GursonFD model stress:
    ScalarT
    compute_Phi(Intrepid::Tensor<ScalarT> & s, ScalarT & p, ScalarT & fvoid,
        ScalarT & Y, ScalarT & isoH, ScalarT & Jacobian);

    void
    compute_ResidJacobian(std::vector<ScalarT> & X, std::vector<ScalarT> & R,
        std::vector<ScalarT> & dRdX, const ScalarT & p, const ScalarT & fvoid,
        const ScalarT & es, Intrepid::Tensor<ScalarT> & s, ScalarT & mu,
        ScalarT & kappa, ScalarT & H, ScalarT & Y, ScalarT & Rd,
        ScalarT & Jacobian);

    //input
    PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> defgrad;
    PHX::MDField<ScalarT, Cell, QuadPoint> J;
    PHX::MDField<ScalarT, Cell, QuadPoint> elasticModulus;
    PHX::MDField<ScalarT, Cell, QuadPoint> poissonsRatio;
    PHX::MDField<ScalarT, Cell, QuadPoint> yieldStrength;
    PHX::MDField<ScalarT, Cell, QuadPoint> hardeningModulus;
    PHX::MDField<ScalarT, Cell, QuadPoint> recoveryModulus;

    std::string fpName, essName, eqpsName, voidVolumeName, isoHardeningName;
    unsigned int numQPs;
    unsigned int numDims;
    unsigned int worksetSize;

    // constant parameters
    RealType f0;
    RealType kw;
    RealType eN;
    RealType sN;
    RealType fN;
    RealType fc;
    RealType ff;
    RealType q1;
    RealType q2;
    RealType q3;

    //output
    PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> stress;
    PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> Fp;
    PHX::MDField<ScalarT, Cell, QuadPoint> ess;
    PHX::MDField<ScalarT, Cell, QuadPoint> eqps;
    PHX::MDField<ScalarT, Cell, QuadPoint> voidVolume;
    PHX::MDField<ScalarT, Cell, QuadPoint> isoHardening;


    Intrepid::FieldContainer<ScalarT> Fpinv;
    Intrepid::FieldContainer<ScalarT> FpinvT;
    Intrepid::FieldContainer<ScalarT> Cpinv;

  };
}

#endif /* GURSONHMR_HPP_ */
