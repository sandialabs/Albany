//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef GURSONSDSTRESS_HPP
#define GURSONSDSTRESS_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Tensor.h"
#include "Sacado.hpp"

namespace LCM {
  /** \brief Gurson small deformation stress response

   This evaluator computes stress based on modified Gurson model.

   */

  template<typename EvalT, typename Traits>
  class GursonSDStress: public PHX::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalT, Traits> {

  public:

    GursonSDStress(const Teuchos::ParameterList& p);

    void postRegistrationSetup(typename Traits::SetupData d,
        PHX::FieldManager<Traits>& vm);

    void evaluateFields(typename Traits::EvalData d);

  private:

    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;

    // all local functions used in computing GursonSD model stress:
    ScalarT compute_Y(ScalarT & epVal, ScalarT & Eor3mu);
    ScalarT compute_Phi(LCM::Tensor<ScalarT> & devVal, ScalarT & pVal,
        ScalarT & fvoidVal, ScalarT & epVal, ScalarT & Eor3mu);
    ScalarT compute_dPhidep(ScalarT & tmp, ScalarT & pN, ScalarT & fvoidN,
        ScalarT & epN, ScalarT & YN, ScalarT & Eor3mu);
    ScalarT compute_dfdgam(ScalarT & depdgam, ScalarT & tmp, ScalarT & pN,
        ScalarT & sigeN, ScalarT & J3N, ScalarT & fvoidN, ScalarT & epN,
        ScalarT & YN);

    //Input
    PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> strain;
    PHX::MDField<ScalarT, Cell, QuadPoint> elasticModulus;
    PHX::MDField<ScalarT, Cell, QuadPoint> poissonsRatio;

    unsigned int numQPs;
    unsigned int numDims;

    double f0;
    double Y0;
    double kw;
    double N;
    double q1;
    double q2;
    double q3;
    double eN;
    double sN;
    double fN;
    double fc;
    double ff;
    double flag;

    std::string strainName, stressName;
    std::string voidVolumeName, epName, yieldStrengthName;

    //output
    PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> stress;
    PHX::MDField<ScalarT, Cell, QuadPoint> voidVolume;
    PHX::MDField<ScalarT, Cell, QuadPoint> ep;
    PHX::MDField<ScalarT, Cell, QuadPoint> yieldStrength;

  };
}

#endif

