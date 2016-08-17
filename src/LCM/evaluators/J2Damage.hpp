//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef J2DAMAGE_HPP
#define J2DAMAGE_HPP

#include <Intrepid2_MiniTensor.h>
#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

namespace LCM {
  /** \brief J2Stress with damage dependent response

   This evaluator computes stress based on a uncoupled J2Stress
   potential

   */

  template<typename EvalT, typename Traits>
  class J2Damage: public PHX::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalT, Traits> {

  public:

    J2Damage(const Teuchos::ParameterList& p);

    void postRegistrationSetup(typename Traits::SetupData d,
        PHX::FieldManager<Traits>& vm);

    void evaluateFields(typename Traits::EvalData d);

  private:

    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;

    // Input:
    PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> defgrad;
    PHX::MDField<ScalarT, Cell, QuadPoint> J;
    PHX::MDField<ScalarT, Cell, QuadPoint> bulkModulus;
    PHX::MDField<ScalarT, Cell, QuadPoint> shearModulus;
    PHX::MDField<ScalarT, Cell, QuadPoint> yieldStrength;
    PHX::MDField<ScalarT, Cell, QuadPoint> hardeningModulus;
    PHX::MDField<ScalarT, Cell, QuadPoint> satMod;
    PHX::MDField<ScalarT, Cell, QuadPoint> satExp;
    PHX::MDField<ScalarT, Cell, QuadPoint> damage;

    // Output:
    PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> stress;
    PHX::MDField<ScalarT, Cell, QuadPoint> dp;
    PHX::MDField<ScalarT, Cell, QuadPoint> seff;
    PHX::MDField<ScalarT, Cell, QuadPoint> energy;
    PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> Fp;
    PHX::MDField<ScalarT, Cell, QuadPoint> eqps;

    std::string fpName, eqpsName;
    unsigned int numQPs;
    unsigned int numDims;
    unsigned int worksetSize;

    Kokkos::DynRankView<ScalarT, PHX::Device> Fpinv;
    Kokkos::DynRankView<ScalarT, PHX::Device> FpinvT;
    Kokkos::DynRankView<ScalarT, PHX::Device> Cpinv;

  };
}

#endif
