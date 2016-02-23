//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef THERMO_MECHANICAL_STRESS_HPP
#define THERMO_MECHANICAL_STRESS_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

namespace LCM {
  /** \brief ThermoMechanical stress response

   This evaluator computes stress based on a uncoupled Neohookean
   Helmholtz potential with temperature dependence

   */

  template<typename EvalT, typename Traits>
  class ThermoMechanicalStress: public PHX::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalT, Traits> {

  public:

    ThermoMechanicalStress(const Teuchos::ParameterList& p);

    void postRegistrationSetup(typename Traits::SetupData d,
        PHX::FieldManager<Traits>& vm);

    void evaluateFields(typename Traits::EvalData d);

  private:

    typedef typename EvalT::ScalarT ScalarT;

    // Input:
    PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> F_array;
    PHX::MDField<ScalarT, Cell, QuadPoint> J_array;
    PHX::MDField<ScalarT, Cell, QuadPoint> shearModulus;
    PHX::MDField<ScalarT, Cell, QuadPoint> bulkModulus;
    PHX::MDField<ScalarT, Cell, QuadPoint> temperature;
    PHX::MDField<ScalarT, Cell, QuadPoint> yieldStrength;
    PHX::MDField<ScalarT, Cell, QuadPoint> hardeningModulus;
    PHX::MDField<ScalarT, Cell, QuadPoint> satMod;
    PHX::MDField<ScalarT, Cell, QuadPoint> satExp;
    PHX::MDField<ScalarT, Dummy> deltaTime;

    // Output:
    PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> stress;
    PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> Fp;
    PHX::MDField<ScalarT, Cell, QuadPoint> eqps;
    PHX::MDField<ScalarT, Cell, QuadPoint> mechSource;

    std::string fpName, eqpsName;
    unsigned int numQPs;
    unsigned int numDims;
    RealType thermalExpansionCoeff;
    RealType refTemperature;

  };
}

#endif
