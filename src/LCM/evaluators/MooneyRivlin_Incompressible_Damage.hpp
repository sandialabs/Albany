/*
 * MooneyRivlin_Incompressible_Damage.hpp
 *
 *  Created on: Aug 10, 2012
 *      Author: jrthune
 */

#ifndef MOONEYRIVLIN_INCOMPRESSIBLE_DAMAGE_HPP
#define MOONEYRIVLIN_INCOMPRESSIBLE_DAMAGE_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

namespace LCM {
/** \brief Nearly incompressible Mooney-Rivlin stress response with damage

    This evaluator computes the Cauchy stress based on a decoupled
    Helmholtz potential. Same as the incompressible Mooney-Rivlin
    material model but with a scalar damage parameter added.

    Based on "Nonlinear Solid Mechanics," Holzapfel and "Nonlinear Continuum
    Mechanics for Finite Element Analysis," Bonet and Wood.

 */

template<typename EvalT, typename Traits>
class MooneyRivlin_Incompressible_Damage : public PHX::EvaluatorWithBaseImpl<Traits>,
        public PHX::EvaluatorDerived<EvalT,Traits> {

public:
  MooneyRivlin_Incompressible_Damage(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
          PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> F;
  PHX::MDField<ScalarT,Cell,QuadPoint> J;
  RealType c1;
  RealType c2;
  RealType mult;

  // Damage parameters
  RealType zeta_inf;
  RealType iota;

  std::string alphaName;

  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> stress; // Cauchy stress
  PHX::MDField<ScalarT,Cell,QuadPoint> alpha;

  unsigned int numQPs;
  unsigned int numDims;
  unsigned int worksetSize;

  // scratch space FCs
  Intrepid::FieldContainer<ScalarT> FT;

};
}

#endif /* MOONEYRIVLIN_INCOMPRESSIBLE_DAMAGE_HPP */
