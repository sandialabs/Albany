//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PISDWDF_HPP
#define PISDWDF_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Sacado_Fad_SLFad.hpp"

namespace LCM {
/** \brief Nonlinear Elasticity Energy Potential

    This evaluator computes a energy density for nonlinear elastic material

*/

template<typename EvalT, typename Traits>
class PisdWdF : public PHX::EvaluatorWithBaseImpl<Traits>,
		public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  PisdWdF(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Stuff needed for AD of energy functional
  //typedef typename Sacado::Fad::SLFad<ScalarT, 9> EnergyFadType;
  typedef typename Sacado::Fad::SLFad<ScalarT, 9> EnergyFadType;
  EnergyFadType computeEnergy(ScalarT& kappa, ScalarT& mu,
                          Kokkos::DynRankView<EnergyFadType, PHX::Device>& W);


  // Input:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> defgrad;
  PHX::MDField<ScalarT,Cell,QuadPoint> elasticModulus;
  PHX::MDField<ScalarT,Cell,QuadPoint> poissonsRatio;

  unsigned int numQPs;
  unsigned int numDims;

  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> P;
};
}

#endif
