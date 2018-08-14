//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef MIXTURE_SPECIFIC_HEAT_HPP
#define MIXTURE_SPECIFIC_HEAT_HPP

#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

namespace LCM {
/** \brief

    This evaluator calculates thermal expansion of a bi-phase
    mixture through volume averaging


*/

template <typename EvalT, typename Traits>
class MixtureSpecificHeat : public PHX::EvaluatorWithBaseImpl<Traits>,
                            public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  MixtureSpecificHeat(const Teuchos::ParameterList& p);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  void
  evaluateFields(typename Traits::EvalData d);

 private:
  typedef typename EvalT::ScalarT     ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<const ScalarT, Cell, QuadPoint> porosity;
  PHX::MDField<const ScalarT, Cell, QuadPoint> gammaSkeleton;
  PHX::MDField<const ScalarT, Cell, QuadPoint> gammaPoreFluid;
  PHX::MDField<const ScalarT, Cell, QuadPoint> densitySkeleton;
  PHX::MDField<const ScalarT, Cell, QuadPoint> densityPoreFluid;
  PHX::MDField<const ScalarT, Cell, QuadPoint> J;

  // Output:
  PHX::MDField<ScalarT, Cell, QuadPoint> mixtureSpecificHeat;

  unsigned int numQPs;
  //  unsigned int numDims;
};
}  // namespace LCM

#endif
