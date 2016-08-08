//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AERAS_XZHYDROSTATIC_PRESSURE_HPP
#define AERAS_XZHYDROSTATIC_PRESSURE_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Aeras_Layouts.hpp"
#include "Aeras_Dimension.hpp"
#include "Aeras_Eta.hpp"

namespace Aeras {
/** \brief Pressure for XZHydrostatic atmospheric model

    This evaluator computes the Pressure for the XZHydrostatic model
    of atmospheric dynamics.

*/
template<typename EvalT, typename Traits>
class XZHydrostatic_Pressure : public PHX::EvaluatorWithBaseImpl<Traits>,
                   public PHX::EvaluatorDerived<EvalT, Traits> {

public:
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  XZHydrostatic_Pressure(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Aeras::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:
  // Input
  PHX::MDField<ScalarT,Cell,Node>       Ps;

  // Output:
  PHX::MDField<ScalarT,Cell,Node,Level> Pressure;
  PHX::MDField<ScalarT,Cell,Node,Level> Pi;

  const int numNodes;
  const int numLevels;
  const Eta<EvalT> &E;

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
public:
  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;

  struct XZHydrostatic_Pressure_Tag{};
  struct XZHydrostatic_Pressure_Pi_Tag{};

  typedef Kokkos::RangePolicy<ExecutionSpace, XZHydrostatic_Pressure_Tag> XZHydrostatic_Pressure_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, XZHydrostatic_Pressure_Pi_Tag> XZHydrostatic_Pressure_Pi_Policy;

  KOKKOS_INLINE_FUNCTION
  void operator() (const XZHydrostatic_Pressure_Tag& tag, const int& i) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (const XZHydrostatic_Pressure_Pi_Tag& tag, const int& i) const;

#endif
};
}

#endif
