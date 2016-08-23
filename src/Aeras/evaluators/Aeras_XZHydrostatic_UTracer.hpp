//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AERAS_XZHYDROSTATIC_UTRACER_HPP
#define AERAS_XZHYDROSTATIC_UTRACER_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Aeras_Layouts.hpp"
#include "Aeras_Dimension.hpp"
#include "Sacado_ParameterAccessor.hpp"

namespace Aeras {
/** \brief kinetic energy for XZHydrostatic atmospheric model

    This evaluator computes the kinetic energy for the XZHydrostatic model
    of atmospheric dynamics.

*/

template<typename EvalT, typename Traits>
class XZHydrostatic_UTracer : public PHX::EvaluatorWithBaseImpl<Traits>,
                              public PHX::EvaluatorDerived<EvalT, Traits> {

public:
  typedef typename EvalT::ScalarT ScalarT;

  XZHydrostatic_UTracer(const Teuchos::ParameterList& p,
                        const Teuchos::RCP<Aeras::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);
private:
  // Input:
  PHX::MDField<ScalarT,Cell,Node,Level,Dim> Velocity;
  PHX::MDField<ScalarT,Cell,Node,Level,Dim> PiVelx;
  PHX::MDField<ScalarT,Cell,Node,Level>     Tracer;
  
  // Output:
  PHX::MDField<ScalarT,Cell,Node,Level,Dim> UTracer;

  const int numDims;
  const int numNodes;
  const int numLevels;

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
public:
  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;

  struct XZHydrostatic_UTracer_Tag{};

  typedef Kokkos::RangePolicy<ExecutionSpace, XZHydrostatic_UTracer_Tag> XZHydrostatic_UTracer_Policy;

  KOKKOS_INLINE_FUNCTION
  void operator() (const XZHydrostatic_UTracer_Tag& tag, const int& i) const;

#endif
};
}

#endif
