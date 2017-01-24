//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AERAS_XZHYDROSTATIC_OMEGA_HPP
#define AERAS_XZHYDROSTATIC_OMEGA_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Aeras_Layouts.hpp"
#include "Aeras_Dimension.hpp"
#include "Aeras_Eta.hpp"

namespace Aeras {
/** \brief Density for XZHydrostatic atmospheric model

    This evaluator computes the density for the XZHydrostatic model
    of atmospheric dynamics.

*/

template<typename EvalT, typename Traits>
class XZHydrostatic_Omega : public PHX::EvaluatorWithBaseImpl<Traits>,
                   public PHX::EvaluatorDerived<EvalT, Traits> {

public:
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  XZHydrostatic_Omega(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Aeras::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:
  // Input:
  PHX::MDField<ScalarT,Cell,Node,Level,Dim> Velocity;
  PHX::MDField<ScalarT,Cell,Node,Level>     density;
  PHX::MDField<ScalarT,Cell,Node,Level>     Cpstar;
  PHX::MDField<ScalarT,Cell,QuadPoint,Level,Dim> gradp;
  PHX::MDField<ScalarT,Cell,QuadPoint,Level>     divpivelx;

  // Output:
  PHX::MDField<ScalarT,Cell,Node,Level>      omega;

  const int numQPs;
  const int numDims;
  const int numLevels;
  const ScalarT Cp;
  const Eta<EvalT> &E;

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  Kokkos::DynRankView<ScalarT, PHX::Device> delta;

public:
  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;

  struct XZHydrostatic_Omega_Tag{};

  typedef Kokkos::RangePolicy<ExecutionSpace, XZHydrostatic_Omega_Tag> XZHydrostatic_Omega_Policy;

  KOKKOS_INLINE_FUNCTION
  void operator() (const XZHydrostatic_Omega_Tag& tag, const int& i) const;

#endif
};
}

#endif
