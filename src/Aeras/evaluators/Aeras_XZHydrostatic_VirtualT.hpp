//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AERAS_XZHYDROSTATIC_VIRTUALT_HPP
#define AERAS_XZHYDROSTATIC_VIRTUALT_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Aeras_Layouts.hpp"
#include "Aeras_Dimension.hpp"

namespace Aeras {
/** \brief Virtual Temperature for XZHydrostatic atmospheric model

    This evaluator computes the virtual temperature 
    for the XZHydrostatic model of atmospheric dynamics.
    Tv = T + (Rv/R -1)*qv*T

*/

template<typename EvalT, typename Traits>
class XZHydrostatic_VirtualT : public PHX::EvaluatorWithBaseImpl<Traits>,
                   public PHX::EvaluatorDerived<EvalT, Traits> {

public:
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  XZHydrostatic_VirtualT(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Aeras::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:
  // Input:
  PHX::MDField<ScalarT,Cell,Node,Level> temperature;
  PHX::MDField<ScalarT,Cell,Node,Level> Pi;
  PHX::MDField<ScalarT,Cell,Node,Level> qv;
  // Output:
  PHX::MDField<ScalarT,Cell,Node,Level> virt_t;
  PHX::MDField<ScalarT,Cell,Node,Level> Cpstar;

  const Teuchos::ArrayRCP<std::string> tracerNames;

  const int numNodes;
  const int numLevels;
  bool vapor;
  const double Cp;
  double Cpv;
  double Cvv;
  ScalarT R;
  ScalarT Rv;
  ScalarT factor;

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
public:
  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;
  using Iterate = Kokkos::Experimental::Iterate;
#if defined(PHX_KOKKOS_DEVICE_TYPE_CUDA)
  static constexpr Iterate IterateDirection = Iterate::Left;
#else
  static constexpr Iterate IterateDirection = Iterate::Right;
#endif

  struct XZHydrostatic_VirtualT_Tag{};
  struct XZHydrostatic_VirtualT_vapor_Tag{};

  using XZHydrostatic_VirtualT_Policy = Kokkos::Experimental::MDRangePolicy<
        Kokkos::Experimental::Rank<3, IterateDirection, IterateDirection>,
        Kokkos::IndexType<int>, XZHydrostatic_VirtualT_Tag>;
  using XZHydrostatic_VirtualT_vapor_Policy = Kokkos::Experimental::MDRangePolicy<
        Kokkos::Experimental::Rank<3, IterateDirection, IterateDirection>,
        Kokkos::IndexType<int>, XZHydrostatic_VirtualT_vapor_Tag>;

  KOKKOS_INLINE_FUNCTION
  void operator() (const XZHydrostatic_VirtualT_Tag& tag, const int cell, const int node, const int level) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (const XZHydrostatic_VirtualT_vapor_Tag& tag, const int cell, const int node, const int level) const;

#endif
};
}

#endif
