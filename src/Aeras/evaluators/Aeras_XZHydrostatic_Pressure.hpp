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
  const ScalarT P0, Ptop;

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  Kokkos::DynRankView<ScalarT, PHX::Device> A, B, delta;

public:
  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;
  using Iterate = Kokkos::Experimental::Iterate;
#if defined(PHX_KOKKOS_DEVICE_TYPE_CUDA)
  static constexpr Iterate IterateDirection = Iterate::Left;
#else
  static constexpr Iterate IterateDirection = Iterate::Right;
#endif

  struct XZHydrostatic_Pressure_Tag{};
  struct XZHydrostatic_Pressure_Pi_Tag{};

  using XZHydrostatic_Pressure_Policy = Kokkos::Experimental::MDRangePolicy<
        Kokkos::Experimental::Rank<3, IterateDirection, IterateDirection>,
        Kokkos::IndexType<int>, XZHydrostatic_Pressure_Tag>;

  using XZHydrostatic_Pressure_Pi_Policy = Kokkos::Experimental::MDRangePolicy<
        Kokkos::Experimental::Rank<3, IterateDirection, IterateDirection>,
        Kokkos::IndexType<int>, XZHydrostatic_Pressure_Pi_Tag>;

#if defined(PHX_KOKKOS_DEVICE_TYPE_CUDA)
  typename XZHydrostatic_Pressure_Policy::tile_type 
    XZHydrostatic_Pressure_TileSize{{256,1,1}};
  typename XZHydrostatic_Pressure_Pi_Policy::tile_type 
    XZHydrostatic_Pressure_Pi_TileSize{{256,1,1}};
#else
  typename XZHydrostatic_Pressure_Policy::tile_type 
    XZHydrostatic_Pressure_TileSize{};
  typename XZHydrostatic_Pressure_Pi_Policy::tile_type 
    XZHydrostatic_Pressure_Pi_TileSize{};
#endif

  KOKKOS_INLINE_FUNCTION
  void operator() (const XZHydrostatic_Pressure_Tag& tag, const int cell, const int node, const int level) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (const XZHydrostatic_Pressure_Pi_Tag& tag, const int cell, const int node, const int level) const;

#endif
};
}

#endif
