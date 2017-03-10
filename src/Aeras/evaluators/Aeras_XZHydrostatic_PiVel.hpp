//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AERAS_XZHYDROSTATIC_PIVEL_HPP
#define AERAS_XZHYDROSTATIC_PIVEL_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Aeras_Layouts.hpp"
#include "Aeras_Dimension.hpp"

namespace Aeras {
/** \brief Density for XZHydrostatic atmospheric model

    This evaluator computes the density for the XZHydrostatic model
    of atmospheric dynamics.

*/

template<typename EvalT, typename Traits>
class XZHydrostatic_PiVel : public PHX::EvaluatorWithBaseImpl<Traits>,
                   public PHX::EvaluatorDerived<EvalT, Traits> {

public:
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  XZHydrostatic_PiVel(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Aeras::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:
  // Output:
  PHX::MDField<ScalarT,Cell,Node,Level>     pi;
  PHX::MDField<ScalarT,Cell,Node,Level,Dim> velocity;
  PHX::MDField<ScalarT,Cell,Node,Level,Dim> pivelx;

  const int numNodes;
  const int numDims;
  const int numLevels;

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
public:
  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;
  using Iterate = Kokkos::Experimental::Iterate;
#if defined(PHX_KOKKOS_DEVICE_TYPE_CUDA)
  static constexpr Iterate IterateDirection = Iterate::Left;
#else
  static constexpr Iterate IterateDirection = Iterate::Right;
#endif

  struct XZHydrostatic_PiVel_Tag{};

  using XZHydrostatic_PiVel_Policy = Kokkos::Experimental::MDRangePolicy<
    Kokkos::Experimental::Rank<3, IterateDirection, IterateDirection>, 
    Kokkos::IndexType<int>, XZHydrostatic_PiVel_Tag>;

#if defined(PHX_KOKKOS_DEVICE_TYPE_CUDA)
  typename XZHydrostatic_PiVel_Policy::tile_type 
    XZHydrostatic_PiVel_TileSize{{256,1,1}};
#else
  typename XZHydrostatic_PiVel_Policy::tile_type 
    XZHydrostatic_PiVel_TileSize{};
#endif

  KOKKOS_INLINE_FUNCTION
  void operator() (const XZHydrostatic_PiVel_Tag &tag, const int cell, const int node, const int level) const;

#endif
};
}

#endif
