//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AERAS_HYDROSTATIC_VELOCITY_HPP
#define AERAS_HYDROSTATIC_VELOCITY_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Aeras_Layouts.hpp"
#include "Aeras_Dimension.hpp"
#include "Sacado_ParameterAccessor.hpp"
#include "Aeras_Eta.hpp"

namespace Aeras {

template<typename EvalT, typename Traits>
class Hydrostatic_Velocity : public PHX::EvaluatorWithBaseImpl<Traits>,
                              public PHX::EvaluatorDerived<EvalT, Traits> {

public:
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  Hydrostatic_Velocity(const Teuchos::ParameterList& p,
                        const Teuchos::RCP<Aeras::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);
private:
  // Input:
  PHX::MDField<ScalarT,Cell,Node,Level,Dim>       Velx;
  PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>    sphere_coord;
  PHX::MDField<ScalarT,Cell,QuadPoint,Level>      pressure;
  
  // Output:
  PHX::MDField<ScalarT,Cell,Node,Level,Dim> Velocity;

  const int numNodes;
  const int numDims;
  const int numLevels;
  const Eta<EvalT> &E;
  Teuchos::RCP<Teuchos::FancyOStream> out;
  
  enum ADVTYPE {UNKNOWN, PRESCRIBED_1_1, PRESCRIBED_1_2};
  ADVTYPE adv_type;
  double time;

  // Prescribed 1-1 parameters
  ScalarT PI, earthRadius, ptop, p0, tau, omega0, k;

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  Kokkos::DynRankView<ScalarT, PHX::Device> B;

public:
  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;

  struct Hydrostatic_Velocity_Tag{};
  struct Hydrostatic_Velocity_PRESCRIBED_1_1_Tag{};
  struct Hydrostatic_Velocity_PRESCRIBED_1_2_Tag{};

#if defined(PHX_KOKKOS_DEVICE_TYPE_CUDA)
  using Hydrostatic_Velocity_Policy =
        Kokkos::Experimental::MDRangePolicy<
        Kokkos::Experimental::Rank<3, Kokkos::Experimental::Iterate::Left,
        Kokkos::Experimental::Iterate::Left >, Kokkos::IndexType<int> ,
        Hydrostatic_Velocity_Tag >;
  typename Hydrostatic_Velocity_Policy::tile_type 
    Hydrostatic_Velocity_TileSize{{256,1,1}};

 using Hydrostatic_Velocity_PRESCRIBED_1_1_Policy =
        Kokkos::Experimental::MDRangePolicy<
        Kokkos::Experimental::Rank<2, Kokkos::Experimental::Iterate::Left,
        Kokkos::Experimental::Iterate::Left >, Kokkos::IndexType<int>,
        Hydrostatic_Velocity_PRESCRIBED_1_1_Tag >;
  typename Hydrostatic_Velocity_PRESCRIBED_1_1_Policy::tile_type 
    Hydrostatic_Velocity_PRESCRIBED_1_1_TileSize{{256,1}};

  using Hydrostatic_Velocity_PRESCRIBED_1_2_Policy =
        Kokkos::Experimental::MDRangePolicy<
        Kokkos::Experimental::Rank<2, Kokkos::Experimental::Iterate::Left,
        Kokkos::Experimental::Iterate::Left >, Kokkos::IndexType<int>, 
        Hydrostatic_Velocity_PRESCRIBED_1_2_Tag >;
  typename Hydrostatic_Velocity_PRESCRIBED_1_2_Policy::tile_type 
    Hydrostatic_Velocity_PRESCRIBED_1_2_TileSize{{256,1}};
#else
  using Hydrostatic_Velocity_Policy =
        Kokkos::Experimental::MDRangePolicy<
        Kokkos::Experimental::Rank<3, Kokkos::Experimental::Iterate::Right,
        Kokkos::Experimental::Iterate::Right >, Kokkos::IndexType<int> ,
        Hydrostatic_Velocity_Tag >;
  typename Hydrostatic_Velocity_Policy::tile_type 
    Hydrostatic_Velocity_TileSize{};

 using Hydrostatic_Velocity_PRESCRIBED_1_1_Policy =
        Kokkos::Experimental::MDRangePolicy<
        Kokkos::Experimental::Rank<2, Kokkos::Experimental::Iterate::Right,
        Kokkos::Experimental::Iterate::Right>, Kokkos::IndexType<int>,
        Hydrostatic_Velocity_PRESCRIBED_1_1_Tag >;
  typename Hydrostatic_Velocity_PRESCRIBED_1_1_Policy::tile_type 
    Hydrostatic_Velocity_PRESCRIBED_1_1_TileSize{};

  using Hydrostatic_Velocity_PRESCRIBED_1_2_Policy =
        Kokkos::Experimental::MDRangePolicy<
        Kokkos::Experimental::Rank<2, Kokkos::Experimental::Iterate::Right,
        Kokkos::Experimental::Iterate::Right >, Kokkos::IndexType<int>,
        Hydrostatic_Velocity_PRESCRIBED_1_2_Tag >;
  typename Hydrostatic_Velocity_PRESCRIBED_1_2_Policy::tile_type 
    Hydrostatic_Velocity_PRESCRIBED_1_2_TileSize{};

#endif 

  KOKKOS_INLINE_FUNCTION
  void operator() (const Hydrostatic_Velocity_Tag& tag, const int cell, const int node, const int level) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (const Hydrostatic_Velocity_PRESCRIBED_1_1_Tag& tag, const int cell, const int node) const;
  
  KOKKOS_INLINE_FUNCTION
  void operator() (const Hydrostatic_Velocity_PRESCRIBED_1_2_Tag& tag, const int cell, const int node) const;

#endif
};
}

#endif
