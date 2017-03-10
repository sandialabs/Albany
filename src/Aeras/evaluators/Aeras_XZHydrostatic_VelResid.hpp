//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AERAS_XZHYDROSTATIC_VELRESID_HPP
#define AERAS_XZHYDROSTATIC_VELRESID_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Aeras_Layouts.hpp"
#include "Aeras_Dimension.hpp"

namespace Aeras {
/** \brief XZHydrostatic equation Residual for atmospheric modeling

    This evaluator computes the residual of the XZHydrostatic equation for
    atmospheric dynamics.

*/

template<typename EvalT, typename Traits>
class XZHydrostatic_VelResid : public PHX::EvaluatorWithBaseImpl<Traits>,
                               public PHX::EvaluatorDerived<EvalT, Traits> {

public:
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  XZHydrostatic_VelResid(const Teuchos::ParameterList& p,
                         const Teuchos::RCP<Aeras::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  // Input:
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint>         wBF;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> wGradBF;

  PHX::MDField<ScalarT,Cell,QuadPoint,Level,Dim>  keGrad;
  PHX::MDField<ScalarT,Cell,QuadPoint,Level,Dim>  PhiGrad;
  PHX::MDField<ScalarT,Cell,Node,Level,Dim>  etadotdVelx;
  PHX::MDField<ScalarT,Cell,QuadPoint,Level,Dim>  pGrad;
  PHX::MDField<ScalarT,Cell,Node,Level,Dim>  uDot;
  PHX::MDField<ScalarT,Cell,QuadPoint,Level,Dim>  DVelx;
  PHX::MDField<ScalarT,Cell,Node,Level>      density;


  // Output:
  PHX::MDField<ScalarT,Cell,Node,Level,Dim> Residual;

  const double viscosity;
  const double hyperviscosity;

  const int numNodes;
  const int numQPs;
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

  struct XZHydrostatic_VelResid_Tag{};

  using XZHydrostatic_VelResid_Policy =
        Kokkos::Experimental::MDRangePolicy<
        Kokkos::Experimental::Rank<3, IterateDirection, IterateDirection>, 
        Kokkos::IndexType<int> >;

#if defined(PHX_KOKKOS_DEVICE_TYPE_CUDA)
  typename XZHydrostatic_VelResid_Policy::tile_type 
    XZHydrostatic_VelResid_TileSize{{256,1,1}};
#else
  typename XZHydrostatic_VelResid_Policy::tile_type 
    XZHydrostatic_VelResid_TileSize{};
#endif

  KOKKOS_INLINE_FUNCTION
  void operator() (const int cell, const int node, const int level) const;

#endif
};
}

#endif
