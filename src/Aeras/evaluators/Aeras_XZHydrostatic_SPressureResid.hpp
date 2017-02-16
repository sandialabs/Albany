//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AERAS_XZHYDROSTATICSPRESSURERESID_HPP
#define AERAS_XZHYDROSTATICSPRESSURERESID_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Aeras_Layouts.hpp"
#include "Aeras_Dimension.hpp"
#include "Sacado_ParameterAccessor.hpp"
#include "Aeras_Eta.hpp"

namespace Aeras {
/** \brief XZHydrostatic equation Residual for atmospheric modeling

    This evaluator computes the residual of the XZHydrostatic surface pressure 
    equation for atmospheric dynamics.

*/

template<typename EvalT, typename Traits>
class XZHydrostatic_SPressureResid : public PHX::EvaluatorWithBaseImpl<Traits>,
                   public PHX::EvaluatorDerived<EvalT, Traits>,
                   public Sacado::ParameterAccessor<EvalT, SPL_Traits>  {

public:
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  XZHydrostatic_SPressureResid(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Aeras::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

  ScalarT& getValue(const std::string &n);

private:

  // Input:
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint>         wBF;

  PHX::MDField<ScalarT,Cell,QuadPoint>       spDot;
  PHX::MDField<ScalarT,Cell,QuadPoint,Level> divpivelx;

  // Output:
  PHX::MDField<ScalarT,Cell,Node> Residual;

  const int numNodes;
  const int numQPs;
  const int numLevels;
  const Eta<EvalT> &E;

  ScalarT sp0;

  bool obtainLaplaceOp;
  bool pureAdvection;

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  Kokkos::DynRankView<ScalarT, PHX::Device> delta;

public:
  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;

  //struct XZHydrostatic_SPressureResid_Tag{};
  struct XZHydrostatic_SPressureResid_pureAdvection_Tag{};

 // typedef Kokkos::RangePolicy<ExecutionSpace, XZHydrostatic_SPressureResid_Tag> XZHydrostatic_SPressureResid_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, XZHydrostatic_SPressureResid_pureAdvection_Tag> XZHydrostatic_SPressureResid_pureAdvection_Policy;

#if defined(PHX_KOKKOS_DEVICE_TYPE_CUDA) 
  using XZHydrostatic_SPressureResid_Policy =
        Kokkos::Experimental::MDRangePolicy<
        Kokkos::Experimental::Rank<2, Kokkos::Experimental::Iterate::Left,
        Kokkos::Experimental::Iterate::Left >, Kokkos::IndexType<int> >;
#else
  using XZHydrostatic_SPressureResid_Policy =
        Kokkos::Experimental::MDRangePolicy<
        Kokkos::Experimental::Rank<2, Kokkos::Experimental::Iterate::Right,
        Kokkos::Experimental::Iterate::Right >, Kokkos::IndexType<int> >;
#endif


  KOKKOS_INLINE_FUNCTION
  void operator() (const int cell, const int qp) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (const XZHydrostatic_SPressureResid_pureAdvection_Tag& tag, const int& i) const;

#endif
};
}

#endif
