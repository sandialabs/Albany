//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AERAS_DOF_INTERPOLATION_HPP
#define AERAS_DOF_INTERPOLATION_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Aeras_Layouts.hpp"

namespace Aeras {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template<typename EvalT, typename Traits>
class DOFInterpolation : public PHX::EvaluatorWithBaseImpl<Traits>,
 			 public PHX::EvaluatorDerived<EvalT, Traits>  {

public:
  typedef typename EvalT::ScalarT ScalarT;

  DOFInterpolation(Teuchos::ParameterList& p,
                   const Teuchos::RCP<Aeras::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                             PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:
  // Input:
  //! Values at nodes
  PHX::MDField<const ScalarT> val_node;
  //! Basis Functions
  PHX::MDField<const RealType,Cell,Node,QuadPoint> BF;

  // Output:
  //! Values at quadrature points
  PHX::MDField<ScalarT> val_qp;

  const int numNodes;
  const int numLevels;
  const int numRank;

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
public:
  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;

  struct DOFInterpolation_numRank2_Tag{};
  struct DOFInterpolation_Tag{};

  typedef Kokkos::RangePolicy<ExecutionSpace, DOFInterpolation_numRank2_Tag> DOFInterpolation_numRank2_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, DOFInterpolation_Tag> DOFInterpolation_Policy;

  KOKKOS_INLINE_FUNCTION
  void operator() (const DOFInterpolation_numRank2_Tag& tag, const int& i) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (const DOFInterpolation_Tag& tag, const int& i) const;

#endif
};
}

#endif
