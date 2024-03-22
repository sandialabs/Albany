//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_GATHER_SOLUTION_SIDE_HPP
#define PHAL_GATHER_SOLUTION_SIDE_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"
#include "Albany_DiscretizationUtils.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_Dimension.hpp"

#include "Teuchos_ParameterList.hpp"

namespace Albany {
  class DOFManager;
}
namespace shards {
  class CellTopology;
}


namespace PHAL {
/** \brief Gathers solution values from the Newton solution vector into
    the nodal fields of the field manager

    Currently makes an assumption that the stride is constant for dofs
    and that the nmber of dofs is equal to the size of the solution
    names vector.

*/
// **************************************************************
// Base Class with Generic Implementations: Specializations for
// Automatic Differentiation Below
// **************************************************************

template<typename EvalT, typename Traits>
class GatherSolutionSide : public PHX::EvaluatorWithBaseImpl<Traits>,
                           public PHX::EvaluatorDerived<EvalT, Traits>
{
public:
  GatherSolutionSide (const Teuchos::ParameterList& p,
                      const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                             PHX::FieldManager<Traits>& vm);

  // This function requires template specialization, in derived class below
  void evaluateFields(typename Traits::EvalData d);

protected:
  typedef typename EvalT::ScalarT ScalarT;
  using ref_t = typename PHAL::Ref<typename EvalT::ScalarT>::type;

  // These functions are used to select the correct field based on rank.
  // They are called from *inside* for loops, but the switch statement
  // is constant for all iterations, so the compiler branch predictor
  // can easily guess the correct branch, making the conditional jump
  // cheap.
  KOKKOS_INLINE_FUNCTION
  ref_t get_ref (const int side, const int node, const int eq) const {
    switch (solRank) {
      case 0:
        return val(side,node);
      case 1:
        return valVec(side,node,eq);
      case 2:
        return valTensor(side,node,eq/numDim,eq%numDim);
    }
    Kokkos::abort("Unsupported tensor rank");
  }

  KOKKOS_INLINE_FUNCTION
  ref_t get_ref_dot (const int side, const int node, const int eq) const {
    switch (solRank) {
      case 0:
        return val_dot(side,node);
      case 1:
        return valVec_dot(side,node,eq);
      case 2:
        return valTensor_dot(side,node,eq/numDim,eq%numDim);
    }
    Kokkos::abort("Unsupported tensor rank");
  }

  KOKKOS_INLINE_FUNCTION
  ref_t get_ref_dotdot (const int side, const int node, const int eq) const {
    switch (solRank) {
      case 0:
        return val_dotdot(side,node);
      case 1:
        return valVec_dotdot(side,node,eq);
      case 2:
        return valTensor_dotdot(side,node,eq/numDim,eq%numDim);
    }
    Kokkos::abort("Unsupported tensor rank");
  }

  void gather_fields_offsets (const Teuchos::RCP<const Albany::DOFManager>& dof_mgr);

  PHX::MDField<ScalarT,Side,Node> val;
  PHX::MDField<ScalarT,Side,Node> val_dot;
  PHX::MDField<ScalarT,Side,Node> val_dotdot;
  PHX::MDField<ScalarT,Side,Node,Dim> valVec;
  PHX::MDField<ScalarT,Side,Node,Dim> valVec_dot;
  PHX::MDField<ScalarT,Side,Node,Dim> valVec_dotdot;
  PHX::MDField<ScalarT,Side,Node,Dim,Dim> valTensor;
  PHX::MDField<ScalarT,Side,Node,Dim,Dim> valTensor_dot;
  PHX::MDField<ScalarT,Side,Node,Dim,Dim> valTensor_dotdot;

  std::string sideSetName;

  int numFields;

  int offset = 0; // Offset of first field being gathered when numFields<neq

  int numDim;

  int solRank;

  bool enableSolution       = false;
  bool enableSolutionDot    = false;
  bool enableSolutionDotDot = false;
};

// Full specialization of the evaluateFields method for all Evaluation types

template<>
void GatherSolutionSide<PHAL::AlbanyTraits::Jacobian,PHAL::AlbanyTraits>::
evaluateFields(PHAL::AlbanyTraits::EvalData d);

template<>
void GatherSolutionSide<PHAL::AlbanyTraits::Tangent,PHAL::AlbanyTraits>::
evaluateFields(PHAL::AlbanyTraits::EvalData d);

} // namespace PHAL

#endif // PHAL_GATHER_SOLUTION_SIDE_HPP
