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

#include "Kokkos_Vector.hpp"

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

  // For f,fdot,fdotdot, can either gather a bunch of scalar fields or one vector field
  std::vector< PHX::MDField<ScalarT,Cell,Side,Node> > val;
  std::vector< PHX::MDField<ScalarT,Cell,Side,Node> > val_dot;
  std::vector< PHX::MDField<ScalarT,Cell,Side,Node> > val_dotdot;
  PHX::MDField<ScalarT,Cell,Side,Node,VecDim> valvec;
  PHX::MDField<ScalarT,Cell,Side,Node,VecDim> valvec_dot;
  PHX::MDField<ScalarT,Cell,Side,Node,VecDim> valvec_dotdot;

  std::string sideSetName;
  std::vector<std::vector<int>> sideNodes;

  int num_side_nodes;

  int numFields;        // Number of fields in std::vector val
  int numFieldsDot;     // Number of fields in std::vector val_dot
  int numFieldsDotDot;  // Number of fields in std::vector val_dotdot

  int offset;           // Offset of first field being gathered when numFields<neq
  int offsetDot;        // Offset of first dot field being gathered when numFields<neq
  int offsetDotDot;     // Offset of first dotdot field being gathered when numFields<neq

  int vecDim;

  bool is_dof_vec;
  bool is_dof_dot_vec;
  bool is_dof_dotdot_vec;

  bool enableSolution;
  bool enableSolutionDot;
  bool enableSolutionDotDot;
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
