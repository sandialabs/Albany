//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_DOF_VEC_GRAD_INTERPOLATION_SIDE_HPP
#define PHAL_DOF_VEC_GRAD_INTERPOLATION_SIDE_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"
#include "Albany_SacadoTypes.hpp"

namespace PHAL {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOFVec values to their
    gradients at quad points.

*/

template<typename EvalT, typename Traits, typename ScalarT>
class DOFVecGradInterpolationSideBase : public PHX::EvaluatorWithBaseImpl<Traits>,
                                        public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  DOFVecGradInterpolationSideBase (const Teuchos::ParameterList& p,
                                   const Teuchos::RCP<Albany::Layouts>& dl_side);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& vm);

  void evaluateFields (typename Traits::EvalData d);

private:

  typedef typename EvalT::MeshScalarT MeshScalarT;
  typedef typename Albany::StrongestScalarType<ScalarT,MeshScalarT>::type OutputScalarT;
  std::string sideSetName;

  // Input:
  // TODO: restore layout template arguments when removing old sideset layout
  //! Values at nodes
  PHX::MDField<const ScalarT> val_node;    // Side, Node, VecDim
  //! Basis Functions
  PHX::MDField<const MeshScalarT> gradBF;  // Side, Node, QuadPoint, Dim

  // Output:
  //! Values at quadrature points
  PHX::MDField<OutputScalarT> grad_qp;

  bool useCollapsedSidesets;

  Albany::LocalSideSetInfo sideSet;

  int numSideNodes;
  int numSideQPs;
  int numDims;
  int vecDim;

public:

  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;
  struct VecGradInterpolationSide_Tag{};

  typedef Kokkos::RangePolicy<ExecutionSpace, VecGradInterpolationSide_Tag> VecGradInterpolationSide_Policy;

  KOKKOS_INLINE_FUNCTION
  void operator() (const VecGradInterpolationSide_Tag& tag, const int& sideSet_idx) const;

};

// Some shortcut names
template<typename EvalT, typename Traits>
using DOFVecGradInterpolationSide = DOFVecGradInterpolationSideBase<EvalT,Traits,typename EvalT::ScalarT>;

template<typename EvalT, typename Traits>
using DOFVecGradInterpolationSideMesh = DOFVecGradInterpolationSideBase<EvalT,Traits,typename EvalT::MeshScalarT>;

template<typename EvalT, typename Traits>
using DOFVecGradInterpolationSideParam = DOFVecGradInterpolationSideBase<EvalT,Traits,typename EvalT::ParamScalarT>;

} // Namespace PHAL

#endif // PHAL_DOF_VEC_GRAD_INTERPOLATION_SIDE_HPP
