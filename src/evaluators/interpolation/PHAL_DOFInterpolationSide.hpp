//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_DOF_INTERPOLATION_SIDE_HPP
#define PHAL_DOF_INTERPOLATION_SIDE_HPP 1

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"
#include "Albany_ScalarOrdinalTypes.hpp"
#include "PHAL_Utilities.hpp"
#include "PHAL_Dimension.hpp"

namespace PHAL {
/** \brief Finite Element InterpolationSide Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template<typename EvalT, typename Traits, typename ScalarT>
class DOFInterpolationSideBase : public PHX::EvaluatorWithBaseImpl<Traits>,
                                 public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  DOFInterpolationSideBase (const Teuchos::ParameterList& p,
                            const Teuchos::RCP<Albany::Layouts>& dl_side);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& vm);

  void evaluateFields (typename Traits::EvalData d);

private:

  std::string sideSetName;

  // Input:
  //! Values at nodes
  PHX::MDField<const ScalarT> val_node;

  //! Basis Functions
  typedef typename EvalT::MeshScalarT MeshScalarT;
  PHX::MDField<const MeshScalarT> BF;

  // Output:
  typedef typename Albany::StrongestScalarType<ScalarT,MeshScalarT>::type OutputScalarT;
  //! Values at quadrature points
  PHX::MDField<OutputScalarT> val_qp;

  int numSideNodes;
  int numSideQPs;

  MDFieldMemoizer<Traits> memoizer;

  Albany::LocalSideSetInfo sideSet;

public:

  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;
  struct InterpolationSide_Tag{};

  typedef Kokkos::RangePolicy<ExecutionSpace, InterpolationSide_Tag> InterpolationSide_Policy;

  KOKKOS_INLINE_FUNCTION
  void operator() (const InterpolationSide_Tag& tag, const int& sideSet_idx) const;
};

// Some shortcut names
template<typename EvalT, typename Traits>
using DOFInterpolationSide = DOFInterpolationSideBase<EvalT,Traits,typename EvalT::ScalarT>;

template<typename EvalT, typename Traits>
using DOFInterpolationSideMesh = DOFInterpolationSideBase<EvalT,Traits,typename EvalT::MeshScalarT>;

template<typename EvalT, typename Traits>
using DOFInterpolationSideParam = DOFInterpolationSideBase<EvalT,Traits,typename EvalT::ParamScalarT>;

} // Namespace PHAL

#endif // PHAL_DOF_INTERPOLATION_SIDE_HPP
