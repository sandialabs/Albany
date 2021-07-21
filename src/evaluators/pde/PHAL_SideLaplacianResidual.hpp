//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_SIDE_LAPLACIAN_RESIDUAL_HPP
#define PHAL_SIDE_LAPLACIAN_RESIDUAL_HPP 1

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "PHAL_Dimension.hpp"
#include "Albany_ScalarOrdinalTypes.hpp"
#include "Albany_Layouts.hpp"

namespace PHAL
{

template<typename EvalT, typename Traits>
class SideLaplacianResidual : public PHX::EvaluatorWithBaseImpl<Traits>,
                           public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  typedef typename EvalT::ScalarT ScalarT;

  SideLaplacianResidual (const Teuchos::ParameterList& p,
                         const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:

  void evaluateFieldsCell (typename Traits::EvalData d);
  void evaluateFieldsSide (typename Traits::EvalData d);

  typedef typename EvalT::MeshScalarT                   MeshScalarT;

  // Input:
  PHX::MDField<RealType>                                BF;
  PHX::MDField<MeshScalarT>                             GradBF;
  PHX::MDField<MeshScalarT>                             w_measure;
  PHX::MDField<MeshScalarT> metric; // Only used in 2D, so we know the layout (Cell,Side,QuadPoint,Dim,Dim)

  PHX::MDField<ScalarT>                                 u;
  PHX::MDField<ScalarT>                                 grad_u;

  // Output:
  PHX::MDField<ScalarT,Cell,Node>                       residual; // Always a 3D residual, so we know the layout

  Albany::LocalSideSetInfo sideSet;

  std::string                     sideSetName;
  Kokkos::View<int**, PHX::Device> sideNodes;

  int spaceDim;
  int gradDim;
  int numNodes;
  int numQPs;

  bool sideSetEquation;

public:

  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;

  struct SideLaplacianResidual_Side_Tag{};
  struct SideLaplacianResidual_Cell_Tag{};

  typedef Kokkos::RangePolicy<ExecutionSpace,SideLaplacianResidual_Side_Tag> SideLaplacianResidual_Side_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,SideLaplacianResidual_Cell_Tag> SideLaplacianResidual_Cell_Policy;

  KOKKOS_INLINE_FUNCTION
  void operator() (const SideLaplacianResidual_Side_Tag& tag, const int& i) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const SideLaplacianResidual_Cell_Tag& tag, const int& i) const;
};

} // Namespace PHAL

#endif // PHAL_SIDE_LAPLACIAN_RESIDUAL_HPP
