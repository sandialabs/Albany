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

  SideLaplacianResidual (const Teuchos::ParameterList& p,
                         const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:

  void evaluateFieldsCell (typename Traits::EvalData d);
  void evaluateFieldsSide (typename Traits::EvalData d);

  using MeshScalarT = typename EvalT::MeshScalarT;
  using ScalarT     = typename EvalT::ScalarT;

  // Input:
  PHX::MDField<const RealType>      BF;
  PHX::MDField<const MeshScalarT>   GradBF;
  PHX::MDField<const MeshScalarT>   w_measure;
  PHX::MDField<const MeshScalarT>   metric; // Only used if sideSetEquation=true
  PHX::MDField<const ScalarT>       u;
  PHX::MDField<const ScalarT>       grad_u;

  // Output: <Cell,Node> when in 2D, <Side,Node> when in 3d
  PHX::MDField<ScalarT>             residual;

  std::string                     sideSetName;

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
