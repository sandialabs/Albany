//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AERAS_DOFDIV_INTERPOLATION_LEVELS_HPP
#define AERAS_DOFDIV_INTERPOLATION_LEVELS_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Aeras_Layouts.hpp"
#include "Aeras_Dimension.hpp"

namespace Aeras {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to their
    divergence at quad points.

*/

template<typename EvalT, typename Traits>
class DOFDivInterpolationLevels : public PHX::EvaluatorWithBaseImpl<Traits>,
 			     public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  DOFDivInterpolationLevels(Teuchos::ParameterList& p,
                       const Teuchos::RCP<Aeras::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  //! Values at nodes
  PHX::MDField<ScalarT,Cell,Node,Level,Dim> val_node;
  //! Basis Functions
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> GradBF;
  //
  PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim,Dim> jacobian_inv;
  //
  PHX::MDField<MeshScalarT,Cell,QuadPoint> jacobian_det;

  // Output:
  //! Values at quadrature points
  PHX::MDField<ScalarT,Cell,QuadPoint,Level> div_val_qp;


  Teuchos::RCP<Intrepid2::Basis<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > > intrepidBasis;
  Teuchos::RCP<Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout,PHX::Device> > > cubature;
  Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device>    refPoints;
  Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device>    refWeights;

  Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device>    grad_at_cub_points;
  Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device>     vcontra;

  const int numNodes;
  const int numDims;
  const int numQPs;
  const int numLevels;

  std::string myName;

  bool originalDiv;

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
public:
  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;

  struct DOFDivInterpolationLevels_originalDiv_Tag{};
  struct DOFDivInterpolationLevels_Tag{};

  typedef Kokkos::RangePolicy<ExecutionSpace, DOFDivInterpolationLevels_originalDiv_Tag> DOFDivInterpolationLevels_originalDiv_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, DOFDivInterpolationLevels_Tag> DOFDivInterpolationLevels_Policy;

  KOKKOS_INLINE_FUNCTION
  void operator() (const DOFDivInterpolationLevels_originalDiv_Tag& tag, const int& i) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (const DOFDivInterpolationLevels_Tag& tag, const int& i) const;

#endif
};
}
#endif
