//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_STOKESFORESID_HPP
#define LANDICE_STOKESFORESID_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"
#include "PHAL_Dimension.hpp"

namespace LandIce {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template<typename EvalT, typename Traits>
class StokesFOResid : public PHX::EvaluatorWithBaseImpl<Traits>,
            public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  StokesFOResid(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
           PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<const MeshScalarT,Cell,Node,QuadPoint>     wBF;
  PHX::MDField<const MeshScalarT,Cell,Node,QuadPoint,Dim> wGradBF;
  PHX::MDField<const ScalarT,Cell,QuadPoint,VecDim>       force;

  PHX::MDField<const ScalarT,Cell,QuadPoint,VecDim>       U;
  PHX::MDField<const ScalarT,Cell,QuadPoint,VecDim,Dim>   Ugrad;
  PHX::MDField<const ScalarT,Cell,QuadPoint>              muLandIce;

  PHX::MDField<const MeshScalarT,Cell,QuadPoint, Dim>     coordVec;

  enum EQNTYPE {LandIce, POISSON, LandIce_XZ};
  EQNTYPE eqn_type;

  // Output:
  PHX::MDField<ScalarT,Cell,Node,VecDim> Residual;

  const unsigned int numNodes, numQPs, numDims;
  const bool useStereographicMap;
  const RealType R2, x_0, y_0;

public:

  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;

  struct LandIce_3D_Tag{};
  struct POISSON_3D_Tag{};
  struct LandIce_2D_Tag{};
  struct LandIce_XZ_2D_Tag{};
  struct POISSON_2D_Tag{};

  typedef Kokkos::RangePolicy<ExecutionSpace,LandIce_3D_Tag> LandIce_3D_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,POISSON_3D_Tag> POISSON_3D_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,LandIce_2D_Tag> LandIce_2D_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,LandIce_XZ_2D_Tag> LandIce_XZ_2D_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,POISSON_2D_Tag> POISSON_2D_Policy;

  KOKKOS_INLINE_FUNCTION
  void operator() (const LandIce_3D_Tag& tag, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const POISSON_3D_Tag& tag, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const LandIce_2D_Tag& tag, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const LandIce_XZ_2D_Tag& tag, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const POISSON_2D_Tag& tag, const int& cell) const;
};

}

#endif
