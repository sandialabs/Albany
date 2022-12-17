//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_STOKES_FO_IMPLICIT_THICKNESS_UPDATE_RESID_HPP
#define LANDICE_STOKES_FO_IMPLICIT_THICKNESS_UPDATE_RESID_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "PHAL_Dimension.hpp"
#include "Albany_Layouts.hpp"

namespace LandIce {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template<typename EvalT, typename Traits>
class StokesFOImplicitThicknessUpdateResid : public PHX::EvaluatorWithBaseImpl<Traits>,
            public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  StokesFOImplicitThicknessUpdateResid(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
           PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<const MeshScalarT,Cell,Node,QuadPoint> wBF;
  PHX::MDField<const MeshScalarT,Cell,Node,QuadPoint,Dim> gradBF;
  PHX::MDField<const ScalarT,Cell,Node> dH;
  PHX::MDField<const ScalarT,Cell,Node> H0;

  // Output:
  PHX::MDField<ScalarT,Cell,Node,VecDim> Residual;

  std::size_t numNodes;
  std::size_t numQPs;
  std::size_t numVecDims;
  std::size_t numCells;

  double rho_g;

  Kokkos::DynRankView<ScalarT, PHX::Device> Res;

public:

  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;
  struct StokesFOImplicitThicknessUpdateResid_Tag{};
  typedef Kokkos::RangePolicy<ExecutionSpace, StokesFOImplicitThicknessUpdateResid_Tag> StokesFOImplicitThicknessUpdateResid_Policy;

  KOKKOS_INLINE_FUNCTION
  void operator() (const StokesFOImplicitThicknessUpdateResid_Tag& tag, const int& cell) const;
};

} // namespace LandIce

#endif // LANDICE_STOKES_FO_IMPLICIT_THICKNESS_UPDATE_RESID_HPP
