/*
 * LandIce_w_Resid.hpp
 *
 *  Created on: Jun 7, 2016
 *      Author: abarone
 */

#ifndef LANDICE_W_RESID_HPP_
#define LANDICE_W_RESID_HPP_

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "PHAL_Dimension.hpp"
#include "Albany_Layouts.hpp"
#include "Albany_ScalarOrdinalTypes.hpp"

namespace LandIce
{

template<typename EvalT, typename Traits, typename VelocityType>
class w_Resid : public PHX::EvaluatorWithBaseImpl<Traits>,
public PHX::EvaluatorDerived<EvalT, Traits>
{
public:
  w_Resid (const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d, PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;
  typedef typename EvalT::ParamScalarT ParamScalarT;

  // Input:
  PHX::MDField<const MeshScalarT,Cell,Node,QuadPoint> wBF;  // [km^3]
  PHX::MDField<const MeshScalarT,Cell,Node,QuadPoint,Dim>  wGradBF; // [km^2]
  PHX::MDField<const RealType> sideBF;  // []
  PHX::MDField<const MeshScalarT> side_w_measure;  // [km^2]
  PHX::MDField<const MeshScalarT>   normals;

  PHX::MDField<const ScalarT> basalVerticalVelocitySideQP; // [m yr^{-1}]
  PHX::MDField<const VelocityType,Cell,QuadPoint,VecDim,Dim>  GradVelocity; // [k^{-1} yr^{-1}]
  PHX::MDField<const VelocityType,Cell,QuadPoint,VecDim>  velocity; // [m yr^{-1}]
  PHX::MDField<const ScalarT,Cell,QuadPoint, Dim> w_z;  // [k^{-1} yr^{-1}]
  PHX::MDField<const ScalarT> side_w_qp; // [m yr^{-1}]
  PHX::MDField<const MeshScalarT,Cell,Node,Dim>  coordVec; // [km]

  // Output
  PHX::MDField<ScalarT,Cell,Node> Residual;

  Albany::LocalSideSetInfo sideSet;

  std::string sideName;
  Kokkos::View<int**, PHX::Device> sideNodes;
  unsigned int numNodes;
  unsigned int numSideNodes;
  unsigned int numQPs;
  unsigned int numSideQPs;

public:

  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;

  struct wResid_Cell_Tag{};
  struct wResid_Side_Tag{};

  typedef Kokkos::RangePolicy<ExecutionSpace,wResid_Cell_Tag> wResid_Cell_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,wResid_Side_Tag> wResid_Side_Policy;

  KOKKOS_INLINE_FUNCTION
  void operator() (const wResid_Cell_Tag& tag, const int& i) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const wResid_Side_Tag& tag, const int& i) const;
};

}	// Namespace LandIce




#endif /* LandIce_VELOCITYZ_HPP_ */
