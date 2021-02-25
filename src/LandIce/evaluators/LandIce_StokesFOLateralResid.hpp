//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_STOKES_FO_LATERAL_RESID_HPP
#define LANDICE_STOKES_FO_LATERAL_RESID_HPP 1

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"
#include "Albany_ScalarOrdinalTypes.hpp"
#include "Albany_SacadoTypes.hpp"
#include "PHAL_Dimension.hpp"

namespace LandIce
{

/** \brief The residual of the lateral BC

    This evaluator evaluates the residual of the Lateral bc for the StokesFO problem
*/

template<typename EvalT, typename Traits, typename ThicknessScalarT>
class StokesFOLateralResid : public PHX::EvaluatorWithBaseImpl<Traits>,
                             public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  typedef typename EvalT::ScalarT       ScalarT;
  typedef typename EvalT::MeshScalarT   MeshScalarT;

  StokesFOLateralResid (const Teuchos::ParameterList& p,
                        const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename Albany::StrongestScalarType<ScalarT,MeshScalarT>::type OutputScalarT;

  void evaluate_with_given_immersed_ratio(typename Traits::EvalData d);
  void evaluate_with_computed_immersed_ratio(typename Traits::EvalData d);

  // Input:
  // TODO: restore layout template arguments when removing old sideset layout
  PHX::MDField<const MeshScalarT>        coords_qp; // Side, Node, Dim
  PHX::MDField<const ThicknessScalarT>   thickness; // Side, QuadPoint
  PHX::MDField<const MeshScalarT>        elevation; // Side, QuadPoint
  PHX::MDField<const RealType>           BF;        // Side, Node, QuadPoint
  PHX::MDField<const MeshScalarT>        normals;   // Side, QuadPoint, Dim
  PHX::MDField<const MeshScalarT>        w_measure; // Side, QuadPoint

  // Output:
  PHX::MDField<OutputScalarT,Cell,Node,VecDim> residual;

  Kokkos::View<int**, PHX::Device> sideNodes;
  std::string                      lateralSideName;

  bool useCollapsedSidesets;
  
  double rho_w;  // [Kg m^{-3}]
  double rho_i;  // [Kg m^{-3}]
  double g;      // [m s^{-2}]
  double given_immersed_ratio; //[]
  double melange_force_value;  //[N m^{-1}]
  double melange_thickness_threshold; //[km]
  double X_0; //[km]
  double Y_0; //[km]
  double R2; //[km]

  bool immerse_ratio_provided;
  bool add_melange_force;
  bool use_stereographic_map;

  unsigned int numSideNodes;
  unsigned int numSideQPs;
  unsigned int vecDimFO;

  Albany::LocalSideSetInfo sideSet;

public:

  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;
  struct GivenImmersedRatio_Tag{};
  struct ComputedImmersedRatio_Tag{};

  typedef Kokkos::RangePolicy<ExecutionSpace, GivenImmersedRatio_Tag> GivenImmersedRatio_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, ComputedImmersedRatio_Tag> ComputedImmersedRatio_Policy;

  KOKKOS_INLINE_FUNCTION
  void operator() (const GivenImmersedRatio_Tag& tag, const int& sideSet_idx) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const ComputedImmersedRatio_Tag& tag, const int& sideSet_idx) const;
  
};

} // Namespace LandIce

#endif // LANDICE_STOKES_FO_LATERAL_RESID_HPP
