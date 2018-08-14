//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_TransportResidual_hpp)
#define LCM_TransportResidual_hpp

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_MDField.hpp>
#include <Phalanx_config.hpp>

#include "Albany_Layouts.hpp"

namespace LCM {
/// \brief
///
///  This evaluator computes the residual for the transport equation
///
template <typename EvalT, typename Traits>
class TransportResidual : public PHX::EvaluatorWithBaseImpl<Traits>,
                          public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  ///
  /// Constructor
  ///
  TransportResidual(
      Teuchos::ParameterList&              p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  ///
  /// Phalanx method to allocate space
  ///
  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  ///
  /// Implementation of physics
  ///
  void
  evaluateFields(typename Traits::EvalData d);

 private:
  typedef typename EvalT::ScalarT     ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  ///
  /// Stress field
  ///
  PHX::MDField<const ScalarT, Cell, QuadPoint, Dim, Dim> stress_;

  // velocity gradient (Lagrangian)
  PHX::MDField<const ScalarT, Cell, QuadPoint, Dim, Dim> vel_grad_;

  // Temporal container used to store P : F_dot
  Kokkos::DynRankView<ScalarT, PHX::Device> term1_;

  ///
  /// Scalar field for transport variable
  ///
  PHX::MDField<const ScalarT, Cell, QuadPoint> scalar_;

  ///
  /// Scalar dot field for transport variable
  ///
  PHX::MDField<const ScalarT, Cell, QuadPoint> scalar_dot_;

  ///
  /// Input: time step
  ///
  PHX::MDField<const ScalarT, Dummy> delta_time_;

  ///
  /// Scalar field for transport variable
  ///
  PHX::MDField<const ScalarT, Cell, QuadPoint, Dim> scalar_grad_;

  ///
  /// Integrations weights
  ///
  PHX::MDField<const MeshScalarT, Cell, QuadPoint> weights_;

  ///
  /// Weighted basis functions
  ///
  PHX::MDField<const MeshScalarT, Cell, Node, QuadPoint> w_bf_;

  ///
  /// Weighted gradients of basis functions
  ///
  PHX::MDField<const MeshScalarT, Cell, Node, QuadPoint, Dim> w_grad_bf_;

  ///
  /// Source term(s)
  ///
  PHX::MDField<const ScalarT, Cell, QuadPoint> source_;
  PHX::MDField<const ScalarT, Cell, QuadPoint> second_source_;

  ///
  /// M operator for contact
  ///
  PHX::MDField<const ScalarT, Cell, QuadPoint> M_operator_;

  ///
  /// Scalar coefficient on the transient transport term
  ///
  PHX::MDField<const ScalarT, Cell, QuadPoint> transient_coeff_;

  ///
  /// Tensor diffusivity
  ///
  PHX::MDField<const ScalarT, Cell, QuadPoint, Dim, Dim> diffusivity_;

  ///
  /// Vector convection term
  ///
  PHX::MDField<const ScalarT, Cell, QuadPoint, Dim> convection_vector_;

  ///
  /// Species coupling term
  ///
  PHX::MDField<const ScalarT, Cell, QuadPoint> species_coupling_;

  ///
  /// Stabilization term
  ///
  PHX::MDField<const ScalarT, Cell, QuadPoint> stabilization_;

  ///
  /// Output residual
  ///
  PHX::MDField<ScalarT, Cell, Node> residual_;

  ///
  ///  Feature flags
  ///
  bool have_source_;
  bool have_second_source_;
  bool have_transient_;
  bool have_diffusion_;
  bool have_convection_;
  bool have_species_coupling_;
  bool have_stabilization_;
  bool have_contact_;
  bool have_mechanics_;

  std::string SolutionType_;

  ///
  /// Data structure dimensions
  ///
  int num_cells_, num_nodes_, num_pts_, num_dims_;

  ///
  /// Scalar name
  ///
  std::string scalar_name_;
};
}  // namespace LCM

#endif
