//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_ConstitutiveModelDriver_hpp)
#define LCM_ConstitutiveModelDriver_hpp

#include "Albany_Layouts.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

namespace LCM {

/// \brief Constitutive Model Driver
template <typename EvalT, typename Traits>
class ConstitutiveModelDriver : public PHX::EvaluatorWithBaseImpl<Traits>,
                                public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  ///
  /// Constructor
  ///
  ConstitutiveModelDriver(
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
  /// residual field
  ///
  PHX::MDField<ScalarT, Cell, Node, Dim, Dim> residual_;

  ///
  /// def grad field
  ///
  PHX::MDField<const ScalarT, Cell, QuadPoint, Dim, Dim> def_grad_;

  ///
  /// Cauchy stress field
  ///
  PHX::MDField<const ScalarT, Cell, QuadPoint, Dim, Dim> stress_;

  ///
  /// prescribed deformation gradient field
  ///
  PHX::MDField<const ScalarT, Cell, QuadPoint, Dim, Dim> prescribed_def_grad_;

  ///
  /// Time field
  ///
  PHX::MDField<ScalarT, Dummy> time_;

  ///
  /// Number of dimensions
  ///
  std::size_t num_dims_;

  ///
  /// Number of integration points
  ///
  std::size_t num_pts_;

  ///
  /// Number of integration points
  ///
  std::size_t num_nodes_;

  ///
  /// Number of integration points
  ///
  std::size_t final_time_;
};

}  // namespace LCM

#endif
