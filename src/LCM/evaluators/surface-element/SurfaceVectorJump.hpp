//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef SURFACEVECTORJUMP_HPP
#define SURFACEVECTORJUMP_HPP

#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_Cubature.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

#include "Albany_Layouts.hpp"

namespace LCM {
/** \brief

 Compute the jump of a vector on a midplane surface

 **/

template <typename EvalT, typename Traits>
class SurfaceVectorJump : public PHX::EvaluatorWithBaseImpl<Traits>,
                          public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  SurfaceVectorJump(
      const Teuchos::ParameterList&        p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  void
  evaluateFields(typename Traits::EvalData d);

 private:
  typedef typename EvalT::ScalarT     ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  //! Numerical integration rule
  Teuchos::RCP<Intrepid2::Cubature<PHX::Device>> cubature_;

  //! Finite element basis for the midplane
  Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>>
      intrepid_basis_;

  //! Vector to take the jump of
  PHX::MDField<const ScalarT, Cell, Vertex, Dim> vector_;

  // Reference Cell Views
  Kokkos::DynRankView<RealType, PHX::Device> ref_values_;

  Kokkos::DynRankView<RealType, PHX::Device> ref_grads_;

  Kokkos::DynRankView<RealType, PHX::Device> ref_points_;

  Kokkos::DynRankView<RealType, PHX::Device> ref_weights_;

  // Output:
  PHX::MDField<ScalarT, Cell, QuadPoint, Dim> jump_;

  unsigned int workset_size_;

  unsigned int num_nodes_;

  unsigned int num_qps_;

  unsigned int num_dims_;

  unsigned int num_plane_nodes_;

  unsigned int num_plane_dims_;
};
}  // namespace LCM

#endif
