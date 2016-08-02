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


  Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> > intrepidBasis;
  Teuchos::RCP<Intrepid2::Cubature<PHX::Device> > cubature;
  Kokkos::DynRankView<RealType, PHX::Device>    refPoints;
  Kokkos::DynRankView<RealType, PHX::Device>    refWeights;

  Kokkos::DynRankView<RealType, PHX::Device>    grad_at_cub_points;
  Kokkos::DynRankView<ScalarT, PHX::Device>     vcontra;

  const int numNodes;
  const int numDims;
  const int numQPs;
  const int numLevels;

  std::string myName;

  bool originalDiv;
};

}
#endif
