//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_DOF_INTERPOLATION_HPP
#define PHAL_DOF_INTERPOLATION_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"
#include "PHAL_Utilities.hpp"

namespace PHAL {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template<typename EvalT, typename Traits, typename ScalarT>
class DOFInterpolationBase : public PHX::EvaluatorWithBaseImpl<Traits>,
                             public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  DOFInterpolationBase (const Teuchos::ParameterList& p,
                        const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  // Input:
  //! Values at nodes
  PHX::MDField<const ScalarT,Cell,Node> val_node;
  //! Basis Functions
  PHX::MDField<const RealType,Cell,Node,QuadPoint> BF;

  // Output:
  //! Values at quadrature points
  PHX::MDField<ScalarT,Cell,QuadPoint> val_qp;

  std::size_t numNodes;
  std::size_t numQPs;

  MDFieldMemoizer<Traits> memoizer;
};

// Some shortcut names
template<typename EvalT, typename Traits>
using DOFInterpolation = DOFInterpolationBase<EvalT,Traits,typename EvalT::ScalarT>;

template<typename EvalT, typename Traits>
using DOFInterpolationMesh = DOFInterpolationBase<EvalT,Traits,typename EvalT::MeshScalarT>;

template<typename EvalT, typename Traits>
using DOFInterpolationParam = DOFInterpolationBase<EvalT,Traits,typename EvalT::ParamScalarT>;

} // Namespace PHAL

#endif // PHAL_DOF_INTERPOLATION_HPP
