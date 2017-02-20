//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_DOF_VEC_GRAD_INTERPOLATION_SIDE_HPP
#define PHAL_DOF_VEC_GRAD_INTERPOLATION_SIDE_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"

namespace PHAL {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOFVec values to their
    gradients at quad points.

*/

template<typename EvalT, typename Traits, typename ScalarT>
class DOFVecGradInterpolationSideBase : public PHX::EvaluatorWithBaseImpl<Traits>,
                                        public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  DOFVecGradInterpolationSideBase (const Teuchos::ParameterList& p,
                                   const Teuchos::RCP<Albany::Layouts>& dl_side);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& vm);

  void evaluateFields (typename Traits::EvalData d);

private:

  typedef typename EvalT::MeshScalarT MeshScalarT;
  std::string sideSetName;

  // Input:
  //! Values at nodes
  PHX::MDField<ScalarT,Side,Cell,Node,VecDim> val_node;
  //! Basis Functions
  PHX::MDField<MeshScalarT,Side,Cell,Node,QuadPoint,Dim> gradBF;

  // Output:
  //! Values at quadrature points
  PHX::MDField<ScalarT,Cell,Side,QuadPoint,VecDim,Dim> grad_qp;

  int numSideNodes;
  int numSideQPs;
  int numDims;
  int vecDim;
};

// Some shortcut names
template<typename EvalT, typename Traits>
using DOFVecGradInterpolationSide = DOFVecGradInterpolationSideBase<EvalT,Traits,typename EvalT::ScalarT>;

template<typename EvalT, typename Traits>
using DOFVecGradInterpolationSideMesh = DOFVecGradInterpolationSideBase<EvalT,Traits,typename EvalT::MeshScalarT>;

template<typename EvalT, typename Traits>
using DOFVecGradInterpolationSideParam = DOFVecGradInterpolationSideBase<EvalT,Traits,typename EvalT::ParamScalarT>;

} // Namespace PHAL

#endif // PHAL_DOF_VEC_GRAD_INTERPOLATION_SIDE_HPP
