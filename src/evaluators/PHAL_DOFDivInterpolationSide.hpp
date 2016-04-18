//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_DOF_DIV_INTERPOLATION_SIDE_HPP
#define PHAL_DOF_DIV_INTERPOLATION_SIDE_HPP 1

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"

namespace PHAL {
/** \brief Finite Element InterpolationSide Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template<typename EvalT, typename Traits, typename ScalarT>
class DOFDivInterpolationSideBase : public PHX::EvaluatorWithBaseImpl<Traits>,
                                    public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  DOFDivInterpolationSideBase (const Teuchos::ParameterList& p,
                               const Teuchos::RCP<Albany::Layouts>& dl_side);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& vm);

  void evaluateFields (typename Traits::EvalData d);

private:

  typedef typename EvalT::MeshScalarT MeshScalarT;

  std::string sideSetName;

  // Input:
  //! Values at nodes
  PHX::MDField<ScalarT,Cell,Side,Node, Dim> val_node;
  //! Basis Functions
  PHX::MDField<MeshScalarT,Cell,Side,Node,QuadPoint,Dim> gradBF;

  // Output:
  //! Values at quadrature points
  PHX::MDField<ScalarT,Cell,Side,QuadPoint> val_qp;

  int numSideNodes;
  int numSideQPs;
  int numDims;
};

// Some shortcut names
template<typename EvalT, typename Traits>
using DOFDivInterpolationSide = DOFDivInterpolationSideBase<EvalT,Traits,typename EvalT::ScalarT>;

template<typename EvalT, typename Traits>
using DOFDivInterpolationSideMesh = DOFDivInterpolationSideBase<EvalT,Traits,typename EvalT::MeshScalarT>;

template<typename EvalT, typename Traits>
using DOFDivInterpolationSideParam = DOFDivInterpolationSideBase<EvalT,Traits,typename EvalT::ParamScalarT>;

} // Namespace PHAL

#endif // PHAL_DOF_DIV_INTERPOLATION_SIDE_HPP
