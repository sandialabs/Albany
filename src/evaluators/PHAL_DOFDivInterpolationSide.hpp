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

template<typename EvalT, typename Traits>
class DOFDivInterpolationSide : public PHX::EvaluatorWithBaseImpl<Traits>,
                                 public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  DOFDivInterpolationSide (const Teuchos::ParameterList& p,
                        const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& vm);

  void evaluateFields (typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
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

} // Namespace PHAL

#endif // PHAL_DOF_DIV_INTERPOLATION_SIDE_HPP
