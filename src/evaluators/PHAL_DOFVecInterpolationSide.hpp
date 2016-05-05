//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_DOF_VEC_INTERPOLATION_SIDE_HPP
#define PHAL_DOF_VEC_INTERPOLATION_SIDE_HPP 1

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"
#include "Albany_DataTypes.hpp"

namespace PHAL {
/** \brief Finite Element InterpolationSide Evaluator

    This evaluator interpolates nodal DOF vector values to quad points.

*/

template<typename EvalT, typename Traits, typename ScalarT>
class DOFVecInterpolationSideBase : public PHX::EvaluatorWithBaseImpl<Traits>,
                                    public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  DOFVecInterpolationSideBase (const Teuchos::ParameterList& p,
                               const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& vm);

  void evaluateFields (typename Traits::EvalData d);

private:

  typedef ScalarT ParamScalarT;

  std::string sideSetName;

  // Input:
  //! Values at nodes
  PHX::MDField<ParamScalarT,Cell,Side,Node,Dim> val_node;
  //! Basis Functions
  PHX::MDField<RealType,Cell,Side,Node,QuadPoint> BF;

  // Output:
  //! Values at quadrature points
  PHX::MDField<ParamScalarT,Side,Cell,QuadPoint,Dim> val_qp;

  int numSideNodes;
  int numSideQPs;
  int vecDim;
};

// Some shortcut names
template<typename EvalT, typename Traits>
using DOFVecInterpolationSide = DOFVecInterpolationSideBase<EvalT,Traits,typename EvalT::ScalarT>;

template<typename EvalT, typename Traits>
using DOFVecInterpolationSideMesh = DOFVecInterpolationSideBase<EvalT,Traits,typename EvalT::MeshScalarT>;

template<typename EvalT, typename Traits>
using DOFVecInterpolationSideParam = DOFVecInterpolationSideBase<EvalT,Traits,typename EvalT::ParamScalarT>;

} // Namespace PHAL

#endif // PHAL_DOF_VEC_INTERPOLATION_SIDE_HPP
