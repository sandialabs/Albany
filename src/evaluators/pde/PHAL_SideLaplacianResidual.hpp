//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_SIDE_LAPLACIAN_RESIDUAL_HPP
#define PHAL_SIDE_LAPLACIAN_RESIDUAL_HPP 1

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace PHAL
{

template<typename EvalT, typename Traits>
class SideLaplacianResidual : public PHX::EvaluatorWithBaseImpl<Traits>,
                           public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  typedef typename EvalT::ScalarT ScalarT;

  SideLaplacianResidual (const Teuchos::ParameterList& p,
                         const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:

  void evaluateFieldsCell (typename Traits::EvalData d);
  void evaluateFieldsSide (typename Traits::EvalData d);

  typedef typename EvalT::MeshScalarT                   MeshScalarT;

  // Input:
  PHX::MDField<RealType>                                BF;
  PHX::MDField<RealType>                                GradBF;
  PHX::MDField<RealType>                                w_measure;
  PHX::MDField<RealType,Cell,Side,QuadPoint,Dim,Dim>    metric; // Only used in 2D, so we know the layout

  PHX::MDField<ScalarT>                                 u;
  PHX::MDField<ScalarT>                                 grad_u;

  // Output:
  PHX::MDField<ScalarT,Cell,Node>                       residual; // Always a 3D residual, so we know the layout

  std::string                     sideSetName;
  std::vector<std::vector<int> >  sideNodes;

  int spaceDim;
  int gradDim;
  int numSideNodes;
  int numSideQPs;
  int numNodes;
  int numQPs;

  bool sideSetEquation;
};

} // Namespace PHAL

#endif // PHAL_SIDE_LAPLACIAN_RESIDUAL_HPP
