//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_ELLIPTIC_2D_RESIDUAL_HPP
#define FELIX_ELLIPTIC_2D_RESIDUAL_HPP 1

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace FELIX
{

template<typename EvalT, typename Traits>
class Elliptic2DResidual : public PHX::EvaluatorWithBaseImpl<Traits>,
                           public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  typedef typename EvalT::ScalarT ScalarT;

  Elliptic2DResidual (const Teuchos::ParameterList& p,
                      const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:

  double getValue(const ScalarT& v);
  typedef typename EvalT::MeshScalarT                   MeshScalarT;

  // Input:
  PHX::MDField<const RealType>                                BF;
  PHX::MDField<const RealType>                                GradBF;
  PHX::MDField<const RealType>                                w_measure;
  PHX::MDField<const RealType,Cell,Side,QuadPoint,Dim,Dim>    inv_metric; // Only used in 2D, so we know the layout

  PHX::MDField<const ScalarT>                                 u;
  PHX::MDField<const ScalarT>                                 grad_u;
  PHX::MDField<const MeshScalarT>                             coords;

  // Output:
  PHX::MDField<ScalarT,Cell,Node>                       residual; // Always a 3D residual, so we know the layout

  std::string                     sideSetName;
  std::vector<std::vector<int> >  sideNodes;

  int gradDim;
  int numSideNodes;
  int numSideQPs;
  int numNodes;
  int numQPs;

  bool sideSetEquation;
};

} // Namespace FELIX

#endif // FELIX_ELLIPTIC_2D_RESIDUAL_HPP
