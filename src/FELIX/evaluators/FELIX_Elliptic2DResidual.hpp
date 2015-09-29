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
  PHX::MDField<RealType,Cell,Side,Node,QuadPoint>       BF_side;
  PHX::MDField<RealType,Cell,Side,Node,QuadPoint,Dim>   GradBF_side;
  PHX::MDField<RealType,Cell,Side,QuadPoint,Dim,Dim>    inv_metric;
  PHX::MDField<RealType,Cell,Side,QuadPoint>            w_measure;
  PHX::MDField<ScalarT,Cell,Node>                       u_node;
  PHX::MDField<ScalarT,Cell,Side,QuadPoint>             u_side;
  PHX::MDField<ScalarT,Cell,Side,QuadPoint,Dim>         grad_u_side;

  PHX::MDField<RealType,Cell,Node,QuadPoint>            BF;
  PHX::MDField<RealType,Cell,Node,QuadPoint>            wBF;
  PHX::MDField<RealType,Cell,Node,QuadPoint,Dim>        wGradBF;
  PHX::MDField<ScalarT,Cell,QuadPoint>                  u;
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim>              grad_u;

  PHX::MDField<MeshScalarT,Cell,Vertex,Dim>             coords;

  // Output:
  PHX::MDField<ScalarT,Cell,Node>                       residual;

  std::string                     sideSetName;
  std::vector<std::vector<int> >  sideNodes;

  unsigned int gradDim;
  unsigned int numSideNodes,numSideQPs;
  unsigned int numNodes, numQPs;

  bool sideSetEquation;
};

} // Namespace FELIX

#endif // FELIX_ELLIPTIC_2D_RESIDUAL_HPP
