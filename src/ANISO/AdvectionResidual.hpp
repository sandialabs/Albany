//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ADVECTION_RESIDUAL_HPP
#define ADVECTION_RESIDUAL_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace ANISO {

template<typename EvalT, typename Traits>
class AdvectionResidual : 
  public PHX::EvaluatorWithBaseImpl<Traits>,
  public PHX::EvaluatorDerived<EvalT, Traits>
{

  public:

    AdvectionResidual(
        const Teuchos::ParameterList& p,
        const Teuchos::RCP<Albany::Layouts>& dl);

    void 
      postRegistrationSetup(
          typename Traits::SetupData d,
          PHX::FieldManager<Traits>& vm);

    void 
      evaluateFields(typename Traits::EvalData d);

  private:

    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;

    int num_nodes;
    int num_qps;
    int num_dims;

    double kappa;
    Teuchos::Array<double> alpha;

    PHX::MDField<MeshScalarT, Cell, Node, QuadPoint> w_bf;
    PHX::MDField<MeshScalarT, Cell, Node, QuadPoint, Dim> w_grad_bf;
    PHX::MDField<ScalarT, Cell, QuadPoint> phi;
    PHX::MDField<ScalarT, Cell, QuadPoint, Dim> grad_phi;
    PHX::MDField<ScalarT, Cell, Node> residual;

};

}

#endif
