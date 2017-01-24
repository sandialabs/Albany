//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ADVECTION_KAPPA_HPP
#define ADVECTION_KAPPA_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace ANISO {

template<typename EvalT, typename Traits>
class AdvectionKappa : 
  public PHX::EvaluatorWithBaseImpl<Traits>,
  public PHX::EvaluatorDerived<EvalT, Traits>
{

  public:

    AdvectionKappa(
        const Teuchos::ParameterList& p,
        const Teuchos::RCP<Albany::Layouts>& dl);

    void postRegistrationSetup(
          typename Traits::SetupData d,
          PHX::FieldManager<Traits>& vm);

    void evaluateFields(typename Traits::EvalData d);

  private:

    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;

    int num_qps;
    int num_dims;
    std::string kappa_val;

    PHX::MDField<MeshScalarT, Cell, QuadPoint, Dim> coord;
    PHX::MDField<ScalarT, Cell, QuadPoint> kappa;

};

}

#endif
