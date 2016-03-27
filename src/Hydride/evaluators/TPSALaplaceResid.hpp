//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef TPSALAPLACERESID_HPP
#define TPSALAPLACERESID_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"

namespace PHAL {

/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template<typename EvalT, typename Traits>
class TPSALaplaceResid : public PHX::EvaluatorWithBaseImpl<Traits>,
  public PHX::EvaluatorDerived<EvalT, Traits>  {

  public:

    TPSALaplaceResid(const Teuchos::ParameterList& p,
                         const Teuchos::RCP<Albany::Layouts>& dl);

    void postRegistrationSetup(typename Traits::SetupData d,
                               PHX::FieldManager<Traits>& vm);

    void evaluateFields(typename Traits::EvalData d);

  private:

    typedef typename EvalT::MeshScalarT MeshScalarT;
    typedef typename EvalT::ScalarT ScalarT;

    // Input:

    //! Coordinate vector at vertices being solved for
    PHX::MDField<ScalarT, Cell, Node, Dim> solnVec;
    PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> GradBF;
    PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> wGradBF;

    // Output:
    PHX::MDField<ScalarT, Cell, Node, Dim> solnResidual;

    unsigned int numQPs, numDims, numNodes, worksetSize;

};

}

#endif
