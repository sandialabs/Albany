//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef TPSLAPLACERESID_HPP
#define TPSLAPLACERESID_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"
#include "Intrepid_CellTools.hpp"
#include "Intrepid_Cubature.hpp"

namespace PHAL {

/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template<typename EvalT, typename Traits>
class TPSLaplaceResid : public PHX::EvaluatorWithBaseImpl<Traits>,
  public PHX::EvaluatorDerived<EvalT, Traits>  {

  public:

    TPSLaplaceResid(const Teuchos::ParameterList& p,
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

    Teuchos::RCP<Intrepid::Cubature<RealType> > cubature;
    Teuchos::RCP<shards::CellTopology> cellType;
    Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > intrepidBasis;

    // Temporary FieldContainers
    Intrepid::FieldContainer<RealType> grad_at_cub_points;
    Intrepid::FieldContainer<RealType> refPoints;
    Intrepid::FieldContainer<RealType> refWeights;
    Intrepid::FieldContainer<ScalarT> jacobian;
    Intrepid::FieldContainer<ScalarT> jacobian_det;

    // Output:
    PHX::MDField<ScalarT, Cell, Node, Dim> solnResidual;

    unsigned int numQPs, numDims, numNodes, worksetSize;

};
}

#endif
