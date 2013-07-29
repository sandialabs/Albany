//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_GATHER_COORDINATE_FROM_SOLUTIONVECTOR_HPP
#define PHAL_GATHER_COORDINATE_FROM_SOLUTIONVECTOR_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Epetra_Vector.h"

namespace PHAL {
/** \brief Gathers Coordinates values from the nodal fields of the field manager

    Currently makes an assumption that the stride is constant for dofs
    and that the nmber of dofs is equal to the size of the coordinates
    names vector.

*/

template<typename EvalT, typename Traits>
class GatherCoordinateFromSolutionVector : public PHX::EvaluatorWithBaseImpl<Traits>,
  public PHX::EvaluatorDerived<EvalT, Traits>  {

  public:

    GatherCoordinateFromSolutionVector(const Teuchos::ParameterList& p,
                                       const Teuchos::RCP<Albany::Layouts>& dl);

    void postRegistrationSetup(typename Traits::SetupData d,
                               PHX::FieldManager<Traits>& vm);

    void evaluateFields(typename Traits::EvalData d);

  private:

    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;

    PHX::MDField<MeshScalarT, Cell, Vertex, Dim> coordVec;
    PHX::MDField<MeshScalarT, Cell, Node, Dim> solutionVec;

    std::size_t worksetSize;
    std::size_t numVertices;
    std::size_t numNodes;
    std::size_t numDim;
};
}

#endif
