//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_UPDATEZCOORDINATE_HPP
#define FELIX_UPDATEZCOORDINATE_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Sacado_ParameterAccessor.hpp" 
#include "Albany_Layouts.hpp"

namespace FELIX {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template<typename EvalT, typename Traits>
class UpdateZCoordinate : public PHX::EvaluatorWithBaseImpl<Traits>,
		    public PHX::EvaluatorDerived<EvalT, Traits> {

public:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  UpdateZCoordinate(const Teuchos::ParameterList& p,
            const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  // Input:
  PHX::MDField<MeshScalarT, Cell, Node,Dim> coordVecIn;
  PHX::MDField<MeshScalarT, Cell, Node> dH;
  PHX::MDField<MeshScalarT, Cell, Node> elevation;
  PHX::MDField<MeshScalarT, Cell, Node> H0;

  // Output:
  PHX::MDField<MeshScalarT, Cell, Node, Dim> coordVecOut;

  unsigned int numDims, numNodes;
};
}

#endif
