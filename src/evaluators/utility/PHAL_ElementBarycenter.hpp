//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_ELEMENT_BARICENTER_HPP
#define PHAL_ELEMENT_BARICENTER_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"
#include "PHAL_Utilities.hpp"

namespace PHAL {

/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/
template<typename EvalT, typename Traits>
class ElementBarycenter : public PHX::EvaluatorWithBaseImpl<Traits>,
 			 public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  ElementBarycenter(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::MeshScalarT MeshScalarT;
  unsigned int numVertices, numDims;
  MDFieldMemoizer<Traits> memoizer;

  // Input coordinate
  PHX::MDField<const MeshScalarT,Cell,Vertex,Dim> coords;

  // Output:
  //! Coordinates (vector, and components) of baricenter
  PHX::MDField<MeshScalarT,Cell,Dim> bary;
  PHX::MDField<MeshScalarT,Cell>     bary_x;
  PHX::MDField<MeshScalarT,Cell>     bary_y;
  PHX::MDField<MeshScalarT,Cell>     bary_z;
};

} // namespace PHAL

#endif // PHAL_ELEMENT_BARICENTER_HPP
