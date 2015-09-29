//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_DOF_SIDE_TO_CELL_HPP
#define PHAL_DOF_SIDE_TO_CELL_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"

namespace PHAL {
/** \brief Finite Element SideToCell Evaluator

    This evaluator creates a field defined cell wise from a celli-side wise field

*/

template<typename EvalT, typename Traits>
class DOFSideToCell : public PHX::EvaluatorWithBaseImpl<Traits>,
                      public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  DOFSideToCell(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;

  std::string                     sideSetName;
  std::vector<std::vector<int> >  sideNodes;

  // Input:
  //! Values at nodes
  PHX::MDField<ScalarT,Cell,Side,Node> val_side;

  // Output:
  //! Values on side
  PHX::MDField<ScalarT,Cell,Node> val_cell;

  int numSideNodes;
};

} // Namespace PHAL

#endif // DOF_SIDE_TO_CELL_HPP
