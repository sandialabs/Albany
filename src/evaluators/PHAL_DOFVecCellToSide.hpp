//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_DOF_VEC_CELL_TO_SIDE_HPP
#define PHAL_DOF_VEC_CELL_TO_SIDE_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"

namespace PHAL {
/** \brief Finite Element CellToSide Evaluator

    This evaluator creates a field defined cell-side wise from a cell wise field

*/

template<typename EvalT, typename Traits>
class DOFVecCellToSide : public PHX::EvaluatorWithBaseImpl<Traits>,
       public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  DOFVecCellToSide(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;

  std::string           sideSetName;

  // Input:
  //! Values at nodes
  PHX::MDField<ScalarT,Cell,Node,VecDim> val_cell;

  // Output:
  //! Values on side
  PHX::MDField<ScalarT,Cell,Side,Node,VecDim> val_side;

  std::vector<std::vector<int> >  sideNodes;

  int numSideNodes;
  int vecDim;
};

} // Namespace PHAL

#endif // DOF_VEC_CELL_TO_SIDE_HPP
