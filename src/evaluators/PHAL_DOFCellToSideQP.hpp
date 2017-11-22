//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_DOF_CELL_TO_SIDE_QP_HPP
#define PHAL_DOF_CELL_TO_SIDE_QP_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"

namespace PHAL {
/** \brief Finite Element CellToSideQP Evaluator

    This evaluator creates a field defined cell-side wise from a cell wise field

*/

template<typename EvalT, typename Traits, typename ScalarT>
class DOFCellToSideQPBase : public PHX::EvaluatorWithBaseImpl<Traits>,
                          public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  DOFCellToSideQPBase (const Teuchos::ParameterList& p,
                     const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  std::string                     sideSetName;
  std::vector<std::vector<int> >  sideNodes;
  std::vector<int>                dims_cell;
  std::vector<int>                dims_side;

  // Input:
  PHX::MDField<const ScalarT>                           val_cell;
  PHX::MDField<const ScalarT,Cell,Side,Node,QuadPoint>  BF;

  // Output:
  PHX::MDField<ScalarT>                           val_side_qp;

  enum LayoutType
  {
    CELL_SCALAR = 1,
    CELL_VECTOR,
    CELL_TENSOR,
    NODE_SCALAR,
    NODE_VECTOR,
    NODE_TENSOR,
  };

  LayoutType layout;
};

// Some shortcut names
template<typename EvalT, typename Traits>
using DOFCellToSideQP = DOFCellToSideQPBase<EvalT,Traits,typename EvalT::ScalarT>;

template<typename EvalT, typename Traits>
using DOFCellToSideQPMesh = DOFCellToSideQPBase<EvalT,Traits,typename EvalT::MeshScalarT>;

template<typename EvalT, typename Traits>
using DOFCellToSideQPParam = DOFCellToSideQPBase<EvalT,Traits,typename EvalT::ParamScalarT>;

} // Namespace PHAL

#endif // DOF_CELL_TO_SIDE_HPP

