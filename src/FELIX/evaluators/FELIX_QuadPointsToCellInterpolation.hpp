//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_QUAD_POINT_TO_CELL_INTERPOLATION_HPP
#define FELIX_QUAD_POINT_TO_CELL_INTERPOLATION_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Sacado_ParameterAccessor.hpp"
#include "Albany_Layouts.hpp"

namespace FELIX
{
/** \brief Average from Qp to Cell

    This evaluator averages the quadrature points values to
    obtain a single value for the whole cell

*/

template<typename EvalT, typename Traits>
class QuadPointsToCellInterpolation : public PHX::EvaluatorWithBaseImpl<Traits>,
                                      public PHX::EvaluatorDerived<PHAL::AlbanyTraits::Residual, Traits>
{
public:

  typedef typename EvalT::ScalarT ScalarT;

  QuadPointsToCellInterpolation (const Teuchos::ParameterList& p,
                                 const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:

  int numQPt;

  // Input:
  PHX::MDField<ScalarT,Cell,QuadPoint>  field_qp;

  // Output:
  PHX::MDField<ScalarT,Cell>            field_cell;
};

} // Namespace FELIX

#endif // FELIX_QUAD_POINT_TO_CELL_INTERPOLATION_HPP
