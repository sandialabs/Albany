//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_COLUMN_COUPLING_TEST_RESIDUAL_HPP
#define LANDICE_COLUMN_COUPLING_TEST_RESIDUAL_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "PHAL_Dimension.hpp"
#include "Albany_Layouts.hpp"
#include "Albany_ScalarOrdinalTypes.hpp"

namespace LandIce
{

template<typename EvalT, typename Traits>
class ColumnCouplingTestResidual : public PHX::EvaluatorWithBaseImpl<Traits>,
                                   public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  typedef typename EvalT::ScalarT ScalarT;

  ColumnCouplingTestResidual (const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:

  std::string sideSetName;

  // Input:
  PHX::MDField<const ScalarT,Side,Node> solution;
  PHX::MDField<const RealType,Side,Node> surf_height;

  // Output:
  PHX::MDField<ScalarT,Cell,Node>       residual;
};

} // Namespace LandIce

#endif // LANDICE_COLUMN_COUPLING_TEST_RESIDUAL_HPP
