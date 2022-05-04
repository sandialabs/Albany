//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_LINEAR_COMBINATION_PARAMETER_HPP
#define PHAL_LINEAR_COMBINATION_PARAMETER_HPP

#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"
#include "Teuchos_ParameterList.hpp"

#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_SharedParameter.hpp"
#include "Albany_UnivariateDistribution.hpp"

namespace PHAL {
///
/// LinearCombinationParameter
///
template<typename EvalT, typename Traits>
class LinearCombinationParameter : 
  public PHX::EvaluatorWithBaseImpl<Traits>,
  public PHX::EvaluatorDerived<EvalT, Traits>
{
 private:
  typedef typename EvalT::ParamScalarT ParamScalarT;

 public:
  typedef typename EvalT::ScalarT   ScalarT;

  LinearCombinationParameter (const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData workset);

protected:
  std::size_t numModes, numNodes;
  PHX::MDField<ScalarT,Cell,Node> val;
  std::vector<PHX::MDField<const ScalarT,Dim>>   coefficients_as_field; // or ParamScalarT
  std::vector<PHX::MDField<const RealType,Cell,Node>>   modes_val;

  bool eval_on_side;
  bool scale;
  std::string sideSetName;
  Teuchos::Array<double> scalar_scale;
};

}  // Namespace PHAL

#endif  // PHAL_LINEAR_COMBINATION_PARAMETER_HPP
