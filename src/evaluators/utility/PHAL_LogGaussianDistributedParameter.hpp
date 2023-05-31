//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_LOG_GAUSSIAN_COMBINATION_PARAMETER_HPP
#define PHAL_LOG_GAUSSIAN_COMBINATION_PARAMETER_HPP

#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"
#include "Teuchos_ParameterList.hpp"

namespace PHAL {
///
/// LogGaussianDistributedParameter
///
template<typename EvalT, typename Traits>
class LogGaussianDistributedParameter : 
  public PHX::EvaluatorWithBaseImpl<Traits>,
  public PHX::EvaluatorDerived<EvalT, Traits>
{
 private:
  typedef typename EvalT::ParamScalarT ParamScalarT;

 public:
  typedef typename EvalT::ScalarT   ScalarT;
  //typedef ParamNameEnum             EnumType;

  LogGaussianDistributedParameter (const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData workset);

protected:
  std::size_t numNodes;
  PHX::MDField<ScalarT,Cell,Node> logGaussian;
  PHX::MDField<const ScalarT,Cell,Node> gaussian;
  PHX::MDField<const RealType,Cell,Node> mean;
  RealType a, b;

  bool eval_on_side, mean_is_field;
  std::string sideSetName;
};

}  // Namespace PHAL

#endif  // PHAL_LOG_GAUSSIAN_COMBINATION_PARAMETER_HPP
