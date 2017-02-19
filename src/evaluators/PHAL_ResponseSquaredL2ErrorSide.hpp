//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_RESPONSE_SQUARED_L2_ERROR_SIDE_HPP
#define PHAL_RESPONSE_SQUARED_L2_ERROR_SIDE_HPP

#include "PHAL_SeparableScatterScalarResponse.hpp"

namespace PHAL {
/**
 * \brief Response Description
 */
template<typename EvalT, typename Traits, typename TargetScalarT>
class ResponseSquaredL2ErrorSideBase : public PHAL::SeparableScatterScalarResponse<EvalT,Traits>
{
public:
  typedef typename EvalT::ScalarT     ScalarT;

  ResponseSquaredL2ErrorSideBase (Teuchos::ParameterList& p,
                                  const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& vm);

  void preEvaluate (typename Traits::PreEvalData d);

  void evaluateFields (typename Traits::EvalData d);

  void postEvaluate (typename Traits::PostEvalData d);

private:

  int getLayout (const Teuchos::RCP<Albany::Layouts>& dl, const std::string& rank, Teuchos::RCP<PHX::DataLayout>& layout);

  std::string sideSetName;

  int sideDim;
  int numQPs;
  int fieldDim;

  bool target_zero;
  RealType scaling;

  PHX::MDField<const ScalarT>                     computedField;
  PHX::MDField<const TargetScalarT>                 targetField;
  PHX::MDField<TargetScalarT>                   targetFieldEval;

  PHX::MDField<const RealType,Cell,Side,QuadPoint>  w_measure;
};

// Some shortcut names
template<typename EvalT, typename Traits>
using ResponseSquaredL2ErrorSideTargetST = ResponseSquaredL2ErrorSideBase<EvalT,Traits,typename EvalT::ScalarT>;

template<typename EvalT, typename Traits>
using ResponseSquaredL2ErrorSideTargetMST = ResponseSquaredL2ErrorSideBase<EvalT,Traits,typename EvalT::MeshScalarT>;

template<typename EvalT, typename Traits>
using ResponseSquaredL2ErrorSideTargetPST = ResponseSquaredL2ErrorSideBase<EvalT,Traits,typename EvalT::ParamScalarT>;

} // Namespace PHAL

#endif // PHAL_RESPONSE_SQUARED_L2_ERROR_SIDE_HPP
