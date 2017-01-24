//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_RESPONSE_SQUARED_L2_DIFFERENCE_SIDE_HPP
#define PHAL_RESPONSE_SQUARED_L2_DIFFERENCE_SIDE_HPP

#include "PHAL_SeparableScatterScalarResponse.hpp"

namespace PHAL {
/**
 * \brief Response Description
 */
template<typename EvalT, typename Traits, typename SourceScalarT, typename TargetScalarT>
class ResponseSquaredL2DifferenceSideBase : public PHAL::SeparableScatterScalarResponse<EvalT,Traits>
{
public:

  ResponseSquaredL2DifferenceSideBase (Teuchos::ParameterList& p,
                                  const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& vm);

  void preEvaluate (typename Traits::PreEvalData d);

  void evaluateFields (typename Traits::EvalData d);

  void postEvaluate (typename Traits::PostEvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;

  int getLayout (const Teuchos::RCP<Albany::Layouts>& dl, const std::string& rank, Teuchos::RCP<PHX::DataLayout>& layout);

  std::string sideSetName;

  int sideDim;
  int numQPs;
  int fieldDim;

  bool target_zero;
  RealType scaling;

  PHX::MDField<SourceScalarT>                     sourceField;
  PHX::MDField<TargetScalarT>                     targetField;

  PHX::MDField<RealType,Cell,Side,QuadPoint>      w_measure;
};

//-- SourceScalarT = ScalarT
template<typename EvalT, typename Traits>
using ResponseSquaredL2DifferenceSideSST_TST  = ResponseSquaredL2DifferenceSideBase<EvalT,Traits,typename EvalT::ScalarT,typename EvalT::ScalarT>;

template<typename EvalT, typename Traits>
using ResponseSquaredL2DifferenceSideSST_TMST = ResponseSquaredL2DifferenceSideBase<EvalT,Traits,typename EvalT::ScalarT,typename EvalT::MeshScalarT>;

template<typename EvalT, typename Traits>
using ResponseSquaredL2DifferenceSideSST_TPST = ResponseSquaredL2DifferenceSideBase<EvalT,Traits,typename EvalT::ScalarT,typename EvalT::ParamScalarT>;

//-- SourceScalarT = ParamScalarT
template<typename EvalT, typename Traits>
using ResponseSquaredL2DifferenceSideSPST_TST  = ResponseSquaredL2DifferenceSideBase<EvalT,Traits,typename EvalT::ParamScalarT,typename EvalT::ScalarT>;

template<typename EvalT, typename Traits>
using ResponseSquaredL2DifferenceSideSPST_TMST = ResponseSquaredL2DifferenceSideBase<EvalT,Traits,typename EvalT::ParamScalarT,typename EvalT::MeshScalarT>;

template<typename EvalT, typename Traits>
using ResponseSquaredL2DifferenceSideSPST_TPST = ResponseSquaredL2DifferenceSideBase<EvalT,Traits,typename EvalT::ParamScalarT,typename EvalT::ParamScalarT>;

//-- SourceScalarT = MeshScalarT
template<typename EvalT, typename Traits>
using ResponseSquaredL2DifferenceSideSMST_TST  = ResponseSquaredL2DifferenceSideBase<EvalT,Traits,typename EvalT::MeshScalarT,typename EvalT::ScalarT>;

template<typename EvalT, typename Traits>
using ResponseSquaredL2DifferenceSideSMST_TMST = ResponseSquaredL2DifferenceSideBase<EvalT,Traits,typename EvalT::MeshScalarT,typename EvalT::MeshScalarT>;

template<typename EvalT, typename Traits>
using ResponseSquaredL2DifferenceSideSMST_TPST = ResponseSquaredL2DifferenceSideBase<EvalT,Traits,typename EvalT::MeshScalarT,typename EvalT::ParamScalarT>;

} // Namespace PHAL

#endif // PHAL_RESPONSE_SQUARED_L2_DIFFERENCE_SIDE_HPP
