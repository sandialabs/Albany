//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_RESPONSE_SQUARED_L2_DIFFERENCE_HPP
#define PHAL_RESPONSE_SQUARED_L2_DIFFERENCE_HPP

#include "PHAL_SeparableScatterScalarResponse.hpp"

namespace PHAL {
/**
 * \brief Response Description
 */
template<typename EvalT, typename Traits, typename SourceScalarT, typename TargetScalarT>
class ResponseSquaredL2DifferenceBase : public PHAL::SeparableScatterScalarResponse<EvalT,Traits>
{
public:
  ResponseSquaredL2DifferenceBase (Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& vm);

  void preEvaluate (typename Traits::PreEvalData d);

  void evaluateFields (typename Traits::EvalData d);

  void postEvaluate (typename Traits::PostEvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;

  int getLayout (const Teuchos::RCP<Albany::Layouts>& dl, const std::string& rank, Teuchos::RCP<PHX::DataLayout>& layout);

  int numQPs;
  int fieldDim;
  std::vector<PHX::Device::size_type> dims;

  bool target_zero;
  RealType scaling;

  PHX::MDField<SourceScalarT>                sourceField;
  PHX::MDField<TargetScalarT>                targetField;

  PHX::MDField<RealType,Cell,QuadPoint>      w_measure;
};

// Some shortcut names

//-- SourceScalarT = ScalarT
template<typename EvalT, typename Traits>
using ResponseSquaredL2DifferenceSST_TST  = ResponseSquaredL2DifferenceBase<EvalT,Traits,typename EvalT::ScalarT,typename EvalT::ScalarT>;

template<typename EvalT, typename Traits>
using ResponseSquaredL2DifferenceSST_TMST = ResponseSquaredL2DifferenceBase<EvalT,Traits,typename EvalT::ScalarT,typename EvalT::MeshScalarT>;

template<typename EvalT, typename Traits>
using ResponseSquaredL2DifferenceSST_TPST = ResponseSquaredL2DifferenceBase<EvalT,Traits,typename EvalT::ScalarT,typename EvalT::ParamScalarT>;

//-- SourceScalarT = ParamScalarT
template<typename EvalT, typename Traits>
using ResponseSquaredL2DifferenceSPST_TST  = ResponseSquaredL2DifferenceBase<EvalT,Traits,typename EvalT::ParamScalarT,typename EvalT::ScalarT>;

template<typename EvalT, typename Traits>
using ResponseSquaredL2DifferenceSPST_TMST = ResponseSquaredL2DifferenceBase<EvalT,Traits,typename EvalT::ParamScalarT,typename EvalT::MeshScalarT>;

template<typename EvalT, typename Traits>
using ResponseSquaredL2DifferenceSPST_TPST = ResponseSquaredL2DifferenceBase<EvalT,Traits,typename EvalT::ParamScalarT,typename EvalT::ParamScalarT>;

//-- SourceScalarT = MeshScalarT
template<typename EvalT, typename Traits>
using ResponseSquaredL2DifferenceSMST_TST  = ResponseSquaredL2DifferenceBase<EvalT,Traits,typename EvalT::MeshScalarT,typename EvalT::ScalarT>;

template<typename EvalT, typename Traits>
using ResponseSquaredL2DifferenceSMST_TMST = ResponseSquaredL2DifferenceBase<EvalT,Traits,typename EvalT::MeshScalarT,typename EvalT::MeshScalarT>;

template<typename EvalT, typename Traits>
using ResponseSquaredL2DifferenceSMST_TPST = ResponseSquaredL2DifferenceBase<EvalT,Traits,typename EvalT::MeshScalarT,typename EvalT::ParamScalarT>;

} // Namespace PHAL

#endif // PHAL_RESPONSE_SQUARED_L2_DIFFERENCE_HPP
