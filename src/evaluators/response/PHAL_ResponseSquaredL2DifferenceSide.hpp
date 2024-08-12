//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_RESPONSE_SQUARED_L2_DIFFERENCE_SIDE_HPP
#define PHAL_RESPONSE_SQUARED_L2_DIFFERENCE_SIDE_HPP

#include "PHAL_SeparableScatterScalarResponse.hpp"

#include "Albany_KokkosUtils.hpp"

namespace PHAL {
/**
 * \brief Response Description
 */
template<typename EvalT, typename Traits, typename SourceScalarT, typename TargetScalarT>
class ResponseSquaredL2DifferenceSideBase : public PHAL::SeparableScatterScalarResponseWithExtrudedParams<EvalT,Traits>
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

  using Base = PHAL::SeparableScatterScalarResponseWithExtrudedParams<EvalT,Traits>;
  typedef typename EvalT::ScalarT     ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  int getLayout (const Teuchos::RCP<Albany::Layouts>& dl, const std::string& rank, Teuchos::RCP<PHX::DataLayout>& layout);

  std::string sideSetName;
  Albany::LocalSideSetInfo sideSet;
  
  int sideDim;
  int numQPs;
  int fieldDim;
  int dims_2;
  int dims_3;
  std::vector<PHX::Device::size_type> dims;

  bool target_value, rmsScaling, extrudedParams, isFieldGradient;
  TargetScalarT target_value_val;
  RealType scaling;

  PHX::MDField<const SourceScalarT> sourceField;
  PHX::MDField<const TargetScalarT> targetField;
  PHX::MDField<const RealType> rootMeanSquareField;

  PHX::MDField<const MeshScalarT,Side,QuadPoint,Dim,Dim>   metric;
  PHX::MDField<const MeshScalarT,Side,QuadPoint>   w_measure;

  bool resp_depends_on_sol_column;

  using ExecutionSpace = typename PHX::Device::execution_space;
  using RangePolicy = Kokkos::RangePolicy<ExecutionSpace>;
};

//-- SourceScalarT = ScalarT
template<typename EvalT, typename Traits>
using ResponseSquaredL2DifferenceSideSST_TST = ResponseSquaredL2DifferenceSideBase<EvalT,Traits,typename EvalT::ScalarT,typename EvalT::ScalarT>;

template<typename EvalT, typename Traits>
using ResponseSquaredL2DifferenceSideSST_TMST = ResponseSquaredL2DifferenceSideBase<EvalT,Traits,typename EvalT::ScalarT,typename EvalT::MeshScalarT>;

template<typename EvalT, typename Traits>
using ResponseSquaredL2DifferenceSideSST_TPST = ResponseSquaredL2DifferenceSideBase<EvalT,Traits,typename EvalT::ScalarT,typename EvalT::ParamScalarT>;

template<typename EvalT, typename Traits>
using ResponseSquaredL2DifferenceSideSST_TRT = ResponseSquaredL2DifferenceSideBase<EvalT,Traits,typename EvalT::ScalarT,RealType>;

//-- SourceScalarT = ParamScalarT
template<typename EvalT, typename Traits>
using ResponseSquaredL2DifferenceSideSPST_TST = ResponseSquaredL2DifferenceSideBase<EvalT,Traits,typename EvalT::ParamScalarT,typename EvalT::ScalarT>;

template<typename EvalT, typename Traits>
using ResponseSquaredL2DifferenceSideSPST_TMST = ResponseSquaredL2DifferenceSideBase<EvalT,Traits,typename EvalT::ParamScalarT,typename EvalT::MeshScalarT>;

template<typename EvalT, typename Traits>
using ResponseSquaredL2DifferenceSideSPST_TPST = ResponseSquaredL2DifferenceSideBase<EvalT,Traits,typename EvalT::ParamScalarT,typename EvalT::ParamScalarT>;

template<typename EvalT, typename Traits>
using ResponseSquaredL2DifferenceSideSPST_TRT = ResponseSquaredL2DifferenceSideBase<EvalT,Traits,typename EvalT::ParamScalarT,RealType>;

//-- SourceScalarT = MeshScalarT
template<typename EvalT, typename Traits>
using ResponseSquaredL2DifferenceSideSMST_TST = ResponseSquaredL2DifferenceSideBase<EvalT,Traits,typename EvalT::MeshScalarT,typename EvalT::ScalarT>;

template<typename EvalT, typename Traits>
using ResponseSquaredL2DifferenceSideSMST_TMST = ResponseSquaredL2DifferenceSideBase<EvalT,Traits,typename EvalT::MeshScalarT,typename EvalT::MeshScalarT>;

template<typename EvalT, typename Traits>
using ResponseSquaredL2DifferenceSideSMST_TPST = ResponseSquaredL2DifferenceSideBase<EvalT,Traits,typename EvalT::MeshScalarT,typename EvalT::ParamScalarT>;

template<typename EvalT, typename Traits>
using ResponseSquaredL2DifferenceSideSMST_TRT = ResponseSquaredL2DifferenceSideBase<EvalT,Traits,typename EvalT::MeshScalarT,RealType>;

} // Namespace PHAL

#endif // PHAL_RESPONSE_SQUARED_L2_DIFFERENCE_SIDE_HPP
