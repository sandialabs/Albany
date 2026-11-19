//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_RESPONSE_SQUARED_L2_DIFFERENCE_HPP
#define PHAL_RESPONSE_SQUARED_L2_DIFFERENCE_HPP

#include "PHAL_SeparableScatterScalarResponse.hpp"
#include "Albany_KokkosUtils.hpp"

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

  typedef typename EvalT::ScalarT     ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  int getLayout (const Teuchos::RCP<Albany::Layouts>& dl, const std::string& rank, Teuchos::RCP<PHX::DataLayout>& layout);

  int numQPs;
  int fieldDim;
  std::vector<PHX::ExecutionSpace::size_type> dims;
  int dims_2;
  int dims_3;

  bool target_value;
  TargetScalarT target_value_val;
  RealType scaling;

  PHX::MDField<const SourceScalarT>             sourceField;
  PHX::MDField<const TargetScalarT>             targetField;

  PHX::MDField<const MeshScalarT,Cell,QuadPoint>   w_measure;

  // Device buffer for the reduction of the cell contributions
  Kokkos::View<RealType*, PHX::Device> m_sums;

public:
  using ExecutionSpace = typename PHX::ExecutionSpace;
  using RangePolicy = Kokkos::RangePolicy<ExecutionSpace>;

  // Sums the value and derivatives of local(:,0) as an array of NComp doubles
  // (NComp = 1 + number of derivatives of a statically sized, first-order AD type)
  template<typename LocalView, int NComp>
  struct SumScalarComponents {
    using value_type = RealType[];
    using size_type  = int;
    LocalView local;
    int value_count = NComp;

    KOKKOS_INLINE_FUNCTION
    void operator() (const int cell, value_type sum) const {
      const auto& x = local(cell,0);
      if constexpr (NComp==1) {
        sum[0] += x;
      } else {
        sum[0] += x.val();
        for (int k=1; k<NComp; ++k)
          sum[k] += x.fastAccessDx(k-1);
      }
    }
    KOKKOS_INLINE_FUNCTION
    void init (value_type sum) const {
      for (int k=0; k<NComp; ++k) sum[k] = 0;
    }
    KOKKOS_INLINE_FUNCTION
    void join (value_type dst, const value_type src) const {
      for (int k=0; k<NComp; ++k) dst[k] += src[k];
    }
  };

  // Adds the NComp doubles in sums (value and derivatives) to global(0)
  template<typename GlobalView, typename SumsView, int NComp>
  struct AddScalarComponents {
    GlobalView global;
    SumsView   sums;

    KOKKOS_INLINE_FUNCTION
    void operator() (const int) const {
      auto&& g = global(0);
      if constexpr (NComp==1) {
        g += sums(0);
      } else {
        g.val() += sums(0);
        for (int k=1; k<NComp; ++k)
          g.fastAccessDx(k-1) += sums(k);
      }
    }
  };
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
