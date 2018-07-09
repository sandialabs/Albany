//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ATO_TENSORAVERAGERESPONSE_HPP
#define ATO_TENSORAVERAGERESPONSE_HPP

#include "PHAL_SeparableScatterScalarResponse.hpp"
#include "ATO_TopoTools.hpp"
#include "Kokkos_Vector.hpp"


/**
 * \brief Response Description
 */
namespace ATO
{
  template<typename EvalT, typename Traits>
  class TensorAverageResponseSpec :
    public PHAL::SeparableScatterScalarResponse<EvalT,Traits>
  {
  public:
    typedef typename EvalT::ScalarT ScalarT;

    void postEvaluate(typename Traits::PostEvalData d);

  protected:
    RealType global_measure;
  };

  /******************************************************************************/
  // Specialization: Jacobian
  /******************************************************************************/
  template<typename Traits>
  class TensorAverageResponseSpec<PHAL::AlbanyTraits::Jacobian,Traits> :
    public PHAL::SeparableScatterScalarResponse<PHAL::AlbanyTraits::Jacobian,Traits>
  {
  public:
    typedef PHAL::AlbanyTraits::Jacobian EvalT;
    typedef typename EvalT::ScalarT ScalarT;

    void postEvaluate(typename Traits::PostEvalData d);

  protected:
    RealType global_measure;
  };
  /******************************************************************************/
  // Specialization: DistParamDeriv
  /******************************************************************************/
  template<typename Traits>
  class TensorAverageResponseSpec<PHAL::AlbanyTraits::DistParamDeriv,Traits> :
    public PHAL::SeparableScatterScalarResponse<PHAL::AlbanyTraits::DistParamDeriv,Traits>
  {
  public:
    typedef PHAL::AlbanyTraits::DistParamDeriv EvalT;
    typedef typename EvalT::ScalarT ScalarT;

    void postEvaluate(typename Traits::PostEvalData d);

  protected:
    RealType global_measure;
  };

  template<typename EvalT, typename Traits>
  class TensorAverageResponse : public TensorAverageResponseSpec<EvalT,Traits>
  {
  public:
    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;
    typedef typename EvalT::ParamScalarT ParamScalarT;

    TensorAverageResponse(Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl);

    void postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& vm);
    void preEvaluate(typename Traits::PreEvalData d);
    void evaluateFields(typename Traits::EvalData d);
    void postEvaluate(typename Traits::PostEvalData d);

  private:

    using TensorAverageResponseSpec<EvalT,Traits>::global_measure;

    std::string FName;
    static const std::string className;
    PHX::MDField<const ScalarT> field;
    PHX::MDField<const MeshScalarT,Cell,QuadPoint> weights;
    PHX::MDField<const RealType,Cell,Node,QuadPoint> BF;
    PHX::MDField<const ParamScalarT,Cell,Node> topo;
    Teuchos::RCP<Topology> topology;
    Teuchos::RCP< PHX::Tag<ScalarT> > objective_tag;

    Kokkos::vector<int> component0, component1;
    int tensorRank;

    int functionIndex;

    RealType local_measure;

  };
}

#endif
