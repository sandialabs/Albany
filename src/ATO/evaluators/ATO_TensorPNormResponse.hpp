//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ATO_TENSORPNORMRESPONSE_HPP
#define ATO_TENSORPNORMRESPONSE_HPP

#include "PHAL_SeparableScatterScalarResponse.hpp"
#include "Albany_StateManager.hpp"
#include "ATO_TopoTools.hpp"
#include <MiniTensor.h>


/** 
 * \brief Response Description
 */
namespace ATO 
{
  template<typename EvalT, typename Traits>
  class TensorPNormResponseSpec : 
    public PHAL::SeparableScatterScalarResponse<EvalT,Traits>
  {
  public:
    typedef typename EvalT::ScalarT ScalarT;

    void postEvaluate(typename Traits::PostEvalData d);
	  
  protected:
    void saveState(RealType&, ScalarT){}
    Albany::StateManager* pStateMgr;
    double pVal;
  };
  /******************************************************************************/
  // Specialization: Residual
  /******************************************************************************/
  template<typename Traits>
  class TensorPNormResponseSpec<PHAL::AlbanyTraits::Residual,Traits> : 
    public PHAL::SeparableScatterScalarResponse<PHAL::AlbanyTraits::Residual,Traits>
  {
  public:
    typedef PHAL::AlbanyTraits::Residual EvalT;
    typedef typename EvalT::ScalarT ScalarT;
    
    void postEvaluate(typename Traits::PostEvalData d);
	  
  protected:
    void saveState(RealType& to, ScalarT from);
    Albany::StateManager* pStateMgr;
    double pVal;
  };

  /******************************************************************************/
  // Specialization: Jacobian
  /******************************************************************************/
  template<typename Traits>
  class TensorPNormResponseSpec<PHAL::AlbanyTraits::Jacobian,Traits> : 
    public PHAL::SeparableScatterScalarResponse<PHAL::AlbanyTraits::Jacobian,Traits>
  {
  public:
    typedef PHAL::AlbanyTraits::Jacobian EvalT;
    typedef typename EvalT::ScalarT ScalarT;
    
    void postEvaluate(typename Traits::PostEvalData d);
	  
  protected:
    void saveState(RealType& to, ScalarT from);
    Albany::StateManager* pStateMgr;
    double pVal;
  };
  /******************************************************************************/
  // Specialization: DistParamDeriv
  /******************************************************************************/
  template<typename Traits>
  class TensorPNormResponseSpec<PHAL::AlbanyTraits::DistParamDeriv,Traits> : 
    public PHAL::SeparableScatterScalarResponse<PHAL::AlbanyTraits::DistParamDeriv,Traits>
  {
  public:
    typedef PHAL::AlbanyTraits::DistParamDeriv EvalT;
    typedef typename EvalT::ScalarT ScalarT;
    
    void postEvaluate(typename Traits::PostEvalData d);
	  
  protected:
    void saveState(RealType&, ScalarT){}
    Albany::StateManager* pStateMgr;
    double pVal;
  };
  /******************************************************************************/
  // Specialization: SGJacobian
  /******************************************************************************/
#ifdef ALBANY_SG
  template<typename Traits>
  class TensorPNormResponseSpec<PHAL::AlbanyTraits::SGJacobian,Traits> : 
    public PHAL::SeparableScatterScalarResponse<PHAL::AlbanyTraits::SGJacobian,Traits>
  {
  public:
    typedef PHAL::AlbanyTraits::SGJacobian EvalT;
    typedef typename EvalT::ScalarT ScalarT;
    
    void postEvaluate(typename Traits::PostEvalData d);
	  
  protected:
    void saveState(RealType&, ScalarT){}
    Albany::StateManager* pStateMgr;
    double pVal;
  };

#endif 
#ifdef ALBANY_ENSEMBLE 

  /******************************************************************************/
  // Specialization: MPJacobian
  /******************************************************************************/
  template<typename Traits>
  class TensorPNormResponseSpec<PHAL::AlbanyTraits::MPJacobian,Traits> : 
    public PHAL::SeparableScatterScalarResponse<PHAL::AlbanyTraits::MPJacobian,Traits>
  {
  public:
    typedef PHAL::AlbanyTraits::MPJacobian EvalT;
    typedef typename EvalT::ScalarT ScalarT;
    
    void postEvaluate(typename Traits::PostEvalData d);
	  
  protected:
    void saveState(RealType&, ScalarT){}
    Albany::StateManager* pStateMgr;
    double pVal;
  };
#endif

	

  template<typename EvalT, typename Traits>
  class TensorPNormResponse : public TensorPNormResponseSpec<EvalT,Traits>
  {
  public:
    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;
    typedef typename EvalT::ParamScalarT ParamScalarT;
    
    TensorPNormResponse(Teuchos::ParameterList& p, 
                        const Teuchos::RCP<Albany::Layouts>& dl,
                        const Albany::MeshSpecsStruct* meshSpecs);
  
    void postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& vm);
    void preEvaluate(typename Traits::PreEvalData d);
    void evaluateFields(typename Traits::EvalData d);
    void postEvaluate(typename Traits::PostEvalData d);

  private:
    using TensorPNormResponseSpec<EvalT,Traits>::pVal;
    using TensorPNormResponseSpec<EvalT,Traits>::pStateMgr;
    using TensorPNormResponseSpec<EvalT,Traits>::saveState;

    void TransformResponse (int Cell, int QP, ScalarT& response_eff);
    std::string FName;
    static const std::string className;
    PHX::MDField<ScalarT> tensor;
    PHX::MDField<MeshScalarT,Cell,QuadPoint> qp_weights;
    PHX::MDField<RealType,Cell,Node,QuadPoint> BF;
    PHX::MDField<ParamScalarT,Cell,Node> topo;
    Teuchos::RCP< PHX::Tag<ScalarT> > objective_tag;

    Teuchos::RCP<Topology> topology;

    minitensor::Matrix<double> Cp;
    minitensor::Matrix<double> Cpp;
    int R;
    double yieldStress;

    int functionIndex;

  };
}

#endif
