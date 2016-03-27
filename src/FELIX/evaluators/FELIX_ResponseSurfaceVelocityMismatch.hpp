//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_RESPONSE_SURFACE_VELOCITY_MISMATCH_HPP
#define FELIX_RESPONSE_SURFACE_VELOCITY_MISMATCH_HPP

//#include "FELIX_MeshRegion.hpp"
#include "PHAL_SeparableScatterScalarResponse.hpp"

namespace FELIX {
/**
 * \brief Response Description
 */
  template<typename EvalT, typename Traits>
  class ResponseSurfaceVelocityMismatch :
    public PHAL::SeparableScatterScalarResponse<EvalT,Traits>
  {
  public:
    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;
    typedef typename EvalT::ParamScalarT ParamScalarT;

    ResponseSurfaceVelocityMismatch(Teuchos::ParameterList& p,
       const std::map<std::string,Teuchos::RCP<Albany::Layouts>>& dls);

    ResponseSurfaceVelocityMismatch(Teuchos::ParameterList& p,
       const Teuchos::RCP<Albany::Layouts>& dl);

    void postRegistrationSetup(typename Traits::SetupData d,
             PHX::FieldManager<Traits>& vm);

    void preEvaluate(typename Traits::PreEvalData d);

    void evaluateFields(typename Traits::EvalData d);

    void postEvaluate(typename Traits::PostEvalData d);

  private:
    Teuchos::RCP<const Teuchos::ParameterList> getValidResponseParameters() const;

    std::string surfaceSideName;
    std::string basalSideName;

    int numSideNodes;
    int numBasalQPs;
    int numSurfaceQPs;
    int numSideDims;

    PHX::MDField<ScalarT,Cell,Side,QuadPoint,VecDim>       velocity;
    PHX::MDField<ParamScalarT,Cell,Side,QuadPoint,VecDim>  observedVelocity;
    PHX::MDField<ParamScalarT,Cell,Side,QuadPoint,VecDim>  observedVelocityRMS;
    PHX::MDField<ParamScalarT,Cell,Side,QuadPoint,Dim>     grad_beta;
    PHX::MDField<MeshScalarT,Cell,Side,QuadPoint>          w_measure_basal;
    PHX::MDField<MeshScalarT,Cell,Side,QuadPoint>          w_measure_surface;

    ScalarT p_resp, p_reg, resp, reg;
    double scaling, alpha, asinh_scaling;
  };

}

#endif
