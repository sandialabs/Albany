//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_RESPONSE_SURFACE_VELOCITY_MISMATCH_HPP
#define LANDICE_RESPONSE_SURFACE_VELOCITY_MISMATCH_HPP

#include "PHAL_SeparableScatterScalarResponse.hpp"

namespace LandIce {
/**
 * \brief Response Description
 */
  template<typename EvalT, typename Traits>
  class ResponseSurfaceVelocityMismatch : public PHAL::SeparableScatterScalarResponseWithExtrudedParams<EvalT,Traits>
  {
  public:
    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;
    typedef typename EvalT::ParamScalarT ParamScalarT;

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
    Albany::LocalSideSetInfo sideSet;

    unsigned int numSideNodes;
    unsigned int numBasalQPs;
    unsigned int numSurfaceQPs;
    unsigned int numSideDims;

    PHX::MDField<const ScalarT,Side,QuadPoint,VecDim>  velocity;
    PHX::MDField<const RealType,Side,QuadPoint,VecDim> observedVelocity;
    PHX::MDField<const RealType,Side,QuadPoint,VecDim> observedVelocityRMS;
    PHX::MDField<const RealType,Side,QuadPoint>        observedVelocityMagnitudeRMS;
    PHX::MDField<const MeshScalarT,Side,QuadPoint>     w_measure_surface;

    //PHX::MDField<const MeshScalarT,Cell,Side,QuadPoint,Dim,Dim>  metric_surface;

    // Stuff for stifferning regularization
    std::string basalSideName;
    PHX::MDField<const ParamScalarT,Side,QuadPoint,Dim>    grad_stiffening;
    PHX::MDField<const ParamScalarT,Side,QuadPoint>        stiffening;
    PHX::MDField<const MeshScalarT,Side,QuadPoint>         w_measure_basal;
    PHX::MDField<const MeshScalarT,Side,QuadPoint,Dim,Dim> metric_basal;

    // Stuff for beta regularization
    std::vector<Teuchos::RCP<Teuchos::ParameterList>> beta_reg_params;
    std::vector<PHX::MDField<const ParamScalarT,Side,QuadPoint,Dim>>    grad_beta_vec;
    std::vector<PHX::MDField<const MeshScalarT,Side,QuadPoint>>         w_measure_beta_vec;
    std::vector<PHX::MDField<const MeshScalarT,Side,QuadPoint,Dim,Dim>> metric_beta_vec;
    Teuchos::RCP<const CellTopologyData> cell_topo;

    PHX::MDField<const ParamScalarT> grad_beta;
    PHX::MDField<const MeshScalarT>  metric;
    PHX::MDField<const MeshScalarT>  w_measure;

    ScalarT p_resp, p_reg, resp, reg, p_reg_stiffening,reg_stiffening;
    double scaling, alpha, asinh_scaling, alpha_stiffening;
    bool scalarRMS;

  };

} // Namespace LandIce

#endif // LANDICE_RESPONSE_SURFACE_VELOCITY_MISMATCH_HPP
