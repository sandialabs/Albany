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
    bool useCollapsedSidesets;    
    Albany::LocalSideSetInfo sideSet;

    unsigned int numSideNodes;
    unsigned int numBasalQPs;
    unsigned int numSurfaceQPs;
    unsigned int numSideDims;

    // TODO: restore layout template arguments when removing old sideset layout
    PHX::MDField<const ScalarT>      velocity;                     // Side, QuadPoint, VecDim
    PHX::MDField<const RealType>     observedVelocity;             // Side, QuadPoint, VecDim
    PHX::MDField<const RealType>     observedVelocityRMS;          // Side, QuadPoint, VecDim
    PHX::MDField<const RealType>     observedVelocityMagnitudeRMS; // Side, QuadPoint
    PHX::MDField<const MeshScalarT>  w_measure_surface;            // Side, QuadPoint

    //PHX::MDField<const MeshScalarT,Cell,Side,QuadPoint,Dim,Dim>  metric_surface;

    // Stuff for stifferning regularization
    std::string basalSideName;
    PHX::MDField<const ParamScalarT> grad_stiffening;  // Side, QuadPoint, Dim
    PHX::MDField<const ParamScalarT> stiffening;       // Side, QuadPoint
    PHX::MDField<const MeshScalarT>  w_measure_basal;  // Side, QuadPoint
    PHX::MDField<const MeshScalarT>  metric_basal;     // Side, QuadPoint, Dim, Dim

    // Stuff for beta regularization
    std::vector<Teuchos::RCP<Teuchos::ParameterList>> beta_reg_params;
    std::vector<PHX::MDField<const ParamScalarT>>     grad_beta_vec;      // Side, QuadPoint, Dim
    std::vector<PHX::MDField<const MeshScalarT>>      w_measure_beta_vec; // Side, QuadPoint
    std::vector<PHX::MDField<const MeshScalarT>>      metric_beta_vec;    // Side, QuadPoint, Dim, Dim
    Teuchos::RCP<const CellTopologyData>              cell_topo;

    PHX::MDField<const ParamScalarT> grad_beta;
    PHX::MDField<const MeshScalarT> metric;
    PHX::MDField<const MeshScalarT> w_measure;

    ScalarT p_resp, p_reg, resp, reg, p_reg_stiffening,reg_stiffening;
    double scaling, alpha, asinh_scaling, alpha_stiffening;
    bool scalarRMS;

  };

} // Namespace LandIce

#endif // LANDICE_RESPONSE_SURFACE_VELOCITY_MISMATCH_HPP
