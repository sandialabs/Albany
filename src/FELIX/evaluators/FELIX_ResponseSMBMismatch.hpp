//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_RESPONSESMBMISMATCH_HPP
#define FELIX_RESPONSESMBMISMATCH_HPP

#include "PHAL_SeparableScatterScalarResponse.hpp"
#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_Cubature.hpp"

namespace FELIX {
/** 
 * \brief Response Description
 */
  template<typename EvalT, typename Traits>
  class ResponseSMBMismatch :
    public PHAL::SeparableScatterScalarResponse<EvalT,Traits>
  {
  public:
    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;
    typedef typename EvalT::ParamScalarT ParamScalarT;
    
    ResponseSMBMismatch(Teuchos::ParameterList& p,
       const std::map<std::string,Teuchos::RCP<Albany::Layouts>>& dls);
    
    ResponseSMBMismatch(Teuchos::ParameterList& p,
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

    PHX::MDField<ScalarT,Cell,Side,QuadPoint,VecDim>       averaged_velocity;
    PHX::MDField<ScalarT,Cell,Side,QuadPoint>              div_averaged_velocity;
    PHX::MDField<ParamScalarT,Cell,Side,QuadPoint>         SMB;
    PHX::MDField<RealType,Cell,Side,Node,QuadPoint>        BF_surface;
    PHX::MDField<ParamScalarT,Cell,Side,QuadPoint>         thickness;
    PHX::MDField<ParamScalarT,Cell,Side,QuadPoint,Dim>     grad_thickness;
    PHX::MDField<MeshScalarT,Cell,Side,QuadPoint>          w_measure_2d;

    ScalarT p_resp, p_reg, resp, reg;
    double scaling, alpha, asinh_scaling;

    Teuchos::RCP<const CellTopologyData> cell_topo;
  };

}

#endif
