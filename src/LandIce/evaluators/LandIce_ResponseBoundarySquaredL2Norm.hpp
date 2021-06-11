//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_RESPONSE_BOUNDARY_SQUARED_L2_NORM_HPP
#define LANDICE_RESPONSE_BOUNDARY_SQUARED_L2_NORM_HPP

//#include "LandIce_MeshRegion.hpp"
#include "PHAL_SeparableScatterScalarResponse.hpp"

namespace LandIce {
/**
 * \brief Response Description
 */
  template<typename EvalT, typename Traits>
  class ResponseBoundarySquaredL2Norm : public PHAL::SeparableScatterScalarResponseWithExtrudedParams<EvalT,Traits>
  {
  public:
    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;
    typedef typename EvalT::ParamScalarT ParamScalarT;

    ResponseBoundarySquaredL2Norm(Teuchos::ParameterList& p,
       const Teuchos::RCP<Albany::Layouts>& dl);

    void postRegistrationSetup(typename Traits::SetupData d,
             PHX::FieldManager<Traits>& vm);

    void preEvaluate(typename Traits::PreEvalData d);

    void evaluateFields(typename Traits::EvalData d);

    void postEvaluate(typename Traits::PostEvalData d);

  private:
    Teuchos::RCP<const Teuchos::ParameterList> getValidResponseParameters() const;

    std::string sideName;

    unsigned int numSideNodes;
    unsigned int numSideQPs;
    unsigned int numSideDims;

    Albany::LocalSideSetInfo sideSet;

    PHX::MDField<const ScalarT,Side,Node>          solution;
    PHX::MDField<const MeshScalarT,Side,QuadPoint> w_side_measure;

    ScalarT p_reg, reg;
    double scaling;
    int offset;

  };

} // Namespace LandIce

#endif // LANDICE_RESPONSE_SURFACE_VELOCITY_MISMATCH_HPP
