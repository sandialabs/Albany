//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_RESPONSE_SMB_MISMATCH_HPP
#define LANDICE_RESPONSE_SMB_MISMATCH_HPP

#include "Albany_SacadoTypes.hpp"
#include "PHAL_SeparableScatterScalarResponse.hpp"
#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_Cubature.hpp"

namespace LandIce {
/**
 * \brief Response Description
 */
template<typename EvalT, typename Traits, typename ThicknessScalarType>
class ResponseSMBMismatch :
  public PHAL::SeparableScatterScalarResponseWithExtrudedParams<EvalT,Traits>
{
public:
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;
  typedef typename EvalT::ParamScalarT ParamScalarT;

  ResponseSMBMismatch(Teuchos::ParameterList& p,
     const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
           PHX::FieldManager<Traits>& vm);

  void preEvaluate(typename Traits::PreEvalData d);

  void evaluateFields(typename Traits::EvalData d);

  void postEvaluate(typename Traits::PostEvalData d);

private:
  using Base = PHAL::SeparableScatterScalarResponseWithExtrudedParams<EvalT,Traits>;

  Teuchos::RCP<const Teuchos::ParameterList> getValidResponseParameters() const;

  using GradThicknessScalarType = typename Albany::StrongestScalarType<MeshScalarT,ThicknessScalarType>::type;

  std::string surfaceSideName;
  std::string basalSideName;

  Albany::LocalSideSetInfo sideSet;

  unsigned int numSideNodes;
  unsigned int numBasalQPs;
  unsigned int numSideDims;

  PHX::MDField<const ScalarT,Side,QuadPoint>                     flux_div;
  PHX::MDField<const RealType,Side,QuadPoint>                    SMB;
  PHX::MDField<const RealType,Side,QuadPoint>                    SMBRMS;
  PHX::MDField<const RealType,Side,QuadPoint>                    obs_thickness;
  PHX::MDField<const RealType,Side,QuadPoint>                    thicknessRMS;
  PHX::MDField<const ThicknessScalarType,Side,QuadPoint>         thickness;
  PHX::MDField<const GradThicknessScalarType,Side,QuadPoint,Dim> grad_thickness;
  PHX::MDField<const MeshScalarT,Side,QuadPoint>                 w_measure_2d;
  PHX::MDField<const MeshScalarT,Side,QuadPoint,Dim,Dim>         tangents;

  ScalarT p_resp, p_reg, resp, reg, p_misH, misH;
  double scaling, alpha, alphaH, alphaSMB;

  Teuchos::RCP<const CellTopologyData> cell_topo;
};

} // namespace LandIce

#endif // LANDICE_RESPONSE_SMB_MISMATCH_HPP
