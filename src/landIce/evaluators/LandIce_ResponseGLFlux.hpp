//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_RESPONSE_GL_FLUX_HPP
#define LANDICE_RESPONSE_GL_FLUX_HPP

#include "Albany_SacadoTypes.hpp"
#include "PHAL_SeparableScatterScalarResponse.hpp"
#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_Cubature.hpp"

namespace LandIce {
/**
 * \brief Response Description
 * It computes the ice flux [kg yr^{-1}] through the grounding line
 */
template<typename EvalT, typename Traits, typename ThicknessST>
class ResponseGLFlux :
  public PHAL::SeparableScatterScalarResponseWithExtrudedParams<EvalT,Traits>
{
public:
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;
  typedef typename EvalT::ParamScalarT ParamScalarT;

  ResponseGLFlux(Teuchos::ParameterList& p,
     const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
           PHX::FieldManager<Traits>& vm);

  void preEvaluate(typename Traits::PreEvalData d);

  void evaluateFields(typename Traits::EvalData d);

  void postEvaluate(typename Traits::PostEvalData d);

private:
  using Base = PHAL::SeparableScatterScalarResponseWithExtrudedParams<EvalT,Traits>;

  Teuchos::RCP<const Teuchos::ParameterList> getValidResponseParameters() const;

  std::string basalSideName;

  unsigned int numSideNodes;
  unsigned int numSideDims;

  using xyST = typename Albany::StrongestScalarType<MeshScalarT,ThicknessST>::type;

  PHX::MDField<const ScalarT,Side,Node,Dim>         avg_vel;     //[m yr^{-1}]
  PHX::MDField<const ThicknessST,Side,Node>         thickness;   //[km]
  PHX::MDField<const ThicknessST,Side,Node>         bed;         //[km]
  PHX::MDField<const MeshScalarT,Side,Node,Dim>     coords;      //[km]

  Kokkos::DynRankView<ThicknessST, PHX::Device>     gl_func,H;
  Kokkos::DynRankView<xyST, PHX::Device>            x,y;
  Kokkos::DynRankView<ScalarT, PHX::Device>         velx,vely;

  double rho_i, rho_w;  //[kg m^{-3}]
  double scaling;       //[adim]

  Albany::LocalSideSetInfo sideSet;

protected:
  using ExecutionSpace = typename PHX::Device::execution_space;
  using RangePolicy = Kokkos::RangePolicy<ExecutionSpace>;
};

} // namespace LandIce

#endif // LANDICE_RESPONSE_GL_FLUX_HPP
