//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_RESPONSEGLFLUX_HPP
#define LANDICE_RESPONSEGLFLUX_HPP

#include "PHAL_SeparableScatterScalarResponse.hpp"
#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_Cubature.hpp"

namespace LandIce {
/**
 * \brief Response Description
 */
  template<typename EvalT, typename Traits>
  class ResponseGLFlux :
    public PHAL::SeparableScatterScalarResponse<EvalT,Traits>
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
    Teuchos::RCP<const Teuchos::ParameterList> getValidResponseParameters() const;

    std::string basalSideName;

    int numSideNodes;
    int numSideDims;

    PHX::MDField<const ScalarT,Cell,Side,Node,Dim>          avg_vel;
    PHX::MDField<const ParamScalarT,Cell,Side,Node>         thickness;
    PHX::MDField<const ParamScalarT,Cell,Side,Node>         bed;
    PHX::MDField<const MeshScalarT,Cell,Side,Node,Dim>      coords;
    Kokkos::DynRankView<ParamScalarT, PHX::Device>          gl_func,H,x,y;
    Kokkos::DynRankView<ScalarT, PHX::Device>               velx,vely;

    double rho_i, rho_w;
    double scaling;

    Teuchos::RCP<const CellTopologyData> cell_topo;
  };

}

#endif
