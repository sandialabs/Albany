//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_RESPONSE_THERMAL_ENERGYT_HPP
#define PHAL_RESPONSE_THERMAL_ENERGYT_HPP

#include "PHAL_SeparableScatterScalarResponseT.hpp"

namespace PHAL {
/** 
 * \brief Response Description
 * This response evaluate the thermal energy : e = int_{\Omega} \rho * c_{p} T
 * Where \Omega is the domain, \rho density, C_{p} specific heat and T is
 * temperature. 
 */
  template<typename EvalT, typename Traits>
  class ResponseThermalEnergyT : 
    public PHAL::SeparableScatterScalarResponseT<EvalT,Traits>
  {
  public:
    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;

    ResponseThermalEnergyT(Teuchos::ParameterList& p,
			  const Teuchos::RCP<Albany::Layouts>& dl);
  
    void postRegistrationSetup(typename Traits::SetupData d,
			       PHX::FieldManager<Traits>& vm);

    void preEvaluate(typename Traits::PreEvalData d);
  
    void evaluateFields(typename Traits::EvalData d);

    void postEvaluate(typename Traits::PostEvalData d);
	  
  private:
    Teuchos::RCP<const Teuchos::ParameterList> getValidResponseParameters() const;

    // temperature
    PHX::MDField<ScalarT> field;
    // time
//    PHX::MDField<ScalarT,Dummy> time;
//    PHX::MDField<ScalarT,Dummy> deltaTime;
    // coordinates
    PHX::MDField<MeshScalarT> coordVec;
    // Quadrature points
    PHX::MDField<MeshScalarT> weights;
    std::vector<PHX::DataLayout::size_type> field_dims;
    std::size_t numQPs;
    std::size_t numDims;
    // density: assumed constant in element block
    ScalarT density;
    // specific heat : assumed constant element block
    ScalarT heat_capacity;

      };
	
}

#endif
