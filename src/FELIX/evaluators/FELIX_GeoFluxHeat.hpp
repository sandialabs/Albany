/*
 * FELIX_GeoFluxHeat.hpp
 *
 *  Created on: May 31, 2016
 *      Author: abarone
 */

#ifndef FELIX_GEOFLUXHEAT_HPP_
#define FELIX_GEOFLUXHEAT_HPP_


#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace FELIX
{

  /** \brief Geotermal Flux Heat Evaluator

    This evaluator evaluates the production of heat coming from the earth
   */

  template<typename EvalT, typename Traits, typename Type>
  class GeoFluxHeat : public PHX::EvaluatorWithBaseImpl<Traits>,
  public PHX::EvaluatorDerived<EvalT, Traits>
  {
  public:
    GeoFluxHeat(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl);

    void postRegistrationSetup (typename Traits::SetupData d, PHX::FieldManager<Traits>& fm);

    void evaluateFields(typename Traits::EvalData d);

  private:
    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;
    typedef typename EvalT::ParamScalarT ParamScalarT;

    // Input:
    PHX::MDField<ParamScalarT,Cell,Side,QuadPoint>    	 geoFlux;     // [W m^{-2}] = [Pa m s^{-1}]
    PHX::MDField<RealType,Cell,Side,Node,QuadPoint>   	 BF;          // []
    PHX::MDField<RealType,Cell,Side,Node,QuadPoint,Dim>  GradBF;      // [km^{-1}
    PHX::MDField<MeshScalarT,Cell,Side,QuadPoint>     	 w_measure;   // [km^2]
    PHX::MDField<Type,Cell,Side,QuadPoint,VecDim>        velocity;    // [m y^{-1}]
    PHX::MDField<ScalarT,Cell,Side,QuadPoint>        	   verticalVel; // [m y^{-1}]

    // Output:
    PHX::MDField<ScalarT,Cell,Node> geoFluxHeat;      // [MW] = [k^{-2} kPa s^{-1} km^3]
    PHX::MDField<ScalarT,Cell,Node> geoFluxHeatSUPG;  // [MW s^{-1}] = [k^{-2} kPa s^{-2} km^3]

    std::vector<std::vector<int> >  sideNodes;
    std::string                     basalSideName;

    int numCellNodes;
    int numSideNodes;
    int numSideQPs;
    int sideDim;
    int vecDimFO;

    bool haveSUPG;
    bool isGeoFluxConst;
  };

}	// end namespace FELIX

#endif /* FELIX_GEOFLUXHEAT_HPP_ */
