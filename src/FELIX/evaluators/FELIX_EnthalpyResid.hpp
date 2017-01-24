/*
 * FELIX_EnthalpyResid.hpp
 *
 *  Created on: May 11, 2016
 *      Author: abarone
 */

#ifndef FELIX_ENTHALPYRESID_HPP_
#define FELIX_ENTHALPYRESID_HPP_

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace FELIX
{

  template<typename EvalT, typename Traits, typename VelocityType>
  class EnthalpyResid : public PHX::EvaluatorWithBaseImpl<Traits>,
  public PHX::EvaluatorDerived<EvalT, Traits>
  {
  public:

    EnthalpyResid(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl);

    void postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& vm);

    void evaluateFields(typename Traits::EvalData d);

  private:
    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;
    typedef typename EvalT::ParamScalarT ParamScalarT;

    enum STABILIZATION_TYPE {SUPG, SU, NONE} ;
    STABILIZATION_TYPE stabilization;
    bool haveSUPG;
    double delta;

    // Input:
    PHX::MDField<MeshScalarT,Cell,Node,QuadPoint> wBF; // [km^3]
    PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> wGradBF; // [km^2]

    PHX::MDField<ScalarT,Cell,QuadPoint> Enthalpy;  //[MW s m^{-3}]
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim> EnthalpyGrad; //[kW s m^{-4}]
    PHX::MDField<ParamScalarT,Cell,QuadPoint> EnthalpyHs;  //[MW s m^{-3}]
    PHX::MDField<ScalarT,Cell,Node> diffEnth;  //[MW s m^{-3}]

    PHX::MDField<VelocityType,Cell,QuadPoint,VecDim> Velocity; //[m yr^{-1}]
    PHX::MDField<VelocityType,Cell,QuadPoint,VecDim, Dim> velGrad; //[m yr^{-1}]
    PHX::MDField<ScalarT,Cell,QuadPoint> verticalVel; //[m yr^{-1}]
    PHX::MDField<MeshScalarT,Cell,Node,Dim> coordVec; // [km]
    PHX::MDField<ScalarT,Cell,QuadPoint> diss;  //[W m^{-3}] = [Pa s^{-1}]
    PHX::MDField<ScalarT,Cell,Node> basalFricHeat;  // [MW] = [k^{-1} Pa s^{-1} km^3], k=1000
    PHX::MDField<ScalarT,Cell,Node> basalFricHeatSUPG; // [MW s^{-1}] = [k^{-1} Pa s^{-2} km^3], k=1000
    PHX::MDField<ScalarT,Cell,Node> geoFluxHeat;     // [MW]
    PHX::MDField<ScalarT,Cell,Node> geoFluxHeatSUPG; // [MW s^{-1}]
    PHX::MDField<ScalarT,Cell,QuadPoint> phi;                //[]
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim> phiGrad;        //[km^{-1}
    PHX::MDField<ParamScalarT,Cell,QuadPoint,Dim> meltTempGrad; // [K km^{-1}]
    PHX::MDField<ScalarT,Cell,Node> basalResid; // [k^{2} W], k =1000
    PHX::MDField<ScalarT,Cell,Node> basalResidSUPG; // [k^{2} W], k =1000

    PHX::MDField<ScalarT,Dim> homotopy;

    // Output:
    PHX::MDField<ScalarT,Cell,Node> Residual; // [k^{4} W]  = [km^3 kPa s^{-1} ], k =1000

    unsigned int numQPs, numNodes, vecDimFO;

    bool needsDiss, needsBasFric;

    double k_i;   //[W m^{-1} K^{-1}], Conductivity of ice
    double c_i;   //[J Kg^{-1} K^{-1}], Heat capacity of ice
    double K_i;   //[m^2 s^{-1}]  := k_i / (rho_i * c_i)
    double k_0;   //[m^2], Permeability factor
    double eta_w; //[Pa s], Viscosity of water
    double nu;	  //[m^2 s^{-1}], Diffusivity temperate ice
    double rho_i; // [kg m^{-3}], density of ice
    double rho_w; // [kg m^{-3}] density of water
    double g;     //[m s^{-2}], Gravity Acceleration
    double L;     //[J kg^{-1} ] Latent heat of fusion", 3e5);
    double a;     // [adim], Diffusivity homotopy exponent
    double drainage_coeff; //[kg s^{-3}]
    double alpha_om;  //[adim], Omega exponent alpha

    ScalarT printedAlpha;
  };

}


#endif /* FELIX_ENTHALPYRESID_HPP_ */
