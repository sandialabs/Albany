/*
 * LandIce_EnthalpyBasalResid.hpp
 *
 *  Created on: May 31, 2016
 *      Author: abarone
 */

#ifndef LANDICE_ENTHALPY_BASAL_RESID_HPP
#define LANDICE_ENTHALPY_BASAL_RESID_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "PHAL_Dimension.hpp"
#include "Albany_Layouts.hpp"
#include "Albany_ScalarOrdinalTypes.hpp"

namespace LandIce
{

/** \brief Geotermal Flux Heat Evaluator

  This evaluator evaluates the production of heat coming from the earth
 */

template<typename EvalT, typename Traits, typename Type>
class EnthalpyBasalResid : public PHX::EvaluatorWithBaseImpl<Traits>,
                           public PHX::EvaluatorDerived<EvalT, Traits>
{
public:
  EnthalpyBasalResid(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d, PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;
  typedef typename EvalT::ParamScalarT ParamScalarT;

  // Input:
  PHX::MDField<const RealType,Cell,Side,Node,QuadPoint>         BF;          // []
  PHX::MDField<const MeshScalarT,Cell,Side,QuadPoint>           w_measure;   // [km^2]
  // PHX::MDField<const RealType,Cell,Side,QuadPoint>              geoFlux;     // [W m^{-2}] = [Pa m s^{-1}]
  // PHX::MDField<const Type,Cell,Side,QuadPoint>                  beta; // [kPa m / yr]
  // PHX::MDField<const ScalarT,Cell,Side,QuadPoint>               basal_dTdz; // [K  km^{-1}]
  // PHX::MDField<const ScalarT,Cell,Side, QuadPoint>              enthalpy;  //[MW s m^{-3}]
  // PHX::MDField<const ParamScalarT,Cell, Side, QuadPoint>        enthalpyHs;  //[MW s m^{-3}]
  // PHX::MDField<const Type,Cell,Side,QuadPoint,VecDim>           velocity; // [m yr^{-1}
  // PHX::MDField<const ScalarT,Cell,Side,QuadPoint>               verticalVel; // [m y^{-1}]
  // PHX::MDField<const MeshScalarT,Cell,Side,Node,QuadPoint,Dim>  GradBF;      // [km^{-1}
  // PHX::MDField<const ScalarT,Cell,Node>                         diffEnth;  //[MW s m^{-3}]
  // PHX::MDField<const ScalarT,Cell,Side,QuadPoint>               phi;  // []
  // PHX::MDField<const ScalarT,Dim>                               homotopy;
  PHX::MDField<const ScalarT,Cell,Side,QuadPoint>               basalMeltRateQP;      // [MW] = [m/yr]

  // Output:
  PHX::MDField<ScalarT,Cell,Node> enthalpyBasalResid;      // [MW] = [k^{-2} kPa s^{-1} km^3]
  // PHX::MDField<ScalarT,Cell,Side, Node> basalMeltRate;      // [MW] = [m/yr]

  std::vector<std::vector<int> >  sideNodes;
  std::string                     basalSideName;

  int numCellNodes;
  int numSideNodes;
  int numSideQPs;
  int sideDim;
  int vecDimFO;

  // double a;
  // double k_i;   //[W m^{-1} K^{-1}], Conductivity of ice
  // double beta_p;  //[K Pa^{-1}]
  // double rho_i;  // [kg m^{-3}]
  // double rho_w;  // [kg m^{-3}]
  // double g;  //[m s^{-2}]
  // double L;       //[J kg^{-1} ] Latent heat of fusion", 3e5)
  // double k_0;     //[m^2], Permeability factor
  // double eta_w;   //[Pa s], Viscosity of water
  // double alpha_om; //[]

  // bool haveSUPG;
};

} // namespace LandIce

#endif // LANDICE_ENTHALPY_BASAL_RESID_HPP
