/*
 * LandIce_EnthalpyResid.hpp
 *
 *  Created on: May 11, 2016
 *      Author: abarone
 */

#ifndef LANDICE_ENTHALPY_RESID_HPP
#define LANDICE_ENTHALPY_RESID_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"
#include "PHAL_Dimension.hpp"

namespace LandIce
{

template<typename EvalT, typename Traits, typename VelocityST, typename MeltTempST>
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

  enum STABILIZATION_TYPE {SU, UPWIND, NONE} ;
  STABILIZATION_TYPE stabilization;
  double delta;

  // Input:
  PHX::MDField<const MeshScalarT,Cell,Node,QuadPoint>           wBF; // [km^3]
  PHX::MDField<const MeshScalarT,Cell,Node,QuadPoint,Dim>       wGradBF; // [km^2]

  PHX::MDField<const ScalarT,Cell,QuadPoint>                    Enthalpy;  //[MW s m^{-3}]
  PHX::MDField<const ScalarT,Cell,QuadPoint,Dim>                EnthalpyGrad; //[kW s m^{-4}]
  PHX::MDField<const MeltTempST,Cell,QuadPoint>                 EnthalpyHs;  //[MW s m^{-3}]
  PHX::MDField<const ScalarT,Cell,Node>                         diffEnth;  //[MW s m^{-3}]

  PHX::MDField<const VelocityST,Cell,QuadPoint,VecDim>          Velocity; //[m yr^{-1}]
  PHX::MDField<const VelocityST,Cell,QuadPoint,VecDim, Dim>     velGrad; //[m yr^{-1}]
  PHX::MDField<const VelocityST,Cell,QuadPoint>                 verticalVel; //[m yr^{-1}]
  PHX::MDField<const MeshScalarT,Cell,Node,Dim>                 coordVec; // [km]
  PHX::MDField<const ScalarT,Cell,QuadPoint>                    diss;  //[W m^{-3}] = [Pa s^{-1}]
  PHX::MDField<const ScalarT,Cell,Node>                         basalFricHeat;  // [MW] = [k^{-1} Pa s^{-1} km^3], k=1000
  PHX::MDField<const ScalarT,Cell,Node>                         basalFricHeatSUPG; // [MW s^{-1}] = [k^{-1} Pa s^{-2} km^3], k=1000
  PHX::MDField<const ScalarT,Cell,Node>                         geoFluxHeat;     // [MW]
  PHX::MDField<const ScalarT,Cell,Node>                         geoFluxHeatSUPG; // [MW s^{-1}]
  PHX::MDField<const ScalarT,Cell,QuadPoint>                    phi;                //[]
  PHX::MDField<const ScalarT,Cell,QuadPoint,Dim>                phiGrad;        //[km^{-1}
  PHX::MDField<const MeltTempST,Cell,QuadPoint,Dim>             meltTempGrad; // [K km^{-1}]
  PHX::MDField<const ScalarT,Cell,Node>                         basalResid; // [k^{2} W], k =1000
  PHX::MDField<const ScalarT,Cell,Node>                         basalResidSUPG; // [k^{2} W], k =1000

  PHX::MDField<const ScalarT,Dim>                               homotopy;

  // Output:
  PHX::MDField<ScalarT,Cell,Node> Residual; // [k^3 W]  = [km^3 Pa s^{-1} ], k =1000

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
  double alpha_om; //[adim], Omega exponent alpha
  double scyr ;    // [s/yr] (3.1536e7);

  double flux_reg_alpha;
  double flux_reg_beta;

  const double powm3 = 1e-3;  //[k], k=1000
  const double powm6 = 1e-6;  //[k^2], k=1000
  const double pow3 = 1e3;  //[k^{-1}], k=1000

  ScalarT flux_reg_coeff;
  ScalarT printedRegCoeff;

public:

  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;

  struct Upwind_Stabilization_Tag{};
  struct SU_Stabilization_Tag{};
  struct Other_Stabilization_Tag{};

  typedef Kokkos::RangePolicy<ExecutionSpace,Upwind_Stabilization_Tag> Upwind_Stabilization_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,SU_Stabilization_Tag> SU_Stabilization_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,Other_Stabilization_Tag> Other_Stabilization_Policy;

  // KOKKOS_INLINE_FUNCTION
  // void stabilizationInitialization(int cell, VelocityST& vmax_xy, ScalarT& vmax, ScalarT& vmax_z, 
  //   MeshScalarT& diam, MeshScalarT& diam_xy, MeshScalarT& diam_z, ScalarT& wSU);
  // KOKKOS_INLINE_FUNCTION
  // void evaluateResidNode(int cell, int node);

  KOKKOS_INLINE_FUNCTION
  void operator() (const Upwind_Stabilization_Tag& tag, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const SU_Stabilization_Tag& tag, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const Other_Stabilization_Tag& tag, const int& cell) const;

};

} // namespace LandIce

#endif /* LandIce_ENTHALPYRESID_HPP_ */
