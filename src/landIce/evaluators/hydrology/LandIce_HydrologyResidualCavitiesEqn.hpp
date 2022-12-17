//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_HYDROLOGY_RESIDUAL_CAVITIES_EQN_HPP
#define LANDICE_HYDROLOGY_RESIDUAL_CAVITIES_EQN_HPP 1

#include "Albany_Layouts.hpp"

#include "Albany_DiscretizationUtils.hpp"
#include "Albany_ScalarOrdinalTypes.hpp"

#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"

namespace LandIce
{

/** \brief Hydrology Mass Equation Residual Evaluator

    This evaluator evaluates the residual of the Hydrology model
*/

/*
 *  The (water) thickness equation has the following (strong) form
 *
 *     dh/dt = m/rho_i + (h_r-h)*|u_b|/l_r - c_creep*A*h*N^3
 *
 *  where h is the water thickness, m the melting rate of the ice,
 *  h_r/l_r typical height/length of bed bumps, u_b the sliding
 *  velocity of the ice, A is the ice softness, N is the
 *  effective pressure, and c_creep is a tuning coefficient.
 *  Also, dh/dt denotes the *partial* time derivative.
 */

template<typename EvalT, typename Traits, bool IsStokes, bool ThermoCoupled>
class HydrologyResidualCavitiesEqn : public PHX::EvaluatorWithBaseImpl<Traits>
{
public:

  typedef typename EvalT::ScalarT         ScalarT;
  typedef typename EvalT::MeshScalarT     MeshScalarT;
  typedef typename EvalT::ParamScalarT    ParamScalarT;

  typedef typename std::conditional<IsStokes,ScalarT,RealType>::type       IceScalarT;
  typedef typename std::conditional<ThermoCoupled,ScalarT,RealType>::type  TempScalarT;

  HydrologyResidualCavitiesEqn (const Teuchos::ParameterList& p,
                                 const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData,
                              PHX::FieldManager<Traits>&) {}

  void evaluateFields(typename Traits::EvalData d);

private:

  void evaluateFieldsCell(typename Traits::EvalData d);
  void evaluateFieldsSide(typename Traits::EvalData d);

  // Input:
  PHX::MDField<const RealType>      BF;           // Basis functions
  PHX::MDField<const MeshScalarT>   w_measure;    // Weighted measure (for quadrature)
  PHX::MDField<const ScalarT>       h;            // Water thickness [m]
  PHX::MDField<const ScalarT>       h_node;       // Water thickness nodal [m]
  PHX::MDField<const ScalarT>       h_dot;        // Water thickness time derivative [m/s]
  PHX::MDField<const ScalarT>       P_dot;        // Water pressure time derivative [kPa/s]
  PHX::MDField<const ScalarT>       N;            // Effective pressure [kPa]
  PHX::MDField<const ScalarT>       m;            // Ice Melting Rate [kg /(m^2 yr)]
  PHX::MDField<const IceScalarT>    u_b;          // Ice sliding velocity [m/yr]
  PHX::MDField<const TempScalarT>   ice_softness; // Flow factor [Pa^-3 s^-1]

  // Output:
  PHX::MDField<ScalarT>             residual;     // Cavity evolution residual [m/yr]

  unsigned int numNodes;
  unsigned int numQPs;

  double rho_i;               // Ice density [kg/m^3]
  double rho_w;               // Water density [kg/m^3]
  double g;                   // Gravity acceleration [m/2^2]
  double eta_i;               // Ice viscosity [Pa s]
  double englacial_phi;       // Englacial porosity [non dimensional]
  double h_r;                 // Bed bumps typical height [m]
  double l_r;                 // Bed bumps typical length [m]
  double c_creep;             // Creep closure coefficient [non dimensional]
  double n;                   // Glen's Law exponent

  // The type of the cavities closure term. Cubic: c_creep*h*A*N^3. Linear: c_creep*h*N/eta_i
  enum ClosureTypeN { Cubic , Linear};

  ClosureTypeN closure;

  bool unsteady;
  bool has_p_dot;
  bool use_melting;
  bool nodal_equation;
  bool use_eff_cavity;

  // Variables necessary for stokes coupling
  bool                            stokes_coupling;
  std::string                     sideSetName;
  std::vector<std::vector<int> >  sideNodes;
  Albany::LocalSideSetInfo sideSet;
};

} // Namespace LandIce

#endif // LANDICE_HYDROLOGY_RESIDUAL_CAVITIES_EQN_HPP
