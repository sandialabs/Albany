/*
 * LandIce_LiquidWaterFraction.hpp
 *
 *  Created on: Jun 6, 2016
 *      Author: abarone
 */

#ifndef LANDICE_LIQUID_WATER_FRACTION_HPP
#define LANDICE_LIQUID_WATER_FRACTION_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_SacadoTypes.hpp"
#include "Albany_Layouts.hpp"
#include "PHAL_Dimension.hpp"
#include "PHAL_Utilities.hpp"

namespace LandIce
{

/** \brief Liquid Water Fraction

  This evaluator computes the liquid water fraction in temperate ice
 */

template<typename EvalT, typename Traits, typename EntalpyType>
class LiquidWaterFraction: public PHX::EvaluatorWithBaseImpl<Traits>,
                           public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  typedef typename EvalT::MeshScalarT MeshScalarT;

  LiquidWaterFraction (const Teuchos::ParameterList& p,
                       const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:
  // This is just to allow ETI machinery to work. In a real setting, ScalarT should always be constructible from Type
  typedef typename Albany::StrongestScalarType<EntalpyType,MeshScalarT>::type OutputScalarT;

  // Input:
  PHX::MDField<const MeshScalarT,Cell,Node> enthalpyHs;  //[MW s m^{-3}]
  PHX::MDField<const EntalpyType,Cell,Node> 	      enthalpy;  //[MW s m^{-3}]

  // Output:
  PHX::MDField<OutputScalarT,Cell,Node>   phi;         //[adim]

  unsigned int numNodes;

  double L;      //[J kg^{-1}] = [ m^2 s^{-2}]
  double rho_w;  //[kg m^{-3}]
  double phi_scaling; //[MW^{-1} s^{-1} m^{3}]

  PHAL::MDFieldMemoizer<Traits> memoizer;

public:

  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;

  typedef Kokkos::RangePolicy< ExecutionSpace > Phi_Policy;

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const;
};

} // Namespace LandIce

#endif // LandIce_LIQUID_WATER_FRACTION_HPP
