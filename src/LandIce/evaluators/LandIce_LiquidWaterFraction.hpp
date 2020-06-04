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

template<typename EvalT, typename Traits, typename Type>
class LiquidWaterFraction: public PHX::EvaluatorWithBaseImpl<Traits>,
                           public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  typedef typename EvalT::ScalarT ScalarT;

  LiquidWaterFraction (const Teuchos::ParameterList& p,
                       const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:
  // This is just to allow ETI machinery to work. In a real setting, ScalarT should always be constructible from Type
  typedef typename Albany::StrongestScalarType<Type,ScalarT>::type OutputScalarT;

  // Input:
  PHX::MDField<const Type,Cell,Node> 		  enthalpyHs;  //[MW s m^{-3}]
  PHX::MDField<const ScalarT,Cell,Node> 	enthalpy;  //[MW s m^{-3}]

  // Output:
  PHX::MDField<OutputScalarT,Cell,Node>   phi;         //[adim]

  int numNodes;

  double L;      //[J kg^{-1}] = [ m^2 s^{-2}]
  double rho_w;  //[kg m^{-3}]

  const double pow6 = 1e6; //[k^{-2}], k =1000

  ScalarT printedAlpha;

  PHAL::MDFieldMemoizer<Traits> memoizer;

  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;

  typedef Kokkos::MDRangePolicy< ExecutionSpace, Kokkos::Rank<2> > Phi_Policy;

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i, const int& j) const;
};

} // Namespace LandIce

#endif // LandIce_LIQUID_WATER_FRACTION_HPP
