/*
 * LandIce_Temperature.hpp
 *
 *  Created on: Jun 6, 2016
 *      Author: abarone
 */

#ifndef LANDICE_TEMPERATURE_HPP
#define LANDICE_TEMPERATURE_HPP

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

/** \brief Temperature

  This evaluator computes the temperature from enthalpy
 */

template<typename EvalT, typename Traits, typename TemperatureST>
class Temperature: public PHX::EvaluatorWithBaseImpl<Traits>,
public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  typedef typename EvalT::MeshScalarT MeshScalarT;
  typedef typename EvalT::ScalarT ScalarT;

  Temperature (const Teuchos::ParameterList& p,
               const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:

  // This is just to allow ETI machinery to work. In a real setting, ScalarT should always be constructible from TemperatureST.
  typedef typename Albany::StrongestScalarType<TemperatureST,MeshScalarT>::type OutputScalarT;

  // Input:
  PHX::MDField<const MeshScalarT,Cell,Node> 		meltingTemp; //[K]
  PHX::MDField<const MeshScalarT,Cell,Node> 		enthalpyHs;  //[MW s m^{-3}]
  PHX::MDField<const TemperatureST,Cell,Node> 	enthalpy;  //[MW s m^{-3}]

  // Output:
  PHX::MDField<OutputScalarT,Cell,Node> 	temperature; //[K]
  PHX::MDField<OutputScalarT,Cell,Node> 	correctedTemp; //[K]
  PHX::MDField<OutputScalarT,Cell,Node> 	diffEnth;    //[MW s m^{-3}]

  unsigned int numNodes;

  double c_i;   //[J Kg^{-1} K^{-1}], Heat capacity of ice
  double rho_i; //[kg m^{-3}]
  double T0;    //[K]
  double Tm; //[K], 273.15
  double temperature_scaling; // [MW^{-1} s^{-1} K m^{3}]

  PHAL::MDFieldMemoizer<Traits> memoizer;

public:

  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;

  typedef Kokkos::RangePolicy< ExecutionSpace > Temperature_Policy;

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const;
};

} // namespace LandIce

#endif // LANDICE_TEMPERATURE_HPP
