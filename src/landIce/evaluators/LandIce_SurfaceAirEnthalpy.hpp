/*
 * LandIce_SurfaceAirEnthalpy.hpp
 *
 *  Created on: March 2, 2020
 *      Author: mperego
 */

#ifndef LANDICE_SURFACE_AIR_ENTHALPY_HPP
#define LANDICE_SURFACE_AIR_ENTHALPY_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"
#include "PHAL_Dimension.hpp"
#include "PHAL_Utilities.hpp"

namespace LandIce
{

/** \brief Pressure-melting enthalpy

  This evaluator computes enthalpy of the ice at pressure-melting temperature Tm(p).
 */

template<typename EvalT, typename Traits, typename SurfTempST>
class SurfaceAirEnthalpy: public PHX::EvaluatorWithBaseImpl<Traits>,
                          public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  SurfaceAirEnthalpy (const Teuchos::ParameterList& p,
                      const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData workset);

private:
  typedef typename EvalT::ParamScalarT ParamScalarT;

  // Input
  PHX::MDField<const SurfTempST,Cell,Node>   surfaceTemp;  //[K]

  // Output:
  PHX::MDField<SurfTempST,Cell,Node>   surfaceEnthalpy;  //[MW s m^{-3}]

  unsigned int numNodes;
  std::string fieldName;

  double c_i;   //[J Kg^{-1} K^{-1}], Heat capacity of ice
  double rho_i; //[kg m^{-3}]
  double T0;    //[K]
  double Tm; //[K], 273.15

  PHAL::MDFieldMemoizer<Traits> memoizer;

protected:
  using ExecutionSpace = typename PHX::Device::execution_space;
  using RangePolicy = Kokkos::RangePolicy<ExecutionSpace>;
};

} // Namespace LandIce


#endif // LANDICE_PRESSURE_MELTING_ENTHALPY_HPP
