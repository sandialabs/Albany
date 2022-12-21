/*
 * LandIce_PressureCorrectedTemperature.hpp
 *
 *  Created on: Jun 6, 2016
 *      Author: abarone
 */

#ifndef LANDICE_PRESSURECORRECTEDTEMPERATURE_HPP_
#define LANDICE_PRESSURECORRECTEDTEMPERATURE_HPP_

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"
#include "PHAL_Utilities.hpp"

namespace LandIce
{

/** \brief Pressure-melting temperature

    This evaluator computes the pressure-melting temperature Tm(p) via the hydrostatic approximation of the pressure.
*/

template<typename EvalT, typename Traits, typename TempST, typename SurfHeightST>
class PressureCorrectedTemperature: public PHX::EvaluatorWithBaseImpl<Traits>,
                              public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  typedef typename EvalT::MeshScalarT MeshScalarT;

  PressureCorrectedTemperature (const Teuchos::ParameterList& p,
                                const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:
  // Input:
  typedef typename Albany::StrongestScalarType<typename Albany::StrongestScalarType<TempST,MeshScalarT>::type, SurfHeightST>::type OutputScalarT;
  PHX::MDField<const SurfHeightST,Cell> sHeight;
  PHX::MDField<const TempST,Cell> temp;
  PHX::MDField<const MeshScalarT,Cell,Dim> coord;

  // Output:
  PHX::MDField<OutputScalarT,Cell> correctedTemp;

  double beta;     //[K Pa^{-1}]
  double rho_i;    //[kg m^{-3}]
  double g;        //[m s^{-2}]
  double coeff;    //[K km^{-1}]
  double meltingT; //[K], 273.15

  PHAL::MDFieldMemoizer<Traits> memoizer;
};

} // Namespace LandIce

#endif /* LandIce_PressureCorrectedTemperature_HPP_ */
