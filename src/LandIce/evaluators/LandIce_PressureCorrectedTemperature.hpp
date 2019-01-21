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

template<typename EvalT, typename Traits, typename TempST, typename SurfHeightST, typename CoordST = typename EvalT::MeshScalarT>
class PressureCorrectedTemperature : public PHX::EvaluatorWithBaseImpl<Traits>,
                                     public PHX::EvaluatorDerived<EvalT, Traits>
{
public :
  // Provide dymmy impl for default case, so that  createEvaluatorWithTwoScalarTypes
  // in LandIce_ProblemUtils can compile also when called with this evaluator.
  // Put exception to catch the wrong usage at runtime.
  PressureCorrectedTemperature (const Teuchos::ParameterList&,
                                const Teuchos::RCP<Albany::Layouts>&) {
    TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error,
                                "Error! Instantiation of PressureCorrectedTemperature with "
                                "incompatible scalar type for temperature, surface height, and mesh coordinates. "
                                "Here are the scalar types:\n"
                                "  - temperature: " + PHX::typeAsString<TempST>() + "\n"
                                "  - surface height: " + PHX::typeAsString<SurfHeightST>() + "\n"
                                "  - coordinates: " + PHX::typeAsString<CoordST>() + "\n");
  }

  void postRegistrationSetup (typename Traits::SetupData,
                              PHX::FieldManager<Traits>&){}

  void evaluateFields(typename Traits::EvalData) {}
};


template<typename EvalT, typename Traits, typename TempST, typename SurfHeightST>
class PressureCorrectedTemperature<EvalT, Traits, TempST, SurfHeightST,
                                   typename std::enable_if<
                                              std::is_convertible<SurfHeightST, TempST>::value &&
                                              std::is_convertible<typename EvalT::MeshScalarT,TempST>::value,
                                              typename EvalT::MeshScalarT
                                            >::type>
                            : public PHX::EvaluatorWithBaseImpl<Traits>,
                              public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  typedef typename EvalT::MeshScalarT MeshScalarT;

  PressureCorrectedTemperature (const Teuchos::ParameterList& p,
                                const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData ,
                              PHX::FieldManager<Traits>& ) {}

  void evaluateFields(typename Traits::EvalData d);

private:
  // Input:
  PHX::MDField<const SurfHeightST,Cell> sHeight;
  PHX::MDField<const TempST,Cell> temp;
  PHX::MDField<const MeshScalarT,Cell,Dim> coord;

  // Output:
  PHX::MDField<TempST,Cell> correctedTemp;

  double beta, rho_i, g, coeff;

  PHAL::MDFieldMemoizer<Traits> memoizer;
};

} // Namespace LandIce

#endif /* LandIce_PressureCorrectedTemperature_HPP_ */
