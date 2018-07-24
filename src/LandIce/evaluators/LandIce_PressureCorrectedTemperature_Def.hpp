/*
 * LandIce_PressureCorrectedTemperature_Def.hpp
 *
 */

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

namespace LandIce
{

template<typename EvalT, typename Traits, typename Type >
PressureCorrectedTemperature<EvalT,Traits,Type,typename std::enable_if<std::is_convertible<typename EvalT::ParamScalarT, Type>::value>::type>::
PressureCorrectedTemperature(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl) :
  sHeight (p.get<std::string> ("Surface Height Variable Name"), dl->cell_scalar2),
  temp (p.get<std::string> ("Temperature Variable Name"), dl->cell_scalar2),
  correctedTemp (p.get<std::string> ("Corrected Temperature Variable Name"), dl->cell_scalar2),
  coord (p.get<std::string> ("Coordinate Vector Variable Name"), dl->cell_gradient),
  physicsList (*p.get<Teuchos::ParameterList*>("LandIce Physical Parameters"))
{
  if (p.isType<bool>("Enable Memoizer") && p.get<bool>("Enable Memoizer")) memoizer.enable_memoizer();

	this->addDependentField(sHeight);
	this->addDependentField(coord);
	this->addDependentField(temp);
	this->addEvaluatedField(correctedTemp);

	this->setName("Pressure Corrected Temperature"+PHX::typeAsString<EvalT>());

  // dummy initialization
  //(we do not want to initialize them now because their values might not be available if the evaluator is not used)
  rho_i = g = beta  = coeff = 0;
}

template<typename EvalT, typename Traits, typename Type >
void PressureCorrectedTemperature<EvalT,Traits,Type, typename std::enable_if<std::is_convertible<typename EvalT::ParamScalarT, Type>::value>::type>::
postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
{
  rho_i = physicsList.get<double>("Ice Density");
  g     = physicsList.get<double>("Gravity Acceleration");
  //p_atm = 101325.0; // kg * m^-1 * s^-2
  beta  = physicsList.get<double>("Clausius-Clapeyron Coefficient");//,0);
  coeff = beta * 1000.0 * rho_i * g;
}

template<typename EvalT, typename Traits, typename Type>
void PressureCorrectedTemperature<EvalT,Traits,Type, typename std::enable_if<std::is_convertible<typename EvalT::ParamScalarT, Type>::value>::type>::
evaluateFields(typename Traits::EvalData d)
{
  if (memoizer.have_stored_data(d)) return;

  for (std::size_t cell = 0; cell < d.numCells; ++cell)
    correctedTemp(cell) = std::min(temp(cell) +coeff * (sHeight(cell) - coord(cell,2)), 273.15);
}

}
