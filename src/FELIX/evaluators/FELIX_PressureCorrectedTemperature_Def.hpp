/*
 * FELIX_PressureCorrectedTemperature_Def.hpp
 *
 */

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

namespace FELIX
{

template<typename EvalT, typename Traits, typename Type>
PressureCorrectedTemperature<EvalT,Traits,Type>::
PressureCorrectedTemperature(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl):
	sHeight       (p.get<std::string> ("Surface Height Variable Name"), dl->cell_scalar2),
	temp (p.get<std::string> ("Temperature Variable Name"), dl->cell_scalar2),
	correctedTemp (p.get<std::string> ("Corrected Temperature Variable Name"), dl->cell_scalar2),
	coord (p.get<std::string> ("Coordinate Vector Variable Name"), dl->cell_gradient)
{


	this->addDependentField(sHeight);
	this->addDependentField(coord);
	this->addDependentField(temp);
	this->addEvaluatedField(correctedTemp);

	this->setName("Pressure Corrected Temperature");

	// Setting parameters
	Teuchos::ParameterList& physics = *p.get<Teuchos::ParameterList*>("FELIX Physical Parameters");

  rho_i = physics.get<double>("Ice Density");
  g     = physics.get<double>("Gravity Acceleration");
  //p_atm = 101325.0; // kg * m^-1 * s^-2
	beta  = physics.get<double>("Clausius-Clapeyron coefficient",0);

	coeff = beta * 1000.0 * rho_i * g;
}

template<typename EvalT, typename Traits, typename Type>
void PressureCorrectedTemperature<EvalT,Traits,Type>::
postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
{
}

template<typename EvalT, typename Traits, typename Type>
void PressureCorrectedTemperature<EvalT,Traits,Type>::
evaluateFields(typename Traits::EvalData d)
{
    for (std::size_t cell = 0; cell < d.numCells; ++cell)
        correctedTemp(cell) = std::min(temp(cell) +coeff * ( sHeight(cell) - coord(cell,2) ), 273.15);
}


}
