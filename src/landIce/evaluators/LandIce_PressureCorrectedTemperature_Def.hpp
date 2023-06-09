/*
 * LandIce_PressureCorrectedTemperature_Def.hpp
 *
 */

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_Print.hpp"
#include "Albany_KokkosUtils.hpp"

#include "LandIce_PressureCorrectedTemperature.hpp"

namespace LandIce
{

template<typename EvalT, typename Traits, typename TempST, typename SurfHeightST>
PressureCorrectedTemperature<EvalT,Traits, TempST, SurfHeightST>::
PressureCorrectedTemperature(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl) :
  sHeight (p.get<std::string> ("Surface Height Variable Name"), dl->cell_scalar2),
  temp (p.get<std::string> ("Temperature Variable Name"), dl->cell_scalar2),
  coord (p.get<std::string> ("Coordinate Vector Variable Name"), dl->cell_gradient),
  correctedTemp (p.get<std::string> ("Corrected Temperature Variable Name"), dl->cell_scalar2)
{
  useP0Temp = p.get<bool>("Use P0 Temperature");
  if(useP0Temp) {
    sHeight = decltype(sHeight)(p.get<std::string> ("Surface Height Variable Name"), dl->cell_scalar2);
    temp = decltype(temp)(p.get<std::string> ("Temperature Variable Name"), dl->cell_scalar2);
    coord = decltype(coord)(p.get<std::string> ("Coordinate Vector Variable Name"), dl->cell_gradient);
    correctedTemp  = decltype(correctedTemp)(p.get<std::string> ("Corrected Temperature Variable Name"), dl->cell_scalar2);
    numQPs = 0;
  } else {
    sHeight = decltype(sHeight)(p.get<std::string> ("Surface Height Variable Name"), dl->qp_scalar);
    temp = decltype(temp)(p.get<std::string> ("Temperature Variable Name"), dl->qp_scalar);
    coord = decltype(coord)(p.get<std::string> ("Coordinate Vector Variable Name"), dl->qp_gradient);
    correctedTemp  = decltype(correctedTemp)(p.get<std::string> ("Corrected Temperature Variable Name"), dl->qp_scalar);
    std::vector<PHX::DataLayout::size_type> dims;
    dl->qp_scalar->dimensions(dims);
    numQPs  = dims[1];
  }

  this->addDependentField(sHeight);
  this->addDependentField(coord);
  this->addDependentField(temp);
  this->addEvaluatedField(correctedTemp);
  this->setName("Pressure Corrected Temperature"+PHX::print<EvalT>());

  auto physicsList = p.get<Teuchos::ParameterList*>("LandIce Physical Parameters");
  rho_i = physicsList->get<double>("Ice Density");
  g     = physicsList->get<double>("Gravity Acceleration");
  //p_atm = 101325.0; // kg * m^-1 * s^-2
  beta  = physicsList->get<double>("Clausius-Clapeyron Coefficient");
  coeff = beta * 1000.0 * rho_i * g;
  meltingT = physicsList->get<double>("Atmospheric Pressure Melting Temperature");
}

//**********************************************************************
template<typename EvalT, typename Traits, typename TempST, typename SurfHeightST>
void PressureCorrectedTemperature<EvalT, Traits, TempST, SurfHeightST>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
  if (d.memoizer_active()) memoizer.enable_memoizer();
}

//**********************************************************************
template<typename EvalT, typename Traits, typename TempST, typename SurfHeightST>
void PressureCorrectedTemperature<EvalT,Traits, TempST, SurfHeightST>::
evaluateFields(typename Traits::EvalData workset)
{
  if (memoizer.have_saved_data(workset,this->evaluatedFields())) return;

  Kokkos::parallel_for(RangePolicy(0, workset.numCells),
                       KOKKOS_CLASS_LAMBDA(const int& cell) {
    if(useP0Temp) {
      correctedTemp(cell) = KU::min(temp(cell) + coeff * (sHeight(cell) - coord(cell,2)), meltingT);
    } else {
      for (int qp = 0; qp < numQPs; ++qp)
        correctedTemp(cell,qp) = KU::min(temp(cell,qp) + coeff * (sHeight(cell,qp) - coord(cell,qp,2)), meltingT);
    }
  });
}

} // namespace LandIce
