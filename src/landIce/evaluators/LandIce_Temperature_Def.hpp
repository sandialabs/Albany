/*
 * LandIce_Temperature_Def.hpp
 *
 *  Created on: Jun 6, 2016
 *      Author: abarone
 */

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_Print.hpp"
#include "Shards_CellTopology.hpp"

#include "Albany_ScalarOrdinalTypes.hpp"
#include "Albany_AbstractDiscretization.hpp"

#include "LandIce_Temperature.hpp"

namespace LandIce
{

template<typename EvalT, typename Traits, typename TemperatureST>
Temperature<EvalT,Traits,TemperatureST>::
Temperature(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl)
 : meltingTemp    (p.get<std::string> ("Melting Temperature Variable Name"), dl->node_scalar)
 , enthalpyHs     (p.get<std::string> ("Enthalpy Hs Variable Name"), dl->node_scalar)
 , enthalpy     (p.get<std::string> ("Enthalpy Variable Name"), dl->node_scalar)
 // , thickness     (p.get<std::string> ("Thickness Variable Name"), dl->node_scalar)
 , temperature    (p.get<std::string> ("Temperature Variable Name"), dl->node_scalar)
 , correctedTemp  (p.get<std::string> ("Corrected Temperature Variable Name"), dl->node_scalar)
 , diffEnth       (p.get<std::string> ("Diff Enthalpy Variable Name"), dl->node_scalar)
{
  Teuchos::RCP<shards::CellTopology> cellType;
  cellType = p.get<Teuchos::RCP <shards::CellTopology> > ("Cell Type");

  // dTdz = decltype(dTdz)(p.get<std::string> ("Basal dTdz Variable Name"), dl_side->node_scalar);

  std::vector<PHX::Device::size_type> dims;
  dl->node_qp_vector->dimensions(dims);

  numNodes = dims[1];

  this->addDependentField(meltingTemp);
  this->addDependentField(enthalpyHs);
  this->addDependentField(enthalpy);
  // this->addDependentField(thickness);

  this->addEvaluatedField(temperature);
  this->addEvaluatedField(correctedTemp);
  this->addEvaluatedField(diffEnth);
  // this->addEvaluatedField(dTdz);
  this->setName("Temperature");

  // Setting parameters
  Teuchos::ParameterList& physics = *p.get<Teuchos::ParameterList*>("LandIce Physical Parameters");
  rho_i   = physics.get<double>("Ice Density"); //916
  c_i   = physics.get<double>("Heat capacity of ice"); //2009
  T0    = physics.get<double>("Reference Temperature"); //265
  Tm    = physics.get<double>("Atmospheric Pressure Melting Temperature"); //273.15
  temperature_scaling = 1e6/(rho_i * c_i);
}

template<typename EvalT, typename Traits, typename TemperatureST>
KOKKOS_INLINE_FUNCTION
void Temperature<EvalT,Traits,TemperatureST>::
operator() (const int &cell) const{

  for (unsigned int node = 0; node < numNodes; ++node) {
    if ( enthalpy(cell,node) < enthalpyHs(cell,node) )
      temperature(cell,node) = temperature_scaling * enthalpy(cell,node) + T0;
    else
      temperature(cell,node) = meltingTemp(cell,node);

    correctedTemp(cell, node) = temperature(cell,node) + Tm - meltingTemp(cell,node);

    diffEnth(cell,node) = enthalpy(cell,node) - enthalpyHs(cell,node);
  }
  

}

template<typename EvalT, typename Traits, typename TemperatureST>
void Temperature<EvalT,Traits,TemperatureST>::
postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(meltingTemp,fm);
  this->utils.setFieldData(enthalpyHs,fm);
  this->utils.setFieldData(enthalpy,fm);
  // this->utils.setFieldData(thickness,fm);

  this->utils.setFieldData(temperature,fm);
  this->utils.setFieldData(correctedTemp,fm);
  this->utils.setFieldData(diffEnth,fm);
  // this->utils.setFieldData(dTdz,fm);

  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
  if (d.memoizer_active()) memoizer.enable_memoizer();
}

template<typename EvalT, typename Traits, typename TemperatureST>
void Temperature<EvalT,Traits,TemperatureST>::
evaluateFields(typename Traits::EvalData workset)
{
  if (memoizer.have_saved_data(workset,this->evaluatedFields())) return;
  
  Kokkos::parallel_for(Temperature_Policy(0, workset.numCells), *this);
}

}  // end namespace LandIce
