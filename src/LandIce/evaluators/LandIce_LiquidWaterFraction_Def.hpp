/*
 * LandIce_LiquidWaterFraction_Def.hpp
 *
 *  Created on: Jun 6, 2016
 *      Author: abarone
 */

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_Print.hpp"

namespace LandIce
{

template<typename EvalT, typename Traits, typename Type>
LiquidWaterFraction<EvalT,Traits,Type>::
LiquidWaterFraction(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl):
  enthalpyHs (p.get<std::string> ("Enthalpy Hs Variable Name"), dl->node_scalar),
  enthalpy   (p.get<std::string> ("Enthalpy Variable Name"), dl->node_scalar),
  phi        (p.get<std::string> ("Water Content Variable Name"), dl->node_scalar)
{
  // Get Dimensions
  std::vector<PHX::DataLayout::size_type> dims;
  dl->node_qp_vector->dimensions(dims);
  numNodes = dims[1];

  this->addDependentField(enthalpyHs);
  this->addDependentField(enthalpy);

  this->addEvaluatedField(phi);
  this->setName("Phi");

  // Setting parameters
  Teuchos::ParameterList& physics = *p.get<Teuchos::ParameterList*>("LandIce Physical Parameters");
  rho_w = physics.get<double>("Water Density");//, 1000.0);
  L = physics.get<double>("Latent heat of fusion");//, 334000.0);

  printedAlpha = -1.0;

}

template<typename EvalT, typename Traits, typename Type>
KOKKOS_INLINE_FUNCTION
void LiquidWaterFraction<EvalT,Traits,Type>::
operator() (const int& node, const int& cell) const{

    phi(cell,node) =  ( enthalpy(cell,node) < enthalpyHs(cell,node) ) ? ScalarT(0) : pow6 * (enthalpy(cell,node) - enthalpyHs(cell,node)) / (rho_w * L);

}

template<typename EvalT, typename Traits, typename Type>
void LiquidWaterFraction<EvalT,Traits,Type>::
postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(enthalpyHs,fm);
  this->utils.setFieldData(enthalpy,fm);

  this->utils.setFieldData(phi,fm);

  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
  if (d.memoizer_active()) memoizer.enable_memoizer();
}

template<typename EvalT, typename Traits, typename Type>
void LiquidWaterFraction<EvalT,Traits,Type>::
evaluateFields(typename Traits::EvalData workset)
{
  
  if (memoizer.have_saved_data(workset,this->evaluatedFields())) return;

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  Kokkos::parallel_for(Phi_Policy({0,0}, {numNodes,workset.numCells}), *this);
#else
  for (std::size_t cell = 0; cell < workset.numCells; ++cell)
  {
    for (std::size_t node = 0; node < numNodes; ++node)
    {
      phi(cell,node) =  ( enthalpy(cell,node) < enthalpyHs(cell,node) ) ? ScalarT(0) : pow6 * (enthalpy(cell,node) - enthalpyHs(cell,node)) / (rho_w * L);
    }
  }
#endif

}


}





