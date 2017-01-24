/*
 * FELIX_PressureMeltingEnthalpy_Def.hpp
 *
 *  Created on: Jun 6, 2016
 *      Author: abarone
 */

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

namespace FELIX
{

  template<typename EvalT, typename Traits, typename Type>
  PressureMeltingEnthalpy<EvalT,Traits,Type>::
  PressureMeltingEnthalpy(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl):
  meltingTemp    (p.get<std::string> ("Melting Temperature Variable Name"), dl->node_scalar),
  surfaceTemp	   (p.get<std::string> ("Surface Air Temperature Name"), dl->node_scalar),
  enthalpyHs     (p.get<std::string> ("Enthalpy Hs Variable Name"), dl->node_scalar),
  surfaceEnthalpy(p.get<std::string> ("Surface Air Enthalpy Name"), dl->node_scalar)

  {
    std::vector<PHX::Device::size_type> dims;
    dl->node_qp_vector->dimensions(dims);

    numNodes = dims[1];

    this->addDependentField(meltingTemp.fieldTag());
    this->addDependentField(surfaceTemp.fieldTag());

    this->addEvaluatedField(enthalpyHs);
    this->addEvaluatedField(surfaceEnthalpy);
    this->setName("Pressure-melting Enthalpy");

    // Setting parameters
    Teuchos::ParameterList& physics = *p.get<Teuchos::ParameterList*>("FELIX Physical Parameters");
    rho_i 	= physics.get<double>("Ice Density", 916.0);
    c_i 	= physics.get<double>("Heat capacity of ice", 2009.0);
    T0 		= physics.get<double>("Reference Temperature", 265.0);
  }

  template<typename EvalT, typename Traits, typename Type>
  void PressureMeltingEnthalpy<EvalT,Traits,Type>::
  postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(meltingTemp,fm);
    this->utils.setFieldData(surfaceTemp,fm);

    this->utils.setFieldData(enthalpyHs,fm);
    this->utils.setFieldData(surfaceEnthalpy,fm);
  }

  template<typename EvalT, typename Traits, typename Type>
  void PressureMeltingEnthalpy<EvalT,Traits,Type>::
  evaluateFields(typename Traits::EvalData d)
  {
    const double powm6 = 1e-6; // [k^2], k=1000

    for (std::size_t cell = 0; cell < d.numCells; ++cell)
      for (std::size_t node = 0; node < numNodes; ++node) {
        enthalpyHs(cell,node) = rho_i * c_i * ( meltingTemp(cell,node) - T0 ) * powm6;
        surfaceEnthalpy(cell,node) = rho_i * c_i * ( surfaceTemp(cell,node) - T0 ) * powm6;
      }
  }


}

