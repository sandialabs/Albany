/*
 * FELIX_VerticalVelocity_Def.hpp
 *
 *  Created on: Jun 15, 2016
 *      Author: abarone
 */

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

namespace FELIX
{


  template<typename EvalT, typename Traits, typename VelocityType>
  VerticalVelocity<EvalT,Traits,VelocityType>::
  VerticalVelocity(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl):
  thickness				(p.get<std::string> ("Thickness Variable Name"), dl->node_scalar),
  int1Dw_z 				(p.get<std::string> ("Integral1D w_z Variable Name"),dl->node_scalar),
  w						(p.get<std::string> ("Vertical Velocity Variable Name"),dl->node_scalar)
  {
    this->addDependentField(thickness.fieldTag());
    this->addDependentField(int1Dw_z.fieldTag());

    this->addEvaluatedField(w);
    this->setName("Vertical Velocity");

    std::vector<PHX::Device::size_type> dims;
    dl->node_qp_vector->dimensions(dims);

    numNodes = dims[1];
  }

  template<typename EvalT, typename Traits, typename VelocityType>
  void VerticalVelocity<EvalT,Traits,VelocityType>::
  postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(thickness,fm);
    this->utils.setFieldData(int1Dw_z,fm);
    this->utils.setFieldData(w,fm);
  }

  template<typename EvalT, typename Traits, typename Type>
  void VerticalVelocity<EvalT,Traits,Type>::
  evaluateFields(typename Traits::EvalData d)
  {
    for (std::size_t cell = 0; cell < d.numCells; ++cell)
      for (std::size_t node = 0; node < numNodes; ++node)
        w(cell,node) = thickness(cell,node) * int1Dw_z(cell,node);
  }

}	// end namespace FELIX
