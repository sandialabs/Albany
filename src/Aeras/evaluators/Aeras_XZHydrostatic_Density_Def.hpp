//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Teuchos_RCP.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"
#include "Aeras_Layouts.hpp"

namespace Aeras {

//**********************************************************************
template<typename EvalT, typename Traits>
XZHydrostatic_Density<EvalT, Traits>::
XZHydrostatic_Density(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Aeras::Layouts>& dl) :
  density    (p.get<std::string> ("Density"),     dl->node_scalar_level),
  pressure   (p.get<std::string> ("Pressure"),    dl->node_scalar_level),
  temperature(p.get<std::string> ("Temperature"), dl->node_scalar_level),

  numNodes   (dl->node_scalar             ->dimension(1)),
  numLevels  (dl->node_scalar_level       ->dimension(2))
{
  this->addDependentField(pressure);
  this->addDependentField(temperature);
  this->addEvaluatedField(density);
  this->setName("Aeras::XZHydrostatic_Density"+PHX::TypeString<EvalT>::value);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_Density<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(density,    fm);
  this->utils.setFieldData(pressure,   fm);
  this->utils.setFieldData(temperature,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_Density<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  const ScalarT R=287;
  for (int cell=0; cell < workset.numCells; ++cell) {
    for (int node=0; node < numNodes; ++node) {
      for (int level=0; level < numLevels; ++level) {
        density(cell,node,level) = 
          pressure(cell,node,level)/(R*temperature(cell,node,level));
      }
    }
  }
}
}
