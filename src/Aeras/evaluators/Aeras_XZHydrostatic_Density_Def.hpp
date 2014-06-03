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
  temperature (p.get<std::string> ("Temperature"), dl->node_scalar_level),
  numNodes ( dl->node_scalar             ->dimension(1)),
  numQPs   ( dl->node_qp_scalar          ->dimension(2)),
  numDims  ( dl->node_qp_gradient        ->dimension(3)),
  numLevels( dl->node_scalar_level       ->dimension(2))
{

  this->addDependentField(temperature);

  this->addEvaluatedField(density);

  this->setName("Aeras::XZHydrostatic_Density"+PHX::TypeString<EvalT>::value);

  density0 = 0.0;

}

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_Density<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(temperature,fm);
  this->utils.setFieldData(density,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_Density<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t node=0; node < numNodes; ++node) {
      for (std::size_t level=0; level < numLevels; ++level) {
        density(cell,node,level) = 1.0;
      }
    }
  }
}

//**********************************************************************
template<typename EvalT,typename Traits>
typename XZHydrostatic_Density<EvalT,Traits>::ScalarT& 
XZHydrostatic_Density<EvalT,Traits>::getValue(const std::string &n)
{
  if (n=="Density") return density0;
}

}
