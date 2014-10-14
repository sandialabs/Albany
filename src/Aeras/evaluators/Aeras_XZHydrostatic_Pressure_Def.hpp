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
#include "Aeras_Eta.hpp"

namespace Aeras {

//**********************************************************************
template<typename EvalT, typename Traits>
XZHydrostatic_Pressure<EvalT, Traits>::
XZHydrostatic_Pressure(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Aeras::Layouts>& dl) :
  Ps        (p.get<std::string> ("Pressure Level 0"), dl->node_scalar),
  Pressure  (p.get<std::string> ("Pressure"),         dl->node_scalar_level),
  Pi        (p.get<std::string> ("Pi"),               dl->node_scalar_level),

  numNodes ( dl->node_scalar          ->dimension(1)),
  numLevels( dl->node_scalar_level    ->dimension(2))
{

  Teuchos::ParameterList* xzhydrostatic_params =
    p.isParameter("XZHydrostatic Problem") ? 
      p.get<Teuchos::ParameterList*>("XZHydrostatic Problem"):
      p.get<Teuchos::ParameterList*>("Hydrostatic Problem");

  this->addDependentField(Ps);

  this->addEvaluatedField(Pressure);
  this->addEvaluatedField(Pi);
  this->setName("Aeras::XZHydrostatic_Pressure"+PHX::TypeString<EvalT>::value);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_Pressure<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(Ps       ,fm);
  this->utils.setFieldData(Pressure ,fm);
  this->utils.setFieldData(Pi       ,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_Pressure<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  const Eta<EvalT> &E = Eta<EvalT>::self();

  for (int cell=0; cell < workset.numCells; ++cell) {
    for (int node=0; node < numNodes; ++node) {
      for (int level=0; level < numLevels; ++level) {
        Pressure(cell,node,level) = E.A(level)*E.p0() + E.B(level)*Ps(cell,node);
      }
      for (int level=0; level < numLevels; ++level) {
        const ScalarT pm   = level             ? 0.5*( Pressure(cell,node,level) + Pressure(cell,node,level-1) ) : E.ptop();
        const ScalarT pp   = level<numLevels-1 ? 0.5*( Pressure(cell,node,level) + Pressure(cell,node,level+1) ) : Ps(cell,node);
        Pi(cell,node,level) = (pp - pm) /E.delta(level);
      }
    }
  }
}
}
