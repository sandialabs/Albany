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
XZHydrostatic_VirtualT_CpStar<EvalT, Traits>::
XZHydrostatic_VirtualT_CpStar(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Aeras::Layouts>& dl) :
  virt_t     (p.get<std::string> ("VirtualT"),    dl->node_scalar_level),
  cpstar     (p.get<std::string> ("CpStar"),      dl->node_scalar_level),
  temperature(p.get<std::string> ("Temperature"), dl->node_scalar_level),
  density    (p.get<std::string> ("Density"),     dl->node_scalar_level),
  tracerNames(p.get< Teuchos::ArrayRCP<std::string> >("Tracer Names")),

  numNodes   (dl->node_scalar             ->dimension(1)),
  numLevels  (dl->node_scalar_level       ->dimension(2))
{

  bool vapor = false;;

  Teuchos::ArrayRCP<std::string> RequiredTracers(1);
  RequiredTracers[0] = "Vapor";
  for (int i=0; i<1; ++i) {
    bool found = false;
    for (int j=0; j<tracerNames.size() && !found; ++j)
      if (RequiredTracers[i] == tracerNames[j]) {
        found = true;
        vapor = true;
      }
    TEUCHOS_TEST_FOR_EXCEPTION(!found, std::logic_error,
      "Aeras::XZHydrostatic_VirtualT_CpStar requires Vapor tracer.");
  }

  numTracers = tracerNames.size();
  if (numTracers > 0 && vapor) {
    qv = PHX::MDField<ScalarT,Cell,Node> ("Vapor",   dl->node_scalar_level);
    this->addDependentField(qv);
  }

  this->addDependentField(temperature);
  this->addDependentField(density);
  
  this->addEvaluatedField(virt_t);
  this->addEvaluatedField(cpstar);
  this->setName("Aeras::XZHydrostatic_VirtualT_CpStar"+PHX::TypeString<EvalT>::value);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_VirtualT_CpStar<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(virt_t,     fm);
  this->utils.setFieldData(cpstar,     fm);
  this->utils.setFieldData(temperature,fm);
  this->utils.setFieldData(density,fm);
  if (numTracers > 0 && vapor) { 
    this->utils.setFieldData(qv,fm);
  }
}

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_VirtualT_CpStar<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  const ScalarT R=287.0;
  const ScalarT Rv=461.5;
  const ScalarT factor = Rv/R - 1.0;
  const ScalarT cp = 1005.4;
  const ScalarT cpv = 1870.0; // Rogers and Yau 3rd ed. 1989

  if (numTracers == 0 || !vapor) {
    for (int cell=0; cell < workset.numCells; ++cell) {
      for (int node=0; node < numNodes; ++node) {
        for (int level=0; level < numLevels; ++level) {
          virt_t(cell,node,level) = temperature(cell,node,level);
          cpstar(cell,node,level) = cp;
        }
      }
    }
  } else { 
    for (int cell=0; cell < workset.numCells; ++cell) {
      for (int node=0; node < numNodes; ++node) {
        for (int level=0; level < numLevels; ++level) {
          virt_t(cell,node,level) = temperature(cell,node,level) 
                                  + factor * temperature(cell,node,level)*qv(cell,node,level)/density(cell,node,level);
          cpstar(cell,node,level) = cp + (cpv - cp)*qv(cell,node,level)/density(cell,node,level);
        }
      }
    }
  }
}
}
