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

template<typename EvalT, typename Traits>
typename EvalT::ScalarT XZHydrostatic_Pressure<EvalT, Traits>::
eta(const int L, const int l) 
{
  static const ScalarT Etatop = Ptop/P0;
  const ScalarT e = Etatop + l*(1-Etatop)/(L-1);
  return e;
}



//**********************************************************************
template<typename EvalT, typename Traits>
XZHydrostatic_Pressure<EvalT, Traits>::
XZHydrostatic_Pressure(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Aeras::Layouts>& dl) :
  Ps        (p.get<std::string> ("QP Pressure Level 0"), dl->qp_scalar),
  Pressure  (p.get<std::string> ("QP Pressure"),         dl->qp_scalar_level),
  Eta       (p.get<std::string> ("QP Eta"),              dl->qp_scalar_level),
  numQPs   ( dl->node_qp_scalar          ->dimension(2)),
  numLevels( dl->node_scalar_level       ->dimension(2)),
  Ptop     (100),
  P0       (100000)
{
  this->addDependentField(Ps);
  this->addEvaluatedField(Pressure);
  this->addEvaluatedField(Eta);
  this->setName("Aeras::XZHydrostatic_Pressure"+PHX::TypeString<EvalT>::value);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_Pressure<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(Ps,      fm);
  this->utils.setFieldData(Pressure,fm);
  this->utils.setFieldData(Eta     ,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_Pressure<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {
      for (std::size_t level=0; level < numLevels; ++level) {
        const ScalarT e = eta(numLevels, level);
        const ScalarT w = ScalarT(level)/(numLevels-1);
        Eta(cell,qp,level) = e;
        Pressure(cell,qp,level) = (1-w)*e*P0 + w*e*Ps(cell,qp);
      }
    }
  }
}
}
