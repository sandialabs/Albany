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
XZHydrostatic_EtaDotPi<EvalT, Traits>::
XZHydrostatic_EtaDotPi(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Aeras::Layouts>& dl) :
  graddvelx  (p.get<std::string> ("Gradient QP DensityVelx"),   dl->qp_scalar_level),
  pdotP0     (p.get<std::string> ("Pressure Dot Level 0"),      dl->qp_scalar),

  etadotpi   (p.get<std::string> ("EtaDotPi"),                  dl->qp_scalar_level),

  numQPs     (dl->node_qp_scalar          ->dimension(2)),
  numLevels  (dl->node_scalar_level       ->dimension(2)),
  Ptop     (100),
  P0       (100000)

{
  this->addDependentField(graddvelx);
  this->addDependentField(pdotP0);

  this->addEvaluatedField(etadotpi);
  this->setName("Aeras::XZHydrostatic_Density"+PHX::TypeString<EvalT>::value);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_EtaDotPi<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(graddvelx   ,   fm);
  this->utils.setFieldData(pdotP0      ,   fm);
  this->utils.setFieldData(etadotpi    ,   fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_EtaDotPi<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  const ScalarT Etatop = Ptop/P0;

  for (int cell=0; cell < workset.numCells; ++cell) {
    for (int qp=0; qp < numQPs; ++qp) {
      for (int level=0; level < numLevels; ++level) {
        ScalarT integral = 0;
        for (int j=0; j<level; ++j) {
          const ScalarT e_jp = Etatop + (1-Etatop)*ScalarT(j+.5)/(numLevels-1);
          const ScalarT e_jm = Etatop + (1-Etatop)*ScalarT(j-.5)/(numLevels-1);
          const ScalarT del_eta = e_jp - e_jm;
          integral += graddvelx(cell,qp,j) * del_eta;
        }  
        const ScalarT e_i = Etatop + (1-Etatop)*ScalarT(level+.5)/(numLevels-1);
        const ScalarT w_i =                     ScalarT(level+.5)/(numLevels-1);
        const ScalarT   B = w_i * e_i;
        if (!level) etadotpi(cell,qp,level) = 0;
        else        etadotpi(cell,qp,level) = -B*pdotP0(cell,qp) - integral;
       
      }
    }
  }
}
}
