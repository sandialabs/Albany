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
XZHydrostatic_Omega<EvalT, Traits>::
XZHydrostatic_Omega(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Aeras::Layouts>& dl) :
  Velx      (p.get<std::string> ("Velx"),                 dl->qp_scalar_level),
  gradp     (p.get<std::string> ("Gradient QP Pressure"), dl->qp_gradient_level),
  density   (p.get<std::string> ("Density"),              dl->qp_scalar_level),
  gradpivelx(p.get<std::string> ("Gradient QP PiVelx"),   dl->qp_gradient_level),
  omega     (p.get<std::string> ("Omega")              ,  dl->qp_scalar_level),

  numQPs     (dl->node_qp_scalar          ->dimension(2)),
  numLevels  (dl->node_scalar_level       ->dimension(2)),
  Cp         (1005.7)

{

  Teuchos::ParameterList* xzhydrostatic_params = p.get<Teuchos::ParameterList*>("XZHydrostatic Problem");
  Cp = xzhydrostatic_params->get<double>("Cp", 1005.7); //Default: Cp=1005.7
  std::cout << "XZHydrostatic_Omega: Cp = " << Cp << std::endl;

  this->addDependentField(Velx);
  this->addDependentField(gradp);
  this->addDependentField(density);
  this->addDependentField(gradpivelx);

  this->addEvaluatedField(omega);

  this->setName("Aeras::XZHydrostatic_Omega"+PHX::TypeString<EvalT>::value);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_Omega<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(Velx      ,   fm);
  this->utils.setFieldData(gradp     ,   fm);
  this->utils.setFieldData(density   ,   fm);
  this->utils.setFieldData(gradpivelx,   fm);
  this->utils.setFieldData(omega     ,   fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_Omega<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  const Eta<EvalT> &E = Eta<EvalT>::self();

  for (int cell=0; cell < workset.numCells; ++cell) {
    for (int qp=0; qp < numQPs; ++qp) {
      for (int level=0; level < numLevels; ++level) {

        ScalarT sum = Velx(cell,qp,level)*gradp(cell,qp,level) + 0.5*gradpivelx(cell,qp,level)*E.delta(level) ;
        for (int j=0; j<level-1; ++j) {
          sum -= gradpivelx(cell,qp,level) * E.delta(level);
        }  
        omega(cell,qp,level) = (1/(density(cell,qp,level)*Cp))*sum;
       
      }
    }
  }
}
}
