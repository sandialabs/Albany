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
XZHydrostatic_EtaDotPi<EvalT, Traits>::
XZHydrostatic_EtaDotPi(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Aeras::Layouts>& dl) :
  gradpivelx     (p.get<std::string> ("Gradient QP PiVelx"),    dl->qp_gradient_level),
  pdotP0         (p.get<std::string> ("Pressure Dot Level 0"),  dl->qp_scalar),
  Pi             (p.get<std::string> ("Pi"),                    dl->qp_scalar_level),
  Temperature    (p.get<std::string> ("QP Temperature"),        dl->qp_scalar_level),
  Velx           (p.get<std::string> ("QP Velx"),               dl->qp_scalar_level),
  tracerNames    (p.get< Teuchos::ArrayRCP<std::string> >("Tracer Names")),
  etadotdtracerNames    (p.get< Teuchos::ArrayRCP<std::string> >("Tracer EtaDotd Names")),
  etadotdT       (p.get<std::string> ("EtaDotdT"),              dl->qp_scalar_level),
  etadotdVelx    (p.get<std::string> ("EtaDotdVelx"),           dl->qp_scalar_level),

  numQPs     (dl->node_qp_scalar          ->dimension(2)),
  numLevels  (dl->node_scalar_level       ->dimension(2))
{

  Teuchos::ParameterList* xzhydrostatic_params = p.get<Teuchos::ParameterList*>("XZHydrostatic Problem");

  this->addDependentField(gradpivelx);
  this->addDependentField(pdotP0);
  this->addDependentField(Pi);
  this->addDependentField(Temperature);
  this->addDependentField(Velx);

  this->addEvaluatedField(etadotdT);
  this->addEvaluatedField(etadotdVelx);

  for (int i = 0; i < tracerNames.size(); ++i) {
    PHX::MDField<ScalarT,Cell,QuadPoint> in   (tracerNames[i],   dl->qp_scalar_level);
    PHX::MDField<ScalarT,Cell,QuadPoint> out  (etadotdtracerNames[i],   dl->qp_scalar_level);
    Tracer[tracerNames[i]]         = in;
    etadotdTracer[tracerNames[i]] = out;
    this->addDependentField(Tracer[tracerNames[i]]);
    this->addEvaluatedField(etadotdTracer[tracerNames[i]]);
  }

  this->setName("Aeras::XZHydrostatic_EtaDotPi"+PHX::TypeString<EvalT>::value);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_EtaDotPi<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(gradpivelx ,   fm);
  this->utils.setFieldData(pdotP0     ,   fm);
  this->utils.setFieldData(Pi         ,   fm);
  this->utils.setFieldData(Temperature,   fm);
  this->utils.setFieldData(Velx       ,   fm);
  this->utils.setFieldData(etadotdT   ,   fm);
  this->utils.setFieldData(etadotdVelx,   fm);
  for (int i = 0; i < Tracer.size();  ++i)       this->utils.setFieldData(Tracer[tracerNames[i]], fm);
  for (int i = 0; i < etadotdTracer.size(); ++i) this->utils.setFieldData(etadotdTracer[tracerNames[i]],fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_EtaDotPi<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  const Eta<EvalT> &E = Eta<EvalT>::self();

  std::vector<ScalarT> etadotpi(numLevels+1);

  for (int cell=0; cell < workset.numCells; ++cell) {
    for (int qp=0; qp < numQPs; ++qp) {
      for (int level=0; level < numLevels; ++level) {
        ScalarT integral = 0;
        for (int j=0; j<level; ++j) integral += gradpivelx(cell,qp,j) * E.delta(j);
        etadotpi[level] = -E.B(level+.5)*pdotP0(cell,qp) - integral;
      }
      etadotpi[0] = etadotpi[numLevels] = 0;

      //Vertical Finite Differencing
      for (int level=0; level < numLevels; ++level) {
        const ScalarT factor     = 1.0/(2.0*Pi(cell,qp,level)*E.delta(level));
        const int level_p = level+1<numLevels ? level+1 : level;
        const int level_m = level             ? level-1 : 0;
        const ScalarT etadotpi_p = etadotpi[level+1];
        const ScalarT etadotpi_m = etadotpi[level  ];

        const ScalarT dT_p       = Temperature(cell,qp,level_p) - Temperature(cell,qp,level);
        const ScalarT dT_m       = Temperature(cell,qp,level)   - Temperature(cell,qp,level_m);
        etadotdT(cell,qp,level) = factor * ( etadotpi_p*dT_p + etadotpi_m*dT_m );

        const ScalarT dVx_p      = Velx(cell,qp,level_p) - Velx(cell,qp,level);
        const ScalarT dVx_m      = Velx(cell,qp,level)   - Velx(cell,qp,level_m);
        etadotdVelx(cell,qp,level) = factor * ( etadotpi_p*dVx_p + etadotpi_m*dVx_m );

        for (int i = 0; i < tracerNames.size(); ++i) {
          const ScalarT dq_p = Tracer[tracerNames[i]](cell,qp,level_p) - Tracer[tracerNames[i]](cell,qp,level);
          const ScalarT dq_m = Tracer[tracerNames[i]](cell,qp,level)   - Tracer[tracerNames[i]](cell,qp,level_m);
          etadotdTracer[tracerNames[i]](cell,qp,level) = factor * ( etadotpi_p*dq_p + etadotpi_m*dq_m );
        }
      }
    }
  }
}
}
