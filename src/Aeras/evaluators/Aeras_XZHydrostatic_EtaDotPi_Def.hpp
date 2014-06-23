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
  numLevels  (dl->node_scalar_level       ->dimension(2)),
  P0(101325.0),
  Ptop(101.325)
{

  Teuchos::ParameterList* xzhydrostatic_params = p.get<Teuchos::ParameterList*>("XZHydrostatic Problem");
  P0   = xzhydrostatic_params->get<double>("P0", 101325.0); //Default: P0=101325.0
  Ptop = xzhydrostatic_params->get<double>("Ptop", 101.325); //Default: Ptop=101.325
  std::cout << "XZHydrostatic_EtaDotPi: P0 = " << P0 << std::endl;
  std::cout << "XZHydrostatic_EtaDotPi: Ptop = " << Ptop << std::endl;

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

  const ScalarT Etatop = Ptop/P0;
  const ScalarT DeltaEta = (1-Etatop)/numLevels;
  std::vector<ScalarT> etadotpi(numLevels);

  for (int cell=0; cell < workset.numCells; ++cell) {
    for (int qp=0; qp < numQPs; ++qp) {
      for (int level=0; level < numLevels; ++level) {
        ScalarT integral = 0;
        for (int j=0; j<level; ++j) {
          //const ScalarT e_jp = Etatop + (1-Etatop)*ScalarT(j+.5)/numLevels;
          //const ScalarT e_jm = Etatop + (1-Etatop)*ScalarT(j-.5)/numLevels;
          //const ScalarT del_eta = (1-Etatop)/numLevels ; // e_jp - e_jm;
          integral += gradpivelx(cell,qp,j) * DeltaEta;
        }  
        const ScalarT e_i = Etatop + (1-Etatop)*ScalarT(level+.5)/numLevels;
        const ScalarT w_i =                     ScalarT(level+.5)/numLevels;
        const ScalarT   B = w_i * e_i;
        if (!level) etadotpi[level] = 0;
        else        etadotpi[level] = -B*pdotP0(cell,qp) - integral;
      }

      //Vertical Finite Differencing
      int level = 0;
      ScalarT factor     = 1.0/(2.0*Pi(cell,qp,level)*DeltaEta);
      ScalarT etadotpi_m = 0.0;
      ScalarT etadotpi_p = etadotpi[level];

      ScalarT dT_m       = 0.0;
      ScalarT dT_p       = Temperature(cell,qp,level+1) - Temperature(cell,qp,level);
      etadotdT(cell,qp,level) = factor * ( etadotpi_p*dT_p + etadotpi_m*dT_m );

      ScalarT dVx_m      = 0.0;
      ScalarT dVx_p      = Velx(cell,qp,level+1) - Velx(cell,qp,level);
      etadotdVelx(cell,qp,level) = factor * ( etadotpi_p*dVx_p + etadotpi_m*dVx_m );

      for (int i = 0; i < tracerNames.size(); ++i) {
        ScalarT dq_m = 0.0;
        ScalarT dq_p = Tracer[tracerNames[i]](cell,qp,level+1) - Tracer[tracerNames[i]](cell,qp,level);
        etadotdTracer[tracerNames[i]](cell,qp,level) = factor * ( etadotpi_p*dq_p + etadotpi_m*dq_m );
      }

      for (level=1; level < numLevels-1; ++level) {
        ScalarT factor     = 1.0/(2.0*Pi(cell,qp,level)*DeltaEta);
        ScalarT etadotpi_m = etadotpi[level-1];
        ScalarT etadotpi_p = etadotpi[level  ];

        ScalarT dT_m       = Temperature(cell,qp,level)   - Temperature(cell,qp,level-1);
        ScalarT dT_p       = Temperature(cell,qp,level+1) - Temperature(cell,qp,level);
        etadotdT(cell,qp,level) = factor * ( etadotpi_p*dT_p + etadotpi_m*dT_m );

        ScalarT dVx_m      = Velx(cell,qp,level)   - Velx(cell,qp,level-1);
        ScalarT dVx_p      = Velx(cell,qp,level+1) - Velx(cell,qp,level);
        etadotdVelx(cell,qp,level) = factor * ( etadotpi_p*dVx_p + etadotpi_m*dVx_m );

        for (int i = 0; i < tracerNames.size(); ++i) {
          ScalarT dq_m = Tracer[tracerNames[i]](cell,qp,level)   - Tracer[tracerNames[i]](cell,qp,level-1);
          ScalarT dq_p = Tracer[tracerNames[i]](cell,qp,level+1) - Tracer[tracerNames[i]](cell,qp,level);
          etadotdTracer[tracerNames[i]](cell,qp,level) = factor * ( etadotpi_p*dq_p + etadotpi_m*dq_m );
        }
      }

      level = numLevels-1;
      factor     = 1.0/(2.0*Pi(cell,qp,level)*DeltaEta);
      etadotpi_m = etadotpi[level];
      etadotpi_p = 0.0;

      dT_m = Temperature(cell,qp,level) - Temperature(cell,qp,level-1);
      dT_p = 0.0;
      etadotdT(cell,qp,level) = factor * (etadotpi_p*dT_p + etadotpi_m*dT_m);

      dVx_m = Velx(cell,qp,level) - Velx(cell,qp,level-1);
      dVx_p = 0.0;
      etadotdVelx(cell,qp,level) = factor * (etadotpi_p*dVx_p + etadotpi_m*dVx_m);

      for (int i = 0; i < tracerNames.size(); ++i) {
        ScalarT dq_m = Tracer[tracerNames[i]](cell,qp,level) - Tracer[tracerNames[i]](cell,qp,level-1);
        ScalarT dq_p = 0.0;
        etadotdTracer[tracerNames[i]](cell,qp,level) = factor * ( etadotpi_p*dq_p + etadotpi_m*dq_m );
      }
    }
  }
}
}
