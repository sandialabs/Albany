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
#include "Aeras_Dimension.hpp"

#include "Aeras_Eta.hpp"

namespace Aeras {

//**********************************************************************
template<typename EvalT, typename Traits>
XZHydrostatic_EtaDotPi<EvalT, Traits>::
XZHydrostatic_EtaDotPi(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Aeras::Layouts>& dl) :
  divpivelx      (p.get<std::string> ("Divergence QP PiVelx"),  dl->qp_scalar_level),
  pdotP0         (p.get<std::string> ("Pressure Dot Level 0"),  dl->qp_scalar),
  Pi             (p.get<std::string> ("Pi"),                    dl->qp_scalar_level),
  Temperature    (p.get<std::string> ("QP Temperature"),        dl->qp_scalar_level),
  Velx           (p.get<std::string> ("QP Velx"),               dl->qp_vector_level),
  tracerNames    (p.get< Teuchos::ArrayRCP<std::string> >("Tracer Names")),
  etadotdtracerNames    (p.get< Teuchos::ArrayRCP<std::string> >("Tracer EtaDotd Names")),
  etadotdT       (p.get<std::string> ("EtaDotdT"),              dl->qp_scalar_level),
  etadotdVelx    (p.get<std::string> ("EtaDotdVelx"),           dl->qp_vector_level),
  Pidot          (p.get<std::string> ("PiDot"),                 dl->qp_scalar_level),

  numQPs     (dl->node_qp_scalar          ->dimension(2)),
  numDims    (dl->node_qp_gradient        ->dimension(3)),
  numLevels  (dl->node_scalar_level       ->dimension(2))
{

  Teuchos::ParameterList* xzhydrostatic_params =
    p.isParameter("XZHydrostatic Problem") ? 
      p.get<Teuchos::ParameterList*>("XZHydrostatic Problem"):
      p.get<Teuchos::ParameterList*>("Hydrostatic Problem");


  this->addDependentField(divpivelx);
  this->addDependentField(pdotP0);
  this->addDependentField(Pi);
  this->addDependentField(Temperature);
  this->addDependentField(Velx);

  this->addEvaluatedField(etadotdT);
  this->addEvaluatedField(etadotdVelx);
  this->addEvaluatedField(Pidot);

  for (int i = 0; i < tracerNames.size(); ++i) {
    PHX::MDField<ScalarT,Cell,QuadPoint,Level> in   (tracerNames[i],          dl->qp_scalar_level);
    PHX::MDField<ScalarT,Cell,QuadPoint,Level> out  (etadotdtracerNames[i],   dl->qp_scalar_level);
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
  this->utils.setFieldData(divpivelx  ,   fm);
  this->utils.setFieldData(pdotP0     ,   fm);
  this->utils.setFieldData(Pi         ,   fm);
  this->utils.setFieldData(Temperature,   fm);
  this->utils.setFieldData(Velx       ,   fm);
  this->utils.setFieldData(etadotdT   ,   fm);
  this->utils.setFieldData(etadotdVelx,   fm);
  this->utils.setFieldData(Pidot,         fm);
  for (int i = 0; i < Tracer.size();  ++i)       this->utils.setFieldData(Tracer[tracerNames[i]], fm);
  for (int i = 0; i < etadotdTracer.size(); ++i) this->utils.setFieldData(etadotdTracer[tracerNames[i]],fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_EtaDotPi<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  const Eta<EvalT> &E = Eta<EvalT>::self();

  //etadotpi(level) shifted by 1/2
  std::vector<ScalarT> etadotpi(numLevels+1);

  for (int cell=0; cell < workset.numCells; ++cell) {
    for (int qp=0; qp < numQPs; ++qp) {
      ScalarT pdotp0 = 0;
      for (int level=0; level < numLevels; ++level) pdotp0 -= divpivelx(cell,qp,level) * E.delta(level);
      for (int level=0; level < numLevels; ++level) {
        ScalarT integral = 0;
        for (int j=0; j<=level; ++j) integral += divpivelx(cell,qp,j) * E.delta(j);
        etadotpi[level] = -E.B(level+.5)*pdotp0 - integral;
      }
      etadotpi[0] = etadotpi[numLevels] = 0;

      //Vertical Finite Differencing
      for (int level=0; level < numLevels; ++level) {
        const ScalarT factor     = 1.0/(2.0*Pi(cell,qp,level)*E.delta(level));
        const int level_m = level             ? level-1 : 0;
        const int level_p = level+1<numLevels ? level+1 : level;
        const ScalarT etadotpi_m = etadotpi[level  ];
        const ScalarT etadotpi_p = etadotpi[level+1];

        const ScalarT dT_m       = Temperature(cell,qp,level)   - Temperature(cell,qp,level_m);
        const ScalarT dT_p       = Temperature(cell,qp,level_p) - Temperature(cell,qp,level);
        etadotdT(cell,qp,level) = factor * ( etadotpi_p*dT_p + etadotpi_m*dT_m );

        for (int dim=0; dim<numDims; ++dim) {
          const ScalarT dVx_m      = Velx(cell,qp,level,dim)   - Velx(cell,qp,level_m,dim);
          const ScalarT dVx_p      = Velx(cell,qp,level_p,dim) - Velx(cell,qp,level,dim);
          etadotdVelx(cell,qp,level,dim) = factor * ( etadotpi_p*dVx_p + etadotpi_m*dVx_m );
        }

        for (int i = 0; i < tracerNames.size(); ++i) {
          const ScalarT q_m = 0.5*( Tracer[tracerNames[i]](cell,qp,level)   / Pi(cell,qp,level)   
                                  + Tracer[tracerNames[i]](cell,qp,level_m) / Pi(cell,qp,level_m) );
          const ScalarT q_p = 0.5*( Tracer[tracerNames[i]](cell,qp,level_p) / Pi(cell,qp,level_p) 
                                  + Tracer[tracerNames[i]](cell,qp,level)   / Pi(cell,qp,level)   );
          etadotdTracer[tracerNames[i]](cell,qp,level) = ( etadotpi_p*q_p - etadotpi_m*q_m ) / E.delta(level);
        }

        Pidot(cell,qp,level) = - divpivelx(cell,qp,level) - (etadotpi_p - etadotpi_m)/E.delta(level);

      }
    }
  }
}
}
