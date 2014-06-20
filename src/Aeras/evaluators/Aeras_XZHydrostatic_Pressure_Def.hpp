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
XZHydrostatic_Pressure<EvalT, Traits>::
XZHydrostatic_Pressure(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Aeras::Layouts>& dl) :
  Ps        (p.get<std::string> ("Pressure Level 0"), dl->node_scalar),
  Pressure  (p.get<std::string> ("Pressure"),         dl->node_scalar_level),
  Eta       (p.get<std::string> ("Eta"),              dl->node_scalar_level),
  DeltaEta  (p.get<std::string> ("DeltaEta"),         dl->node_scalar_level),
  Pi        (p.get<std::string> ("Pi"),               dl->node_scalar_level),

  numNodes ( dl->node_scalar          ->dimension(1)),
  numLevels( dl->node_scalar_level    ->dimension(2)),
  P0(101325.0),
  Ptop(101.325)
{

  Teuchos::ParameterList* xzhydrostatic_params = p.get<Teuchos::ParameterList*>("XZHydrostatic Problem");
  P0   = xzhydrostatic_params->get<double>("P0", 101325.0); //Default: P0=101325.0
  Ptop = xzhydrostatic_params->get<double>("Ptop", 101.325); //Default: Ptop=101.325
  std::cout << "XZHydrostatic_Pressure: P0 = " << P0 << std::endl;
  std::cout << "XZHydrostatic_Pressure: Ptop = " << Ptop << std::endl;

  this->addDependentField(Ps);

  this->addEvaluatedField(Pressure);
  this->addEvaluatedField(Eta);
  this->addEvaluatedField(DeltaEta);
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
  this->utils.setFieldData(Eta      ,fm);
  this->utils.setFieldData(DeltaEta ,fm);
  this->utils.setFieldData(Pi       ,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_Pressure<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  const ScalarT Etatop = Ptop/P0;

  for (int cell=0; cell < workset.numCells; ++cell) {
    for (int node=0; node < numNodes; ++node) {
      for (int level=0; level < numLevels; ++level) {
        const ScalarT e = Etatop + (1-Etatop)*ScalarT(level)/(numLevels-1);
        const ScalarT w =                     ScalarT(level)/(numLevels-1);
        Eta(cell,node,level) = e;
        Pressure(cell,node,level) = (1-w)*e*P0 + w*e*Ps(cell,node);
      }

      //level 0
      int level = 0;
      ScalarT pp   = 0.5*( Pressure(cell,node,level) + Pressure(cell,node,level+1) );
      ScalarT pm   = Ptop;
      ScalarT etap = 0.5*( Eta(cell,node,level) + Eta(cell,node,level+1) );
      ScalarT etam = Etatop;
      Pi(cell,node,level) = (pp - pm) / (etap - etam);
      DeltaEta(cell,node,level) = etap - etam;
      
      for (level=1; level < numLevels-1; ++level) {
        pp   = 0.5*( Pressure(cell,node,level) + Pressure(cell,node,level+1) );
        pm   = 0.5*( Pressure(cell,node,level) + Pressure(cell,node,level-1) );
        etap = 0.5*( Eta(cell,node,level) + Eta(cell,node,level+1) );
        etam = 0.5*( Eta(cell,node,level) + Eta(cell,node,level-1) );
        Pi(cell,node,level) = (pp - pm) / (etap - etam);
        DeltaEta(cell,node,level) = etap - etam;
      }

      //level numLevels-1
      level = numLevels-1;
      pp   = Ps(cell,node);
      pm   = 0.5*( Pressure(cell,node,level) + Pressure(cell,node,level-1) );
      etap = 1.0; 
      etam = 0.5*( Eta(cell,node,level) + Eta(cell,node,level-1) );
      Pi(cell,node,level) = (pp - pm) / (etap - etam);
      DeltaEta(cell,node,level) = etap - etam;
    }
  }
}
}
