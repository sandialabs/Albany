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
XZHydrostatic_GeoPotential<EvalT, Traits>::
XZHydrostatic_GeoPotential(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Aeras::Layouts>& dl) :
  density   (p.get<std::string> ("Density")     , dl->node_scalar_level),
  Eta       (p.get<std::string> ("Eta")         , dl->node_scalar_level),
  DeltaEta  (p.get<std::string> ("DeltaEta")    , dl->node_scalar_level),
  Pi        (p.get<std::string> ("Pi")          , dl->node_scalar_level),
  Phi       (p.get<std::string> ("GeoPotential"), dl->node_scalar_level),

  numNodes ( dl->node_scalar          ->dimension(1)),
  numLevels( dl->node_scalar_level    ->dimension(2)),
  P0(101.325),
  Ptop(101325.0),
  Phi0(0.0)
{

  Teuchos::ParameterList* xzhydrostatic_params = p.get<Teuchos::ParameterList*>("XZHydrostatic Problem");
  P0   = xzhydrostatic_params->get<double>("P0", 101325.0); //Default: P0=101325.0
  std::cout << "XZHydrostatic_GeoPotential: P0 = " << P0 << std::endl;
  Ptop = xzhydrostatic_params->get<double>("Ptop", 101.325); //Default: Ptop=101.325
  std::cout << "XZHydrostatic_GeoPotential: Ptop = " << Ptop << std::endl;
  Phi0 = xzhydrostatic_params->get<double>("Phi0", 0.0); //Default: Phi0=0.0
  std::cout << "XZHydrostatic_GeoPotential: Phi0 = " << Phi0 << std::endl;

  this->addDependentField(density);
  this->addDependentField(Eta);
  this->addDependentField(DeltaEta);
  this->addDependentField(Pi);

  this->addEvaluatedField(Phi);

  this->setName("Aeras::XZHydrostatic_GeoPotential"+PHX::TypeString<EvalT>::value);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_GeoPotential<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(density  , fm);
  this->utils.setFieldData(DeltaEta , fm);
  this->utils.setFieldData(Pi       , fm);
  this->utils.setFieldData(Phi      , fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_GeoPotential<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  const ScalarT Etatop = Ptop/P0;
  for (int cell=0; cell < workset.numCells; ++cell) {
    for (int node=0; node < numNodes; ++node) {
      for (int level=0; level < numLevels; ++level) {

        Phi(cell,node,level) = Phi0 + 0.5*(1/density(cell,node,level))*Pi(cell,node,level)*DeltaEta(cell,node,level);
        ScalarT sum = 0.0;
 
        for (int j=level+1; j < numLevels; ++j) {
          sum += (1/density(cell,node,level))*Pi(cell,node,level)*DeltaEta(cell,node,level);
        }
        Phi(cell,node,level) += sum;
      }
    }
  }
}
}
