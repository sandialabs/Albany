//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Teuchos_RCP.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"
#include "Aeras_Layouts.hpp"
#include "Aeras_Dimension.hpp"

#include "Aeras_Eta.hpp"

namespace Aeras {

//**********************************************************************
template<typename EvalT, typename Traits>
Hydrostatic_EtaDot<EvalT, Traits>::
Hydrostatic_EtaDot(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Aeras::Layouts>& dl) :
  sphere_coord  (p.get<std::string>  ("Spherical Coord Name"), dl->qp_gradient ),
  etadot       (p.get<std::string> ("EtaDot"),              dl->qp_scalar_level),
  numQPs     (dl->node_qp_scalar          ->dimension(2)),
  numDims    (dl->node_qp_gradient        ->dimension(3)),
  numLevels  (dl->node_scalar_level       ->dimension(2))
{

  Teuchos::ParameterList* xzhydrostatic_params =
      p.get<Teuchos::ParameterList*>("Hydrostatic Problem");

  std::string advType = xzhydrostatic_params->get("Advection Type", "Unknown");
  
  if (advType == "Unknown")
    adv_type = UNKNOWN;
  else if (advType == "Prescribed 1-1") 
    adv_type = PRESCRIBED_1_1; 
  else if (advType == "Prescribed 1-2")
    adv_type = PRESCRIBED_1_2; 
  else 
    TEUCHOS_TEST_FOR_EXCEPTION(true,
 		               Teuchos::Exceptions::InvalidParameter,"Aeras::Hydrostatic_Velocity: " 
                               << "Advection Type = " << advType << " is invalid!"); 

  this->addDependentField(sphere_coord);
  this->addDependentField(etadot);
  this->setName("Aeras::Hydrostatic_EtaDot"+PHX::typeAsString<EvalT>());

  pureAdvection = xzhydrostatic_params->get<bool>("Pure Advection", false);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void Hydrostatic_EtaDot<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(sphere_coord  ,   fm);
  this->utils.setFieldData(etadot  ,   fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void Hydrostatic_EtaDot<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  double time = workset.current_time; 
  switch (adv_type) {
    case PRESCRIBED_1_1: //etadot is prescribed to that of 1-1 test
      //FIXME: Pete, Tom will fill in
      for (int cell=0; cell < workset.numCells; ++cell) {
        for (int qp=0; qp < numQPs; ++qp) {
          const MeshScalarT lambda = sphere_coord(cell, qp, 0);
          const MeshScalarT theta = sphere_coord(cell, qp, 1);
          for (int level=0; level < numLevels; ++level) {
            etadot(cell,qp,level) = 0.0; 
          }
        }
      }
    break;
    case PRESCRIBED_1_2: //etadot is prescribed to that of 1-2 test
      //FIXME: Pete, Tom will fill in
      for (int cell=0; cell < workset.numCells; ++cell) {
        for (int qp=0; qp < numQPs; ++qp) {
          const MeshScalarT lambda = sphere_coord(cell, qp, 0);
          const MeshScalarT theta = sphere_coord(cell, qp, 1);
          for (int level=0; level < numLevels; ++level) {
            etadot(cell,qp,level) = 0.0; 
          }
        }
      }
      break;
    default: //constant advection
      for (int cell=0; cell < workset.numCells; ++cell) {
        for (int qp=0; qp < numQPs; ++qp) {
          for (int level=0; level < numLevels; ++level) {
            etadot(cell,qp,level) = 0.0; 
          }
        }
      }
      break; 
    }
}
}
