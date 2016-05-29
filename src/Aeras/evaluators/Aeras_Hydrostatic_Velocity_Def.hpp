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

namespace Aeras {

//**********************************************************************
template<typename EvalT, typename Traits>
Hydrostatic_Velocity<EvalT, Traits>::
Hydrostatic_Velocity(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Aeras::Layouts>& dl) :
  Velx     (p.get<std::string> ("Velx Name"),dl->node_vector_level),
  sphere_coord  (p.get<std::string>  ("Spherical Coord Name"), dl->qp_gradient ),
  Velocity  (p.get<std::string> ("Velocity"),  dl->node_vector_level),
  numNodes ( dl->node_scalar             ->dimension(1)),
  numDims  ( dl->node_qp_gradient        ->dimension(3)),
  numLevels( dl->node_scalar_level       ->dimension(2)),
  out(Teuchos::VerboseObjectBase::getDefaultOStream())
{
  Teuchos::ParameterList* hs_list = p.get<Teuchos::ParameterList*>("Hydrostatic Problem");
  
  std::string advType = hs_list->get("Advection Type", "Unknown");
  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());

  *out << "Advection Type = " << advType << std::endl; 
  
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

  this->addDependentField(Velx);
  this->addDependentField(sphere_coord);
  this->addEvaluatedField(Velocity);

  this->setName("Aeras::Hydrostatic_Velocity" );
}

//**********************************************************************
template<typename EvalT, typename Traits>
void Hydrostatic_Velocity<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(Velx,   fm);
  this->utils.setFieldData(sphere_coord,   fm);
  this->utils.setFieldData(Velocity,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void Hydrostatic_Velocity<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  double time = workset.current_time; 
  //*out << "Aeras::Hydrostatic_Velocity time = " << time << std::endl; 
  switch (adv_type) {
    case UNKNOWN: //velocity is an unknown that we solve for (not prescribed)
      for (int cell=0; cell < workset.numCells; ++cell) 
        for (int node=0; node < numNodes; ++node) 
          for (int level=0; level < numLevels; ++level) 
            for (int dim=0; dim < numDims; ++dim)  
              Velocity(cell,node,level,dim) = Velx(cell,node,level,dim); 
      break; 
    case PRESCRIBED_1_1: //velocity is prescribed to that of 1-1 test
      //FIXME: Pete, Tom - please fill in
      //const MeshScalarT lambda = sphere_coord(cell, qp, 0);
      //const MeshScalarT theta = sphere_coord(cell, qp, 1);
      break; 
    case PRESCRIBED_1_2: //velocity is prescribed to that of 1-2 test
      //FIXME: Pete, Tom - please fill in
      //const MeshScalarT lambda = sphere_coord(cell, qp, 0);
      //const MeshScalarT theta = sphere_coord(cell, qp, 1);
      break; 
  }
}
}
