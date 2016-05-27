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
  numLevels( dl->node_scalar_level       ->dimension(2))
{
  //FIXME: add parameter list input string for velocity type to use 
  //nonlinear velocity computed by the VelResid evaluator or a prescribed velocity
  //hard-coded here.
  //Time will need to be obtained from the workset to hard-code a transient velocity.
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
  //FIXME: add switch statement here to set velocity depending on input string
  for (int cell=0; cell < workset.numCells; ++cell) 
    for (int node=0; node < numNodes; ++node) 
      for (int level=0; level < numLevels; ++level) 
        for (int dim=0; dim < numDims; ++dim)  
          Velocity(cell,node,level,dim) = Velx(cell,node,level,dim); 
}
}
