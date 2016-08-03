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
XZHydrostatic_Velocity<EvalT, Traits>::
XZHydrostatic_Velocity(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Aeras::Layouts>& dl) :
  Velx     (p.get<std::string> ("Velx Name"),dl->node_vector_level),
  Velocity  (p.get<std::string> ("Velocity"),  dl->node_vector_level),
  numNodes ( dl->node_scalar             ->dimension(1)),
  numDims  ( dl->node_qp_gradient        ->dimension(3)),
  numLevels( dl->node_scalar_level       ->dimension(2))
{
  this->addDependentField(Velx);
  this->addEvaluatedField(Velocity);

  this->setName("Aeras::XZHydrostatic_Velocity" + PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_Velocity<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(Velx,   fm);
  this->utils.setFieldData(Velocity,fm);
}

//**********************************************************************
// Kokkos kernels
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void XZHydrostatic_Velocity<EvalT, Traits>::
operator() (const XZHydrostatic_Velocity_Tag& tag, const int& cell) const{
  for (int node=0; node < numNodes; ++node) 
    for (int level=0; level < numLevels; ++level) 
      for (int dim=0; dim < numDims; ++dim)  
        Velocity(cell,node,level,dim) = Velx(cell,node,level,dim); 
}

#endif

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_Velocity<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  for (int cell=0; cell < workset.numCells; ++cell) 
    for (int node=0; node < numNodes; ++node) 
      for (int level=0; level < numLevels; ++level) 
        for (int dim=0; dim < numDims; ++dim)  
          Velocity(cell,node,level,dim) = Velx(cell,node,level,dim);

#else
  Kokkos::parallel_for(XZHydrostatic_Velocity_Policy(0,workset.numCells),*this);

#endif
}
}
