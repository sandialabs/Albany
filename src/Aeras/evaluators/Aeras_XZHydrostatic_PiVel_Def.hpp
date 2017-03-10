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
XZHydrostatic_PiVel<EvalT, Traits>::
XZHydrostatic_PiVel(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Aeras::Layouts>& dl) :
  pi         (p.get<std::string> ("Pi"),          dl->node_scalar_level),
  velocity   (p.get<std::string> ("Velocity"),    dl->node_vector_level),
  pivelx     (p.get<std::string> ("PiVelx"),      dl->node_vector_level),

  numDims    (dl->node_qp_gradient        ->dimension(3)),
  numNodes   (dl->node_scalar             ->dimension(1)),
  numLevels  (dl->node_scalar_level       ->dimension(2))
{
  this->addDependentField(pi);
  this->addDependentField(velocity);

  this->addEvaluatedField(pivelx);
  this->setName("Aeras::XZHydrostatic_PiVel"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_PiVel<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(pi     , fm);
  this->utils.setFieldData(velocity  , fm);
  this->utils.setFieldData(pivelx , fm);
}

//**********************************************************************
// Kokkos kernels
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void XZHydrostatic_PiVel<EvalT, Traits>::
operator() (const XZHydrostatic_PiVel_Tag &tag, const int cell, const int node, const int level) const{
      for (int dim=0; dim < numDims; ++dim)  
        pivelx(cell,node,level,dim) = pi(cell,node,level)*velocity(cell,node,level,dim);
}

#endif

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_PiVel<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  for (int cell=0; cell < workset.numCells; ++cell) 
    for (int node=0; node < numNodes; ++node) 
      for (int level=0; level < numLevels; ++level) 
        for (int dim=0; dim < numDims; ++dim) {
          pivelx(cell,node,level,dim) = pi(cell,node,level)*velocity(cell,node,level,dim);
             //std::cout << "pivelx: " << cell << " " << node << " " << level << " " << dim << " " << PiVelx(cell,node,level,dim) << std::endl;
            //std::cout << "Tracer " << Tracer(cell,node,level) << std::endl;
          }
#else
  XZHydrostatic_PiVel_Policy range(
      {0,0,0}, {(int)workset.numCells,(int)numNodes,(int)numLevels},
      XZHydrostatic_PiVel_TileSize);
  Kokkos::Experimental::md_parallel_for(range,*this);

#endif
}
}
