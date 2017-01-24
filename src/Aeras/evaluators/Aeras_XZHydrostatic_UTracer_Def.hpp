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
XZHydrostatic_UTracer<EvalT, Traits>::
XZHydrostatic_UTracer(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Aeras::Layouts>& dl) :
  Velocity (p.get<std::string> ("Velocity"),dl->node_vector_level),
  PiVelx   (p.get<std::string> ("PiVelx"),   dl->node_vector_level),
  Tracer   (p.get<std::string> ("Tracer"),   dl->node_scalar_level),
  UTracer  (p.get<std::string> ("UTracer"),  dl->node_vector_level),

  numDims  (dl->node_qp_gradient         ->dimension(3)),
  numNodes ( dl->node_scalar             ->dimension(1)),
  numLevels( dl->node_scalar_level       ->dimension(2))
{
  this->addDependentField(Velocity);
  this->addDependentField(PiVelx);
  this->addDependentField(Tracer);
  this->addEvaluatedField(UTracer);

  this->setName("Aeras::XZHydrostatic_UTracer" + PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_UTracer<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(Velocity,   fm);
  this->utils.setFieldData(PiVelx, fm);
  this->utils.setFieldData(Tracer, fm);
  this->utils.setFieldData(UTracer,fm);
}

//**********************************************************************
// Kokkos kernels
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void XZHydrostatic_UTracer<EvalT, Traits>::
operator() (const XZHydrostatic_UTracer_Tag& tag, const int& cell) const{
  for (int node=0; node < numNodes; ++node) 
    for (int level=0; level < numLevels; ++level) 
      for (int dim=0; dim < numDims; ++dim)
        UTracer(cell,node,level,dim) = Velocity(cell,node,level,dim)*Tracer(cell,node,level);
}

#endif

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_UTracer<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  for (int cell=0; cell < workset.numCells; ++cell) 
    for (int node=0; node < numNodes; ++node) 
      for (int level=0; level < numLevels; ++level) 
        for (int dim=0; dim < numDims; ++dim) {
          UTracer(cell,node,level,dim) = Velocity(cell,node,level,dim)*Tracer(cell,node,level);
          //UTracer(cell,node,level,dim) = PiVelx(cell,node,level,dim)*Tracer(cell,node,level);
            //std::cout << "pivelx: " << cell << " " << node << " " << level << " " << dim << " " << PiVelx(cell,node,level,dim) << std::endl;
            //std::cout << "Tracer " << Tracer(cell,node,level) << std::endl;
          }

#else
  Kokkos::parallel_for(XZHydrostatic_UTracer_Policy(0,workset.numCells),*this);

#endif
}
}
