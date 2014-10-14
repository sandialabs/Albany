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
XZHydrostatic_UTracer<EvalT, Traits>::
XZHydrostatic_UTracer(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Aeras::Layouts>& dl) :
  PiVelx   (p.get<std::string> ("PiVelx"),  dl->node_vector_level),
  Tracer   (p.get<std::string> ("Tracer"),  dl->node_scalar_level),
  UTracer  (p.get<std::string> ("UTracer"), dl->node_vector_level),

  numDims    (dl->node_qp_gradient        ->dimension(3)),
  numNodes ( dl->node_scalar             ->dimension(1)),
  numLevels( dl->node_scalar_level       ->dimension(2))
{
  this->addDependentField(PiVelx);
  this->addDependentField(Tracer);
  this->addEvaluatedField(UTracer);

  this->setName("Aeras::XZHydrostatic_UTracer"+PHX::TypeString<EvalT>::value);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_UTracer<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(PiVelx, fm);
  this->utils.setFieldData(Tracer, fm);
  this->utils.setFieldData(UTracer,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_UTracer<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  for (int cell=0; cell < workset.numCells; ++cell) 
    for (int node=0; node < numNodes; ++node) 
      for (int level=0; level < numLevels; ++level) 
        for (int dim=0; dim < numDims; ++dim) 
          UTracer(cell,node,level,dim) = PiVelx(cell,node,level,dim)*Tracer(cell,node,level);
}
}
