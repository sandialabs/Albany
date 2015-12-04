//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"
#include "Albany_Layouts.hpp"

//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN

namespace FELIX {

const double pi = 3.1415926535897932385;

//**********************************************************************
template<typename EvalT, typename Traits>
CismSurfaceGradFO<EvalT, Traits>::
CismSurfaceGradFO(const Teuchos::ParameterList& p,
            const Teuchos::RCP<Albany::Layouts>& dl) :
  dsdx_node (p.get<std::string> ("CISM Surface Height Gradient X Variable Name"), dl->node_scalar),
  dsdy_node (p.get<std::string> ("CISM Surface Height Gradient Y Variable Name"), dl->node_scalar),
  BF        (p.get<std::string> ("BF Variable Name"), dl->node_qp_scalar),
  gradS_qp  (p.get<std::string> ("Surface Height Gradient QP Variable Name"), dl->qp_gradient )
{

  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());

  this->addDependentField(dsdx_node);
  this->addDependentField(dsdy_node);
  this->addDependentField(BF);
  this->addEvaluatedField(gradS_qp);

  this->setName("CismSurfaceGradFO"+PHX::typeAsString<EvalT>());

  std::vector<PHX::DataLayout::size_type> dims;
  BF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];
}

//**********************************************************************
template<typename EvalT, typename Traits>
void CismSurfaceGradFO<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(dsdx_node,fm);
  this->utils.setFieldData(dsdy_node,fm);
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(gradS_qp,fm);
}

template<typename EvalT,typename Traits>
typename CismSurfaceGradFO<EvalT,Traits>::ScalarT&
CismSurfaceGradFO<EvalT,Traits>::getValue(const std::string &n)
{
  return dummyParam;
}

//**********************************************************************
template<typename EvalT, typename Traits>
void CismSurfaceGradFO<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {
      gradS_qp(cell,qp,0) = dsdx_node(cell, 0) * BF(cell, 0, qp);
      gradS_qp(cell,qp,1) = dsdy_node(cell, 0) * BF(cell, 0, qp);
      for (std::size_t node=1; node < numNodes; ++node) {
        gradS_qp(cell,qp,0) += dsdx_node(cell, node) * BF(cell, node, qp);
        gradS_qp(cell,qp,1) += dsdy_node(cell, node) * BF(cell, node, qp);
      }
    }
  }
}

} // Namespace FELIX
