//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Teuchos_RCP.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp" 

#include "Intrepid_FunctionSpaceTools.hpp"
#include "Albany_Layouts.hpp"
#include "Aeras_ShallowWaterConstants.hpp"

namespace Aeras {

//**********************************************************************
template<typename EvalT, typename Traits>
ShallowWaterHyperViscosity<EvalT, Traits>::
ShallowWaterHyperViscosity(const Teuchos::ParameterList& p,
            const Teuchos::RCP<Albany::Layouts>& dl) :
  wGradBF  (p.get<std::string> ("Weighted Gradient BF Name"),dl->node_qp_gradient),
  Ugrad    (p.get<std::string> ("Gradient QP Variable Name"), dl->qp_vecgradient),
  hyperViscosity (p.get<std::string> ("Hyperviscosity Name"), dl->node_vector)
{
  Teuchos::ParameterList* shallowWaterList =
   p.get<Teuchos::ParameterList*>("Parameter List");

  useHyperViscosity = shallowWaterList->get<bool>("Use Hyperviscosity", false); //Default: false
  std::string hvTypeString = shallowWaterList->get<std::string>("Hyperviscosity Type", "Constant");
  hvTau = shallowWaterList->get<double>("Hyperviscosity Tau", 1.0);

  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());

  if (hvTypeString == "Constant")
    hvType = CONSTANT; 
  else {
  TEUCHOS_TEST_FOR_EXCEPTION( true, std::logic_error,
                              "Unknown shallow water hyperviscosity type: " << hvTypeString
                               << std::endl);
  }

  this->addEvaluatedField(hyperViscosity);
  this->addEvaluatedField(wGradBF);
  this->addEvaluatedField(Ugrad);

  std::vector<PHX::DataLayout::size_type> dims;
  
  dl->qp_gradient->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];
  dl->qp_vector->dimensions(dims);
  vecDim  = dims[2]; //# of dofs/node
  wGradBF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];

 #define ALBANY_VERBOSE
#ifdef ALBANY_VERBOSE
  std::cout << "In hyperviscosity constructor!" << std::endl;
  std::cout << "useHyperviscosity? " << useHyperViscosity <<std::endl; 
  std::cout << "hvTau: " << hvTau << std::endl;
  std::cout << "hvTypeString: " << hvTypeString << std::endl;  
  std::cout << "vecDim: " << vecDim << std::endl;  
  std::cout << "numNodes: " << numNodes << std::endl;  
#endif

  this->setName("ShallowWaterHyperViscosity"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ShallowWaterHyperViscosity<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(hyperViscosity,fm);
  this->utils.setFieldData(wGradBF,fm);
  this->utils.setFieldData(Ugrad,fm);
}

//**********************************************************************
//A concrete (non-virtual) implementation of getValue is needed for code to compile. 
//Do we really need it though for this problem...?
template<typename EvalT,typename Traits>
typename ShallowWaterHyperViscosity<EvalT,Traits>::ScalarT& 
ShallowWaterHyperViscosity<EvalT,Traits>::getValue(const std::string &n)
{
  static ScalarT junk(0);
  return junk;
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ShallowWaterHyperViscosity<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  if (useHyperViscosity == false) { //no hyperviscosity
    for (std::size_t cell=0; cell < workset.numCells; ++cell) 
      for (int node=0; node < numNodes; ++node) 
        for (int i=0; i<vecDim; ++i) 
           hyperViscosity(cell, node, i) = 0.0;       
  }
  else { //hyperviscosity
    if (hvType == CONSTANT) {
      for (std::size_t cell=0; cell < workset.numCells; ++cell) {
        for (int qp=0; qp < numQPs; ++qp) {
          for (int node=0; node < numNodes; ++node) {
            for (int i=0; i<vecDim; ++i) {
              //FIXME: Oksana will implement hyperviscosity here.
              hyperViscosity(cell, node, i) = hvTau*0.0;      
              //e.g., if your hyperviscosity is hvTau*d(solution vector)/dx*dw/dx + hvTau*d(solution vector)/dy*dw/dy, you'd put
              //hyperViscosity(cell, node, i) = hvTau*Ugrad(cell,qp,i,0)*wGradBF(cell,node,qp,0) 
              //                              + hvTau*Ugrad(cell,qp,i,1)*wGradBF(cell,node,qp,1); 
            }
          }
        }
      }
    }
  }
#else
  //FIXME, IKT: the following needs to be converted to the above (Irina D.).
  if (hvType == CONSTANT) {
      hyperViscosity.deep_copy(0.0);
  }
#endif

}
}
