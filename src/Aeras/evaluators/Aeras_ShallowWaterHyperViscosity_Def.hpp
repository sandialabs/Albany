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
  hyperviscosity (p.get<std::string> ("Hyperviscosity Name"), dl->qp_vector)
{
  Teuchos::ParameterList* shallowWaterList =
   p.get<Teuchos::ParameterList*>("Parameter List");

  useHyperviscosity =  ( shallowWaterList->get<bool>("Use Explicit Hyperviscosity", false) ||
		  shallowWaterList->get<bool>("Use Implicit Hyperviscosity", false) );

  //Default: false
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

  this->addEvaluatedField(hyperviscosity);

  std::vector<PHX::DataLayout::size_type> dims;
  
  dl->qp_gradient->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];
  dl->qp_vector->dimensions(dims);
  vecDim  = dims[2]; //# of dofs/node

 //#define ALBANY_VERBOSE
#ifdef ALBANY_VERBOSE
  std::cout << "In hyperviscosity constructor!" << std::endl;
  std::cout << "useHyperviscosity? " << useHyperviscosity <<std::endl;
  std::cout << "hvTau: " << hvTau << std::endl;
  std::cout << "hvTypeString: " << hvTypeString << std::endl;  
  std::cout << "vecDim: " << vecDim << std::endl;  
#endif

  this->setName("ShallowWaterHyperViscosity"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ShallowWaterHyperViscosity<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(hyperviscosity,fm);
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
  // WARNING: Don't use this if hyperviscosity is non-constant.
  if (memoizer_.haveStoredData(workset)) return;

#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  if (useHyperviscosity == false) { //no hyperviscosity
    for(std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for(std::size_t qp = 0; qp < numQPs; ++qp) {
        for (int i=0; i<vecDim; ++i) 
           hyperviscosity(cell, qp, i) = 0.0;
      }
    }       
  }
  else { //hyperviscosity
    if (hvType == CONSTANT) {
    for(std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for(std::size_t qp = 0; qp < numQPs; ++qp) {
        for (int i=0; i<vecDim; ++i) 
           hyperviscosity(cell, qp, i) = hvTau;
        }
      }
    }
  }
#else
  if (useHyperviscosity == false) //no hyperviscosity 
      hyperviscosity.deep_copy(0.0);
  else {//hyperviscosity 
    if (hvType == CONSTANT) 
      hyperviscosity.deep_copy(hvTau);
  }
#endif

}
}
