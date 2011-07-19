/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


#include <fstream>
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"

template<typename EvalT, typename Traits>
QCAD::ResponseFieldIntegral<EvalT, Traits>::
ResponseFieldIntegral(Teuchos::ParameterList& p) :
  PHAL::ResponseBase<EvalT, Traits>(p),
  weights(p.get<std::string>("Weights Name"),
	p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout"))
{
  //! get and validate Response parameter list
  Teuchos::ParameterList* plist = 
    p.get<Teuchos::ParameterList*>("Parameter List");

  Teuchos::RCP<const Teuchos::ParameterList> reflist = 
    this->getValidResponseParameters();
  plist->validateParameters(*reflist,0);

  //! User-specified parameters
  ebName = plist->get<std::string>("Element Block Name");
  fieldName = plist->get<std::string>("Field Name");

  // passed down from main list
  length_unit_in_m = p.get<double>("Length unit in m");

  //! number of quad points per cell
  Teuchos::RCP<PHX::DataLayout> scalar_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
  numQPs = scalar_dl->dimension(1);
  
  //! obtain number of dimensions
  Teuchos::RCP<PHX::DataLayout> vector_dl =
      p.get< Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numDims = dims[2];

  //! add dependent fields
  PHX::MDField<ScalarT,Cell,QuadPoint> f(fieldName, scalar_dl); 
  field = f;
  this->addDependentField(field);
  this->addDependentField(weights);
  
  //! set initial values
  std::vector<double> initVals(1); initVals[0] = 0.0;
  PHAL::ResponseBase<EvalT, Traits>::setInitialValues(initVals);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::ResponseFieldIntegral<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(field,fm);
  this->utils.setFieldData(weights,fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::ResponseFieldIntegral<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  PHAL::ResponseBase<EvalT, Traits>::beginEvaluateFields(workset);

  // Scaling factors
  double X0 = length_unit_in_m/1e-2; // length scaling to get to [cm]
  double scaling = 0.0; 
  
  if (numDims == 1)
    scaling = X0; 
  else if (numDims == 2)
    scaling = X0*X0; 
  else if (numDims == 3)
    scaling = X0*X0*X0; 
  else      
    TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, std::endl 
			  << "Error! Invalid number of dimensions: " << numDims << std::endl);
    
  ScalarT val;
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {
      val = field(cell,qp) * weights(cell,qp) * scaling;
      PHAL::ResponseBase<EvalT, Traits>::local_g[0] += val;
    }
  }

  PHAL::ResponseBase<EvalT, Traits>::endEvaluateFields(workset);
}

// **********************************************************************
template<typename EvalT,typename Traits>
Teuchos::RCP<const Teuchos::ParameterList>
QCAD::ResponseFieldIntegral<EvalT,Traits>::getValidResponseParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
     	rcp(new Teuchos::ParameterList("Valid ResponseFieldIntegral Params"));;

  validPL->set<string>("Type", "", "Response type");
  validPL->set<string>("Element Block Name", "", 
  		"Name of the element block to use as the integration domain");
  validPL->set<string>("Field Name", "", "Field to integrate");
  return validPL;
}

// **********************************************************************

