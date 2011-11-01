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
QCAD::ResponseCenterOfMass<EvalT, Traits>::
ResponseCenterOfMass(Teuchos::ParameterList& p) :
  PHAL::ResponseBase<EvalT, Traits>(p),
  coordVec(p.get<string>("Coordinate Vector Name"),
	   p.get< Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout")),
  weights(p.get<std::string>("Weights Name"),
	p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout"))
{
  //! get and validate Response parameter list
  Teuchos::ParameterList* plist = 
    p.get<Teuchos::ParameterList*>("Parameter List");

  Teuchos::RCP<const Teuchos::ParameterList> reflist = 
    this->getValidResponseParameters();
  plist->validateParameters(*reflist,0);

  //! number of quad points per cell and dimension of space
  Teuchos::RCP<PHX::DataLayout> scalar_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
  Teuchos::RCP<PHX::DataLayout> vector_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout");
  
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  //! User-specified parameters
  fieldName  = plist->get<std::string>("Field Name");
  opDomain     = plist->get<std::string>("Operation Domain", "box");

  if(opDomain == "box") {
    limitX = limitY = limitZ = false;

    if( plist->isParameter("x min") && plist->isParameter("x max") ) {
      limitX = true; TEUCHOS_TEST_FOR_EXCEPT(numDims <= 0);
      xmin = plist->get<double>("x min");
      xmax = plist->get<double>("x max");
    }
    if( plist->isParameter("y min") && plist->isParameter("y max") ) {
      limitY = true; TEUCHOS_TEST_FOR_EXCEPT(numDims <= 1);
      ymin = plist->get<double>("y min");
      ymax = plist->get<double>("y max");
    }
    if( plist->isParameter("z min") && plist->isParameter("z max") ) {
      limitZ = true; TEUCHOS_TEST_FOR_EXCEPT(numDims <= 2);
      zmin = plist->get<double>("z min");
      zmax = plist->get<double>("z max");
    }
  }
  else if(opDomain == "element block") {
    ebName = plist->get<string>("Element Block Name");
  }
  else TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, std::endl 
             << "Error!  Invalid operation domain type " << opDomain << std::endl); 


  //! setup field
  PHX::MDField<ScalarT> f(fieldName, scalar_dl); field = f;

  //! add dependent fields
  this->addDependentField(field);
  this->addDependentField(coordVec);
  this->addDependentField(weights);

  //! set initial values: ( <com_x>, <com_y>, <com_z>, <field integral normalized (=1) > )
  std::vector<double> initVals(4, 0.0); 
  PHAL::ResponseBase<EvalT, Traits>::setInitialValues(initVals);

  //! set post processing parameters (used to reconcile values across multiple processors)
  Teuchos::ParameterList ppParams;
  ppParams.set("Processing Type","SumThenNormalize");
  ppParams.set("Normalizer Index",3);

  PHAL::ResponseBase<EvalT, Traits>::setPostProcessingParams(ppParams);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::ResponseCenterOfMass<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(field,fm);
  this->utils.setFieldData(coordVec,fm);
  this->utils.setFieldData(weights,fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::ResponseCenterOfMass<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  PHAL::ResponseBase<EvalT, Traits>::beginEvaluateFields(workset);

  std::vector<ScalarT>& local_g = PHAL::ResponseBase<EvalT, Traits>::local_g;
  ScalarT integral, moment;

  if(opDomain == "element block" && workset.EBName != ebName) 
  {
      PHAL::ResponseBase<EvalT, Traits>::endEvaluateFields(workset);
      return;
  }

  for (std::size_t cell=0; cell < workset.numCells; ++cell) 
  {
    // If operation domain is a "box", check whether the current cell is 
    //  at least partially contained within the box
    if(opDomain == "box") {
      bool cellInBox = false;
      for (std::size_t qp=0; qp < numQPs; ++qp) {
        if( (!limitX || (coordVec(cell,qp,0) >= xmin && coordVec(cell,qp,0) <= xmax)) &&
            (!limitY || (coordVec(cell,qp,1) >= ymin && coordVec(cell,qp,1) <= ymax)) &&
            (!limitZ || (coordVec(cell,qp,2) >= zmin && coordVec(cell,qp,2) <= zmax)) ) {
          cellInBox = true; break; }
      }
      if( !cellInBox ) continue;
    }

    // Add to running total volume and mass moment
    for (std::size_t qp=0; qp < numQPs; ++qp) {
      integral = field(cell,qp) * weights(cell,qp);
      local_g[3] += integral;

      for(std::size_t i=0; i<numDims && i<3; i++) {
	moment = field(cell,qp) * weights(cell,qp) * coordVec(cell,qp,i);
	local_g[i] += moment;
      }
    }

  }

  PHAL::ResponseBase<EvalT, Traits>::endEvaluateFields(workset);
}

// **********************************************************************
template<typename EvalT,typename Traits>
Teuchos::RCP<const Teuchos::ParameterList>
QCAD::ResponseCenterOfMass<EvalT,Traits>::getValidResponseParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
     	rcp(new Teuchos::ParameterList("Valid ResponseCenterOfMass Params"));;

  validPL->set<string>("Field Name", "", "Scalar field from which to compute center of mass");
  validPL->set<string>("Operation Domain", "box", "Region to perform operation: 'box' or 'element block'");

  validPL->set<double>("x min", 0.0, "Box domain minimum x coordinate");
  validPL->set<double>("x max", 0.0, "Box domain maximum x coordinate");
  validPL->set<double>("y min", 0.0, "Box domain minimum y coordinate");
  validPL->set<double>("y max", 0.0, "Box domain maximum y coordinate");
  validPL->set<double>("z min", 0.0, "Box domain minimum z coordinate");
  validPL->set<double>("z max", 0.0, "Box domain maximum z coordinate");

  validPL->set<string>("Element Block Name", "", "Element block name that specifies domain");

  return validPL;
}

// **********************************************************************

