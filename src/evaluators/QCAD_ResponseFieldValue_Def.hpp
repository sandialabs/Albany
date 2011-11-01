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
QCAD::ResponseFieldValue<EvalT, Traits>::
ResponseFieldValue(Teuchos::ParameterList& p) :
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
  operation    = plist->get<std::string>("Operation");

  bOpFieldIsVector = false;
  if(plist->isParameter("Operation Vector Field Name")) {
    opFieldName  = plist->get<std::string>("Operation Vector Field Name");
    bOpFieldIsVector = true;
  }
  else opFieldName  = plist->get<std::string>("Operation Field Name");
 
  bRetFieldIsVector = false;
  if(plist->isParameter("Return Vector Field Name")) {
    retFieldName  = plist->get<std::string>("Return Vector Field Name");
    bRetFieldIsVector = true;
  }
  else retFieldName = plist->get<std::string>("Return Field Name", opFieldName);
  bReturnOpField = (opFieldName == retFieldName);

  opDomain     = plist->get<std::string>("Operation Domain", "box");
  opX = plist->get<bool>("Operate on x-component", true) && (numDims > 0);
  opY = plist->get<bool>("Operate on y-component", true) && (numDims > 1);
  opZ = plist->get<bool>("Operate on z-component", true) && (numDims > 2);

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


  //! setup operation field and return field (if it's a different field)
  if(bOpFieldIsVector) {
    PHX::MDField<ScalarT> f(opFieldName, vector_dl); opField = f; }
  else {
    PHX::MDField<ScalarT> f(opFieldName, scalar_dl); opField = f; }

  if(!bReturnOpField) {
    if(bRetFieldIsVector) {
      PHX::MDField<ScalarT> f(retFieldName, vector_dl); retField = f; }
    else {
      PHX::MDField<ScalarT> f(retFieldName, scalar_dl); retField = f; }
  }

  //! add dependent fields
  this->addDependentField(opField);
  this->addDependentField(coordVec);
  this->addDependentField(weights);
  if(!bReturnOpField) this->addDependentField(retField); //when return field is *different* from op field
  
  //! set initial values: ( <returnValue>, <opFieldValue>, <x>, <y>, <z> ) for any dimension
  std::vector<double> initVals(5, 0.0); 

  // Set sentinal values for max/min problems 
  if( operation == "Maximize" ) initVals[1] = -1e200;
  else if( operation == "Minimize" ) initVals[1] = 1e100;
  else TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, std::endl 
             << "Error!  Invalid operation type " << operation << std::endl); 
  
  PHAL::ResponseBase<EvalT, Traits>::setInitialValues(initVals);

  //! set post processing parameters (used to reconcile values across multiple processors)
  Teuchos::ParameterList ppParams;
  if( operation == "Maximize" ) ppParams.set("Processing Type","Max");
  else if( operation == "Minimize" ) ppParams.set("Processing Type","Min");
  ppParams.set("Index",1);

  PHAL::ResponseBase<EvalT, Traits>::setPostProcessingParams(ppParams);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::ResponseFieldValue<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(opField,fm);
  this->utils.setFieldData(coordVec,fm);
  this->utils.setFieldData(weights,fm);
  if(!bReturnOpField) this->utils.setFieldData(retField,fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::ResponseFieldValue<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  PHAL::ResponseBase<EvalT, Traits>::beginEvaluateFields(workset);

  std::vector<ScalarT>& local_g = PHAL::ResponseBase<EvalT, Traits>::local_g;
  ScalarT opVal, qpVal, cellVol;

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

    // Get the cell volume, used for averaging over a cell
    cellVol = 0.0;
    for (std::size_t qp=0; qp < numQPs; ++qp)
      cellVol += weights(cell,qp);

    // Get the scalar value of the field being operated on which will be used
    //  in the operation (all operations just deal with scalar data so far)
    opVal = 0.0;
    for (std::size_t qp=0; qp < numQPs; ++qp) {
      qpVal = 0.0;
      if(bOpFieldIsVector) {
        if(opX) qpVal += opField(cell,qp,0) * opField(cell,qp,0);
        if(opY) qpVal += opField(cell,qp,1) * opField(cell,qp,1);
        if(opZ) qpVal += opField(cell,qp,2) * opField(cell,qp,2);
      }
      else qpVal = opField(cell,qp);
      opVal += qpVal * weights(cell,qp);
    }
    opVal /= cellVol;  
    // opVal = the average value of the field operated on over the current cell

      
    // Check if the currently stored min/max value needs to be updated
    if( (operation == "Maximize" && opVal > local_g[1]) ||
        (operation == "Minimize" && opVal < local_g[1]) ) {

      // set g[0] = value of return field at the current cell (avg)
      local_g[0]=0.0;
      if(bReturnOpField) {
        for (std::size_t qp=0; qp < numQPs; ++qp) {
          qpVal = 0.0;
          if(bOpFieldIsVector) {
            for(std::size_t i=0; i<numDims; i++) {
              qpVal += opField(cell,qp,i)*opField(cell,qp,i);
            }
          }
          else qpVal = opField(cell,qp);
          local_g[0] += qpVal * weights(cell,qp);
        }
      }
      else {
        for (std::size_t qp=0; qp < numQPs; ++qp) {
          qpVal = 0.0;
          if(bRetFieldIsVector) {
            for(std::size_t i=0; i<numDims; i++) {
              qpVal += retField(cell,qp,i)*retField(cell,qp,i);
            }
          }
          else qpVal = retField(cell,qp);
          local_g[0] += qpVal * weights(cell,qp);
        }
      }
      local_g[0] /= cellVol;

      // set g[1] = value of the field operated on at the current cell (avg)
      local_g[1] = opVal;

      // set g[2+] = average qp coordinate values of the current cell
      for(std::size_t i=0; i<numDims; i++) {
        local_g[i+2] = 0.0;
        for (std::size_t qp=0; qp < numQPs; ++qp) local_g[i+2] += coordVec(cell,qp,i);
          local_g[i+2] /= numQPs;
      }
    }

  } // end of loop over cells

  PHAL::ResponseBase<EvalT, Traits>::endEvaluateFields(workset);
}

// **********************************************************************
template<typename EvalT,typename Traits>
Teuchos::RCP<const Teuchos::ParameterList>
QCAD::ResponseFieldValue<EvalT,Traits>::getValidResponseParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
     	rcp(new Teuchos::ParameterList("Valid ResponseFieldValue Params"));;

  validPL->set<string>("Type", "", "Response type");
  validPL->set<string>("Operation", "Maximize", "Operation to perform");
  validPL->set<string>("Operation Field Name", "", "Scalar field to perform operation on");
  validPL->set<string>("Operation Vector Field Name", "", "Vector field to perform operation on");
  validPL->set<string>("Return Field Name", "<operation field name>",
		       "Scalar field to return value from");
  validPL->set<string>("Return Vector Field Name", "<operation vector field name>",
		       "Vector field to return value from");

  validPL->set<string>("Operation Domain", "box", "Region to perform operation: 'box' or 'element block'");
  validPL->set<bool>("Operate on x-component", true, 
		     "Whether to perform operation on x component of vector field");
  validPL->set<bool>("Operate on y-component", true, 
		     "Whether to perform operation on y component of vector field");
  validPL->set<bool>("Operate on z-component", true, 
		     "Whether to perform operation on z component of vector field");

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

