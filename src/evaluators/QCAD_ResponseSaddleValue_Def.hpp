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
QCAD::ResponseSaddleValue<EvalT, Traits>::
ResponseSaddleValue(Teuchos::ParameterList& p) :
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

  //! get pointer to response function object
  svResponseFn = p.get<Teuchos::RCP<QCAD::SaddleValueResponseFunction> >
    ("Response Function");

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
  retFieldName = plist->get<std::string>("Return Field Name", fieldName);
  bReturnSameField = (fieldName == retFieldName);

  domain = plist->get<std::string>("Domain", "box");

  if(domain == "box") {
    limitX = limitY = limitZ = false;

    if( plist->isParameter("x min") && plist->isParameter("x max") ) {
      limitX = true; TEST_FOR_EXCEPT(numDims <= 0);
      xmin = plist->get<double>("x min");
      xmax = plist->get<double>("x max");
    }
    if( plist->isParameter("y min") && plist->isParameter("y max") ) {
      limitY = true; TEST_FOR_EXCEPT(numDims <= 1);
      ymin = plist->get<double>("y min");
      ymax = plist->get<double>("y max");
    }
    if( plist->isParameter("z min") && plist->isParameter("z max") ) {
      limitZ = true; TEST_FOR_EXCEPT(numDims <= 2);
      zmin = plist->get<double>("z min");
      zmax = plist->get<double>("z max");
    }
  }
  else if(domain == "element block") {
    ebName = plist->get<string>("Element Block Name");
  }
  else TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, std::endl 
             << "Error!  Invalid domain type " << domain << std::endl); 


  //! setup operation field and return field (if it's a different field)
  PHX::MDField<ScalarT> f(fieldName, scalar_dl); field = f;

  if(!bReturnSameField) {
    PHX::MDField<ScalarT> fr(retFieldName, scalar_dl); retField = fr; }

  //! add dependent fields
  this->addDependentField(field);
  this->addDependentField(coordVec);
  this->addDependentField(weights);
  if(!bReturnSameField) this->addDependentField(retField);

  
  //! response evaluator must evaluate dummy operation
  std::string responseID = p.get<string>("Response ID");
  Teuchos::RCP<PHX::DataLayout> dummy_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("Dummy Data Layout");
  
  response_operation = Teuchos::rcp(new PHX::Tag<ScalarT>(responseID, dummy_dl));
  this->addEvaluatedField(*response_operation);

  this->setName(responseID + PHX::TypeString<EvalT>::value);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::ResponseSaddleValue<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(field,fm);
  this->utils.setFieldData(coordVec,fm);
  this->utils.setFieldData(weights,fm);
  if(!bReturnSameField) this->utils.setFieldData(retField,fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::ResponseSaddleValue<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  ScalarT fieldVal, retFieldVal, cellVol;
  std::vector<ScalarT> avgCoord(3, 0.0);

  if(domain == "element block" && workset.EBName != ebName) return;

  for (std::size_t cell=0; cell < workset.numCells; ++cell) 
  {
    // If operation domain is a "box", check whether the current cell is 
    //  at least partially contained within the box
    if(domain == "box") {
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

    // Get the scalar value of the field being operated on (cell average)
    //  and the average coordinates of the cell (average qp coords)
    fieldVal = 0.0;
    for (std::size_t k=0; k < numDims; ++k) avgCoord[k] = 0.0;

    for (std::size_t qp=0; qp < numQPs; ++qp) {
      fieldVal += field(cell,qp) * weights(cell,qp);
      
      for (std::size_t k=0; k < numDims; ++k) 
	avgCoord[k] += coordVec(cell,qp,k);
    }
    fieldVal /= cellVol;  
    for (std::size_t k=0; k < numDims; ++k) avgCoord[k] /= numDims;


    // Get the cell average of the field to return
    if(bReturnSameField) {
      retFieldVal = fieldVal;
    }
    else {
      retFieldVal = 0.0;
      for (std::size_t qp=0; qp < numQPs; ++qp)
	retFieldVal += retField(cell,qp) * weights(cell,qp);
      retFieldVal /= cellVol;  
    }

    // Add data to the response function object
    double fv, rfv, cv, coords[3]; 
    for (std::size_t k=0; k < numDims; ++k) 
      coords[k] = QCAD::EvaluatorTools<EvalT,Traits>::getDoubleValue(avgCoord[k]);

    fv  = QCAD::EvaluatorTools<EvalT,Traits>::getDoubleValue(fieldVal);
    rfv = QCAD::EvaluatorTools<EvalT,Traits>::getDoubleValue(retFieldVal);
    cv  = QCAD::EvaluatorTools<EvalT,Traits>::getDoubleValue(cellVol);

    svResponseFn->addFieldData(fv, rfv, coords, cv);
  } // end of loop over cells
}

// **********************************************************************
template<typename EvalT,typename Traits>
Teuchos::RCP<const Teuchos::ParameterList>
QCAD::ResponseSaddleValue<EvalT,Traits>::getValidResponseParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
     	rcp(new Teuchos::ParameterList("Valid ResponseSaddleValue Params"));;

  validPL->set<string>("Type", "", "Response type");
  validPL->set<string>("Field Name", "", "Field to find saddle point for");
  validPL->set<string>("Return Field Name", "<operation field name>",
		       "Scalar field to return value from");
  validPL->set<string>("Domain", "box", "Region to perform operation: 'box' or 'element block'");
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
