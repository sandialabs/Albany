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
QCAD::ResponseSaveField<EvalT, Traits>::
ResponseSaveField(Teuchos::ParameterList& p) :
  PHAL::ResponseBase<EvalT, Traits>(p)
{
  // States Not Saved for Generic Type, only Specializations
}

// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::ResponseSaveField<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  // States Not Saved for Generic Type, only Specializations
}

// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::ResponseSaveField<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // States Not Saved for Generic Type, only Specializations
}


// **********************************************************************
// RESIDUAL specialization
// **********************************************************************

template<typename Traits>
QCAD::ResponseSaveField<PHAL::AlbanyTraits::Residual, Traits>::
ResponseSaveField(Teuchos::ParameterList& p) :
  PHAL::ResponseBase<PHAL::AlbanyTraits::Residual, Traits>(p)
{
  //! get and validate Response parameter list
  Teuchos::ParameterList* plist = 
    p.get<Teuchos::ParameterList*>("Parameter List");

  Teuchos::RCP<const Teuchos::ParameterList> reflist = 
    this->getValidResponseParameters();
  plist->validateParameters(*reflist,0);

  //! User-specified parameters
  fieldName = plist->get<std::string>("Field Name");
  stateName = plist->get<std::string>("State Name", fieldName);
  outputToExodus = plist->get<bool>("Output to Exodus", true);

  //! number of quad points per cell
  Teuchos::RCP<PHX::DataLayout> scalar_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
  numQPs = scalar_dl->dimension(1);

  //! add dependent fields
  PHX::MDField<ScalarT,Cell,QuadPoint> f(fieldName, scalar_dl); field = f;
  this->addDependentField(field);

  //! set initial values
  std::vector<double> initVals(1); initVals[0] = 0.0; //Response is a dummy 0.0
  PHAL::ResponseBase<PHAL::AlbanyTraits::Residual, Traits>::
    setInitialValues(initVals);
}

// **********************************************************************
template<typename Traits>
void QCAD::ResponseSaveField<PHAL::AlbanyTraits::Residual, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(field,fm);
}

// **********************************************************************
template<typename Traits>
void QCAD::ResponseSaveField<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  PHAL::ResponseBase<PHAL::AlbanyTraits::Residual, Traits>::
    beginEvaluateFields(workset);

  // Get shards Array (from STK) for this state
  // Need to check if we can just copy full size -- can assume same ordering?
  Albany::MDArray sta = (*workset.stateArrayPtr)[stateName];
  std::vector<int> dims;
  sta.dimensions(dims);
  int size = dims.size();

  switch (size) {
  case 2:     
    for (int cell = 0; cell < dims[0]; ++cell)
      for (int qp = 0; qp < dims[1]; ++qp)
	sta(cell, qp) = field(cell,qp);
    break;
  case 3:     
    for (int cell = 0; cell < dims[0]; ++cell)
      for (int qp = 0; qp < dims[1]; ++qp)
	for (int i = 0; i < dims[2]; ++i)
	  sta(cell, qp, i) = field(cell,qp,i);
    break;
  case 4:     
    for (int cell = 0; cell < dims[0]; ++cell)
      for (int qp = 0; qp < dims[1]; ++qp)
	for (int i = 0; i < dims[2]; ++i)
	  for (int j = 0; j < dims[3]; ++j)
	    sta(cell, qp, i, j) = field(cell,qp,i,j);
    break;
  default:
    TEST_FOR_EXCEPTION(size<2||size>4, std::logic_error,
	"Unexpected Array dimensions in SaveField response Evaluator: " << size);
  }

  PHAL::ResponseBase<PHAL::AlbanyTraits::Residual, Traits>::
    endEvaluateFields(workset);
}

// **********************************************************************
template<typename Traits>
Teuchos::RCP<const Teuchos::ParameterList>
QCAD::ResponseSaveField<PHAL::AlbanyTraits::Residual,Traits>::getValidResponseParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
     	rcp(new Teuchos::ParameterList("Valid ResponseSaveField Params"));;

  validPL->set<string>("Type", "", "Response type");
  validPL->set<string>("Field Name", "", "Field to save");
  validPL->set<string>("State Name", "<Field Name>", "State name to save field as");
  validPL->set<bool>("Output to Exodus", true, "Whether state should be output in STK dump to exodus");

  return validPL;
}

// **********************************************************************

