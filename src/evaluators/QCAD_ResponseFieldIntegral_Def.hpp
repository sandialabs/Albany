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
#include "Teuchos_CommHelpers.hpp"

//Utility function to split a std::string by a delimiter, so far only used here
void split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while(std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
}


template<typename EvalT, typename Traits>
QCAD::ResponseFieldIntegral<EvalT, Traits>::
ResponseFieldIntegral(Teuchos::ParameterList& p,
		      const Teuchos::RCP<Albany::Layouts>& dl) :
  coordVec("Coord Vec", dl->qp_vector),
  weights("Weights", dl->qp_scalar)
{
  //! get and validate Response parameter list
  Teuchos::ParameterList* plist = 
    p.get<Teuchos::ParameterList*>("Parameter List");
  Teuchos::RCP<const Teuchos::ParameterList> reflist = 
    this->getValidResponseParameters();
  plist->validateParameters(*reflist,0);

  // passed down from main list
  length_unit_in_m = p.get<double>("Length unit in m");

  //! number of quad points per cell
  Teuchos::RCP<PHX::DataLayout> scalar_dl = dl->qp_scalar;
  numQPs = scalar_dl->dimension(1);
  
  //! obtain number of dimensions
  Teuchos::RCP<PHX::DataLayout> vector_dl = dl->qp_vector;
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numDims = dims[2];

  //! User-specified parameters
  std::string ebNameStr = plist->get<std::string>("Element Block Name","");
  if(ebNameStr.length() > 0) split(ebNameStr,',',ebNames);
  fieldName = plist->get<std::string>("Field Name","");
  bPositiveOnly = plist->get<bool>("Positive Return Only",false);

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

  //! add dependent fields
  if( fieldName.length() > 0 ) {
    PHX::MDField<ScalarT,Cell,QuadPoint> f(fieldName, scalar_dl); 
    field = f;
    this->addDependentField(field);
  }
  this->addDependentField(coordVec);
  this->addDependentField(weights);
  this->setName(fieldName+" Response Field Integral"+PHX::TypeString<EvalT>::value);
  
  // Setup scatter evaluator
  p.set("Stand-alone Evaluator", false);
  std::string local_response_name = 
    fieldName + " Local Response Field Integral";
  std::string global_response_name = 
    fieldName + " Global Response Field Integral";
  PHX::Tag<ScalarT> local_response_tag(local_response_name, 
				       dl->cell_scalar);
  PHX::Tag<ScalarT> global_response_tag(global_response_name, 
					dl->workset_scalar);
  p.set("Local Response Field Tag", local_response_tag);
  p.set("Global Response Field Tag", global_response_tag);
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::setup(p,dl);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::ResponseFieldIntegral<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  if( fieldName.length() > 0 )
    this->utils.setFieldData(field,fm);
  this->utils.setFieldData(coordVec,fm);
  this->utils.setFieldData(weights,fm);
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::postRegistrationSetup(d,fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::ResponseFieldIntegral<EvalT, Traits>::
preEvaluate(typename Traits::PreEvalData workset)
{
  for (typename PHX::MDField<ScalarT>::size_type i=0; 
       i<this->global_response.size(); i++)
    this->global_response[i] = 0.0;

  // Do global initialization
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::preEvaluate(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::ResponseFieldIntegral<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Zero out local response
  for (typename PHX::MDField<ScalarT>::size_type i=0; 
       i<this->local_response.size(); i++)
    this->local_response[i] = 0.0;

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
    TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, std::endl 
			  << "Error! Invalid number of dimensions: " << numDims << std::endl);
    
  if( ebNames.size() == 0 || 
      std::find(ebNames.begin(), ebNames.end(), workset.EBName) != ebNames.end() ) {

    ScalarT val;
    bool bFieldIsValid = (fieldName.length() > 0);
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {

      bool cellInBox = false;
      for (std::size_t qp=0; qp < numQPs; ++qp) {
        if( (!limitX || (coordVec(cell,qp,0) >= xmin && coordVec(cell,qp,0) <= xmax)) &&
            (!limitY || (coordVec(cell,qp,1) >= ymin && coordVec(cell,qp,1) <= ymax)) &&
            (!limitZ || (coordVec(cell,qp,2) >= zmin && coordVec(cell,qp,2) <= zmax)) ) {
          cellInBox = true; break; }
      }
      if( !cellInBox ) continue;

      for (std::size_t qp=0; qp < numQPs; ++qp) {
	
	if( bFieldIsValid )
	  val = field(cell,qp) * weights(cell,qp) * scaling;
	else
	  val = weights(cell,qp) * scaling; //integrate volume

        this->local_response(cell) += val;
	this->global_response(0) += val;
      }
    }
  }

  // Do any local-scattering necessary
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::evaluateFields(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::ResponseFieldIntegral<EvalT, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
  // Add contributions across processors
  Teuchos::RCP< Teuchos::ValueTypeSerializer<int,ScalarT> > serializer =
    workset.serializerManager.template getValue<EvalT>();
  Teuchos::reduceAll(
    *workset.comm, *serializer, Teuchos::REDUCE_SUM,
    this->global_response.size(), &this->global_response[0], 
    &this->global_response[0]);

  if (bPositiveOnly && this->global_response[0] < 1e-6) {
    this->global_response[0] = 1e+100;
  }
  
  // Do global scattering
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::postEvaluate(workset);
}

// **********************************************************************
template<typename EvalT,typename Traits>
Teuchos::RCP<const Teuchos::ParameterList>
QCAD::ResponseFieldIntegral<EvalT,Traits>::getValidResponseParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
     	rcp(new Teuchos::ParameterList("Valid ResponseFieldIntegral Params"));;
  Teuchos::RCP<const Teuchos::ParameterList> baseValidPL =
    PHAL::SeparableScatterScalarResponse<EvalT,Traits>::getValidResponseParameters();
  validPL->setParameters(*baseValidPL);

  validPL->set<string>("Name", "", "Name of response function");
  validPL->set<int>("Phalanx Graph Visualization Detail", 0, "Make dot file to visualize phalanx graph");
  validPL->set<string>("Type", "", "Response type");
  validPL->set<string>("Element Block Name", "", 
  		"Name of the element block to use as the integration domain");
  validPL->set<string>("Field Name", "", "Field to integrate");
  validPL->set<bool>("Positive Return Only",false);

  validPL->set<double>("x min", 0.0, "Integration domain minimum x coordinate");
  validPL->set<double>("x max", 0.0, "Integration domain maximum x coordinate");
  validPL->set<double>("y min", 0.0, "Integration domain minimum y coordinate");
  validPL->set<double>("y max", 0.0, "Integration domain maximum y coordinate");
  validPL->set<double>("z min", 0.0, "Integration domain minimum z coordinate");
  validPL->set<double>("z max", 0.0, "Integration domain maximum z coordinate");

  return validPL;
}

// **********************************************************************

