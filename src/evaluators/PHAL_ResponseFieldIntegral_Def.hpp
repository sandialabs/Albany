//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Teuchos_TestForException.hpp"
#include "Teuchos_CommHelpers.hpp"
#include "PHAL_Utilities.hpp"

namespace PHAL {

//Utility function to split a std::string by a delimiter, so far only used here
static void split(const std::string &s, char delim, std::vector<std::string> &elems) {
  std::stringstream ss(s);
  std::string item;
  while(std::getline(ss, item, delim)) {
    elems.push_back(item);
  }
}

}

template<typename EvalT, typename Traits>
PHAL::ResponseFieldIntegral<EvalT, Traits>::
ResponseFieldIntegral(Teuchos::ParameterList& p,
		      const Teuchos::RCP<Albany::Layouts>& dl) :
  coordVec("Coord Vec", dl->qp_gradient),
  weights("Weights", dl->qp_scalar)
{
  // get and validate Response parameter list
  Teuchos::ParameterList* plist = 
    p.get<Teuchos::ParameterList*>("Parameter List");
  Teuchos::RCP<const Teuchos::ParameterList> reflist = 
    this->getValidResponseParameters();
  plist->validateParameters(*reflist,0);

  // Get field type and corresponding layouts
  std::string field_name = plist->get<std::string>("Field Name");
  std::string fieldType = plist->get<std::string>("Field Type", "Scalar");
  if (plist->isType< Teuchos::Array<int> >("Field Components"))
    field_components = plist->get< Teuchos::Array<int> >("Field Components");
  Teuchos::RCP<PHX::DataLayout> field_layout;
  Teuchos::RCP<PHX::DataLayout> local_response_layout;
  Teuchos::RCP<PHX::DataLayout> global_response_layout;
  if (fieldType == "Scalar") {
    field_layout = dl->qp_scalar;
    local_response_layout = dl->cell_scalar;
    global_response_layout = dl->workset_scalar;
  }
  else if (fieldType == "Vector") {
    field_layout = dl->qp_vector;
    if (field_components.size() == 0) {
      local_response_layout = dl->cell_vector;
      global_response_layout = dl->workset_vector;
    }
    else {
      int worksetSize = dl->cell_scalar->dimension(0);
      local_response_layout = 
	Teuchos::rcp(new PHX::MDALayout<Cell,Dim>(worksetSize, 
						  field_components.size()));
      global_response_layout = 
	Teuchos::rcp(new PHX::MDALayout<Dim>(field_components.size()));
    }
  }
  else if (fieldType == "Tensor") {
    TEUCHOS_TEST_FOR_EXCEPTION(
      true, std::logic_error,
      "local_ and global_response must have rank 2. However, this code path "
      "makes them rank 3. Needs to be fixed.");
    field_layout = dl->qp_tensor;
    local_response_layout = dl->cell_tensor;
    global_response_layout = dl->workset_tensor;
  }
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(
      true, 
      Teuchos::Exceptions::InvalidParameter,
      "Invalid field type " << fieldType << ".  Support values are " << 
      "Scalar, Vector, and Tensor." << std::endl);
  }
  field = decltype(field)(field_name, field_layout);
  field_layout->dimensions(field_dims);
  field_rank = field_layout->rank();
  if (field_components.size() == 0) {
    int num_components = field_dims[field_rank-1];
    field_components.resize(num_components);
    for (int i=0; i<num_components; i++)
      field_components[i] = i;
  }

  // coordinate dimensions
  std::vector<PHX::DataLayout::size_type> coord_dims;
  dl->qp_vector->dimensions(coord_dims);
  numQPs = coord_dims[1];
  numDims = coord_dims[2];
 
  // User-specified parameters
  std::string ebNameStr = plist->get<std::string>("Element Block Name","");
  if(ebNameStr.length() > 0) split(ebNameStr,',',ebNames);

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

  // length scaling
  double X0 = plist->get<double>("Length Scaling", 1.0);
  if (numDims == 1)
    scaling = X0; 
  else if (numDims == 2)
    scaling = X0*X0; 
  else if (numDims == 3)
    scaling = X0*X0*X0; 
  else      
    TEUCHOS_TEST_FOR_EXCEPTION(
      true, Teuchos::Exceptions::InvalidParameter, 
      std::endl << "Error! Invalid number of dimensions: " << numDims << 
      std::endl);

  // add dependent fields
  this->addDependentField(field);
  this->addDependentField(coordVec);
  this->addDependentField(weights);
  this->setName(field_name+" Response Field Integral"+PHX::typeAsString<EvalT>());

  // Setup scatter evaluator
  p.set("Stand-alone Evaluator", false);
  std::string local_response_name = 
    field_name + " Local Response Field Integral";
  std::string global_response_name = 
    field_name + " Global Response Field Integral";
  PHX::Tag<ScalarT> local_response_tag(local_response_name, 
				       local_response_layout);
  PHX::Tag<ScalarT> global_response_tag(global_response_name, 
					global_response_layout);
  p.set("Local Response Field Tag", local_response_tag);
  p.set("Global Response Field Tag", global_response_tag);
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::setup(p,dl);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void PHAL::ResponseFieldIntegral<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(field,fm);
  this->utils.setFieldData(coordVec,fm);
  this->utils.setFieldData(weights,fm);
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::postRegistrationSetup(d,fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void PHAL::ResponseFieldIntegral<EvalT, Traits>::
preEvaluate(typename Traits::PreEvalData workset)
{
  PHAL::set(this->global_response_eval, 0.0);
  // Do global initialization
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::preEvaluate(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void PHAL::ResponseFieldIntegral<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{   
  // Zero out local response
  PHAL::set(this->local_response_eval, 0.0);

  if( ebNames.size() == 0 || 
      std::find(ebNames.begin(), ebNames.end(), workset.EBName) != ebNames.end() ) {

    ScalarT s;
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
	if (field_rank == 2) {
	  s = field(cell,qp) * weights(cell,qp) * scaling;
	  this->local_response_eval(cell,0) += s;
	  this->global_response_eval(0) += s;
	}
	else if (field_rank == 3) {
	  for (std::size_t dim=0; dim < field_components.size(); ++dim) {
	    s = field(cell,qp,field_components[dim]) * weights(cell,qp) * scaling;
	    this->local_response_eval(cell,dim) += s;
	    this->global_response_eval(dim) += s;
	  }
	}
	else if (field_rank == 4) {
	  for (std::size_t dim1=0; dim1 < field_dims[2]; ++dim1) {
	    for (std::size_t dim2=0; dim2 < field_dims[3]; ++dim2) {
	      s = field(cell,qp,dim1,dim2) * weights(cell,qp) * scaling;
	      this->local_response_eval(cell,dim1,dim2) += s;
	      this->global_response_eval(dim1,dim2) += s;
	    }
	  }
	}
      }
    }
  }

  // Do any local-scattering necessary
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::evaluateFields(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void PHAL::ResponseFieldIntegral<EvalT, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
  PHAL::reduceAll<ScalarT>(*workset.comm, Teuchos::REDUCE_SUM,
                           this->global_response_eval);
  // Do global scattering
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::postEvaluate(workset);
}

// **********************************************************************
template<typename EvalT,typename Traits>
Teuchos::RCP<const Teuchos::ParameterList>
PHAL::ResponseFieldIntegral<EvalT,Traits>::
getValidResponseParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
     	rcp(new Teuchos::ParameterList("Valid ResponseFieldIntegral Params"));
  Teuchos::RCP<const Teuchos::ParameterList> baseValidPL =
    PHAL::SeparableScatterScalarResponse<EvalT,Traits>::getValidResponseParameters();
  validPL->setParameters(*baseValidPL);

  validPL->set<std::string>("Name", "", "Name of response function");
  validPL->set<int>("Phalanx Graph Visualization Detail", 0, "Make dot file to visualize phalanx graph");
  validPL->set<std::string>("Field Type", "", "Type of field (scalar, vector, ...)");
  validPL->set<std::string>(
    "Element Block Name", "", 
    "Name of the element block to use as the integration domain");
  validPL->set<std::string>("Field Name", "", "Field to integrate");
  validPL->set<bool>("Positive Return Only",false);

  validPL->set<double>("Length Scaling", 1.0, "Length Scaling");
  validPL->set<double>("x min", 0.0, "Integration domain minimum x coordinate");
  validPL->set<double>("x max", 0.0, "Integration domain maximum x coordinate");
  validPL->set<double>("y min", 0.0, "Integration domain minimum y coordinate");
  validPL->set<double>("y max", 0.0, "Integration domain maximum y coordinate");
  validPL->set<double>("z min", 0.0, "Integration domain minimum z coordinate");
  validPL->set<double>("z max", 0.0, "Integration domain maximum z coordinate");

  validPL->set< Teuchos::Array<int> >("Field Components", Teuchos::Array<int>(),
				      "Field components to scatter");

  return validPL;
}

// **********************************************************************

