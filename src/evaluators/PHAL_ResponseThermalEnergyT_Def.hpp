//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Teuchos_TestForException.hpp"
#include "Teuchos_CommHelpers.hpp"
#include "PHAL_Utilities.hpp"
#include <typeinfo>


template<typename EvalT, typename Traits>
PHAL::ResponseThermalEnergyT<EvalT, Traits>::
ResponseThermalEnergyT(Teuchos::ParameterList& p,
		      const Teuchos::RCP<Albany::Layouts>& dl) :
  coordVec("Coord Vec", dl->qp_gradient),
  weights("Weights", dl->qp_scalar)
//  time("Time",dl->workset_scalar),
//  deltaTime("Delta Time",dl->workset_scalar)
{
  // get and validate Response parameter list
  Teuchos::ParameterList* plist = 
    p.get<Teuchos::ParameterList*>("Parameter List");
  Teuchos::RCP<const Teuchos::ParameterList> reflist = 
    this->getValidResponseParameters();
  plist->validateParameters(*reflist,0);
  
  // get parameters from problem
  Teuchos::RCP<Teuchos::ParameterList> pFromProb = 
    p.get<Teuchos::RCP<Teuchos::ParameterList>>("Parameters From Problem");
  // get properties
  density = pFromProb->get<RealType>("Density");
  heat_capacity = pFromProb->get<RealType>("Heat Capacity");
  
  
  // Get field type and corresponding layouts
  std::string field_name = plist->get<std::string>("Field Name");
  std::string fieldType = plist->get<std::string>("Field Type");
  
  Teuchos::RCP<PHX::DataLayout> field_layout;
  Teuchos::RCP<PHX::DataLayout> local_response_layout;
  Teuchos::RCP<PHX::DataLayout> global_response_layout;
  if (fieldType == "Scalar") {
    field_layout = dl->qp_scalar;
    local_response_layout = dl->cell_scalar;
    global_response_layout = dl->workset_scalar;
  }
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(
      true, 
      Teuchos::Exceptions::InvalidParameter,
      "Invalid field type " << fieldType << ".  Support value is " << 
      "Scalar." << std::endl);
  }
  field = PHX::MDField<ScalarT>(field_name, field_layout);
  field_layout->dimensions(field_dims);

  // coordinate dimensions
  std::vector<PHX::DataLayout::size_type> coord_dims;
  dl->qp_vector->dimensions(coord_dims);
  numQPs = coord_dims[1];
  numDims = coord_dims[2];

  // add dependent fields
  this->addDependentField(field);
  this->addDependentField(coordVec);
//  this->addDependentField(time);
//  this->addDependentField(deltaTime);
  this->addDependentField(weights);
  this->setName(field_name+" Response Field IntegralT"+PHX::typeAsString<EvalT>());

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
  PHAL::SeparableScatterScalarResponseT<EvalT,Traits>::setup(p,dl);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void PHAL::ResponseThermalEnergyT<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(field,fm);
  this->utils.setFieldData(coordVec,fm);
  this->utils.setFieldData(weights,fm);
//  this->utils.setFieldData(time,fm);
//  this->utils.setFieldData(deltaTime,fm);
  PHAL::SeparableScatterScalarResponseT<EvalT,Traits>::postRegistrationSetup(d,fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void PHAL::ResponseThermalEnergyT<EvalT, Traits>::
preEvaluate(typename Traits::PreEvalData workset)
{
  PHAL::set(this->global_response, 0.0);
  // Do global initialization
  PHAL::SeparableScatterScalarResponseT<EvalT,Traits>::preEvaluate(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void PHAL::ResponseThermalEnergyT<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
    // Zero out local response
    PHAL::set(this->local_response, 0.0);
    
    ScalarT s;
    for (std::size_t cell = 0; cell < workset.numCells; ++cell)
    {
        for (std::size_t qp = 0; qp < numQPs; ++qp)
        {
            s = density * heat_capacity * field(cell, qp) * weights(cell, qp);
            this->local_response(cell, 0) += s;
            this->global_response(0) += s;
        }
    }

    // Do any local-scattering necessary
    PHAL::SeparableScatterScalarResponseT<EvalT, Traits>::evaluateFields(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void PHAL::ResponseThermalEnergyT<EvalT, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
  PHAL::reduceAll<ScalarT>(*workset.comm, Teuchos::REDUCE_SUM,
                           this->global_response);
  PHAL::SeparableScatterScalarResponseT<EvalT,Traits>::postEvaluate(workset);
}

// **********************************************************************
template<typename EvalT,typename Traits>
Teuchos::RCP<const Teuchos::ParameterList>
PHAL::ResponseThermalEnergyT<EvalT,Traits>::
getValidResponseParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
     	rcp(new Teuchos::ParameterList("Valid ResponseThermalEnergyT Params"));
  Teuchos::RCP<const Teuchos::ParameterList> baseValidPL =
    PHAL::SeparableScatterScalarResponseT<EvalT,Traits>::getValidResponseParameters();
  validPL->setParameters(*baseValidPL);

  validPL->set<std::string>("Name", "", "Name of response function");
  validPL->set<int>("Phalanx Graph Visualization Detail", 0, "Make dot file to visualize phalanx graph");
  validPL->set<std::string>("Field Type", "", "Type of field (scalar)");
  validPL->set<std::string>("Field Name", "", "Field to integrate");

  return validPL;
}

// **********************************************************************

