//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Teuchos_TestForException.hpp"
#include "Teuchos_CommHelpers.hpp"
#include "Aeras_ShallowWaterConstants.hpp"
#include "PHAL_Utilities.hpp"

namespace Aeras {

}


template<typename EvalT, typename Traits>
Aeras::ShallowWaterResponseL2Norm<EvalT, Traits>::
ShallowWaterResponseL2Norm(Teuchos::ParameterList& p,
		      const Teuchos::RCP<Albany::Layouts>& dl) :
  weighted_measure("Weights", dl->qp_scalar),
  flow_state_field("Flow State", dl->node_vector), 
  BF("BF",dl->node_qp_scalar)
{
  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());
  // get and validate Response parameter list
  Teuchos::ParameterList* plist = 
    p.get<Teuchos::ParameterList*>("Parameter List");
  Teuchos::RCP<const Teuchos::ParameterList> reflist = 
    this->getValidResponseParameters();
  plist->validateParameters(*reflist,0);

  std::string fieldName = "Flow Field"; //field to integral is the flow field 

  // coordinate dimensions
  std::vector<PHX::DataLayout::size_type> coord_dims;
  dl->qp_vector->dimensions(coord_dims);
  numQPs = coord_dims[1]; //# quad points
  numDims = coord_dims[2]; //# spatial dimensions
  std::vector<PHX::DataLayout::size_type> dims;
  flow_state_field.fieldTag().dataLayout().dimensions(dims);
  vecDim = dims[2]; //# dofs per node
  numNodes =  dims[1]; //# nodes per element

 
  // User-specified parameters
  //None right now  

  // add dependent fields
  this->addDependentField(flow_state_field);
  this->addDependentField(weighted_measure);
  this->addDependentField(BF);
  this->setName(fieldName+" Aeras Shallow Water L2 Norm"+PHX::typeAsString<EvalT>());
  
  using PHX::MDALayout;

  // Setup scatter evaluator
  p.set("Stand-alone Evaluator", false);
  std::string local_response_name = fieldName + " Local Response Aeras Shallow Water L2 Norm";
  std::string global_response_name = fieldName + " Global Response Aeras Shallow Water L2 Norm";
  int worksetSize = dl->qp_scalar->dimension(0);
  //There are four components of the response returned by this function: 
  //1.) |h|
  //2.) |u|
  //3.) |v|
  //4.) |solution vector|
  responseSize = 4; 
  Teuchos::RCP<PHX::DataLayout> local_response_layout = Teuchos::rcp(new MDALayout<Cell, Dim>(worksetSize, responseSize));
  Teuchos::RCP<PHX::DataLayout> global_response_layout = Teuchos::rcp(new MDALayout<Dim>(responseSize));
  PHX::Tag<ScalarT> local_response_tag(local_response_name, local_response_layout);
  PHX::Tag<ScalarT> global_response_tag(global_response_name, global_response_layout);
  p.set("Local Response Field Tag", local_response_tag);
  p.set("Global Response Field Tag", global_response_tag);
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::setup(p,dl);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void Aeras::ShallowWaterResponseL2Norm<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(flow_state_field,fm);
  this->utils.setFieldData(weighted_measure,fm);
  this->utils.setFieldData(BF,fm);
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::postRegistrationSetup(d,fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void Aeras::ShallowWaterResponseL2Norm<EvalT, Traits>::
preEvaluate(typename Traits::PreEvalData workset)
{
  PHAL::set(this->global_response, 0.0);
  // Do global initialization
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::preEvaluate(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void Aeras::ShallowWaterResponseL2Norm<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{   
  // Zero out local response
  PHAL::set(this->local_response, 0.0);

  Intrepid::FieldContainer<ScalarT> flow_state_field_qp(workset.numCells, numQPs, vecDim); //flow_state_field at quad points
  
  //Interpolate flow_state_field from nodes -> quadrature points.  
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    this->local_response(cell,3) = 0.0;  
    this->global_response(3) = 0.0; 
    for (std::size_t qp=0; qp < numQPs; ++qp) {
      for (std::size_t i=0; i<vecDim; i++) {
        // Zero out for node==0; then += for node = 1 to numNodes
        flow_state_field_qp(cell,qp,i) = 0.0;
        flow_state_field_qp(cell,qp,i) = flow_state_field(cell, 0, i)*BF(cell, 0, qp); 
        for (std::size_t node=1; node < numNodes; ++node) {
          flow_state_field_qp(cell,qp,i) += flow_state_field(cell,node,i)*BF(cell,node,qp); 
        }
       }
     }
   }

  //Get final time from workset.  This is for setting time-dependent exact solution. 
  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());
  const RealType final_time  = workset.current_time;
  *out << "final time = " << final_time << std::endl; 
 
  //Calculate L2 norm squared of each component of solution
  ScalarT wm;
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {
      wm = weighted_measure(cell,qp);
      for (std::size_t dim=0; dim<vecDim; ++dim) {
        this->local_response(cell,dim) += wm*flow_state_field_qp(cell,qp,dim)*flow_state_field_qp(cell,qp,dim);
        this->global_response(dim) += wm*flow_state_field_qp(cell,qp,dim)*flow_state_field_qp(cell,qp,dim);
      }
    }
  }
  
  // Do any local-scattering necessary
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::evaluateFields(workset);
}

//***********************************************************************
// **********************************************************************
template<typename EvalT, typename Traits>
void Aeras::ShallowWaterResponseL2Norm<EvalT, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());
#if 0
  // Add contributions across processors
  Teuchos::RCP< Teuchos::ValueTypeSerializer<int,ScalarT> > serializer =
    workset.serializerManager.template getValue<EvalT>();

  // we cannot pass the same object for both the send and receive buffers in reduceAll call
  // creating a copy of the global_response, not a view
  std::vector<ScalarT> partial_vector(&this->global_response[0],&this->global_response[0]+this->global_response.size()); //needed for allocating new storage
  PHX::MDField<ScalarT> partial_response(this->global_response);
  partial_response.setFieldData(Teuchos::ArrayRCP<ScalarT>(partial_vector.data(),0,partial_vector.size(),false));


  //perform reduction for each of the components of the response
  Teuchos::reduceAll(
      *workset.comm, *serializer, Teuchos::REDUCE_SUM,
      this->global_response.size(), &partial_response[0],
      &this->global_response[0]);
#else
  //amb reduceAll workaround
  PHAL::reduceAll(*workset.comm, Teuchos::REDUCE_SUM, this->global_response);
#endif
  
#if 0
  ScalarT abs_err_sq = this->global_response[0];
  ScalarT norm_ref_sq = this->global_response[1];
  this-> global_response[0] = sqrt(abs_err_sq); //absolute error in solution w.r.t. reference solution.
  this-> global_response[1] = sqrt(norm_ref_sq); //norm of reference solution
  this-> global_response[2] = sqrt(abs_err_sq/norm_ref_sq); //relative error in solution w.r.t. reference solution.
#else
  //amb op[] bracket workaround
  PHAL::MDFieldIterator<ScalarT> gr(this->global_response);
  ScalarT h_norm_sq = *gr;
  *gr = sqrt(h_norm_sq); 
  ++gr;
  ScalarT u_norm_sq = *gr;
  *gr = sqrt(u_norm_sq); 
  ++gr;
  ScalarT v_norm_sq = *gr;
  *gr = sqrt(v_norm_sq); 
  ++gr;
  *gr = sqrt(h_norm_sq + u_norm_sq + v_norm_sq); //norm of full solution
#endif

  // Do global scattering
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::postEvaluate(workset);
}

// **********************************************************************
template<typename EvalT,typename Traits>
Teuchos::RCP<const Teuchos::ParameterList>
Aeras::ShallowWaterResponseL2Norm<EvalT,Traits>::
getValidResponseParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
     	rcp(new Teuchos::ParameterList("Valid ShallowWaterResponseL2Norm Params"));
  Teuchos::RCP<const Teuchos::ParameterList> baseValidPL =
    PHAL::SeparableScatterScalarResponse<EvalT,Traits>::getValidResponseParameters();
  validPL->setParameters(*baseValidPL);

  validPL->set<std::string>("Name", "", "Name of response function");
  validPL->set<int>("Phalanx Graph Visualization Detail", 0, "Make dot file to visualize phalanx graph");

  return validPL;
}

// **********************************************************************

