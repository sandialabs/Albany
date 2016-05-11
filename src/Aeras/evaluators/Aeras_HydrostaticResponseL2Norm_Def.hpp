//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef AERAS_HYDROSTATIC_RESPONSE_L2NORM_DEF_HPP_
#define AERAS_HYDROSTATIC_RESPONSE_L2NORM_DEF_HPP_

#include <fstream>
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Teuchos_CommHelpers.hpp"
#include "Phalanx.hpp"
#include "PHAL_Utilities.hpp"

namespace Aeras {
template<typename EvalT, typename Traits>
HydrostaticResponseL2Norm<EvalT, Traits>::
HydrostaticResponseL2Norm(Teuchos::ParameterList& p,
                     const Teuchos::RCP<Aeras::Layouts>& dl) :
  weighted_measure("Weights", dl->qp_scalar),
  velocity("Velx",  dl->qp_vector_level),
  temperature("Temperature",dl->qp_scalar_level),
  spressure("SPressure",dl->qp_scalar),
  numLevels(dl->node_scalar_level->dimension(2)), 
  out(Teuchos::VerboseObjectBase::getDefaultOStream())
{
  Teuchos::ParameterList* plist =
    p.get<Teuchos::ParameterList*>("Parameter List");
  Teuchos::RCP<const Teuchos::ParameterList> reflist =
    this->getValidResponseParameters();
  plist->validateParameters(*reflist,0);

  *out << "in Hydrostatic_Response_L2Norm! \n";

  // number of quad points per cell and dimension of space
  Teuchos::RCP<PHX::DataLayout> scalar_dl = dl->qp_scalar;
  Teuchos::RCP<PHX::DataLayout> vector_dl = dl->qp_vector;

  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  *out << "numQPs, numDims, numLevels: " << numQPs << ", " << numDims << ", " << numLevels << std::endl; 
  this->addDependentField(weighted_measure);
  this->addDependentField(velocity);
  this->addDependentField(temperature);
  this->addDependentField(spressure);

  this->setName("Aeras Hydrostatic Response L2 Norm");

  using PHX::MDALayout;

  // Setup scatter evaluator
  p.set("Stand-alone Evaluator", false);
  std::string local_response_name = "Local Response Aeras Hydrostatic Response L2 Norm";
  std::string global_response_name = "Global Response Aeras Hydrostatic Response L2 Norm";
  int worksetSize = scalar_dl->dimension(0);

  //FIXME: extend responseSize to have tracers 
  responseSize = 3*numLevels + 1; //there are 2 velocities and 1 temperature variable on each level
                                      //surface pressure is on 1st level only
                                      //the ordering is: Sp0, u0, v0, T0, u1, v1, T1, etc

  Teuchos::RCP<PHX::DataLayout> local_response_layout = Teuchos::rcp(
      new MDALayout<Cell,Dim>(worksetSize, responseSize));
  PHX::Tag<ScalarT> local_response_tag(local_response_name,
                                       local_response_layout);
  p.set("Local Response Field Tag", local_response_tag);

  Teuchos::RCP<PHX::DataLayout> global_response_layout = Teuchos::rcp(
      new MDALayout<Dim>(responseSize));
  PHX::Tag<ScalarT> global_response_tag(global_response_name,
                                        global_response_layout);
  p.set("Global Response Field Tag", global_response_tag);
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::setup(p,dl);

}

// **********************************************************************
template<typename EvalT, typename Traits>
void HydrostaticResponseL2Norm<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(weighted_measure,fm);
  this->utils.setFieldData(velocity,fm);
  this->utils.setFieldData(temperature,fm);
  this->utils.setFieldData(spressure,fm);

  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::postRegistrationSetup(d,fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void HydrostaticResponseL2Norm<EvalT, Traits>::
preEvaluate(typename Traits::PreEvalData workset)
{
  const int imax = this->global_response.size();

  PHAL::set(this->global_response, 0.0);

  // Do global initialization
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::preEvaluate(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void HydrostaticResponseL2Norm<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  *out << "HydrostaticResponseL2Norm evaluateFields() \n" << std::endl;

  //Zero out local response 
  PHAL::set(this->local_response, 0.0);

  //Calculate L2 norm squared of each component of solution.  We do not need to do 
  //an interpolation from the nodes to the QPs of the solution since nodes = QPs for Aeras
  //spectral elements. 
  ScalarT wm; 
  std::size_t dim; 
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp)  {
      wm = weighted_measure(cell,qp);
      //surface pressure field: dof 0
      dim = 0;
      this->local_response(cell,dim) += wm*spressure(cell,qp)*spressure(cell,qp); 
      this->global_response(dim) += wm*spressure(cell,qp)*spressure(cell,qp);
      for (std::size_t level=0; level < numLevels; ++level) {
        //u-velocity field: dof 1, 4, 7, ...
        //v-velocity field: dof 2, 5, 8, ...
        for (std::size_t i=0; i < 2; ++i) {
          dim = 1 + i + level*3; 
          this->local_response(cell,dim) += wm*velocity(cell,qp,level,i)*velocity(cell,qp,level,i); 
          this->global_response(dim) += wm*velocity(cell,qp,level,i)*velocity(cell,qp,level,i);
        }
        //temperature field: dof 3, 6, 9, .... 
        dim = 3 + level*3; 
        this->local_response(cell,dim) += wm*temperature(cell,qp,level)*temperature(cell,qp,level); 
        this->global_response(dim) += wm*temperature(cell,qp,level)*temperature(cell,qp,level);
        //FIXME: ultimately, will want to add tracers. 
      } 
    }
  }

  // Do any local-scattering necessary
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::evaluateFields(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void HydrostaticResponseL2Norm<EvalT, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{

  *out << "HydrostaticResponseL2Norm postEvaluate() \n" << std::endl;
#if 0
  // Add contributions across processors
  Teuchos::RCP< Teuchos::ValueTypeSerializer<int,ScalarT> > serializer =
    workset.serializerManager.template getValue<EvalT>();

  // we cannot pass the same object for both the send and receive buffers in reduceAll call
  // creating a copy of the global_response, not a view
  std::vector<ScalarT> partial_vector(&this->global_response[0],&this->global_response[0]+this->global_response.size()); //needed for allocating new storage
  PHX::MDField<ScalarT> partial_response(this->global_response);
  partial_response.setFieldData(Teuchos::ArrayRCP<ScalarT>(partial_vector.data(),0,partial_vector.size(),false));

  Teuchos::reduceAll(
    *workset.comm, *serializer, Teuchos::REDUCE_SUM,
    this->global_response.size(), &partial_response[0],
    &this->global_response[0]);
#else
  //amb reduceAll workaround.
  PHAL::reduceAll(*workset.comm, Teuchos::REDUCE_SUM, this->global_response);
#endif

  // Do global scattering
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::postEvaluate(workset);

#if 0
#else
  PHAL::MDFieldIterator<ScalarT> gr(this->global_response);
  for (int i=0; i < responseSize; ++i) {
    ScalarT norm_sq = *gr; 
    *gr = sqrt(norm_sq);  
    ++gr; 
  } 
#endif
}

// **********************************************************************
template<typename EvalT,typename Traits>
Teuchos::RCP<const Teuchos::ParameterList>
HydrostaticResponseL2Norm<EvalT,Traits>::getValidResponseParameters() const
{

  Teuchos::RCP<Teuchos::ParameterList> validPL =
        rcp(new Teuchos::ParameterList("Valid Hydrostatic Response L2 Norm Params"));;
  Teuchos::RCP<const Teuchos::ParameterList> baseValidPL =
    PHAL::SeparableScatterScalarResponse<EvalT,Traits>::getValidResponseParameters();
  validPL->setParameters(*baseValidPL);

  validPL->set<std::string>("Name", "", "Name of response function");
  validPL->set<int>("Phalanx Graph Visualization Detail", 0, "Make dot file to visualize phalanx graph");
  validPL->set<std::string>("Field Name", "", "Scalar field from which to compute center of mass");
  validPL->set<std::string>("Description", "", "Description of this response used by post processors");

  return validPL;
}

}
#endif /* AERAS_HYDROSTATIC_RESPONSE_L2NORM_DEF_HPP_ */
