/*
 * Aeras_TotalVolume_Def.hpp
 *
 *  Created on: Jul 11, 2014
 *      Author: swbova
 */

#ifndef AERAS_TOTALVOLUME_DEF_HPP_
#define AERAS_TOTALVOLUME_DEF_HPP_

#include <fstream>
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Teuchos_CommHelpers.hpp"
#include "Phalanx.hpp"

namespace Aeras {
template<typename EvalT, typename Traits>
TotalVolume<EvalT, Traits>::
TotalVolume(Teuchos::ParameterList& p,
                     const Teuchos::RCP<Aeras::Layouts>& dl) :
  coordVec("Coord Vec", dl->qp_vector),
  weighted_measure("Weights", dl->qp_scalar),
  density ("Density", dl->qp_scalar_level),
  velocity("Velx",  dl->qp_vector_level),
  temperature("Temperature",dl->qp_scalar_level),
  Cpstar("Cpstar",dl->qp_scalar_level),
  pie("Pi",  dl->qp_scalar_level),
   numLevels(dl->node_scalar_level->dimension(2))


{
  Teuchos::ParameterList* plist =
    p.get<Teuchos::ParameterList*>("Parameter List");
  Teuchos::RCP<const Teuchos::ParameterList> reflist =
    this->getValidResponseParameters();
  plist->validateParameters(*reflist,0);

  Phi0 = 0;
   std::cout << "Total_Volume: Phi0 = " << Phi0 << std::endl;

  // number of quad points per cell and dimension of space
  Teuchos::RCP<PHX::DataLayout> scalar_dl = dl->qp_scalar;
  Teuchos::RCP<PHX::DataLayout> vector_dl = dl->qp_vector;

  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

//  this->addDependentField(coordVec);
  this->addDependentField(weighted_measure);
  this->addDependentField(density);
  this->addDependentField(velocity);
  this->addDependentField(temperature);
  this->addDependentField(Cpstar);
  this->addDependentField(pie);

  this->setName("Aeras Total Volume");

  using PHX::MDALayout;

  // Setup scatter evaluator
  p.set("Stand-alone Evaluator", false);
  std::string local_response_name = "Local Response Aeras Total Volume";
  std::string global_response_name = "Global Response Aeras Total Volume";
  int worksetSize = scalar_dl->dimension(0);

  int responseSize = 3;

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
void TotalVolume<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
//  this->utils.setFieldData(coordVec,fm);
  this->utils.setFieldData(weighted_measure,fm);
  this->utils.setFieldData(density,fm);
  this->utils.setFieldData(velocity,fm);
  this->utils.setFieldData(temperature,fm);
  this->utils.setFieldData(Cpstar,fm);
  this->utils.setFieldData(pie,fm);

  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::postRegistrationSetup(d,fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void TotalVolume<EvalT, Traits>::
preEvaluate(typename Traits::PreEvalData workset)
{
  const int imax = this->global_response.size();

  for (typename PHX::MDField<ScalarT>::size_type i=0;
       i< imax; i++)
    this->global_response[i] = 0.0;

  // Do global initialization
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::preEvaluate(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void TotalVolume<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  std::cout << "TotalVolume evaluateFields()" << std::endl;

  for (typename PHX::MDField<ScalarT>::size_type i=0;
       i<this->local_response.size(); i++)
    this->local_response[i] = 0.0;

  ScalarT volume;
  ScalarT mass;
  ScalarT energy;

    for (std::size_t cell=0; cell < workset.numCells; ++cell)
    {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
        volume = weighted_measure(cell,qp);
        for(std::size_t ell = 0; ell < numLevels; ++ell) {
          this->local_response(cell, 0) += volume;
          this->global_response(0) += volume;

          mass = volume*density(cell, qp, ell);
          this->local_response(cell, 1) += mass;
          this->global_response(1) += mass;

          energy = pie(cell, qp, ell)*(0.5*velocity(cell, qp, ell)*velocity(cell,qp,ell) +
              Cpstar(cell,qp,ell)*temperature(cell,qp,ell) + Phi0 )*volume;

          this->local_response(cell, 2) += energy;
          this->global_response(2) += energy;
 }

    }
  }

  // Do any local-scattering necessary
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::evaluateFields(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void TotalVolume<EvalT, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{

  std::cout << "TotalVolume postEvaluate()" << std::endl;

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

  // Do global scattering
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::postEvaluate(workset);

  std::cout << "Total Volume is " << this->global_response(0) << std::endl;
  std::cout << "Total Mass is " << this->global_response(1) << std::endl;
  std::cout << "Total Energy is " << this->global_response(2) << std::endl;

}

// **********************************************************************
template<typename EvalT,typename Traits>
Teuchos::RCP<const Teuchos::ParameterList>
TotalVolume<EvalT,Traits>::getValidResponseParameters() const
{

  Teuchos::RCP<Teuchos::ParameterList> validPL =
        rcp(new Teuchos::ParameterList("Valid Total Volume Params"));;
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
#endif /* AERAS_TOTALVOLUME_DEF_HPP_ */
