//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_Layouts.hpp"

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits>
LogGaussianDistributedParameter<EvalT, Traits>::
LogGaussianDistributedParameter (const Teuchos::ParameterList& p,
                                 const Teuchos::RCP<Albany::Layouts>& dl)
{
  std::string log_gaussian_field_name   = p.get<std::string>("Log Gaussian Name");
  std::string gaussian_field_name       = p.get<std::string>("Gaussian Name");
  std::string mean_field_name;

  mean_is_field = false;
  if (p.isParameter("Mean Name")) {
    mean_field_name = p.get<std::string>("Mean Name");
    mean_is_field = true;
  }

  eval_on_side = false;
  if (p.isParameter("Side Set Name")) {
    sideSetName = p.get<std::string>("Side Set Name");
    eval_on_side = true;
  }

  logGaussian = decltype(logGaussian)(log_gaussian_field_name,dl->node_scalar);
  numNodes = 0;

  gaussian = decltype(gaussian)(gaussian_field_name,dl->node_scalar);

  if (mean_is_field) {
    mean = decltype(mean)(mean_field_name,dl->node_scalar);
    this->addDependentField(mean);
  }
  else {
    RealType mean = p.get<RealType>("mean");
    RealType deviation = p.get<RealType>("deviation");
    a = log(mean/sqrt(1.+deviation*deviation));
    b = sqrt(log(1+deviation*deviation));
  }  
  this->addEvaluatedField(logGaussian);
  this->addDependentField(gaussian);

  this->setName("Log Gaussian " + log_gaussian_field_name + PHX::print<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void LogGaussianDistributedParameter<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(logGaussian,fm);
  numNodes = logGaussian.extent(1);
  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields(),false);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void LogGaussianDistributedParameter<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  std::size_t n_cells;
  
  if (this->eval_on_side) {
    if (workset.sideSetViews->find(this->sideSetName)==workset.sideSetViews->end()) return;
    n_cells = workset.sideSetViews->at(this->sideSetName).size;
  } else {
    n_cells = workset.numCells;
  }

  for (std::size_t cell = 0; cell < n_cells; ++cell) {
    for (std::size_t node = 0; node < this->numNodes; ++node) {
      if (this->mean_is_field)
        (this->logGaussian)(cell, node) = exp(log((this->mean)(cell, node)) + (this->gaussian)(cell, node));
      else
        (this->logGaussian)(cell, node) = exp(this->a + this->b * (this->gaussian)(cell, node));
    }
  }
}

}  // Namespace PHAL
