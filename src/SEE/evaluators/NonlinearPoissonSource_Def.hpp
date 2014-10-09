//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

namespace SEE {

//**********************************************************************
template<typename EvalT, typename Traits>
NonlinearPoissonSource<EvalT, Traits>::
NonlinearPoissonSource(const Teuchos::ParameterList& p,
                         const Teuchos::RCP<Albany::Layouts>& dl) :
  coord_      (p.get<std::string>("Coordinate Name"),
               dl->qp_vector),
  source_     (p.get<std::string>("Source Name"),
               dl->qp_scalar)
{

  this->addDependentField(coord_);
  
  this->addEvaluatedField(source_);
 
  Teuchos::RCP<PHX::DataLayout> scalar_dl = dl->qp_scalar;
  std::vector<PHX::DataLayout::size_type> dims;
  scalar_dl->dimensions(dims);
  workset_size_ = dims[0];
  num_qps_      = dims[1];

  this->setName("NonlinearPoissonSource"+PHX::TypeString<EvalT>::value);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void NonlinearPoissonSource<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(coord_,fm);

  this->utils.setFieldData(source_,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void NonlinearPoissonSource<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

  // source function
  for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
    for (std::size_t qp = 0; qp < num_qps_; ++qp) {
      MeshScalarT* X = &coord_(cell,qp,0);
      source_(cell,qp) =
        -2.0 * X[0];
    }
  }

}

//**********************************************************************
}
