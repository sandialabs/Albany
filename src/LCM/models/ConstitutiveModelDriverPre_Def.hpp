//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_RCP.hpp"
#include "Phalanx_DataLayout.hpp"

namespace LCM
{

//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
ConstitutiveModelDriverPre<EvalT, Traits>::
ConstitutiveModelDriverPre(Teuchos::ParameterList& p,
                           const Teuchos::RCP<Albany::Layouts>& dl):
  solution_(p.get<std::string>("Solution Name"),dl->node_tensor),
  def_grad_(p.get<std::string>("F Name"),dl->qp_tensor),
  j_(p.get<std::string>("J Name"),dl->qp_scalar)
{
  std::vector<PHX::DataLayout::size_type> dims;
  dl->node_qp_tensor->dimensions(dims);
  num_nodes_ = dims[1];
  num_pts_   = dims[2];
  num_dims_  = dims[3];
  this->setName("ConstitutiveModelDriverPre" + PHX::TypeString<EvalT>::value);
}

//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
void ConstitutiveModelDriverPre<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(solution_, fm);
  this->utils.setFieldData(def_grad_, fm);
  this->utils.setFieldData(j_, fm);
}

//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
void ConstitutiveModelDriverPre<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  Intrepid::Tensor<ScalarT> F(num_dims_);

  for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
    for (std::size_t pt = 0; pt < num_pts_; ++pt) {    
      for (std::size_t node = 0; node < num_nodes_; ++node) {
        for (std::size_t dim = 0; dim < num_dims_; ++dim) {
          def_grad_(cell, pt, ) = 0.0;
        }
      }
    }
  }
}

//------------------------------------------------------------------------------

}

