//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <fstream>
#include "Albany_Utils.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "Teuchos_TestForException.hpp"

namespace LCM {

//
//
//
template <typename EvalT, typename Traits>
BodyForce<EvalT, Traits>::
BodyForce(Teuchos::ParameterList & p, Teuchos::RCP<Albany::Layouts> dl)
    : body_force_("Body Force", dl->qp_vector)
{
  Teuchos::RCP<PHX::DataLayout>
  vector_dl = dl->qp_vector;

  std::vector<PHX::DataLayout::size_type>
  dims;

  vector_dl->dimensions(dims);

  num_qp_ = dims[1];
  num_dim_ = dims[2];

  std::string const &
  type = p.get("Body Force Type", "Constant");

  if (type == "Constant") {
    is_constant_ = true;
    constant_value_ = p.get<Teuchos::Array<RealType>>("Value");
  }
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true, Teuchos::Exceptions::InvalidParameter,
        "Invalid body force type " << type);
  }

  this->addEvaluatedField(body_force_);
  this->setName("Body Force" + PHX::typeAsString<EvalT>());
}

//
//
//
template <typename EvalT, typename Traits>
void
BodyForce<EvalT, Traits>::
postRegistrationSetup(
    typename Traits::SetupData d, PHX::FieldManager<Traits> & fm)
{
  this->utils.setFieldData(body_force_, fm);
  if (is_constant_ == false) this->utils.setFieldData(coordinates_, fm);
}

//
//
//
template <typename EvalT, typename Traits>
void
BodyForce<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  int const
  num_cells = workset.numCells;

  if (is_constant_ == true) {
    for (int cell = 0; cell < num_cells; ++cell) {
      for (int qp = 0; qp < num_qp_; ++qp) {
        for (int dim = 0; dim < num_dim_; ++dim) {
          body_force_(cell, qp, dim) = constant_value_[dim];
        }
      }
    }
  }
}

}
