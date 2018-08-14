//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <fstream>
#include "Albany_Utils.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Teuchos_TestForException.hpp"

namespace LCM {

template <typename EvalT, typename Traits>
ACEporosity<EvalT, Traits>::ACEporosity(Teuchos::ParameterList& p)
    : porosity_(
          p.get<std::string>("ACE Porosity"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout"))
{
  Teuchos::ParameterList* porosity_list =
      p.get<Teuchos::ParameterList*>("Parameter List");

  Teuchos::RCP<PHX::DataLayout> vector_dl =
      p.get<Teuchos::RCP<PHX::DataLayout>>("QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  num_qps_  = dims[1];
  num_dims_ = dims[2];

  Teuchos::RCP<ParamLib> paramLib =
      p.get<Teuchos::RCP<ParamLib>>("Parameter Library", Teuchos::null);

  // Read input deck values
  std::string const& type = porosity_list->get<std::string>("Porosity Type");
  if (type == "Constant") {
    is_constant_    = true;
    constant_value_ = porosity_list->get<double>("Value");
  } else if (type == "Depth-Dependent") {
    is_constant_      = false;
    surface_porosity_ = porosity_list->get<double>("Surface Porosity");
    efolding_depth_   = porosity_list->get<double>("E-Depth");
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        Teuchos::Exceptions::InvalidParameter,
        "Invalid Porosity Type " << type);
  }

  // Add porosity as a Sacado-ized parameter
  this->registerSacadoParameter("ACE Porosity", paramLib);

  // List evaluated fields
  this->addEvaluatedField(porosity_);

  this->setName("ACE Porosity" + PHX::typeAsString<EvalT>());
}

//
template <typename EvalT, typename Traits>
void
ACEporosity<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(porosity_, fm);
  return;
}

// This function calculates the depth-dependent porosity,
// or assigns a constant value.
// Based on Athy's Law (Athy, 1930).
template <typename EvalT, typename Traits>
void
ACEporosity<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  int    num_cells = workset.numCells;
  double z = 1.0;  // this is the depth -> how to construct this from cell & qp?

  for (int cell = 0; cell < num_cells; ++cell) {
    for (int qp = 0; qp < num_qps_; ++qp) {
      if (is_constant_) {  // constant porosity:
        porosity_(cell, qp) = constant_value_;
      } else {  // depth-dependent porosity:
        porosity_(cell, qp) =
            surface_porosity_ * exp(-1.0 * z / efolding_depth_);
      }
    }
  }

  return;
}

//
template <typename EvalT, typename Traits>
typename ACEporosity<EvalT, Traits>::ScalarT&
ACEporosity<EvalT, Traits>::getValue(const std::string& n)
{
  if (n == "Surface Porosity") { return surface_porosity_; }
  if (n == "E-Depth") { return efolding_depth_; }

  ALBANY_ASSERT(false, "Invalid request for value of ACE Porosity Input");

  return surface_porosity_;  // does it matter what we return here?
}

}  // namespace LCM
