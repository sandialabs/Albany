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
Density<EvalT, Traits>::Density(Teuchos::ParameterList& p)
    : density(
          p.get<std::string>("Cell Variable Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("Cell Scalar Data Layout"))
{
  Teuchos::ParameterList* density_param_list =
      p.get<Teuchos::ParameterList*>("Parameter List");

  constant_value = density_param_list->get<double>("Value");

  this->addEvaluatedField(density);
  this->setName("Density" + PHX::typeAsString<EvalT>());
}

// **********************************************************************
template <typename EvalT, typename Traits>
void
Density<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(density, fm);
}

// **********************************************************************
template <typename EvalT, typename Traits>
void
Density<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  int numCells = workset.numCells;

  for (int cell = 0; cell < numCells; ++cell) {
    density(cell) = constant_value;
  }
}

// **********************************************************************
template <typename EvalT, typename Traits>
typename Density<EvalT, Traits>::ScalarT&
Density<EvalT, Traits>::getValue(const std::string& n)
{
  if (n == "Density") { return constant_value; }

  TEUCHOS_TEST_FOR_EXCEPTION(
      true,
      Teuchos::Exceptions::InvalidParameter,
      std::endl
          << "Error! Logic error in getting paramter " << n
          << " in Density::getValue()" << std::endl);
  return constant_value;
}

// **********************************************************************
// **********************************************************************
}  // namespace LCM
