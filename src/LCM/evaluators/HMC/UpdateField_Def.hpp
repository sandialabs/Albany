//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <string>
#include <vector>

#include "Phalanx_DataLayout.hpp"
#include "Teuchos_TestForException.hpp"

namespace LCM {

// **********************************************************************
template <typename EvalT, typename Traits>
UpdateField<EvalT, Traits>::UpdateField(const Teuchos::ParameterList& p)
    : field_Nplus1(
          p.get<std::string>("Updated Field Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("Field Layout")),
      field_Inc(
          p.get<std::string>("Increment Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("Field Layout"))
{
  this->addDependentField(field_Inc);
  this->addEvaluatedField(field_Nplus1);

  this->name_N            = p.get<std::string>("Current State Name");
  std::string name_Nplus1 = p.get<std::string>("Updated Field Name");
  std::string name_Inc    = p.get<std::string>("Increment Name");
  this->setName(
      "Update " + name_N + " to " + name_Nplus1 + " by " + name_Inc +
      PHX::typeAsString<EvalT>());
}

// **********************************************************************
template <typename EvalT, typename Traits>
void
UpdateField<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(field_Nplus1, fm);
  this->utils.setFieldData(field_Inc, fm);
}
// **********************************************************************
template <typename EvalT, typename Traits>
void
UpdateField<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  Albany::StateArray::const_iterator it;
  it = workset.stateArrayPtr->find(name_N);

  TEUCHOS_TEST_FOR_EXCEPTION(
      (it == workset.stateArrayPtr->end()),
      std::logic_error,
      std::endl
          << "Error: cannot locate " << name_N << " in UpdateField_Def"
          << std::endl);

  Albany::MDArray                         state_N = it->second;
  std::vector<PHX::DataLayout::size_type> dims;
  state_N.dimensions(dims);
  int size = dims.size();

  switch (size) {
    case 1:
      for (int i = 0; i < dims[0]; ++i)
        field_Nplus1(i) = state_N(i) + field_Inc(i);
      break;
    case 2:
      for (int i = 0; i < dims[0]; ++i)
        for (int j = 0; j < dims[1]; ++j)
          field_Nplus1(i, j) = state_N(i, j) + field_Inc(i, j);
      break;
    case 3:
      for (int i = 0; i < dims[0]; ++i)
        for (int j = 0; j < dims[1]; ++j)
          for (int k = 0; k < dims[2]; ++k)
            field_Nplus1(i, j, k) = state_N(i, j, k) + field_Inc(i, j, k);
      break;
    case 4:
      for (int i = 0; i < dims[0]; ++i)
        for (int j = 0; j < dims[1]; ++j)
          for (int k = 0; k < dims[2]; ++k)
            for (int l = 0; l < dims[3]; ++l)
              field_Nplus1(i, j, k, l) =
                  state_N(i, j, k, l) + field_Inc(i, j, k, l);
      break;
    case 5:
      for (int i = 0; i < dims[0]; ++i)
        for (int j = 0; j < dims[1]; ++j)
          for (int k = 0; k < dims[2]; ++k)
            for (int l = 0; l < dims[3]; ++l)
              for (int m = 0; m < dims[4]; ++m)
                field_Nplus1(i, j, k, l, m) =
                    state_N(i, j, k, l, m) + field_Inc(i, j, k, l, m);
      break;
    default:
      TEUCHOS_TEST_FOR_EXCEPT_MSG(
          size < 1 || size > 5,
          "Unexpected Array dimensions in UpdateField: " << size);
  }
}

}  // namespace LCM
