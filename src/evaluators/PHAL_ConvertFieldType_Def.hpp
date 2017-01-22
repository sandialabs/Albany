//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"
//#include "Kokkos_DynRankView_Fad.hpp"

#include "PHAL_Workset.hpp"

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits, typename InputType, typename OutputType>
ConvertFieldType<EvalT, Traits, InputType, OutputType>::ConvertFieldType(const Teuchos::ParameterList& p) {
  std::string input_field_name, output_field_name;
  if(p.isParameter("Input Field Name") && p.isParameter("Output Field Name")) {
    input_field_name = p.get<std::string>("Input Field Name"); 
    output_field_name = p.get<std::string>("Output Field Name"); 
  } else if (p.isParameter("Field Name")) {
    input_field_name = output_field_name = p.get<std::string>("Field Name");
  } else TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, "Field Name not set.\n");
  in_field = PHX::MDField<InputType>(input_field_name, p.get<Teuchos::RCP<PHX::DataLayout> >("Data Layout"));
  out_field = PHX::MDField<OutputType>(output_field_name, p.get<Teuchos::RCP<PHX::DataLayout> >("Data Layout"));
    
  this->addDependentField(in_field);
  this->addEvaluatedField(out_field);

  this->setName("ConvertFieldType");
  
  PHX::MDField<InputType> in_field;
}

//**********************************************************************
template<typename EvalT, typename Traits, typename InputType, typename OutputType>
void ConvertFieldType<EvalT, Traits, InputType, OutputType>::postRegistrationSetup(
    typename Traits::SetupData d, PHX::FieldManager<Traits>& fm) {
  this->utils.setFieldData(in_field, fm);
  this->utils.setFieldData(out_field, fm);
}

//**********************************************************************
template<typename EvalT, typename Traits, typename InputType, typename OutputType>
void ConvertFieldType<EvalT, Traits, InputType, OutputType>::evaluateFields(
    typename Traits::EvalData workset) {
  std::vector<int> dims;
  in_field.dimensions(dims);
  int size = dims.size();
  switch (size) {
  case 1:
    for (int i = 0; i < dims[0]; ++i)
      out_field(i) = in_field(i);
    break;
  case 2:
    for (int i = 0; i < dims[0]; ++i)
      for (int j = 0; j < dims[1]; ++j)
        out_field(i, j) = in_field(i, j);
    break;
  case 3:
    for (int i = 0; i < dims[0]; ++i)
      for (int j = 0; j < dims[1]; ++j)
        for (int k = 0; k < dims[2]; ++k)
          out_field(i, j, k) = in_field(i, j, k);
    break;
  case 4:
    for (int i = 0; i < dims[0]; ++i)
      for (int j = 0; j < dims[1]; ++j)
        for (int k = 0; k < dims[2]; ++k)
          for (int l = 0; l < dims[3]; ++l)
            out_field(i, j, k, l) = in_field(i, j, k, l);
    break;
  case 5:
    for (int i = 0; i < dims[0]; ++i)
      for (int j = 0; j < dims[1]; ++j)
        for (int k = 0; k < dims[2]; ++k)
          for (int l = 0; l < dims[3]; ++l)
            for (int m = 0; m < dims[4]; ++m)
              out_field(i, j, k, l, m) = in_field(i, j, k, l, m);
    break;
  case 6:
    for (int i = 0; i < dims[0]; ++i)
      for (int j = 0; j < dims[1]; ++j)
        for (int k = 0; k < dims[2]; ++k)
          for (int l = 0; l < dims[3]; ++l)
            for (int m = 0; m < dims[4]; ++m)
              for (int n = 0; n < dims[5]; ++n)
              out_field(i, j, k, l, m, n) = in_field(i, j, k, l, m, n);
    break;
  default:
    TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, "Implemented only up to 6 dimensions in data layout.\n");
  }

//  all the switch above might be substituted in the future with the simple line:
//  Kokkos::deep_copy(out_field.get_view(),in_field.get_view());
}
//**********************************************************************

}

