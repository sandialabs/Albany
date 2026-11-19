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
  in_field = decltype(in_field)(input_field_name, p.get<Teuchos::RCP<PHX::DataLayout> >("Data Layout"));
  out_field = decltype(out_field)(output_field_name, p.get<Teuchos::RCP<PHX::DataLayout> >("Data Layout"));
    
  this->addDependentField(in_field);
  this->addEvaluatedField(out_field);

  this->setName("ConvertFieldType");
}

//**********************************************************************
template<typename EvalT, typename Traits, typename InputType, typename OutputType>
void ConvertFieldType<EvalT, Traits, InputType, OutputType>::
postRegistrationSetup(typename Traits::SetupData /* d */,
                      PHX::FieldManager<Traits>& fm) {
  this->utils.setFieldData(in_field, fm);
  this->utils.setFieldData(out_field, fm);
}

//**********************************************************************
template<typename EvalT, typename Traits, typename InputType, typename OutputType>
void ConvertFieldType<EvalT, Traits, InputType, OutputType>::
evaluateFields(typename Traits::EvalData /* workset */) {
  using ExecutionSpace = PHX::Device::execution_space;
  using Policy2 = Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2>>;
  using Policy3 = Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<3>>;
  using Policy4 = Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<4>>;
  using Policy5 = Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<5>>;
  using Policy6 = Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<6>>;

  std::vector<int> dims;
  in_field.dimensions(dims);
  int size = dims.size();

  // The fields live in device memory: do the copy (with type conversion) in a kernel
  const auto in  = in_field.get_view();
  const auto out = out_field.get_view();
  const std::string name = this->getName();
  switch (size) {
  case 1:
    Kokkos::parallel_for(name, Kokkos::RangePolicy<ExecutionSpace>(0,dims[0]),
                         KOKKOS_LAMBDA(const int i) {
      out(i) = in(i);
    });
    break;
  case 2:
    Kokkos::parallel_for(name, Policy2({0,0},{dims[0],dims[1]}),
                         KOKKOS_LAMBDA(const int i, const int j) {
      out(i, j) = in(i, j);
    });
    break;
  case 3:
    Kokkos::parallel_for(name, Policy3({0,0,0},{dims[0],dims[1],dims[2]}),
                         KOKKOS_LAMBDA(const int i, const int j, const int k) {
      out(i, j, k) = in(i, j, k);
    });
    break;
  case 4:
    Kokkos::parallel_for(name, Policy4({0,0,0,0},{dims[0],dims[1],dims[2],dims[3]}),
                         KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
      out(i, j, k, l) = in(i, j, k, l);
    });
    break;
  case 5:
    Kokkos::parallel_for(name, Policy5({0,0,0,0,0},{dims[0],dims[1],dims[2],dims[3],dims[4]}),
                         KOKKOS_LAMBDA(const int i, const int j, const int k, const int l, const int m) {
      out(i, j, k, l, m) = in(i, j, k, l, m);
    });
    break;
  case 6:
    Kokkos::parallel_for(name, Policy6({0,0,0,0,0,0},{dims[0],dims[1],dims[2],dims[3],dims[4],dims[5]}),
                         KOKKOS_LAMBDA(const int i, const int j, const int k, const int l, const int m, const int n) {
      out(i, j, k, l, m, n) = in(i, j, k, l, m, n);
    });
    break;
  default:
    TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, "Implemented only up to 6 dimensions in data layout.\n");
  }
}
//**********************************************************************

}

