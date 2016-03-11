//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/12/14: no Epetra

#include "Albany_PUMIQPData.hpp"

template<typename DataType, unsigned Dim, class traits>
Albany::PUMIQPData<DataType, Dim, traits>::PUMIQPData(const std::string& name_,
               const std::vector<PHX::DataLayout::size_type>& dim, const bool output_) :
  name(name_),
  output(output_),
  dims(dim),
  nfield_dofs(1),
  beginning_index(0)
{

  for(std::size_t i = 1; i < dims.size(); i++) // multiply it by the number of dofs per node

    nfield_dofs *= dims[i];

}

template<typename DataType, unsigned Dim, class traits>
void
Albany::PUMIQPData<DataType, Dim, traits>::reAllocateBuffer(const std::size_t nelems){

  std::size_t total_size = nelems * nfield_dofs;

  buffer.resize(total_size);

  beginning_index = 0;

  return;

}

template<typename DataType, unsigned Dim, class traits>
Albany::MDArray
Albany::PUMIQPData<DataType, Dim, traits>::getMDA(const std::size_t nelems){

  unsigned total_size = nelems * nfield_dofs;

  field_type the_array = traits_type::buildArray(&buffer[beginning_index], nelems, dims);

  beginning_index += total_size;

  return the_array;

}
