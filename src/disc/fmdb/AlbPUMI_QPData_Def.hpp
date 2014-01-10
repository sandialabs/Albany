//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "AlbPUMI_QPData.hpp"

template<typename DataType, unsigned Dim, class traits>
AlbPUMI::QPData<DataType, Dim, traits>::QPData(const std::string& name_,
               const std::vector<int>& dim, const bool output_) :
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
AlbPUMI::QPData<DataType, Dim, traits>::reAllocateBuffer(const std::size_t nelems){

  std::size_t total_size = nelems * nfield_dofs;

  buffer.resize(total_size);

  beginning_index = 0;

  return;

}

template<typename DataType, unsigned Dim, class traits>
Albany::MDArray
AlbPUMI::QPData<DataType, Dim, traits>::getMDA(const std::size_t nelems){

  unsigned total_size = nelems * nfield_dofs;

  field_type the_array = traits_type::buildArray(&buffer[beginning_index], nelems, dims);

  beginning_index += total_size;

  return the_array;

}
