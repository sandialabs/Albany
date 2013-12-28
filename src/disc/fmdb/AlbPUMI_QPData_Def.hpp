//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "AlbPUMI_QPData.hpp"

template<unsigned Dim, class traits>
AlbPUMI::QPData<Dim, traits>::QPData(const std::string& name_, const std::vector<int>& dim) :
  name(name_),
  dims(dim),
  nfield_dofs(1),
  beginning_index(0)
{

  for(std::size_t i = 1; i < dims.size(); i++) // multiply it by the number of dofs per node

    nfield_dofs *= dims[i];

}

template<unsigned Dim, class traits>
AlbPUMI::QPData<Dim, traits>::~QPData(){

/*
  // clear array of pointers into the buffer
  while (!shArray.empty()) {
   delete shArray.back();  
   shArray.pop_back();
  }
*/

}

template<unsigned Dim, class traits>
void
AlbPUMI::QPData<Dim, traits>::reAllocateBuffer(const std::size_t nelems){

  unsigned total_size = nelems * nfield_dofs;

  buffer.resize(total_size); 

  beginning_index = 0;

/*
  // clear array of pointers into the buffer
  while (!shArray.empty()) {
   delete shArray.back();  
   shArray.pop_back();
  }
*/

  return;

}

template<unsigned Dim, class traits>
//typename traits::field_type*
Albany::MDArray
AlbPUMI::QPData<Dim, traits>::getMDA(const std::size_t nelems){

  unsigned total_size = nelems * nfield_dofs;

  field_type the_array = traits_type::buildArray(&buffer[beginning_index], nelems, dims);

  beginning_index += total_size;

  return the_array;

}

/*
template<unsigned Dim, class traits>
//Albany::MDArray *
typename traits::field_type *
AlbPUMI::QPData<Dim, traits>::allocateArray(unsigned nelems){

  unsigned total_size = nelems;
  for(std::size_t i = 1; i < dims.size(); i++)

    total_size *= dims[i];

  double *buf = new double[total_size]; 

  field_type *the_array = traits_type::buildArray(buf, nelems, dims);

  // save the pointers
  buffer.push_back(buf);
  shArray.push_back(the_array);

  return the_array;

}

template<unsigned Dim, class traits>
//Albany::MDArray *
typename traits::field_type *
AlbPUMI::QPData<Dim, traits>::allocateArray(double *buf, unsigned nelems){

  field_type *the_array = traits_type::buildArray(buf, nelems, dims);

  // save the pointers
  shArray.push_back(the_array);

  return the_array;

}
*/
