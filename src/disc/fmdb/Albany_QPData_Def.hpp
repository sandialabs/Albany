//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_QPData.hpp"

template<unsigned Dim, class traits>
Albany::QPData<Dim, traits>::QPData(const std::string& name_, const std::vector<int>& dim) :
  name(name_)
{

  dims = dim;

}

template<unsigned Dim, class traits>
Albany::QPData<Dim, traits>::~QPData(){

  for(std::size_t i = 0; i < shArray.size(); i++)

    delete shArray[i];

  for(std::size_t i = 0; i < buffer.size(); i++)

    delete [] buffer[i];

}

template<unsigned Dim, class traits>
//Albany::MDArray *
typename traits::field_type *
Albany::QPData<Dim, traits>::allocateArray(unsigned nelems){

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
Albany::QPData<Dim, traits>::allocateArray(double *buf, unsigned nelems){

  field_type *the_array = traits_type::buildArray(buf, nelems, dims);

  // save the pointers
  shArray.push_back(the_array);

  return the_array;

}
