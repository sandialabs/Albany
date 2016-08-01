//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "StaticAllocator.hpp"

using namespace utility;

StaticAllocator::StaticAllocator(std::size_t size)
  : size_(size), buffer_(new unsigned char[size]), ptr_(buffer_)
{
  
}

StaticAllocator::~StaticAllocator()
{
  delete[] buffer_;
}


