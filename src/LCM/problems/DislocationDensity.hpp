//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#if !defined(DislocationDensity_hpp)
#define DislocationDensity_hpp

namespace LCM
{

namespace DislocationDensity
{

const
int
NUM_SLIP_D1 = 2;

const
int
NUM_SLIP_D2 = 4;

const
int
NUM_SLIP_D3 = 24;


constexpr
int
get_num_slip(int const num_dims)
{
  return num_dims == 3 ? NUM_SLIP_D3 : num_dims == 2 ? NUM_SLIP_D2 : NUM_SLIP_D1;
}

}
}

#endif //!defined(DislocationDensity_hpp)
