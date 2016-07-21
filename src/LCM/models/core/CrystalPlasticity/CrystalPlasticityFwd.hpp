//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(CrystalPlasticityFwd_hpp)
#define CrystalPlasticityFwd_hpp

namespace CP
{
static constexpr Intrepid2::Index 
MAX_DIM = 3;

static constexpr Intrepid2::Index
MAX_SLIP = 12;

static constexpr Intrepid2::Index
NLS_DIM = 2 * MAX_SLIP;

static constexpr Intrepid2::Index
MAX_FAMILY = 3;


struct FlowParameterBase;

template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT>
struct HardeningParameterBase;

template<Intrepid2::Index NumDimT>
struct SlipSystem;

template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT>
struct SlipFamily;
}

#endif