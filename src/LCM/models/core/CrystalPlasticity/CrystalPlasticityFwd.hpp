//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(CrystalPlasticityFwd_hpp)
#define CrystalPlasticityFwd_hpp

namespace CP
{
static constexpr RealType
MACHINE_EPS = std::numeric_limits<RealType>::epsilon();

static constexpr RealType
TINY = std::numeric_limits<RealType>::min();

static constexpr RealType
HUGE_ = std::numeric_limits<RealType>::max();

static const RealType
LOG_HUGE = std::log(HUGE_);

static constexpr minitensor::Index 
MAX_DIM = 3;

static constexpr minitensor::Index
MAX_SLIP = 48;

static constexpr minitensor::Index
MAX_FAMILY = 3;

template<minitensor::Index NumSlipT>
struct NlsDim
{
  static constexpr minitensor::Index value{NumSlipT * 2};
};

static constexpr minitensor::Index
NLS_DIM = NlsDim<MAX_SLIP>::value;

enum class IntegrationScheme
{
  UNDEFINED = 0, 
  EXPLICIT = 1, 
  IMPLICIT = 2
};

enum class ResidualType
{
  UNDEFINED = 0, 
  SLIP = 1, 
  SLIP_HARDNESS = 2
};

struct FlowParameterBase;

template<minitensor::Index NumDimT, minitensor::Index NumSlipT>
struct HardeningParameterBase;

template<minitensor::Index NumDimT>
struct SlipSystem;

template<minitensor::Index NumDimT, minitensor::Index NumSlipT>
struct SlipFamily;
}

#endif
