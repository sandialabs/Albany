//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_HardeningLaw_hpp)
#define LCM_HardeningLaw_hpp

#include "../../../../utility/StaticAllocator.hpp"

namespace CP
{
/**
 *	Various types of hardening laws that can be used.
 *
 *	HardeningLawType can be used to drive the factory function hardeningLawFactory.
 */
enum class HardeningLawType
{
  UNDEFINED = 0, 
  LINEAR_MINUS_RECOVERY = 1, 
  SATURATION = 2, 
  DISLOCATION_DENSITY = 3
};


/**
 *	Factory returning a pointer to a hardening parameters object.
 *	
 *	\tparam	NumDimT		Static number of elements in a slip system
 *	\tparam NumSlipT	Static number of slip systems in a slip family
 *	\param type_hardening_law	Which hardening law to instantiate.
 */
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT>
std::shared_ptr<HardeningParameterBase<NumDimT, NumSlipT>>
hardeningParameterFactory(HardeningLawType type_hardening_law);


/**
 *	Hardening parameters base class.
 *
 *	Hardening parameters specify the various parameters used by a particular
 *	hardening law. All hardening law parameters derive from this class.
 *
 *	\tparam	NumDimT		Static number of elements in a slip system
 *	\tparam NumSlipT	Static number of slip systems in a slip family
 */
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT>
struct HardeningParameterBase
{
  using ParamIndex = int;
  
  virtual
  void
  createLatentMatrix(
    SlipFamily<NumDimT, NumSlipT> & slip_family, 
    std::vector<SlipSystem<NumDimT>> const & slip_systems) = 0;

  void
  setParameter(ParamIndex const index_param, RealType const value_param)
  {
    hardening_params_[index_param] = value_param;
  }

  RealType
  getParameter(ParamIndex const index_param)
  {
    return hardening_params_[index_param];
  }

  virtual
  ~HardeningParameterBase() {}

  std::map<std::string, ParamIndex>
  param_map_;

  Intrepid2::Vector<RealType>
  hardening_params_;
};


/**
 *	Parameters for the linear hardening with recovery law.
 *
 *	\tparam	NumDimT		Static number of elements in a slip system
 *	\tparam NumSlipT	Static number of slip systems in a slip family
 */
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT>
struct LinearMinusRecoveryHardeningParameters final :
  public HardeningParameterBase<NumDimT, NumSlipT>
{
  using ParamIndex = typename HardeningParameterBase<NumDimT, NumSlipT>::ParamIndex;

  enum HardeningParamTypes : ParamIndex
  {
    MODULUS_HARDENING,
    MODULUS_RECOVERY,
    STATE_HARDENING_INITIAL,
    NUM_PARAMS
  };

  LinearMinusRecoveryHardeningParameters()
  {
    this->param_map_["Hardening Modulus"] = MODULUS_HARDENING;
    this->param_map_["Recovery Modulus"] = MODULUS_RECOVERY;
    this->param_map_["Initial Hardening State"] = STATE_HARDENING_INITIAL;
    this->hardening_params_.set_dimension(NUM_PARAMS);
    this->hardening_params_.fill(Intrepid2::ZEROS);
  }

  virtual
  void
  createLatentMatrix(
    SlipFamily<NumDimT, NumSlipT> & slip_family, 
    std::vector<SlipSystem<NumDimT>> const & slip_systems) override;

  virtual
  ~LinearMinusRecoveryHardeningParameters() {}
};


/**
 *	Parameters for the saturation hardening law.
 *
 *	\tparam	NumDimT		Static number of elements in a slip system
 *	\tparam NumSlipT	Static number of slip systems in a slip family
 */
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT>
struct SaturationHardeningParameters final :
  public HardeningParameterBase<NumDimT, NumSlipT>
{
  using ParamIndex = typename HardeningParameterBase<NumDimT, NumSlipT>::ParamIndex;

  enum HardeningParamTypes : ParamIndex
  {
    RATE_HARDENING,
    STRESS_SATURATION_INITIAL,
    EXPONENT_SATURATION,
    RATE_SLIP_REFERENCE,
    STATE_HARDENING_INITIAL,
    NUM_PARAMS
  };

  SaturationHardeningParameters()
  {
    this->param_map_["Hardening Rate"] = RATE_HARDENING;
    this->param_map_["Initial Saturation Stress"] = STRESS_SATURATION_INITIAL;
    this->param_map_["Saturation Exponent"] = EXPONENT_SATURATION;
    this->param_map_["Reference Slip Rate"] = RATE_SLIP_REFERENCE;
    this->param_map_["Initial Hardening State"] = STATE_HARDENING_INITIAL;
    this->hardening_params_.set_dimension(NUM_PARAMS);
    this->hardening_params_.fill(Intrepid2::ZEROS);
  }

  virtual
  void
  createLatentMatrix(
    SlipFamily<NumDimT, NumSlipT> & slip_family, 
    std::vector<SlipSystem<NumDimT>> const & slip_systems) override;

  virtual
  ~SaturationHardeningParameters() {}
};


/**
 *	Parameters for the dislocation-density based hardening law.
 *
 *	\tparam	NumDimT		Static number of elements in a slip system
 *	\tparam NumSlipT	Static number of slip systems in a slip family
 */
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT>
struct DislocationDensityHardeningParameters final :
  public HardeningParameterBase<NumDimT, NumSlipT>
{
  using ParamIndex = typename HardeningParameterBase<NumDimT, NumSlipT>::ParamIndex;

  enum HardeningParamTypes : ParamIndex
  {
    FACTOR_GEOMETRY_DISLOCATION,
    FACTOR_GENERATION,
    FACTOR_ANNIHILATION,
    MODULUS_SHEAR,
    MAGNITUDE_BURGERS,
    STATE_HARDENING_INITIAL,
    NUM_PARAMS
  };

  DislocationDensityHardeningParameters()
  {
    this->param_map_["Geometric Factor"] = FACTOR_GEOMETRY_DISLOCATION;
    this->param_map_["Generation Factor"] = FACTOR_GENERATION;
    this->param_map_["Annihilation Factor"] = FACTOR_ANNIHILATION;
    this->param_map_["Shear Modulus"] = MODULUS_SHEAR;
    this->param_map_["Burgers Vector Magnitude"] = MAGNITUDE_BURGERS;
    this->param_map_["Initial Hardening State"] = STATE_HARDENING_INITIAL;
    this->hardening_params_.set_dimension(NUM_PARAMS);
    this->hardening_params_.fill(Intrepid2::ZEROS);
  }

  virtual
  void
  createLatentMatrix(
    SlipFamily<NumDimT, NumSlipT> & slip_family, 
    std::vector<SlipSystem<NumDimT>> const & slip_systems) override;

  virtual
  ~DislocationDensityHardeningParameters() {}
};


/**
 *	Parameters for no hardening.
 *
 *	\tparam	NumDimT		Static number of elements in a slip system
 *	\tparam NumSlipT	Static number of slip systems in a slip family
 */
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT>
struct NoHardeningParameters final :
  public HardeningParameterBase<NumDimT, NumSlipT>
{
  NoHardeningParameters()
  {
    return;
  }

  virtual
  void
  createLatentMatrix(
    SlipFamily<NumDimT, NumSlipT> & slip_family, 
    std::vector<SlipSystem<NumDimT>> const & slip_systems) override;

  virtual
  ~NoHardeningParameters() {}
};


/**
 *  Base class for hardening laws.
 *
 *	\tparam	NumDimT		Static number of elements in a slip system
 *	\tparam NumSlipT	Static number of slip systems in a slip family
 *	\tparam ArgT		Scalar type used for hardening
 */
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename ArgT>
struct HardeningLawBase
{
  HardeningLawBase() {}

  virtual
  void
  harden(
    SlipFamily<NumDimT, NumSlipT> const & slip_family,
    std::vector<SlipSystem<NumDimT>> const & slip_systems,
    RealType dt,
    Intrepid2::Vector<ArgT, NumSlipT> const & rate_slip,
    Intrepid2::Vector<RealType, NumSlipT> const & state_hardening_n,
    Intrepid2::Vector<ArgT, NumSlipT> & state_hardening_np1,
    Intrepid2::Vector<ArgT, NumSlipT> & slip_resistance) = 0;

  virtual
  ~HardeningLawBase() {}
};


/**
 *  Factory class for instantiating hardening laws.
 */
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT>
class HardeningLawFactory
{

public:

  explicit HardeningLawFactory();

  template<typename ArgT>
  utility::StaticPointer<HardeningLawBase<NumDimT, NumSlipT, ArgT>>
  createHardeningLaw(HardeningLawType type_hardening_law) const;

private:

  mutable utility::StaticStackAllocator<sizeof(std::uintptr_t)> allocator_;
};


/**
 *  Linear hardening with recovery law.
 *
 *	\tparam	NumDimT		Static number of elements in a slip system
 *	\tparam NumSlipT	Static number of slip systems in a slip family
 *	\tparam ArgT		Scalar type used for hardening
 */
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename ArgT>
struct LinearMinusRecoveryHardeningLaw final : public HardeningLawBase<NumDimT, NumSlipT, ArgT>
{
  virtual
  void
  harden(
    SlipFamily<NumDimT, NumSlipT> const & slip_family, 
    std::vector<SlipSystem<NumDimT>> const & slip_systems,
    RealType dt,
    Intrepid2::Vector<ArgT, NumSlipT> const & rate_slip,
    Intrepid2::Vector<RealType, NumSlipT> const & state_hardening_n,
    Intrepid2::Vector<ArgT, NumSlipT> & state_hardening_np1,
    Intrepid2::Vector<ArgT, NumSlipT> & slip_resistance);

  virtual
  ~LinearMinusRecoveryHardeningLaw() {}
};


/**
 *  Saturation hardening law.
 *
 *	\tparam	NumDimT		Static number of elements in a slip system
 *	\tparam NumSlipT	Static number of slip systems in a slip family
 *	\tparam ArgT		Scalar type used for hardening
 */
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename ArgT>
struct SaturationHardeningLaw final : public HardeningLawBase<NumDimT, NumSlipT, ArgT>
{
  virtual
  void
  harden(
    SlipFamily<NumDimT, NumSlipT> const & slip_family,
    std::vector<SlipSystem<NumDimT>> const & slip_systems,
    RealType dt,
    Intrepid2::Vector<ArgT, NumSlipT> const & rate_slip,
    Intrepid2::Vector<RealType, NumSlipT> const & state_hardening_n,
    Intrepid2::Vector<ArgT, NumSlipT> & state_hardening_np1,
    Intrepid2::Vector<ArgT, NumSlipT> & slip_resistance);

  virtual
  ~SaturationHardeningLaw() {}
};


/**
 *  Dislocation density hardening law.
 *
 *	\tparam	NumDimT		Static number of elements in a slip system
 *	\tparam NumSlipT	Static number of slip systems in a slip family
 *	\tparam ArgT		Scalar type used for hardening
 */
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename ArgT>
struct DislocationDensityHardeningLaw final : public HardeningLawBase<NumDimT, NumSlipT, ArgT>
{
  virtual
  void
  harden(
    SlipFamily<NumDimT, NumSlipT> const & slip_family,
    std::vector<SlipSystem<NumDimT>> const & slip_systems,
    RealType dt,
    Intrepid2::Vector<ArgT, NumSlipT> const & rate_slip,
    Intrepid2::Vector<RealType, NumSlipT> const & state_hardening_n,
    Intrepid2::Vector<ArgT, NumSlipT> & state_hardening_np1,
    Intrepid2::Vector<ArgT, NumSlipT> & slip_resistance);

  virtual
  ~DislocationDensityHardeningLaw() {}
};


/**
 *  No hardening law.
 *
 *	\tparam	NumDimT		Static number of elements in a slip system
 *	\tparam NumSlipT	Static number of slip systems in a slip family
 *	\tparam ArgT		Scalar type used for hardening
 */
template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename ArgT>
struct NoHardeningLaw final : public HardeningLawBase<NumDimT, NumSlipT, ArgT>
{
  virtual
  void
  harden(
    SlipFamily<NumDimT, NumSlipT> const & slip_family,
    std::vector<SlipSystem<NumDimT>> const & slip_systems,
    RealType dt,
    Intrepid2::Vector<ArgT, NumSlipT> const & rate_slip,
    Intrepid2::Vector<RealType, NumSlipT> const & state_hardening_n,
    Intrepid2::Vector<ArgT, NumSlipT> & state_hardening_np1,
    Intrepid2::Vector<ArgT, NumSlipT> & slip_resistance);

  virtual
  ~NoHardeningLaw() {}
};
}

#include "HardeningLaw_Def.hpp"

#endif
