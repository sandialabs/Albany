//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_CrystalPlasticityModel_ParameterReader_hpp)
#define LCM_CrystalPlasticityModel_ParameterReader_hpp

#include <Teuchos_ParameterList.hpp>
#include <MiniTensor_Solvers.h>
#include "CrystalPlasticityFwd.hpp"
#include "../../../../utility/ParameterEnum.hpp"
#include "../../../../utility/StaticAllocator.hpp"

namespace CP
{
  template<typename EvalT, typename Traits>
  class ParameterReader
  {
  public:

    using ScalarT = typename EvalT::ScalarT;
    using ValueT = typename Sacado::ValueType<ScalarT>::type;
    using Minimizer = minitensor::Minimizer<ValueT, CP::NLS_DIM>;
    using RolMinimizer = ROL::MiniTensor_Minimizer<ValueT, CP::NLS_DIM>;

    ParameterReader(Teuchos::ParameterList* p);

    IntegrationScheme
    getIntegrationScheme() const;

    ResidualType
    getResidualType() const;

    PredictorSlip
    getPredictorSlip() const;

    minitensor::StepType
    getStepType() const;

    Minimizer
    getMinimizer() const;

    RolMinimizer
    getRolMinimizer() const;

    SlipFamily<CP::MAX_DIM, CP::MAX_SLIP>
    getSlipFamily(int index);

    Verbosity
    getVerbosity() const;

  private:

    Teuchos::ParameterList* p_;
  };
}

#include "ParameterReader_Def.hpp"

#endif

