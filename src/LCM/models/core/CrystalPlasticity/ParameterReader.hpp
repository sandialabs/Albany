//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_CrystalPlasticityModel_ParameterReader_hpp)
#define LCM_CrystalPlasticityModel_ParameterReader_hpp

#include <Teuchos_ParameterList.hpp>
#include <Intrepid2_MiniTensor_Solvers.h>
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

		using Minimizer = Intrepid2::Minimizer<ValueT, CP::NLS_DIM>;

		ParameterReader(Teuchos::ParameterList* p, utility::StaticAllocator & alloc);
		
		IntegrationScheme	getIntegrationScheme() const;
		ResidualType getResidualType() const;
		Intrepid2::StepType getStepType() const;
		Minimizer getMinimizer() const;

		SlipFamily<CP::MAX_DIM, CP::MAX_SLIP> getSlipFamily(int index);

	private:

		Teuchos::ParameterList* p_;
    utility::StaticAllocator & allocator_;
	};
}

#include "ParameterReader_Def.hpp"

#endif

