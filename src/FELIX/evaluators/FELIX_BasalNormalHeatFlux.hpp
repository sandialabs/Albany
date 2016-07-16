/*
 * FELIX_BasalNormalHeatFlux.hpp
 *
 *  Created on: Jun 15, 2016
 *      Author: abarone
 */

#ifndef FELIX_BASALNORMALHEATFLUX_HPP_
#define FELIX_BASALNORMALHEATFLUX_HPP_

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace FELIX
{
/** \brief Basal normal heat flux Evaluator

    This evaluator computes the heat flux at the base (useful for computing the basal melt rate)
*/
template<typename EvalT, typename Traits>
class BasalNormalHeatFlux : public PHX::EvaluatorWithBaseImpl<Traits>,
                    	    public PHX::EvaluatorDerived<EvalT, Traits>
{
	public:
		BasalNormalHeatFlux (const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl);

		void postRegistrationSetup (typename Traits::SetupData d, PHX::FieldManager<Traits>& fm);

		void evaluateFields(typename Traits::EvalData d);

	private:
		typedef typename EvalT::ScalarT ScalarT;
		//typedef typename EvalT::MeshScalarT MeshScalarT;
		typedef typename EvalT::ParamScalarT ParamScalarT;

		// Input:
		PHX::MDField<ParamScalarT,Cell,QuadPoint,Dim>    gradMeltTemp;
		PHX::MDField<ScalarT,Cell,QuadPoint,Dim>   	  gradEnthalpy;
		PHX::MDField<ParamScalarT,Cell,QuadPoint,Dim>    normal;

		// Output:
		PHX::MDField<ScalarT,Cell,Node>	  basalNormalHeatCold;
		PHX::MDField<ScalarT,Cell,Node>	  basalNormalHeatTemperate;

		int numNodes, numQPs, numDims;
};

}

#endif /* FELIX_BASALNORMALHEATFLUX_HPP_ */
