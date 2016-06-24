/*
 * FELIX_BasalMeltRate.hpp
 *
 *  Created on: Jun 16, 2016
 *      Author: abarone
 */

#ifndef FELIX_BASALMELTRATE_HPP_
#define FELIX_BASALMELTRATE_HPP_

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace FELIX
{

template<typename EvalT, typename Traits, typename VelocityType>
class BasalMeltRate : public PHX::EvaluatorWithBaseImpl<Traits>,
					  public PHX::EvaluatorDerived<EvalT, Traits>
{
	public:

		BasalMeltRate(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl_basal);

		void postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& vm);

		void evaluateFields(typename Traits::EvalData d);

	private:
		typedef typename EvalT::ScalarT ScalarT;
		typedef typename EvalT::MeshScalarT MeshScalarT;
		typedef typename EvalT::ParamScalarT ParamScalarT;
		//typedef VelocityType ParamScalarT;

		// Input:
		PHX::MDField<ParamScalarT,Cell,Side,Node> basalNormalHeatCold;
		PHX::MDField<ParamScalarT,Cell,Side,Node> basalNormalHeatTemperate;
		PHX::MDField<ParamScalarT,Cell,Side,Node> omega;
		PHX::MDField<ParamScalarT,Cell,Side,Node> basal_heat_flux;

		PHX::MDField<VelocityType,Cell,Side,Node,Dim> velocity;
		PHX::MDField<ParamScalarT,Cell,Side,Node> basal_friction;

		PHX::MDField<ParamScalarT,Cell,Side,Node> meltEnthalpy;
		PHX::MDField<ParamScalarT,Cell,Side,Node> Enthalpy;

		// Output:
		PHX::MDField<ScalarT,Cell,Side,Node> basalMeltRate;

		//unsigned int numQPs, numNodes;

		std::vector<std::vector<int> >  sideNodes;
		std::string                     basalSideName;

		int numCellNodes, numSideNodes, numSideQPs, sideDim;

		double L;
};

}




#endif /* FELIX_BASALMELTRATE_HPP_ */
