/*
 * FELIX_BasalFrictionHeat.hpp
 *
 *  Created on: May 25, 2016
 *      Author: abarone
 */

#ifndef FELIX_BASALFRICTIONHEAT_HPP_
#define FELIX_BASALFRICTIONHEAT_HPP_

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace FELIX
{
/** \brief Basal friction heat Evaluator

    This evaluator evaluates the production of heat caused by basal friction
*/
template<typename EvalT, typename Traits>
class BasalFrictionHeat : public PHX::EvaluatorWithBaseImpl<Traits>,
                    	  public PHX::EvaluatorDerived<EvalT, Traits>
{
	public:
		BasalFrictionHeat (const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl);

		void postRegistrationSetup (typename Traits::SetupData d, PHX::FieldManager<Traits>& fm);

		void evaluateFields(typename Traits::EvalData d);

	private:
		typedef typename EvalT::ScalarT ScalarT;
		typedef typename EvalT::MeshScalarT MeshScalarT;
		typedef typename EvalT::ParamScalarT ParamScalarT;

		// Input:
		PHX::MDField<ScalarT,Cell,Side,QuadPoint> beta;
		PHX::MDField<ScalarT,Cell,Side,QuadPoint,VecDim> velocity;

		// Output:
		PHX::MDField<ScalarT,Cell,Node> basalFricHeat;

		std::vector<std::vector<int> >  sideNodes;
		std::string                     basalSideName;

		int numCellNodes;
		int numSideNodes;
		int numSideQPs;
		int numCellQPs;
		int sideDim;
		int vecDim;
		int vecDimFO;
};

}	// Namespace FELIX

#endif /* FELIX_BASALFRICTIONHEAT_HPP_ */
