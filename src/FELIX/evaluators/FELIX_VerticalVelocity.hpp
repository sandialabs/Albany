/*
 * FELIX_VerticalVelocity.hpp
 *
 *  Created on: Jun 15, 2016
 *      Author: abarone
 */

#ifndef FELIX_VERTICALVELOCITY_HPP_
#define FELIX_VERTICALVELOCITY_HPP_

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace FELIX
{

template<typename EvalT, typename Traits, typename VelocityType>
class VerticalVelocity : public PHX::EvaluatorWithBaseImpl<Traits>,
					     public PHX::EvaluatorDerived<EvalT, Traits>
{
	public:

		VerticalVelocity(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl);

		void postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& vm);

		void evaluateFields(typename Traits::EvalData d);

	private:
		typedef typename EvalT::ScalarT ScalarT;
		typedef typename EvalT::ParamScalarT ParamScalarT;

		// Input:
		//PHX::MDField<ParamScalarT,Cell,Side,QuadPoint> basalMeltRate;
		//PHX::MDField<ScalarT,Cell,Side,QuadPoint> int1Ddrainage;
		//PHX::MDField<ParamScalarT,Cell,Side,QuadPoint,Dim> surface_height_grad;
		//PHX::MDField<ParamScalarT,Cell,Side,QuadPoint,Dim> thickness_grad;
		//PHX::MDField<VelocityType,Cell,Side,QuadPoint,Dim> velocity;
		PHX::MDField<ParamScalarT,Cell,Node> thickness;
		PHX::MDField<ScalarT,Cell,Node> int1Dw_z;

		// Output:
		PHX::MDField<ScalarT,Cell,Node> w;

		unsigned int numNodes;

		//std::vector<std::vector<int> >  sideNodes;
		//std::string                     basalSideName;

		//int numCellNodes, numSideNodes, numSideQPs, sideDim;
};

}

#endif /* FELIX_VERTICALVELOCITY_HPP_ */
