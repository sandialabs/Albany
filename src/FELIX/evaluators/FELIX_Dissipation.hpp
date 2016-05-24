/*
 * FELIX_Dissipation.hpp
 *
 *  Created on: May 19, 2016
 *      Author: abarone
 */

#ifndef FELIX_DISSIPATION_HPP_
#define FELIX_DISSIPATION_HPP_

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace FELIX
{
/** \brief Dissipation Evaluator

    This evaluator evaluates the dissipation due to deformation
*/

template<typename EvalT, typename Traits>
class Dissipation : public PHX::EvaluatorWithBaseImpl<Traits>,
                    public PHX::EvaluatorDerived<EvalT, Traits>
{
	public:
		Dissipation (const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl);

		void postRegistrationSetup (typename Traits::SetupData d, PHX::FieldManager<Traits>& fm);

		void evaluateFields(typename Traits::EvalData d);

	private:
		typedef typename EvalT::ScalarT ScalarT;
		typedef typename EvalT::MeshScalarT MeshScalarT;
		typedef typename EvalT::ParamScalarT ParamScalarT;

		// Input:
		PHX::MDField<ScalarT,Cell,QuadPoint> mu;
		PHX::MDField<ScalarT,Cell,QuadPoint> epsilonSq;

		// Output:
		PHX::MDField<ScalarT,Cell,QuadPoint> diss;

		unsigned int numQPs;
};

}	// Namespace FELIX

#endif /* FELIX_DISSIPATION_HPP_ */
