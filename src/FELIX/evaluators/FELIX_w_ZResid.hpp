/*
 * FELIX_VelocityZ.hpp
 *
 *  Created on: Jun 7, 2016
 *      Author: abarone
 */

#ifndef FELIX_VELOCITYZ_HPP_
#define FELIX_VELOCITYZ_HPP_

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace FELIX
{

template<typename EvalT, typename Traits, typename Type>
class w_ZResid : public PHX::EvaluatorWithBaseImpl<Traits>,
                  public PHX::EvaluatorDerived<EvalT, Traits>
{
	public:
		w_ZResid (const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl);

		void postRegistrationSetup (typename Traits::SetupData d, PHX::FieldManager<Traits>& fm);

		void evaluateFields(typename Traits::EvalData d);

	private:
		typedef typename EvalT::ScalarT ScalarT;
		typedef typename EvalT::MeshScalarT MeshScalarT;
		typedef typename EvalT::ParamScalarT ParamScalarT;

		// Input:
		PHX::MDField<MeshScalarT,Cell,Node,QuadPoint> wBF;
		PHX::MDField<ParamScalarT,Cell,QuadPoint,VecDim,Dim>  GradVelocity;
		PHX::MDField<ScalarT,Cell,QuadPoint> w_z;

		// Output
		PHX::MDField<ScalarT,Cell,Node> Residual;

		unsigned int numQPs, numDims, numNodes;

};

}	// Namespace FELIX




#endif /* FELIX_VELOCITYZ_HPP_ */
