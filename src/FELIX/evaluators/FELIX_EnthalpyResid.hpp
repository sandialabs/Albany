/*
 * FELIX_EnthalpyResid.hpp
 *
 *  Created on: May 11, 2016
 *      Author: abarone
 */

#ifndef FELIX_ENTHALPYRESID_HPP_
#define FELIX_ENTHALPYRESID_HPP_

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace FELIX
{

template<typename EvalT, typename Traits>
class EnthalpyResid : public PHX::EvaluatorWithBaseImpl<Traits>,
					  public PHX::EvaluatorDerived<EvalT, Traits>
{
	public:

		EnthalpyResid(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl);

		void postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& vm);

		void evaluateFields(typename Traits::EvalData d);

	private:
		typedef typename EvalT::ScalarT ScalarT;
		typedef typename EvalT::MeshScalarT MeshScalarT;
		typedef typename EvalT::ParamScalarT ParamScalarT;

		ParamScalarT k;   //viscosity coefficient

		bool haveSUPG;
		ParamScalarT delta;

		// Input:
		PHX::MDField<MeshScalarT,Cell,Node,QuadPoint> wBF;
		PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> wGradBF;

		PHX::MDField<ScalarT,Cell,QuadPoint> Enthalpy;
		PHX::MDField<ScalarT,Cell,QuadPoint,Dim> EnthalpyGrad;

		PHX::MDField<ParamScalarT,Cell,QuadPoint,VecDim> Velocity;
        PHX::MDField<MeshScalarT,Cell,Node,Dim> coordVec;
		PHX::MDField<ScalarT,Cell,QuadPoint> diss;

		// Output:
		PHX::MDField<ScalarT,Cell,Node> Residual;

		unsigned int numQPs, numDims, numNodes;

};

}


#endif /* FELIX_ENTHALPYRESID_HPP_ */
