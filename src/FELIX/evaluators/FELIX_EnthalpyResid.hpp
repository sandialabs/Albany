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

template<typename EvalT, typename Traits, typename VelocityType>
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
		//typedef VelocityType ParamScalarT;

		bool haveSUPG;
		double delta;

		// Input:
		PHX::MDField<MeshScalarT,Cell,Node,QuadPoint> wBF;
		PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> wGradBF;

		PHX::MDField<ScalarT,Cell,QuadPoint> Enthalpy;
		PHX::MDField<ScalarT,Cell,QuadPoint,Dim> EnthalpyGrad;
		PHX::MDField<ParamScalarT,Cell,QuadPoint> EnthalpyHs;

		PHX::MDField<VelocityType,Cell,QuadPoint,VecDim> Velocity;
        PHX::MDField<MeshScalarT,Cell,Node,Dim> coordVec;
		PHX::MDField<ScalarT,Cell,QuadPoint> diss;
		PHX::MDField<ParamScalarT,Cell,QuadPoint> basalFricHeat;
		PHX::MDField<ParamScalarT,Cell,QuadPoint> basalFricHeatSUPG;
		PHX::MDField<ParamScalarT,Cell,QuadPoint> geoFluxHeat;
		PHX::MDField<ParamScalarT,Cell,QuadPoint> geoFluxHeatSUPG;

		// Output:
		PHX::MDField<ScalarT,Cell,Node> Residual;

		unsigned int numQPs, numDims, numNodes;

		bool needsDiss, needsBasFric;

		double k_i, c_i, K_i;   // for computing K_i
		double K_0;	 // diffusivity temperate ice
		double rho; 	// density of ice
};

}


#endif /* FELIX_ENTHALPYRESID_HPP_ */
