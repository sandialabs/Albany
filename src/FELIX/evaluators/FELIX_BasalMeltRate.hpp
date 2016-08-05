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

		// Input:
		PHX::MDField<ScalarT,Cell,Side,Node> 				phi;
		PHX::MDField<ParamScalarT,Cell,Side,Node>    		beta;
		PHX::MDField<VelocityType,Cell,Side,Node,VecDim>	velocity;
		PHX::MDField<ParamScalarT,Cell,Side,Node> 			geoFluxHeat;
		PHX::MDField<ScalarT,Cell,Side,Node> 				Enthalpy;
		PHX::MDField<ParamScalarT,Cell,Side,Node> 			EnthalpyHs;

		// Output:
		PHX::MDField<ScalarT,Cell,Side,Node> basalMeltRate;

		PHX::MDField<ScalarT,Dim> homotopy;

		std::vector<std::vector<int> >  sideNodes;
		std::string                     basalSideName;

		int numCellNodes, numSideNodes, sideDim;

		double rho_w; 	// density of water
		double rho_i; 	// density of ice
		double L, g, a;

		double k_0, eta_w;
		double alpha_om;

};

}




#endif /* FELIX_BASALMELTRATE_HPP_ */
