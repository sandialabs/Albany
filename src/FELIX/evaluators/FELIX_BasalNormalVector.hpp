/*
 * FELIX_BasalNormalVector.hpp
 *
 *  Created on: Jun 9, 2016
 *      Author: abarone
 */

#ifndef FELIX_BASALNORMALVECTOR_HPP_
#define FELIX_BASALNORMALVECTOR_HPP_


#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace FELIX
{
/** \brief Basal normal vector Evaluator

    This evaluator computes the normal vector at the base
*/
template<typename EvalT, typename Traits>
class BasalNormalVector : public PHX::EvaluatorWithBaseImpl<Traits>,
                    	  public PHX::EvaluatorDerived<EvalT, Traits>
{
	public:
		BasalNormalVector (const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl);

		void postRegistrationSetup (typename Traits::SetupData d, PHX::FieldManager<Traits>& fm);

		void evaluateFields(typename Traits::EvalData d);

	private:
		typedef typename EvalT::ScalarT ScalarT;
		//typedef typename EvalT::MeshScalarT MeshScalarT;
		typedef typename EvalT::ParamScalarT ParamScalarT;

		// Input:
		PHX::MDField<ParamScalarT,Cell,Side,QuadPoint,Dim>    gradSurfHeight;
		PHX::MDField<ParamScalarT,Cell,Side,QuadPoint,Dim>    gradThickness;

		// Output:
		PHX::MDField<ParamScalarT,Cell,QuadPoint,Dim>	  normal;

		std::vector<std::vector<int> >  sideNodes;
		std::string                     basalSideName;

		int numSideNodes;
		int numSideQPs;
		int sideDim;

		int numNodes, numQPs, numDims;

};

}	// Namespace FELIX

#endif /* FELIX_BASALNORMALVECTOR_HPP_ */
