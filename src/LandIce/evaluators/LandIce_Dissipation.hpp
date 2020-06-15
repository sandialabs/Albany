/*
 * LandIce_Dissipation.hpp
 *
 *  Created on: May 19, 2016
 *      Author: abarone
 */

#ifndef LANDICE_DISSIPATION_HPP_
#define LANDICE_DISSIPATION_HPP_

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"
#include "PHAL_Utilities.hpp"

namespace LandIce
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
		typedef typename EvalT::ParamScalarT ParamScalarT;

		// Input:
		PHX::MDField<const ScalarT,Cell,QuadPoint> mu; // [k^2 Pa yr], k=1000
		PHX::MDField<const ScalarT,Cell,QuadPoint> epsilonSq; // [(k yr)^{-2}],  k=1000

		// Output:
		PHX::MDField<ScalarT,Cell,QuadPoint> diss; // [W m^{-3}] = [Pa s^{-1}]

		double scyr ;    // [s/yr] (3.1536e7);

		unsigned int numQPs;
    	
		PHAL::MDFieldMemoizer<Traits> memoizer;

		typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;

		typedef Kokkos::MDRangePolicy< ExecutionSpace, Kokkos::Rank<2> > Dissipation_Policy;

		KOKKOS_INLINE_FUNCTION
		void operator() (const int& i, const int& j) const;
};

}	// Namespace LandIce

#endif /* LandIce_DISSIPATION_HPP_ */
