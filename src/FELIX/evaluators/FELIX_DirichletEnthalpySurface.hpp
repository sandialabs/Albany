/*
 * FELIX_DirichletEnthalpySurface.hpp
 *
 *  Created on: Jun 6, 2016
 *      Author: abarone
 */

#ifndef FELIX_DIRICHLETENTHALPYSURFACE_HPP_
#define FELIX_DIRICHLETENTHALPYSURFACE_HPP_

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace FELIX
{

/** \brief Dirichlet Enthalpy Surface

    This evaluator computes the enthalpy field on the surface from the temperature used as Dirichlet condition
*/

template<typename EvalT, typename Traits>
class DirichletEnthalpySurface: public PHX::EvaluatorWithBaseImpl<Traits>,
                   	   	   	    public PHX::EvaluatorDerived<EvalT, Traits>
{
public:
  //typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::ParamScalarT ParamScalarT;
  //typedef typename EvalT::MeshScalarT MeshScalarT;

  DirichletEnthalpySurface (const Teuchos::ParameterList& p,
            	   	   	   	const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:
  // Input:
  PHX::MDField<ParamScalarT,Cell,Node> dirTemp;

  // Output:
  PHX::MDField<ParamScalarT,Cell,Node> dirEnth;

  int numNodes;

  double c_i;
  double T0;
};

} // Namespace FELIX

#endif /* FELIX_DIRICHLETENTHALPYSURFACE_HPP_ */


