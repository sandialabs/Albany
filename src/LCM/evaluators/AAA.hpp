/*
 * AAA.hpp
 *
 *  Created on: Aug 13, 2012
 *      Author: jrthune
 */

#ifndef AAA_HPP
#define AAA_HPP


#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

namespace LCM {
/** \brief Nearly Incompressible AAA model

    This evaluator computes the Cauchy stress based on a decoupled
    Helmholtz potential. Models a hyperelastic material for use in
    modeling Abdominal Aortic Aneurysm (AAA).

    Material model is given in Raghaven and Vorp, Journal of
    Biomechanics 33 (2000) 475-482.

    Special case of the generalized power law neo-Hookean
    model given by eg Zhang and Rajagopal, 1992.
 */

template<typename EvalT, typename Traits>
class AAA : public PHX::EvaluatorWithBaseImpl<Traits>,
  public PHX::EvaluatorDerived<EvalT,Traits> {

public:
  AAA(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
         PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> F;
  PHX::MDField<ScalarT,Cell,QuadPoint> J;
  RealType alpha;
  RealType beta;
  RealType mult;

  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> stress; // Cauchy stress

  unsigned int numQPs;
  unsigned int numDims;
  unsigned int worksetSize;

  // scratch space FCs
  Intrepid::FieldContainer<ScalarT> FT;

};

} // namespace LCM

#endif /* AAA_HPP */
