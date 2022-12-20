//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_VISCOSITY_HPP
#define LANDICE_VISCOSITY_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace LandIce {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template<typename EvalT, typename Traits>
class Viscosity : public PHX::EvaluatorWithBaseImpl<Traits>,
                  public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  typedef typename EvalT::ScalarT ScalarT;

  Viscosity(const Teuchos::ParameterList& p,
            const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::MeshScalarT MeshScalarT;

  //coefficients for Glen's law
  double A;
  double n;

  // Input:
  PHX::MDField<const ScalarT,Cell,QuadPoint,Dim,Dim> VGrad;
  PHX::MDField<const MeshScalarT,Cell,QuadPoint,Dim> coordVec;
  PHX::MDField<const ScalarT,Dim>                    homotopyParam;

  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint> mu;

  unsigned int numQPs, numDims, numNodes;

  enum VISCTYPE {CONSTANT, GLENSLAW};
  VISCTYPE visc_type;
};

} // Namespace LandIce

#endif // LANDICE_VISCOSITY_HPP
