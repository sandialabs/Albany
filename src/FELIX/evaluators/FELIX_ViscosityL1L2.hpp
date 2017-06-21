//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_VISCOSITYL1L2_HPP
#define FELIX_VISCOSITYL1L2_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace FELIX {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template<typename EvalT, typename Traits>
class ViscosityL1L2 : public PHX::EvaluatorWithBaseImpl<Traits>,
        public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  typedef typename EvalT::ScalarT ScalarT;

  ViscosityL1L2(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::MeshScalarT MeshScalarT;

  //coefficients for Glen's law
  double A;
  double n;

  //coefficients for ISMIP-HOM test cases
  double L;
  double alpha;

  std::size_t numQPsZ; //number of quadrature points for z-integral
  std::string surfType; //type of surface, e.g., Test A

  // Input:
  PHX::MDField<const MeshScalarT,Cell,QuadPoint,Dim> coordVec;
  PHX::MDField<const ScalarT,Cell,QuadPoint>         epsilonB;
  PHX::MDField<const ScalarT,Dim>                    homotopyParam;

  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint> mu;

  unsigned int numQPs, numDims, numNodes;

  enum VISCTYPE {CONSTANT, GLENSLAW};
  VISCTYPE visc_type;
  enum SURFTYPE {BOX, TESTA};
  SURFTYPE surf_type;

};
}

#endif
