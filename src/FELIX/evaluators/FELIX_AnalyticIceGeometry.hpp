//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_ANALYTIC_ICE_GEOMETRY_HPP
#define FELIX_ANALYTIC_ICE_GEOMETRY_HPP 1

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

#include <vector>

namespace FELIX
{

/** \brief Hydrology Residual Evaluator

    This evaluator evaluates the residual of the Hydrology model
*/

template<typename EvalT, typename Traits>
class AnalyticIceGeometry : public PHX::EvaluatorWithBaseImpl<Traits>,
                          public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  typedef typename EvalT::ScalarT       ScalarT;
  typedef typename EvalT::MeshScalarT   MeshScalarT;

  AnalyticIceGeometry (const Teuchos::ParameterList& p,
                       const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:
  // Input:
  PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim> coordVec;

  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint> H;
  PHX::MDField<ScalarT,Cell,QuadPoint> z_s;

  unsigned int numQp, numDim;

  double rho,g;

  MeshScalarT L,dx;
};

} // Namespace FELIX

#endif // FELIX_ANALYTIC_ICE_GEOMETRY_HPP
