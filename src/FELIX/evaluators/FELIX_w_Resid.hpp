/*
 * FELIX_w_Resid.hpp
 *
 *  Created on: Jun 7, 2016
 *      Author: abarone
 */

#ifndef FELIX_W_RESID_HPP_
#define FELIX_W_RESID_HPP_

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace FELIX
{

  template<typename EvalT, typename Traits, typename VelocityType>
  class w_Resid : public PHX::EvaluatorWithBaseImpl<Traits>,
  public PHX::EvaluatorDerived<EvalT, Traits>
  {
  public:
    w_Resid (const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl);

    void postRegistrationSetup (typename Traits::SetupData d, PHX::FieldManager<Traits>& fm);

    void evaluateFields(typename Traits::EvalData d);

  private:
    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;
    typedef typename EvalT::ParamScalarT ParamScalarT;

    // Input:
    PHX::MDField<const MeshScalarT,Cell,Node,QuadPoint> wBF;  // [km^3]
    PHX::MDField<const ScalarT,Cell,Node> basalMeltRate; // [m yr^{-1}]
    PHX::MDField<const VelocityType,Cell,QuadPoint,VecDim,Dim>  GradVelocity; // [k^{-1} yr^{-1}]
    PHX::MDField<const ScalarT,Cell,QuadPoint, Dim> w_z;  // [k^{-1} yr^{-1}]
    PHX::MDField<const ScalarT,Cell,Node> w; // [m yr^{-1}]

    // Output
    PHX::MDField<ScalarT,Cell,Node> Residual;

    std::string sideName;
    std::vector<std::vector<int> >  sideNodes;
    int numNodes;
    int numSideNodes;
    int numQPs;

  };

}	// Namespace FELIX




#endif /* FELIX_VELOCITYZ_HPP_ */
