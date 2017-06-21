/*
 * FELIX_BasalFrictionHeat.hpp
 *
 *  Created on: May 25, 2016
 *      Author: abarone
 */

#ifndef FELIX_BASALFRICTIONHEAT_HPP_
#define FELIX_BASALFRICTIONHEAT_HPP_

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace FELIX
{
  /** \brief Basal friction heat Evaluator

    This evaluator evaluates the production of heat caused by basal friction
   */
  template<typename EvalT, typename Traits, typename Type>
  class BasalFrictionHeat : public PHX::EvaluatorWithBaseImpl<Traits>,
  public PHX::EvaluatorDerived<EvalT, Traits>
  {
  public:
    BasalFrictionHeat (const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl);

    void postRegistrationSetup (typename Traits::SetupData d, PHX::FieldManager<Traits>& fm);

    void evaluateFields(typename Traits::EvalData d);

  private:
    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;
    typedef typename EvalT::ParamScalarT ParamScalarT;

    // Input:
    PHX::MDField<const ParamScalarT,Cell,Side,QuadPoint>       beta; // [kPa m / yr]
    PHX::MDField<const Type,Cell,Side,QuadPoint,VecDim>        velocity; // [m yr^{-1}]
    PHX::MDField<const RealType,Cell,Side,Node,QuadPoint>   	 BF;
    PHX::MDField<const RealType,Cell,Side,Node,QuadPoint,Dim>  GradBF;   // [km^{-1}]
    PHX::MDField<const MeshScalarT,Cell,Side,QuadPoint>     	 w_measure;// [km^2]

    // Output:
    PHX::MDField<ScalarT,Cell,Node> basalFricHeat; //  [MW] = [k^{-1} Pa s^{-1} km^3], k=1000
    PHX::MDField<ScalarT,Cell,Node> basalFricHeatSUPG; // [MW s^{-1}] = [k^{-1} Pa s^{-2} km^3], k=1000

    std::vector<std::vector<int> >  sideNodes;
    std::string                     basalSideName;

    int numCellNodes;
    int numSideNodes;
    int numSideQPs;
    int sideDim;
    int vecDimFO;

    bool haveSUPG;
  };

}	// Namespace FELIX

#endif /* FELIX_BASALFRICTIONHEAT_HPP_ */
