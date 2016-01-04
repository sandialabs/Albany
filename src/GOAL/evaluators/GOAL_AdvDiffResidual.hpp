//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef GOAL_ADVDIFFRESIDUAL_HPP
#define GOAL_ADVDIFFRESIDUAL_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"

namespace apf {
class Mesh;
class MeshEntity;
}

namespace GOAL {

template<typename EvalT, typename Traits>
class AdvDiffResidual :
  public PHX::EvaluatorWithBaseImpl<Traits>,
  public PHX::EvaluatorDerived<EvalT, Traits>
{

  public:

    AdvDiffResidual(
        const Teuchos::ParameterList& p,
        const Teuchos::RCP<Albany::Layouts>& dl);

    void postRegistrationSetup(
        typename Traits::SetupData d,
        PHX::FieldManager<Traits>& vm);

    void evaluateFields(typename Traits::EvalData d);

  private:

    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;

    double k;
    Teuchos::Array<double> a;
    double aMagnitude;

    int numDims;
    int numNodes;
    int numQPs;

    apf::Mesh* mesh;
    std::vector<std::vector<apf::MeshEntity*> > buckets;
    double computeTau(apf::MeshEntity* e);

    // Input
    PHX::MDField<ScalarT,Cell,QuadPoint> u;
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim> gradU;
    PHX::MDField<MeshScalarT,Cell,Node,QuadPoint> wBF;
    PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> wGradBF;
    Teuchos::RCP<Albany::Application> app;
    bool useSUPG;

    // Output
    PHX::MDField<ScalarT,Cell,Node> residual;

};

}

#endif
