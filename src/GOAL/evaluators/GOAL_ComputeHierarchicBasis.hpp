//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef GOAL_COMPUTEHIERARCHICBASIS_HPP
#define GOAL_COMPUTEHIERARCHICBASIS_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"

#include <apf.h>

namespace Albany {
class Application;
}

namespace GOAL {

template<typename EvalT, typename Traits>
class ComputeHierarchicBasis :
  public PHX::EvaluatorWithBaseImpl<Traits>,
  public PHX::EvaluatorDerived<EvalT, Traits>
{

  public:

    ComputeHierarchicBasis(
        const Teuchos::ParameterList& p,
        const Teuchos::RCP<Albany::Layouts>& dl);

    void postRegistrationSetup(
        typename Traits::SetupData d,
        PHX::FieldManager<Traits>& vm);

    void evaluateFields(typename Traits::EvalData d);

  private:

    typedef typename EvalT::MeshScalarT MeshScalarT;
    int numDims, numNodes, numQPs;

    // Input
    Teuchos::RCP<Albany::Application> app;
    int cubatureDegree;
    int polynomialOrder;

    // Used for basis function computation
    int wsIndex;
    apf::Mesh* mesh;
    apf::FieldShape* shape;
    apf::Vector3 point;
    apf::NewArray<double> bf;
    apf::NewArray<apf::Vector3> gbf;
    std::vector<std::vector<apf::MeshEntity*> > buckets;

    // Output:
    PHX::MDField<MeshScalarT,Cell,QuadPoint> detJ;
    PHX::MDField<MeshScalarT,Cell,QuadPoint> weightedDV;
    PHX::MDField<RealType,Cell,Node,QuadPoint> BF;
    PHX::MDField<MeshScalarT,Cell,Node,QuadPoint> wBF;
    PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> GradBF;
    PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> wGradBF;

};

}

#endif
