//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef GOAL_FIELDMANAGERBUNDLE_HPP
#define GOAL_FIELDMANAGERBUNDLE_HPP

#include "Teuchos_RCP.hpp"
#include "Phalanx.hpp"

namespace PHAL {
class Workset;
class AlbanyTraits;
}

namespace Albany {
class Application;
class AbstractProblem;
class StateManager;
class MeshSpecsStruct;
}

namespace GOAL {

class BCManager;

struct ProblemBundle
{
  ProblemBundle(
      Teuchos::ParameterList p,
      Teuchos::RCP<Albany::Application> a,
      Teuchos::RCP<Albany::AbstractProblem> pr,
      Teuchos::RCP<Albany::StateManager> sm,
      Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> > ms) :
    params(p), application(a), problem(pr),
    stateManager(sm), meshSpecs(ms) {}
  bool enrich;
  Teuchos::ParameterList params;
  Teuchos::RCP<Albany::Application> application;
  Teuchos::RCP<Albany::AbstractProblem> problem;
  Teuchos::RCP<Albany::StateManager> stateManager;
  Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> > meshSpecs;
};

class FieldManagerBundle
{
  public:
    FieldManagerBundle(
        Teuchos::RCP<BCManager>& mgr,
        Teuchos::RCP<ProblemBundle>& bundle);
    ~FieldManagerBundle();
    void writePHXGraphs();
    void evaluateJacobian(PHAL::Workset& workset);
  private:
    void createFieldManagers();
    Teuchos::RCP<ProblemBundle> pb; 
    Teuchos::RCP<BCManager> bcm;
    Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits> > dfm;
    Teuchos::ArrayRCP<Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits> > > fm;
};

}

#endif
