//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ADVECTION_PROBLEM_HPP
#define ADVECTION_PROBLEM_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_TestForException.hpp"

#include "Albany_AbstractProblem.hpp"

#include "Phalanx.hpp"
#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"
#include "Albany_ProblemUtils.hpp"
#include "QCAD_MaterialDatabase.hpp"

namespace Albany {

class AdvectiveProblem : public AbstractProblem {

  public:
 
    AdvectiveProblem(
      const Teuchos::RCP<Teuchos::ParameterList>& params,
      const Teuchos::RCP<ParamLib>& param_lib,
      const int num_dims,
      Teuchos::RCP<const Teuchos::Comm<int> >& commT);

    ~AdvectiveProblem();

    int spatialDimension() const { return num_dims_; }

    void buildProblem(
        Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
        StateManager& stateMgr);

    Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> > buildEvaluators(
        PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
        const Albany::MeshSpecsStruct& meshSpecs,
        Albany::StateManager& stateMgr,
        Albany::FieldManagerChoice fmchoice,
        const Teuchos::RCP<Teuchos::ParameterList>& responseList);

    Teuchos::RCP<const Teuchos::ParameterList>
    getValidProblemParameters() const;

  private:

    AdvectiveProblem(const AdvectiveProblem&);
    
    AdvectiveProblem& operator=(const AdvectiveProblem&);

  public:

    template <typename EvalT>
    Teuchos::RCP<const PHX::FieldTag> constructEvaluators(
        PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
        const Albany::MeshSpecsStruct& meshSpecs,
        Albany::StateManager& stateMgr,
        Albany::FieldManagerChoice fmchoice,
        const Teuchos::RCP<Teuchos::ParameterList>& responseList);

    void constructDirichletEvaluators(
        const std::vector<std::string>& nodeSetIDs);

    void constructNeumannEvaluators(
        const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs);

  protected:

    int num_dims_;

    Teuchos::RCP<QCAD::MaterialDatabase> material_db_;

    Teuchos::RCP<Albany::Layouts> dl_;

};

}

#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "Albany_Utils.hpp"
#include "Albany_BCUtils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_EvaluatorUtils.hpp"
#include "Albany_ResponseUtilities.hpp"
#include "PHAL_SaveStateField.hpp"

template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
Albany::AdvectiveProblem::constructEvaluators(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fieldManagerChoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList) {

  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::ParameterList;
  using PHX::DataLayout;
  using PHX::MDALayout;
  using std::vector;
  using std::string;
  using PHAL::AlbanyTraits;

}

#endif
