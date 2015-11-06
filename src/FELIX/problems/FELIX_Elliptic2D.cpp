//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "FELIX_Elliptic2D.hpp"

#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "PHAL_FactoryTraits.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include <string>

namespace FELIX
{

Elliptic2D::Elliptic2D (const Teuchos::RCP<Teuchos::ParameterList>& params,
                        const Teuchos::RCP<ParamLib>& paramLib,
                        const int numEq) :
  Albany::AbstractProblem (params, paramLib,numEq)
{
  numDim = params->get<int>("Dimension");
  TEUCHOS_TEST_FOR_EXCEPTION (numDim!=2 && numDim!=3,std::logic_error,"Problem supports only 2D and 3D");

  dof_names.resize(1);
  resid_names.resize(1);

  dof_names[0] = "u";
  resid_names[0] = "res";

  if (numDim==3)
  {
    this->requirements.push_back("surface_height");
    this->requirements.push_back("thickness");

    sideSetName = params->get<std::string>("Side Set Name");
    this->sideSetEquations[0].push_back(sideSetName);
  }
}

Elliptic2D::~Elliptic2D()
{
  // Nothing to be done here
}

void Elliptic2D::buildProblem (Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
                               Albany::StateManager& stateMgr)
{
  TEUCHOS_TEST_FOR_EXCEPTION (meshSpecs.size()!=1,std::logic_error,"Problem supports one Material Block");

  fm.resize(1);
  fm[0]  = Teuchos::rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);

  // Build evaluators
  buildEvaluators(*fm[0], *meshSpecs[0], stateMgr, Albany::BUILD_RESID_FM, Teuchos::null);

  // Build dirichlet evaluator (for 3D case)
  if (numDim==3)
  {
    constructDirichletEvaluators (*meshSpecs[0]);
  }
}

Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
Elliptic2D::buildEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                             const Albany::MeshSpecsStruct& meshSpecs,
                             Albany::StateManager& stateMgr,
                             Albany::FieldManagerChoice fmchoice,
                             const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructeEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  Albany::ConstructEvaluatorsOp<Elliptic2D> op(*this, fm0, meshSpecs, stateMgr, fmchoice, responseList);
  Sacado::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes> fe(op);

  return *op.tags;
}

void Elliptic2D::constructDirichletEvaluators(const Albany::MeshSpecsStruct& meshSpecs)
{
  // Construct Dirichlet evaluators for all nodesets and names
  std::vector<std::string> dirichletNames(1,"U");

  Albany::BCUtils<Albany::DirichletTraits> dirUtils;
  dfm = dirUtils.constructBCEvaluators(meshSpecs.nsNames, dirichletNames, this->params, this->paramLib);
}

Teuchos::RCP<const Teuchos::ParameterList>
Elliptic2D::getValidProblemParameters () const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = this->getGenericProblemParams("ValidElliptic2DProblemParams");

  validPL->set<int>("Dimension",1,"");
  validPL->set<int>("Side Cubature Degree",1,"");
  validPL->set<std::string>("Side Set Name","","");

  return validPL;
}

} // Namespace FELIX
