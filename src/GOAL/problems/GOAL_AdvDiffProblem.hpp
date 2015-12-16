//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef GOAL_ADVDIFFPROBLEM_HPP
#define GOAL_ADVDIFFPROBLEM_HPP

#include "Albany_AbstractProblem.hpp"

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Phalanx.hpp"
#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"
#include "PHAL_AlbanyTraits.hpp"

namespace Albany {

//! mechanics problem
class GOALAdvDiffProblem: public Albany::AbstractProblem
{
  public:

    //! constructor
    GOALAdvDiffProblem(
        const Teuchos::RCP<Teuchos::ParameterList>& params,
        const Teuchos::RCP<ParamLib>& param_lib,
        const int numDim,
        Teuchos::RCP<const Teuchos::Comm<int> >& commT);

    //! destructor
    virtual ~GOALAdvDiffProblem();

    //! return number of spatial dimensions
    int spatialDimension() const {return numDims;}

    //! get the offset corresponding to a variable name
    int getOffset(std::string const& var);

    //! build the pde instantiations for the primal problem
    void buildProblem(
        Teuchos::ArrayRCP<Teuchos::RCP<MeshSpecsStruct> > meshSpecs,
        StateManager& stateMgr);

    //! build evaluators
    Teuchos::Array<Teuchos::RCP<const PHX::FieldTag> > buildEvaluators(
        PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
        const MeshSpecsStruct& meshSpecs,
        StateManager& stateMgr,
        FieldManagerChoice fmchoice,
        const Teuchos::RCP<Teuchos::ParameterList>& responseList);

    //! each problem must generate its own list of valid parameters
    Teuchos::RCP<const Teuchos::ParameterList>
      getValidProblemParameters() const;

  private:
    
    //! private to prohibit copying
    GOALAdvDiffProblem(const GOALAdvDiffProblem&);
    GOALAdvDiffProblem& operator=(const GOALAdvDiffProblem&);

  public:

    //! setup for problem evaluators
    template<typename EvalT>
    Teuchos::RCP<const PHX::FieldTag> constructEvaluators(
        PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
        const Albany::MeshSpecsStruct& meshSpecs,
        Albany::StateManager& stateMgr,
        Albany::FieldManagerChoice fmChoice,
        const Teuchos::RCP<Teuchos::ParameterList>& responseList);

    //! setup for dirichlet bcs
    void constructDirichletEvaluators(
        const MeshSpecsStruct& meshSpecs,
        Teuchos::RCP<Teuchos::ParameterList>& params);

    //! setup for neumann bcs
    void constructNeumannEvaluators(
        const Teuchos::RCP<MeshSpecsStruct>& meshSpecs);

  protected:

    //! a map of the dof offsets
    std::map<std::string, int> offsets;

    //! number of spatial dimensions
    int numDims;

    //! qoi parameters
    Teuchos::RCP<Teuchos::ParameterList> qoiParams;

    //! data layouts
    Teuchos::RCP<Layouts> dl;

    //! diffusivity coefficient
    double k;

    //! advection vector
    Teuchos::Array<double> a;

    //! use supg stabilization
    bool useSUPG;

};

}

#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_ResponseUtilities.hpp"
#include "Albany_EvaluatorUtils.hpp"

#include "GOAL_ComputeHierarchicBasis.hpp"
#include "GOAL_AdvDiffResidual.hpp"

#include <apf.h>
#include <apfMesh.h>
#include <apfShape.h>

template<typename EvalT>
Teuchos::RCP<const PHX::FieldTag> Albany::GOALAdvDiffProblem::
constructEvaluators(
    PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
    const Albany::MeshSpecsStruct& meshSpecs,
    Albany::StateManager& stateMgr,
    Albany::FieldManagerChoice fmChoice,
    const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{

  using Teuchos::rcp;
  using Teuchos::RCP;
  using Teuchos::ArrayRCP;
  using Teuchos::ParameterList;

  // get the name of the current element block
  std::string ebName = meshSpecs.ebName;

  // name variables
  ArrayRCP<std::string> dofNames(1);
  ArrayRCP<std::string> residNames(1);
  dofNames[0] = "U";
  residNames[0] = dofNames[0] + " Residual";

  // do some work to create a data layout
  int pOrder = meshSpecs.polynomialOrder;
  int qOrder = meshSpecs.cubatureDegree;
  apf::Mesh::Type type = apf::Mesh::simplexTypes[numDims];
  apf::FieldShape* shape = apf::getHierarchic(pOrder);
  apf::EntityShape* eShape = shape->getEntityShape(type);

  // create a data layout
  int numNodes = eShape->countNodes();
  int numVertices = numDims + 1; // simplex assumption
  int numQPs = apf::countGaussPoints(type, qOrder);
  const int worksetSize = meshSpecs.worksetSize;

  dl = rcp(new Albany::Layouts(
        worksetSize, numVertices, numNodes, numQPs, numDims));

  // construct standard FEM evaluators
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
  RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;

  fm0.template registerEvaluator<EvalT>(
      evalUtils.constructGatherSolutionEvaluator_noTransient(
        false, dofNames));

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructGatherCoordinateVectorEvaluator());

  fm0.template registerEvaluator<EvalT>(
      evalUtils.constructDOFInterpolationEvaluator(dofNames[0]));

  fm0.template registerEvaluator<EvalT>(
      evalUtils.constructDOFGradInterpolationEvaluator(dofNames[0]));

  fm0.template registerEvaluator<EvalT>(
      evalUtils.constructScatterResidualEvaluator(false, residNames));

  // store velocity and acceleration
  RCP<ParameterList> pFromProb = rcp(
      new ParameterList("Response Parameters from Problem"));

  { // hierarchic basis functions

    // input
    RCP<ParameterList> p = rcp(new ParameterList("Compute Hierarchic Basis"));
    p->set<RCP<Albany::Application> >("Application", this->getApplication());
    p->set<int>("Cubature Degree", meshSpecs.cubatureDegree);
    p->set<int>("Polynomial Order", meshSpecs.polynomialOrder);

    // output
    p->set<std::string>("Jacobian Det Name", "Jacobian Det");
    p->set<std::string>("Weights Name", "Weights");
    p->set<std::string>("BF Name", "BF");
    p->set<std::string>("Weighted BF Name", "wBF");
    p->set<std::string>("Gradient BF Name", "Grad BF");
    p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");

    // register evaluator
    ev = rcp(new GOAL::ComputeHierarchicBasis<EvalT, PHAL::AlbanyTraits>(*p, dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  { // residual

    // input
    RCP<ParameterList> p = rcp(new ParameterList("U Residual"));
    p->set<double>("Diffusivity Coefficient", k);
    p->set<Teuchos::Array<double> >("Advection Vector", a);
    p->set<std::string>("U Name", "U");
    p->set<std::string>("Gradient U Name", "U Gradient");
    p->set<std::string>("Weighted BF Name", "wBF");
    p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
    p->set<RCP<Albany::Application> >("Application", this->getApplication());
    p->set<bool>("Use SUPG", useSUPG);

    // output
    p->set<std::string>("Residual Name", residNames[0]);

    // register evaluator
    ev = rcp(new GOAL::AdvDiffResidual<EvalT, PHAL::AlbanyTraits>(*p, dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (fmChoice == Albany::BUILD_RESID_FM)
  {
    PHX::Tag<typename EvalT::ScalarT> resTag("Scatter", dl->dummy);
    fm0.requireField<EvalT>(resTag);
    return resTag.clone();
  }
  else if (fmChoice == Albany::BUILD_RESPONSE_FM)
  {
    Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
    return respUtils.constructResponses(
        fm0, *responseList, pFromProb, stateMgr, &meshSpecs);
  }
  else
    return Teuchos::null;
}

#endif
