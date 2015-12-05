//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef GOAL_MECHANICSPROBLEM_HPP
#define GOAL_MECHANICSPROBLEM_HPP

#include "Albany_AbstractProblem.hpp"

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Phalanx.hpp"
#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"
#include "PHAL_AlbanyTraits.hpp"

namespace Albany {

//! mechanics problem
class GOALMechanicsProblem: public Albany::AbstractProblem
{
  public:

    //! constructor
    GOALMechanicsProblem(
        const Teuchos::RCP<Teuchos::ParameterList>& params,
        const Teuchos::RCP<ParamLib>& param_lib,
        const int numDim,
        Teuchos::RCP<const Teuchos::Comm<int> >& commT);

    //! destructor
    virtual ~GOALMechanicsProblem();

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

    //! retrieve the state data
    void getAllocatedStates(
        Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP
        <Intrepid::FieldContainer<RealType> > > > oldSt,
        Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP
        <Intrepid::FieldContainer<RealType> > > > newSt) const;

  private:
    
    //! private to prohibit copying
    GOALMechanicsProblem(const GOALMechanicsProblem&);
    GOALMechanicsProblem& operator=(const GOALMechanicsProblem&);

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

    //! number of spatial dimensions
    int numDims;

    //! qoi parameters
    Teuchos::RCP<Teuchos::ParameterList> qoiParams;

    //! a map of the dof offsets
    std::map<std::string, int> offsets;

    //! data layouts
    Teuchos::RCP<Layouts> dl;

    //! material database
    Teuchos::RCP<QCAD::MaterialDatabase> materialDB;

    //! old state data
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP
      <Intrepid::FieldContainer<RealType> > > > oldState;

    //! new state data
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP
      <Intrepid::FieldContainer<RealType> > > > newState;

};

}

#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_ResponseUtilities.hpp"
#include "Albany_EvaluatorUtils.hpp"

#include "PHAL_NSMaterialProperty.hpp"
#include "PHAL_Source.hpp"
#include "PHAL_SaveStateField.hpp"

#include "../../LCM/problems/FieldNameMap.hpp"

#include "Time.hpp"
#include "CurrentCoords.hpp"

#include "FirstPK.hpp"
#include "Kinematics.hpp"
#include "MechanicsResidual.hpp"
#include "ConstitutiveModelInterface.hpp"
#include "ConstitutiveModelParameters.hpp"

#include "GOAL_LpStress.hpp"
#include "GOAL_ScatterQoI.hpp"
#include "GOAL_ComputeHierarchicBasis.hpp"

#include <apf.h>
#include <apfMesh.h>
#include <apfShape.h>

template<typename EvalT>
Teuchos::RCP<const PHX::FieldTag> Albany::GOALMechanicsProblem::
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

  typedef Intrepid::FieldContainer<RealType> FieldContainer;
  typedef RCP<Intrepid::Basis<RealType, FieldContainer> > Basis;
  typedef Intrepid::DefaultCubatureFactory<RealType> CubatureFactory;
  typedef RCP<Intrepid::Cubature<RealType> > Cubature;

  // get the name of the current element block
  std::string ebName = meshSpecs.ebName;

  // get the name of the material model
  std::string matModelName = materialDB->getElementBlockSublist(
      ebName, "Material Model").get<std::string>("Model Name");

  TEUCHOS_TEST_FOR_EXCEPTION(
      matModelName.length() == 0,
      std::logic_error,
      "A material model must be defined for block: " + ebName);

  // set flag for small strain option
  bool smallStrain = false;
  if (matModelName == "Linear Elastic")
    smallStrain = true;

  // name variables
  ArrayRCP<std::string> dofNames(1);
  ArrayRCP<std::string> dofDotNames(1);
  ArrayRCP<std::string> dofDotDotNames(1);
  ArrayRCP<std::string> residNames(1);
  dofNames[0] = "Displacement";
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
        true, dofNames));

  fm0.template registerEvaluator<EvalT>(
      evalUtils.constructGatherCoordinateVectorEvaluator());

  fm0.template registerEvaluator<EvalT>(
      evalUtils.constructDOFVecInterpolationEvaluator(dofNames[0]));

  fm0.template registerEvaluator<EvalT>(
      evalUtils.constructDOFVecGradInterpolationEvaluator(dofNames[0]));

  fm0.template registerEvaluator<EvalT>(
      evalUtils.constructScatterResidualEvaluator(true, residNames));

  // store velocity and acceleration
  RCP<ParameterList> pFromProb = rcp(
      new ParameterList("Response Parameters from Problem"));
  pFromProb->set<std::string>("x Field Name", "xField");

  // define field names
  LCM::FieldNameMap FNM(false);
  RCP<std::map<std::string, std::string> > fnm = FNM.getMap();
  std::string cauchy = (*fnm)["Cauchy_Stress"];
  std::string firstPK = (*fnm)["FirstPK"];
  std::string Fp = (*fnm)["Fp"];
  std::string eqps = (*fnm)["eqps"];
  std::string temperature = (*fnm)["Temperature"];
  std::string pressure = (*fnm)["Pressure"];
  std::string mech_source = (*fnm)["Mechanical_Source"];
  std::string defgrad = (*fnm)["F"];
  std::string J = (*fnm)["J"];

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

  { // time

    // input
    RCP<ParameterList> p = rcp(new ParameterList("Time"));
    p->set<std::string>("Delta Time Name", "Delta Time");
    p->set<RCP<PHX::DataLayout> >(
        "Workset Scalar Data Layout", dl->workset_scalar);
    p->set<bool>("Disable Transient", true);

    // output
    p->set<std::string>("Time Name", "Time");
    p->set<RCP<ParamLib> >("Parameter Library", paramLib);

    // register evaluator
    ev = rcp(new LCM::Time<EvalT, PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

    // register state variable
    p = stateMgr.registerStateVariable(
        "Time", dl->workset_scalar, dl->dummy, ebName, "scalar", 0.0, true);
    ev = rcp(new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  { // constitutive model parameters

    // input
    RCP<ParameterList> p =
      rcp(new ParameterList("Constitutive Model Parameters"));
    std::string matName = 
      materialDB->getElementBlockParam<std::string>(ebName, "material");
    ParameterList& mp =
      materialDB->getElementBlockSublist(ebName, matName);
    p->set<ParameterList*>("Material Parameters", &mp);

    // register evaluator
    RCP<LCM::ConstitutiveModelParameters<EvalT, PHAL::AlbanyTraits> > cmp =
        rcp(new LCM::ConstitutiveModelParameters<EvalT, PHAL::AlbanyTraits>(
              *p,dl));
    fm0.template registerEvaluator<EvalT>(cmp);
  }

  { // constitutive model interface

    // input
    RCP<ParameterList> p =
      rcp(new ParameterList("Constitutive Model Interface"));
    std::string matName =
      materialDB->getElementBlockParam<std::string>(ebName, "material");
    ParameterList& mp =
        materialDB->getElementBlockSublist(ebName, matName);

    // set all of the extra mechanics stuff off
    mp.set<bool>("Have Temperature", false);
    mp.set<bool>("Have Total Concentration", false);
    mp.set<bool>("Have Bubble Volume Fraction", false);
    mp.set<bool>("Have Total Bubble Density", false);
    mp.set<RCP<std::map<std::string, std::string> > >("Name Map", fnm);
    p->set<ParameterList*>("Material Parameters", &mp);
    p->set<bool>("Volume Average Pressure", false);

    // register evaluator
    RCP<LCM::ConstitutiveModelInterface<EvalT, PHAL::AlbanyTraits> > cmi =
        rcp(new LCM::ConstitutiveModelInterface<EvalT, PHAL::AlbanyTraits>(
              *p, dl));
    fm0.template registerEvaluator<EvalT>(cmi);

    // register state variables
    for (int sv(0); sv < cmi->getNumStateVars(); ++sv)
    {
      cmi->fillStateVariableStruct(sv);
      p = stateMgr.registerStateVariable(
          cmi->getName(), cmi->getLayout(), dl->dummy, ebName,
          cmi->getInitType(), cmi->getInitValue(), cmi->getStateFlag(),
          cmi->getOutputFlag());
      ev = rcp(new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }
  }

  { // kinematics quantities

    // input
    RCP<ParameterList> p = rcp(new ParameterList("Kinematics"));
    p->set<bool>("Weighted Volume Average J", false);
    p->set<RCP<PHX::DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set<RCP<PHX::DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);
    p->set<std::string>("Weights Name", "Weights");
    p->set<std::string>("Gradient QP Variable Name", "Displacement Gradient");
    if (smallStrain) p->set<std::string>("Strain Name", "Strain");

    // output
    p->set<std::string>("DefGrad Name", defgrad);
    p->set<std::string>("DetDefGrad Name", J);

    // register evaluator
    ev = rcp(new LCM::Kinematics<EvalT, PHAL::AlbanyTraits>(*p, dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  { // first piola-kirchoff stress tensor

    // input
    RCP<ParameterList> p = rcp(new ParameterList("First PK Stress"));
    p->set<std::string>("Stress Name", cauchy);
    p->set<std::string>("DefGrad Name", defgrad);
    if (smallStrain) p->set<bool>("Small Strain", true);

    // output
    p->set<std::string>("First PK Stress Name", firstPK);
    p->set<RCP<ParamLib> >("Parameter Library", paramLib);

    // register evaluator
    ev = rcp(new LCM::FirstPK<EvalT, PHAL::AlbanyTraits>(*p, dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  { // residual

    // input
    RCP<ParameterList> p = rcp(new ParameterList("Displacement Residual"));
    p->set<std::string>("Stress Name", firstPK);
    p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
    p->set<std::string>("Weighted BF Name", "wBF");
    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    p->set<bool>("Disable Dynamics", true);

    // output
    p->set<std::string>("Residual Name", "Displacement Residual");

    // register evaluator
    ev = rcp(new LCM::MechanicsResidual<EvalT, PHAL::AlbanyTraits>(*p, dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (isAdjoint) {

    std::string qoiName = qoiParams->get<std::string>("Name","");

    if (qoiName == "Lp Stress")
    {{

       // input
       RCP<ParameterList> p = rcp(new ParameterList("Lp Stress"));
       p->set<int>("Order", qoiParams->get<int>("p",1));
       p->set<std::string>("Weights Name", "Weights");
       p->set<std::string>("Stress Name", cauchy);

       // output
       p->set<std::string>("Lp Stress Name", qoiName);

       // register evaluator
       ev = rcp(new GOAL::LpStress<EvalT, PHAL::AlbanyTraits>(*p, dl));
       fm0.template registerEvaluator<EvalT>(ev);

     }}

    else
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
          "Invalid quantity of interest name: " + qoiName);

    { // scatter qoi

      // input
      RCP<ParameterList> p = rcp(new ParameterList("Scatter QoI"));
      p->set<std::string>("QoI Name", qoiName);

      // register evaluator
      ev = rcp(new GOAL::ScatterQoI<EvalT, PHAL::AlbanyTraits>(*p, dl));
      fm0.template registerEvaluator<EvalT>(ev);
    }

  }

  if (fmChoice == Albany::BUILD_RESID_FM)
  {
    RCP<const PHX::FieldTag> retTag;
    PHX::Tag<typename EvalT::ScalarT> resTag("Scatter", dl->dummy);
    fm0.requireField<EvalT>(resTag);
    retTag = resTag.clone();
    return retTag;
  }
  else if (fmChoice == Albany::BUILD_RESPONSE_FM)
  {
    if (isAdjoint)
    {
      PHX::Tag<typename EvalT::ScalarT> resTag("Scatter", dl->dummy);
      fm0.requireField<EvalT>(resTag);
      PHX::Tag<typename EvalT::ScalarT> qoiTag("Scatter QoI", dl->dummy);
      fm0.requireField<EvalT>(qoiTag);
    }
    Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
    return respUtils.constructResponses(
        fm0, *responseList, pFromProb, stateMgr, &meshSpecs);
  }
  else
    return Teuchos::null;
}

#endif
