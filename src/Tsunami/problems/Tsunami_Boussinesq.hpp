//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef TSUNAMI_BOUSSINESQ_HPP
#define TSUNAMI_BOUSSINESQ_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_AbstractProblem.hpp"
#include "Albany_GeneralPurposeFieldsNames.hpp"

#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"

namespace Tsunami {

  /*!
   * \brief Abstract interface for representing a 1-D finite element
   * problem.
   */
  class Boussinesq : public Albany::AbstractProblem {
  public:

    //! Default constructor
    Boussinesq(const Teuchos::RCP<Teuchos::ParameterList>& params,
     const Teuchos::RCP<ParamLib>& paramLib,
     const int numDim_); 

    //! Destructor
    ~Boussinesq();

    //! Return number of spatial dimensions
    virtual int spatialDimension() const { return numDim; }

    //! Get boolean telling code if SDBCs are utilized
    virtual bool useSDBCs() const {return use_sdbcs_; }

    //! Build the PDE instantiations, boundary conditions, and initial solution
    virtual void buildProblem(
      Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
      Albany::StateManager& stateMgr);

    // Build evaluators
    virtual Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
    buildEvaluators(
      PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
      const Albany::MeshSpecsStruct& meshSpecs,
      Albany::StateManager& stateMgr,
      Albany::FieldManagerChoice fmchoice,
      const Teuchos::RCP<Teuchos::ParameterList>& responseList);

    //! Each problem must generate it's list of valide parameters
    Teuchos::RCP<const Teuchos::ParameterList> getValidProblemParameters() const;

  private:

    //! Private to prohibit copying
    Boussinesq(const Boussinesq&);

    //! Private to prohibit copying
    Boussinesq& operator=(const Boussinesq&);

  public:

    //! Main problem setup routine. Not directly called, but indirectly by following functions
    template <typename EvalT> Teuchos::RCP<const PHX::FieldTag>
    constructEvaluators(
      PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
      const Albany::MeshSpecsStruct& meshSpecs,
      Albany::StateManager& stateMgr,
      Albany::FieldManagerChoice fmchoice,
      const Teuchos::RCP<Teuchos::ParameterList>& responseList);

    void constructDirichletEvaluators(const Albany::MeshSpecsStruct& meshSpecs);
    void constructNeumannEvaluators(const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs);


  protected:

    //! Enumerated type describing how a variable appears
    enum NS_VAR_TYPE {
      NS_VAR_TYPE_NONE,      //! Variable does not appear
      NS_VAR_TYPE_CONSTANT,  //! Variable is a constant
      NS_VAR_TYPE_DOF        //! Variable is a degree-of-freedom
    };

    void getVariableType(Teuchos::ParameterList& paramList,
       const std::string& defaultType,
       NS_VAR_TYPE& variableType,
       bool& haveVariable,
       bool& haveEquation);
    std::string variableTypeToString(const NS_VAR_TYPE variableType);

  protected:

    int numDim;        //! number of spatial dimensions

    NS_VAR_TYPE flowType; //! type of flow variables

    bool haveFlow;     //! have flow variables (momentum+continuity)

    bool haveFlowEq;     //! have flow equations (momentum+continuity)

    bool haveSource;   //! have source term in heat equation

    Teuchos::RCP<Albany::Layouts> dl;

    /// Boolean marking whether SDBCs are used
    bool use_sdbcs_;

    double mu, rho; //viscosity and density

    std::string elementBlockName;

    int neq; 
    
    bool use_params_on_mesh; //boolean to indicate whether to use parameters (viscosity, density) on mesh 
  };

}

#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"

#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_EvaluatorUtils.hpp"
#include "Albany_ResponseUtilities.hpp"

#include "PHAL_Neumann.hpp"
//#include "Tsunami_BoussinesqBodyForce.hpp"
//#include "Tsunami_BoussinesqParameters.hpp"
#include "Tsunami_BoussinesqResid.hpp"

template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
Tsunami::Boussinesq::constructEvaluators(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fieldManagerChoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::ParameterList;
  using PHX::DataLayout;
  using PHX::MDALayout;
  using std::vector;
  using std::string;
  using std::map;
  using PHAL::AlbanyTraits;

  RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> >
    intrepidBasis = Albany::getIntrepid2Basis(meshSpecs.ctd);
  RCP<shards::CellTopology> cellType = rcp(new shards::CellTopology (&meshSpecs.ctd));

  //The following, when set to true, will load parameters (rho and mu) only once at the beginning
  //of the simulation to save time 
  const bool enableMemoizer = this->params->get<bool>("Use MDField Memoization", true);

  const int numNodes = intrepidBasis->getCardinality();
  const int worksetSize = meshSpecs.worksetSize;

  Intrepid2::DefaultCubatureFactory cubFactory;
  RCP <Intrepid2::Cubature<PHX::Device> > cubature = cubFactory.create<PHX::Device, RealType, RealType>(*cellType, meshSpecs.cubatureDegree);

  const int numQPts = cubature->getNumPoints();
  const int numVertices = cellType->getNodeCount();

  int vecDim = neq;

  *out << "Field Dimensions: Workset=" << worksetSize
       << ", Vertices= " << numVertices
       << ", Nodes= " << numNodes
       << ", QuadPts= " << numQPts
       << ", vecDim= " << vecDim
       << ", Dim= " << numDim << std::endl;


  dl = rcp(new Albany::Layouts(worksetSize,numVertices,numNodes,numQPts,numDim,vecDim));
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
  bool supportsTransient=true;
  int offset=0;

  // Temporary variable used numerous times below
  Teuchos::RCP<PHX::Evaluator<AlbanyTraits> > ev;

  // Define Field Names

  if (haveFlowEq) {
    Teuchos::ArrayRCP<std::string> dof_names(1);
    Teuchos::ArrayRCP<std::string> dof_names_dot(1);
    Teuchos::ArrayRCP<std::string> resid_names(1);
    dof_names[0] = "EtaUE";
    dof_names_dot[0] = dof_names[0]+"_dot";
    resid_names[0] = "EtaUE Residual";
    fm0.template registerEvaluator<EvalT>
       (evalUtils.constructGatherSolutionEvaluator(true, dof_names, dof_names_dot, offset));

    fm0.template registerEvaluator<EvalT>
       (evalUtils.constructDOFVecInterpolationEvaluator(dof_names[0], offset));

    fm0.template registerEvaluator<EvalT>
       (evalUtils.constructDOFVecInterpolationEvaluator(dof_names_dot[0], offset));

    fm0.template registerEvaluator<EvalT>
       (evalUtils.constructDOFVecGradInterpolationEvaluator(dof_names[0], offset));

    fm0.template registerEvaluator<EvalT>
       (evalUtils.constructScatterResidualEvaluator(true, resid_names,offset, "Scatter EtaUE Residual"));
  }

  fm0.template registerEvaluator<EvalT>
     (evalUtils.constructGatherCoordinateVectorEvaluator());

  fm0.template registerEvaluator<EvalT>
     (evalUtils.constructMapToPhysicalFrameEvaluator(cellType, cubature));

  fm0.template registerEvaluator<EvalT>
     (evalUtils.constructComputeBasisFunctionsEvaluator(cellType, intrepidBasis, cubature));
 
  //Declare density as nodal field 
  Albany::StateStruct::MeshFieldEntity entity = Albany::StateStruct::NodalDataToElemNode;
  {
    std::string stateName("density");
    std::string fieldName = "density";
    RCP<ParameterList> p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName,true, &entity);
    p->set<std::string>("Field Name", fieldName);
    ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }
  // Intepolate density from nodes to QPs
  ev = evalUtils.getPSTUtils().constructDOFInterpolationEvaluator("density");
  fm0.template registerEvaluator<EvalT> (ev);

  //Declare viscosity as nodal field 
  entity = Albany::StateStruct::NodalDataToElemNode;
  {
    std::string stateName("viscosity");
    std::string fieldName = "viscosity";
    RCP<ParameterList> p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName,true, &entity);
    p->set<std::string>("Field Name", fieldName);
    ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }
  // Intepolate viscosity from nodes to QPs
  ev = evalUtils.getPSTUtils().constructDOFInterpolationEvaluator("viscosity");
  fm0.template registerEvaluator<EvalT> (ev);


/*  if (haveFlowEq) { // Body Force
    RCP<ParameterList> p = rcp(new ParameterList("Body Force"));

    //Input
    p->set<std::string>("Coordinate Vector Name", "Coord Vec");

    Teuchos::ParameterList& paramList = params->sublist("Body Force");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);
    p->set<std::string>("Fluid Viscosity QP Name", "Viscosity Field");

    //Output
    p->set<std::string>("Body Force Name", "Body Force");

    ev = rcp(new Tsunami::BoussinesqBodyForce<EvalT,AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (haveFlowEq) { // Parameters
    RCP<ParameterList> p = rcp(new ParameterList("Parameters"));

    //Input

    Teuchos::ParameterList& paramList = params->sublist("Parameters");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);
    p->set<std::string>("Fluid Viscosity In QP Name", "viscosity");
    p->set<std::string>("Fluid Density In QP Name", "density");
    p->set<double>("Viscosity", mu); 
    p->set<double>("Density", rho); 
    p->set<bool>("Use Parameters on Mesh", use_params_on_mesh); 
    p->set<bool>("Enable Memoizer", enableMemoizer);

    //Output
    p->set<std::string>("Fluid Viscosity QP Name", "Viscosity Field");
    p->set<std::string>("Fluid Density QP Name", "Density Field");

    ev = rcp(new Tsunami::BoussinesqParameters<EvalT,AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }
*/

  if (haveFlowEq) { // Boussinesq Resid
    RCP<ParameterList> p = rcp(new ParameterList("EtaUE Resid"));

    //Input
    p->set<std::string>("Weighted BF Name", Albany::weighted_bf_name);
    p->set<std::string>("Weighted Gradient BF Name", Albany::weighted_grad_bf_name);
    p->set<std::string>("EtaUE QP Variable Name", "EtaUE");
    p->set<std::string>("EtaUE Gradient QP Variable Name", "EtaUE Gradient");
    
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_vecgradient);
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", dl->node_qp_scalar);
    p->set< RCP<DataLayout> >("Node QP Gradient Data Layout", dl->node_qp_gradient);
    //IKT FIXME: add udot term 
    //IKT, FIXME?  add body force evaluator
    //p->set<std::string>("Body Force Name", "Body Force");
    //p->set<std::string>("Fluid Viscosity QP Name", "Viscosity Field");

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);

    //Output
    p->set<std::string>("Residual Name", "EtaUE Residual");
    p->set< RCP<DataLayout> >("Node Vector Data Layout", dl->node_vector);

    ev = rcp(new Tsunami::BoussinesqResid<EvalT,AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (fieldManagerChoice == Albany::BUILD_RESID_FM)  {
    Teuchos::RCP<const PHX::FieldTag> ret_tag;
    if (haveFlowEq) {
      PHX::Tag<typename EvalT::ScalarT> mom_tag("Scatter EtaUE Residual", dl->dummy);
      fm0.requireField<EvalT>(mom_tag);
      ret_tag = mom_tag.clone();
    }
    return ret_tag;
  }
  else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM) {
    Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
    return respUtils.constructResponses(fm0, *responseList, stateMgr);
  }

  return Teuchos::null;
}
#endif // TSUNAMI_BOUSSINESQ_HPP
