//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef TSUNAMI_NAVIERSTOKES_HPP
#define TSUNAMI_NAVIERSTOKES_HPP

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
  class NavierStokes : public Albany::AbstractProblem {
  public:

    //! Default constructor
    NavierStokes(const Teuchos::RCP<Teuchos::ParameterList>& params,
     const Teuchos::RCP<ParamLib>& paramLib,
     const int numDim_, 
     const bool haveAdvection_=true,
     const bool haveUnsteady_=true);

    //! Destructor
    ~NavierStokes();

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
    NavierStokes(const NavierStokes&);

    //! Private to prohibit copying
    NavierStokes& operator=(const NavierStokes&);

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

    int numDim;        //! number of spatial dimensions

    bool haveSource;   //! have source term in heat equation
    bool havePSPG;     //! have pressure stabilization
    bool haveSUPG;     //! have SUPG stabilization (for convection-dominated flows) 

    bool haveAdvection; //! turns on nonlinear convection terms in NS
    bool haveUnsteady;  //! turns on time-dependent terms in NS 

    Teuchos::RCP<Albany::Layouts> dl;

    /// Boolean marking whether SDBCs are used
    bool use_sdbcs_;

    double mu, rho; //viscosity and density

    std::string stabType; //stabilization type
    
    std::string elementBlockName;

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
#include "Tsunami_NavierStokesContravarientMetricTensor.hpp"
#include "Tsunami_NavierStokesBodyForce.hpp"
#include "Tsunami_NavierStokesParameters.hpp"
#include "Tsunami_NavierStokesRm.hpp"
#include "Tsunami_NavierStokesContinuityResid.hpp"
#include "Tsunami_NavierStokesMomentumResid.hpp"
#include "Tsunami_NavierStokesTau.hpp"

template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
Tsunami::NavierStokes::constructEvaluators(
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

  *out << "Field Dimensions: Workset=" << worksetSize
       << ", Vertices= " << numVertices
       << ", Nodes= " << numNodes
       << ", QuadPts= " << numQPts
       << ", Dim= " << numDim << std::endl;


  dl = rcp(new Albany::Layouts(worksetSize,numVertices,numNodes,numQPts,numDim, numDim));
  TEUCHOS_TEST_FOR_EXCEPTION(dl->vectorAndGradientLayoutsAreEquivalent==false, std::logic_error,
                              "Data Layout Usage in Stokes problem assumes vecDim = numDim");
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
  bool supportsTransient=true;
  int offset=0;

  // Temporary variable used numerous times below
  Teuchos::RCP<PHX::Evaluator<AlbanyTraits> > ev;

  // Define Field Names

  Teuchos::ArrayRCP<std::string> dof_names(1);
  Teuchos::ArrayRCP<std::string> dof_names_dot(1);
  Teuchos::ArrayRCP<std::string> resid_names(1);

  //Velocity/momentum field names
  dof_names[0] = "Velocity";
  dof_names_dot[0] = dof_names[0]+"_dot";
  resid_names[0] = "Momentum Residual";
  fm0.template registerEvaluator<EvalT>
     (evalUtils.constructGatherSolutionEvaluator(true, dof_names, dof_names_dot, offset));

  fm0.template registerEvaluator<EvalT>
     (evalUtils.constructDOFVecInterpolationEvaluator(dof_names[0], offset));

  fm0.template registerEvaluator<EvalT>
     (evalUtils.constructDOFVecInterpolationEvaluator(dof_names_dot[0], offset));

  fm0.template registerEvaluator<EvalT>
     (evalUtils.constructDOFVecGradInterpolationEvaluator(dof_names[0], offset));

  fm0.template registerEvaluator<EvalT>
     (evalUtils.constructScatterResidualEvaluator(true, resid_names,offset, "Scatter Momentum"));
  offset += numDim;

  //Pressure/continuity field names 
  dof_names[0] = "Pressure";
  dof_names_dot[0] = dof_names[0]+"_dot";
  resid_names[0] = "Continuity Residual";
  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructGatherSolutionEvaluator(false, dof_names, dof_names_dot, offset));

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructDOFInterpolationEvaluator(dof_names[0], offset));

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructDOFInterpolationEvaluator(dof_names_dot[0], offset));

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructDOFGradInterpolationEvaluator(dof_names[0], offset));

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructScatterResidualEvaluator(false, resid_names,offset, "Scatter Continuity"));
  offset ++;


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

  if (havePSPG || haveSUPG) { // Compute Contravarient Metric Tensor
    RCP<ParameterList> p =
      rcp(new ParameterList("Contravarient Metric Tensor"));

    // Inputs: X, Y at nodes, Cubature, and Basis
    p->set<std::string>("Coordinate Vector Name","Coord Vec");
    p->set< RCP<Intrepid2::Cubature<PHX::Device> > >("Cubature", cubature);

    p->set<RCP<shards::CellTopology> >("Cell Type", cellType);

    // Outputs: BF, weightBF, Grad BF, weighted-Grad BF, all in physical space
    p->set<std::string>("Contravarient Metric Tensor Name", "Gc");

    ev = rcp(new Tsunami::NavierStokesContravarientMetricTensor<EvalT,AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }


  // Body Force
  {
    RCP<ParameterList> p = rcp(new ParameterList("Body Force"));

    //Input
    p->set<std::string>("Coordinate Vector Name", "Coord Vec");

    Teuchos::ParameterList& paramList = params->sublist("Body Force");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);
    p->set<std::string>("Fluid Viscosity QP Name", "Viscosity Field");

    //Output
    p->set<std::string>("Body Force Name", "Body Force");

    ev = rcp(new Tsunami::NavierStokesBodyForce<EvalT,AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  { //Parameters
    RCP<ParameterList> p = rcp(new ParameterList("Parameters"));
    //Input
    p->set<std::string>("Fluid Viscosity In QP Name", "viscosity");
    p->set<std::string>("Fluid Density In QP Name", "density");
    p->set<double>("Viscosity", mu); 
    p->set<double>("Density", rho); 
    p->set<bool>("Use Parameters on Mesh", use_params_on_mesh); 
    p->set<bool>("Enable Memoizer", enableMemoizer);

    //Output
    p->set<std::string>("Fluid Viscosity QP Name", "Viscosity Field");
    p->set<std::string>("Fluid Density QP Name", "Density Field");

    ev = rcp(new Tsunami::NavierStokesParameters<EvalT,AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  { // Rm
    RCP<ParameterList> p = rcp(new ParameterList("Rm"));

    //Input
    p->set<std::string>("Velocity QP Variable Name", "Velocity");
    p->set<std::string>("Velocity Dot QP Variable Name", "Velocity_dot");
    p->set<std::string>("Velocity Gradient QP Variable Name", "Velocity Gradient");
    p->set<std::string>("Pressure Gradient QP Variable Name", "Pressure Gradient");
    p->set<std::string>("Body Force QP Variable Name", "Body Force");
    p->set<std::string>("Coordinate Vector Name", "Coord Vec");
    p->set<bool>("Have Advection Term", haveAdvection); 
    p->set<bool>("Have Transient Term", haveUnsteady); 
    p->set<std::string>("Fluid Density QP Name", "Density Field");

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);

    //Output
    p->set<std::string>("Rm Name", "Rm");

    ev = rcp(new Tsunami::NavierStokesRm<EvalT,AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (havePSPG) { // Tau PSPG/SUPG
    RCP<ParameterList> p = rcp(new ParameterList("Tau"));

    //Input
    p->set<std::string>("Velocity QP Variable Name", "Velocity");
    p->set<std::string>("Contravarient Metric Tensor Name", "Gc");
    p->set<std::string>("Jacobian Det Name", "Jacobian Det");
    p->set<string>("Jacobian Name",          "Jacobian");
    p->set<string>("Jacobian Inv Name",      "Jacobian Inv");
    p->set<std::string>("Fluid Viscosity QP Name", "Viscosity Field");
    p->set<std::string>("Fluid Density QP Name", "Density Field");
    p->set<std::string>("Stabilization Type", stabType); 

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Tau");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    //Output
    p->set<std::string>("Tau Name", "Tau");

    ev = rcp(new Tsunami::NavierStokesTau<EvalT,AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }
  

  { // Momentum Resid
    RCP<ParameterList> p = rcp(new ParameterList("Momentum Resid"));

    //Input
    p->set<std::string>("Weighted BF Name", Albany::weighted_bf_name);
    p->set<std::string>("Weighted Gradient BF Name", Albany::weighted_grad_bf_name);
    p->set<std::string>("Velocity QP Variable Name", "Velocity");
    p->set<std::string>("Velocity Gradient QP Variable Name", "Velocity Gradient");
    p->set<std::string>("Pressure QP Variable Name", "Pressure");
    p->set<std::string>("Pressure Gradient QP Variable Name", "Pressure Gradient");
    p->set<std::string>("Body Force Name", "Body Force");
    p->set<std::string> ("Rm Name", "Rm");
    p->set<std::string> ("Tau Name", "Tau");
    p->set<std::string>("Fluid Viscosity QP Name", "Viscosity Field");

    p->set<std::string>("Velocity QP Variable Name", "Velocity");
    p->set<bool> ("Have SUPG", haveSUPG);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);

    //Output
    p->set<std::string>("Residual Name", "Momentum Residual");

    ev = rcp(new Tsunami::NavierStokesMomentumResid<EvalT,AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }


  { // Continuity Resid
    RCP<ParameterList> p = rcp(new ParameterList("Continuity Resid"));

    //Input
    p->set<std::string>("Weighted BF Name", Albany::weighted_bf_name);
    p->set<std::string>("Velocity QP Variable Name", "Velocity");
    p->set<std::string>("Gradient QP Variable Name", "Velocity Gradient");
    p->set<bool> ("Have PSPG", havePSPG);
    p->set<std::string>("Fluid Density QP Name", "Density Field");

    p->set<std::string>("Weighted Gradient BF Name", Albany::weighted_grad_bf_name);
    p->set<std::string> ("Tau Name", "Tau");
    p->set<std::string> ("Rm Name", "Rm");

    //Output
    p->set<std::string>("Residual Name", "Continuity Residual");

    ev = rcp(new Tsunami::NavierStokesContinuityResid<EvalT,AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (fieldManagerChoice == Albany::BUILD_RESID_FM)  {
    Teuchos::RCP<const PHX::FieldTag> ret_tag;
    PHX::Tag<typename EvalT::ScalarT> mom_tag("Scatter Momentum", dl->dummy);
    fm0.requireField<EvalT>(mom_tag);
    PHX::Tag<typename EvalT::ScalarT> con_tag("Scatter Continuity", dl->dummy);
    fm0.requireField<EvalT>(con_tag);
    ret_tag = mom_tag.clone();
    return ret_tag;
  }
  else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM) {
    Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
    return respUtils.constructResponses(fm0, *responseList, stateMgr);
  }

  return Teuchos::null;
}
#endif // TSUNAMI_NAVIERSTOKES_HPP
