//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_STOKES_HPP
#define LANDICE_STOKES_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_AbstractProblem.hpp"
#include "Albany_GeneralPurposeFieldsNames.hpp"

#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"

namespace LandIce {

  /*!
   * \brief Abstract interface for representing a 1-D finite element
   * problem.
   */
  class Stokes : public Albany::AbstractProblem {
  public:

    //! Default constructor
    Stokes(const Teuchos::RCP<Teuchos::ParameterList>& params,
     const Teuchos::RCP<ParamLib>& paramLib,
     const int numDim_);

    //! Destructor
    ~Stokes();

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
    Stokes(const Stokes&);

    //! Private to prohibit copying
    Stokes& operator=(const Stokes&);

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

    unsigned int numDim;        //! number of spatial dimensions

    NS_VAR_TYPE flowType; //! type of flow variables

    bool haveFlow;     //! have flow variables (momentum+continuity)

    bool haveFlowEq;     //! have flow equations (momentum+continuity)

    bool haveSource;   //! have source term in heat equation
    bool havePSPG;     //! have pressure stabilization

    Teuchos::RCP<Albany::Layouts> dl;

    /// Boolean marking whether SDBCs are used
    bool use_sdbcs_;

    //! Problem PL 
    const Teuchos::RCP<Teuchos::ParameterList> params;

  };

}

#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"

#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_EvaluatorUtils.hpp"
#include "LandIce_ResponseUtilities.hpp"

#include "LandIce_StokesContravarientMetricTensor.hpp"
#include "PHAL_Neumann.hpp"
#include "PHAL_SharedParameter.hpp"
#include "LandIce_ParamEnum.hpp"
#include "LandIce_StokesBodyForce.hpp"
#include "LandIce_StokesRm.hpp"
#include "LandIce_StokesTauM.hpp"
#include "LandIce_StokesMomentumResid.hpp"
#include "LandIce_StokesContinuityResid.hpp"
#include "LandIce_Viscosity.hpp"

template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
LandIce::Stokes::constructEvaluators(
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
   int offset=0;

   // Temporary variable used numerous times below
   Teuchos::RCP<PHX::Evaluator<AlbanyTraits> > ev;

   // Define Field Names

   if (haveFlowEq) {
     Teuchos::ArrayRCP<std::string> dof_names(1);
     Teuchos::ArrayRCP<std::string> dof_names_dot(1);
     Teuchos::ArrayRCP<std::string> resid_names(1);
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
   }

   if (haveFlowEq) {
     Teuchos::ArrayRCP<std::string> dof_names(1);
     Teuchos::ArrayRCP<std::string> dof_names_dot(1);
     Teuchos::ArrayRCP<std::string> resid_names(1);
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
   }


   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructGatherCoordinateVectorEvaluator());

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructMapToPhysicalFrameEvaluator(cellType, cubature));

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructComputeBasisFunctionsEvaluator(cellType, intrepidBasis, cubature));

  if (havePSPG) { // Compute Contravarient Metric Tensor
    RCP<ParameterList> p =
      rcp(new ParameterList("Contravarient Metric Tensor"));

    // Inputs: X, Y at nodes, Cubature, and Basis
    p->set<std::string>("Coordinate Vector Name","Coord Vec");
    p->set< RCP<Intrepid2::Cubature<PHX::Device> > >("Cubature", cubature);

    p->set<RCP<shards::CellTopology> >("Cell Type", cellType);

    // Outputs: BF, weightBF, Grad BF, weighted-Grad BF, all in physical space
    p->set<std::string>("Contravarient Metric Tensor Name", "Gc");

    ev = rcp(new LandIce::StokesContravarientMetricTensor<EvalT,AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }



  if (haveFlowEq) { // Body Force
    RCP<ParameterList> p = rcp(new ParameterList("Body Force"));

    //Input
    p->set<std::string>("LandIce Viscosity QP Variable Name", "LandIce Viscosity");
    p->set<std::string>("Coordinate Vector Name", "Coord Vec");

    Teuchos::ParameterList& paramList = params->sublist("Body Force");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);


    //Output
    p->set<std::string>("Body Force Name", "Body Force");

    ev = rcp(new LandIce::StokesBodyForce<EvalT,AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }



  if (haveFlowEq) { // Rm
    RCP<ParameterList> p = rcp(new ParameterList("Rm"));

    //Input
    p->set<std::string>("Velocity QP Variable Name", "Velocity");
    p->set<std::string>("Velocity Dot QP Variable Name", "Velocity_dot");
    p->set<std::string>("Velocity Gradient QP Variable Name", "Velocity Gradient");
    p->set<std::string>("Pressure Gradient QP Variable Name", "Pressure Gradient");
    p->set<std::string>("Body Force QP Variable Name", "Body Force");
    p->set<std::string>("Coordinate Vector Name", "Coord Vec");

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);

    //Output
    p->set<std::string>("Rm Name", "Rm");

    ev = rcp(new LandIce::StokesRm<EvalT,AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  //IK, 7/24/2012
  if (haveFlowEq) { // LandIce viscosity
    RCP<ParameterList> p = rcp(new ParameterList("LandIce Viscosity"));

    //Input
    p->set<std::string>("Velocity Gradient QP Variable Name", "Velocity Gradient");
    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("LandIce Viscosity");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);
    p->set<std::string>("Coordinate Vector Name", "Coord Vec");

    //Output
    p->set<std::string>("LandIce Viscosity QP Variable Name", "LandIce Viscosity");

    ev = rcp(new LandIce::Viscosity<EvalT,AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);

    //--- Shared Parameter for Continuation:  ---//
    p = rcp(new ParameterList("Homotopy Parameter"));

    std::string param_name = "Glen's Law Homotopy Parameter";
    p->set<std::string>("Parameter Name", param_name);
    p->set< RCP<ParamLib> >("Parameter Library", paramLib);
    p->set<const Teuchos::ParameterList*>("Parameters List", &params->sublist("Parameters"));
    p->set<double>("Default Nominal Value", params->sublist("LandIce Viscosity").get<double>(param_name,-1.0));

    RCP<PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Homotopy>> ptr_homotopy;
    ptr_homotopy = rcp(new PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Homotopy>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ptr_homotopy);
  }


  if (haveFlowEq && havePSPG) { // Tau M
    RCP<ParameterList> p = rcp(new ParameterList("Tau M"));

    //Input
    p->set<std::string>("Velocity QP Variable Name", "Velocity");
    p->set<std::string>("Contravarient Metric Tensor Name", "Gc");
    p->set<std::string>("Jacobian Det Name", "Jacobian Det");
    p->set<string>("Jacobian Name",          "Jacobian");
    p->set<string>("Jacobian Inv Name",          "Jacobian Inv");
    p->set<std::string>("LandIce Viscosity QP Variable Name", "LandIce Viscosity");

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Tau M");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    //Output
    p->set<std::string>("Tau M Name", "Tau M");

    ev = rcp(new LandIce::StokesTauM<EvalT,AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (haveFlowEq) { // Momentum Resid
    RCP<ParameterList> p = rcp(new ParameterList("Momentum Resid"));

    //Input
    p->set<std::string>("Weighted BF Name", Albany::weighted_bf_name);
    p->set<std::string>("Weighted Gradient BF Name", Albany::weighted_grad_bf_name);
    p->set<std::string>("Velocity QP Variable Name", "Velocity");
    p->set<std::string>("Velocity Gradient QP Variable Name", "Velocity Gradient");
    p->set<std::string>("Pressure QP Variable Name", "Pressure");
    p->set<std::string>("Pressure Gradient QP Variable Name", "Pressure Gradient");
    p->set<std::string>("LandIce Viscosity QP Variable Name", "LandIce Viscosity");
    p->set<std::string>("Body Force Name", "Body Force");

    p->set<std::string>("Velocity QP Variable Name", "Velocity");
    p->set<std::string>("Density QP Variable Name", "Density");
    p->set<std::string> ("Tau M Name", "Tau M");

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);

    //Output
    p->set<std::string>("Residual Name", "Momentum Residual");

    ev = rcp(new LandIce::StokesMomentumResid<EvalT,AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }


  if (haveFlowEq) { // Continuity Resid
    RCP<ParameterList> p = rcp(new ParameterList("Continuity Resid"));

    //Input
    p->set<std::string>("Weighted BF Name", Albany::weighted_bf_name);
    p->set<std::string>("Velocity QP Variable Name", "Velocity");
    p->set<std::string>("Gradient QP Variable Name", "Velocity Gradient");
    p->set<std::string>("Density QP Variable Name", "Density");

    p->set<bool>("Have PSPG", havePSPG);
    p->set<std::string>("Weighted Gradient BF Name", Albany::weighted_grad_bf_name);
    p->set<std::string> ("Tau M Name", "Tau M");
    p->set<std::string> ("Rm Name", "Rm");

    //Output
    p->set<std::string>("Residual Name", "Continuity Residual");

    ev = rcp(new LandIce::StokesContinuityResid<EvalT,AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }


  Albany::StateStruct::MeshFieldEntity entity= Albany::StateStruct::NodalDistParameter;

  //basal friction
  {
    std::string elementBlockName = meshSpecs.ebName;
    std::string stateName("basal_friction");
    RCP<ParameterList> p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName,true, &entity, "");
  }

  entity= Albany::StateStruct::NodalDataToElemNode;

  //surface_height
  {
    std::string elementBlockName = meshSpecs.ebName;
    std::string stateName("surface_height");
    RCP<ParameterList> p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName,true, &entity);

    ev = rcp(new PHAL::LoadStateField<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }


  if (fieldManagerChoice == Albany::BUILD_RESID_FM)  {
    Teuchos::RCP<const PHX::FieldTag> ret_tag;
    if (haveFlowEq) {
      PHX::Tag<typename EvalT::ScalarT> mom_tag("Scatter Momentum", dl->dummy);
      fm0.requireField<EvalT>(mom_tag);
      PHX::Tag<typename EvalT::ScalarT> con_tag("Scatter Continuity", dl->dummy);
      fm0.requireField<EvalT>(con_tag);
      ret_tag = mom_tag.clone();
    }
    return ret_tag;
  }
  else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM) {
    LandIce::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
    return respUtils.constructResponses(fm0, *responseList, stateMgr);
  }

  return Teuchos::null;
}
#endif // LANDICE_STOKES_HPP
