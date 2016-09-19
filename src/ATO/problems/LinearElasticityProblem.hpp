//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LINEARELASTICITYPROBLEM_HPP
#define LINEARELASTICITYPROBLEM_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_AbstractProblem.hpp"
#include "Albany_BCUtils.hpp"
#include "ATO_OptimizationProblem.hpp"
#include "ATO_Mixture.hpp"
#include "ATO_Utils.hpp"

#include "Phalanx.hpp"
#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"
#include "PHAL_AlbanyTraits.hpp"

#ifdef ATO_USES_COGENT
#include <Cogent_Integrator.hpp>
#endif

namespace Albany {

  /*!
   * \brief Abstract interface for representing a 2-D finite element
   * problem.
   */
  class LinearElasticityProblem : 
    public ATO::OptimizationProblem ,
    public virtual Albany::AbstractProblem {
  public:
  
    //! Default constructor
    LinearElasticityProblem(
		      const Teuchos::RCP<Teuchos::ParameterList>& params_,
		      const Teuchos::RCP<ParamLib>& paramLib_,
		      const int numDim_);

    //! Destructor
    virtual ~LinearElasticityProblem();

    //! Return number of spatial dimensions
    virtual int spatialDimension() const { return numDim; }

    //! Build the PDE instantiations, boundary conditions, and initial solution
    virtual void buildProblem(
      Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
      StateManager& stateMgr);

    // Build evaluators
    virtual Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
    buildEvaluators(
      PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
      const Albany::MeshSpecsStruct& meshSpecs,
      Albany::StateManager& stateMgr,
      Albany::FieldManagerChoice fmchoice,
      const Teuchos::RCP<Teuchos::ParameterList>& responseList);

    //! Each problem must generate it's list of valid parameters
    Teuchos::RCP<const Teuchos::ParameterList> getValidProblemParameters() const;

    void getAllocatedStates(
      Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Kokkos::DynRankView<RealType, PHX::Device> > > > oldState_,
      Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Kokkos::DynRankView<RealType, PHX::Device> > > > newState_) const;

  private:

    //! Private to prohibit copying
    LinearElasticityProblem(const LinearElasticityProblem&);
    
    //! Private to prohibit copying
    LinearElasticityProblem& operator=(const LinearElasticityProblem&);

  public:

    //! Main problem setup routine. Not directly called, but indirectly by following functions
    template <typename EvalT> 
    Teuchos::RCP<const PHX::FieldTag>
    constructEvaluators(
      PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
      const Albany::MeshSpecsStruct& meshSpecs,
      Albany::StateManager& stateMgr,
      Albany::FieldManagerChoice fmchoice,
      const Teuchos::RCP<Teuchos::ParameterList>& responseList);

    void constructDirichletEvaluators(const Albany::MeshSpecsStruct& meshSpecs);
    void constructNeumannEvaluators(const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs);


  protected:

    int numDim;

    Teuchos::RCP<Albany::Layouts> dl;

    Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Kokkos::DynRankView<RealType, PHX::Device> > > > oldState;
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Kokkos::DynRankView<RealType, PHX::Device> > > > newState;

//    Teuchos::RCP<QCAD::MaterialDatabase> material_db_;

  };

}

#include "Albany_SolutionAverageResponseFunction.hpp"
#include "Albany_SolutionTwoNormResponseFunction.hpp"
#include "Albany_SolutionMaxValueResponseFunction.hpp"
#include "Albany_Utils.hpp"
#include "Albany_BCUtils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_ResponseUtilities.hpp"
#include "Albany_EvaluatorUtils.hpp"

#include "PHAL_GatherScalarNodalParameter.hpp"
#include "PHAL_Source.hpp"
#include "Strain.hpp"
#include "ATO_Stress.hpp"
#include "ATO_BodyForce.hpp"
#include "ATO_AddForce.hpp"
#include "ATO_TopologyFieldWeighting.hpp"
#include "ATO_TopologyWeighting.hpp"
#ifdef ATO_USES_COGENT
#include "ATO_ComputeBasisFunctions.hpp"
#endif
#include "PHAL_SaveStateField.hpp"
#include "ElasticityResid.hpp"

#include "Time.hpp"

template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
Albany::LinearElasticityProblem::constructEvaluators(
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
   using PHAL::AlbanyTraits;

  // get the name of the current element block
   std::string elementBlockName = meshSpecs.ebName;

   RCP<shards::CellTopology> cellType = rcp(new shards::CellTopology (&meshSpecs.ctd));
   RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> >
     intrepidBasis = Albany::getIntrepid2Basis(meshSpecs.ctd);

   const int numNodes = intrepidBasis->getCardinality();
   const int worksetSize = meshSpecs.worksetSize;

   int cubatureDegree = meshSpecs.cubatureDegree;

   std::string blockType = "Body";

#ifdef ATO_USES_COGENT
   bool isNonconformal = false;
   Teuchos::ParameterList geomSpec, blockSpec;
   if(params->isSublist("Configuration")){
     if(params->sublist("Configuration").isType<bool>("Nonconformal"))
       isNonconformal = params->sublist("Configuration").get<bool>("Nonconformal");
   }
   RCP<Cogent::Integrator> projector;
   if(isNonconformal){
     // find geom spec
     Teuchos::ParameterList& blocksParams = params->sublist("Configuration").sublist("Element Blocks");
     int nBlocks = blocksParams.get<int>("Number of Element Blocks");
     std::string specName;
     for(int i=0; i<nBlocks; i++){
       std::string specName_i = Albany::strint("Element Block", i);
       blockSpec = blocksParams.sublist(specName_i);
       if( blockSpec.get<std::string>("Name") == elementBlockName ){
         geomSpec = blockSpec.sublist("Geometry Construction");
         if( geomSpec.get<bool>("Uniform Quadrature") ){
           specName = specName_i;
           break;
         } else {
           TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                                    "Nonconformal method requires 'Uniform Quadrature'");
         }
       }
     }
     Cogent::IntegratorFactory integratorFactory;
     projector = integratorFactory.create(cellType, intrepidBasis, geomSpec);
//     projector = rcp(new Cogent::Integrator(cellType, intrepidBasis, geomSpec));

     int projectionOrder = geomSpec.get<int>("Projection Order");
     cubatureDegree = 2*projectionOrder;

     blockType = geomSpec.get<std::string>("Type");

   }
#endif

   Intrepid2::DefaultCubatureFactory cubFactory;
   RCP <Intrepid2::Cubature<PHX::Device> > cubature = cubFactory.create<PHX::Device, RealType, RealType>(*cellType, cubatureDegree);

   const int numQPts = cubature->getNumPoints();
   const int numVertices = cellType->getNodeCount();
   const int numDim = cellType->getDimension();


   *out << "Field Dimensions: Workset=" << worksetSize 
        << ", Vertices= " << numVertices
        << ", Nodes= " << numNodes
        << ", QuadPts= " << numQPts
        << ", Dim= " << numDim << std::endl;


   // Construct standard FEM evaluators with standard field names                              
   dl = rcp(new Albany::Layouts(worksetSize,numVertices,numNodes,numQPts,numDim));
   TEUCHOS_TEST_FOR_EXCEPTION(dl->vectorAndGradientLayoutsAreEquivalent==false, std::logic_error,
                              "Data Layout Usage in Mechanics problems assume vecDim = numDim");

   Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
   ATO::Utils<EvalT, PHAL::AlbanyTraits> atoUtils(dl, numDim);


   // Temporary variable used numerous times below
   Teuchos::RCP<PHX::Evaluator<AlbanyTraits> > ev;

#ifdef ATO_USES_COGENT
   if( isNonconformal ){
     Teuchos::Array<std::string> topoNames = geomSpec.get<Teuchos::Array<std::string> >("Level Set Names");
     int numNames = topoNames.size();
     for(int i=0; i<numNames; i++){
       RCP<ParameterList> p = rcp(new ParameterList);
       Albany::StateStruct::MeshFieldEntity entity = Albany::StateStruct::NodalDataToElemNode;
       p = stateMgr.registerStateVariable(topoNames[i], dl->node_scalar, "all", true, &entity);
     }
   }
#endif

   std::string stressName("Stress");
   std::string strainName("Strain");

   std::string bodyForceName("Body Force");
   std::string boundaryForceName("Boundary Force");
   std::string residStressName("Not Set");


   Teuchos::ArrayRCP<std::string> dof_names(1);
   dof_names[0] = "Displacement";
   Teuchos::ArrayRCP<std::string> resid_names(1);
   resid_names[0] = dof_names[0]+" Residual";
   

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructDOFVecInterpolationEvaluator(dof_names[0]));

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructDOFVecGradInterpolationEvaluator(dof_names[0]));

   fm0.template registerEvaluator<EvalT>
       (evalUtils.constructGatherSolutionEvaluator_noTransient(/*is_vector_dof=*/ true, dof_names));

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructGatherCoordinateVectorEvaluator());

     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructMapToPhysicalFrameEvaluator(cellType, cubature));

#ifdef ATO_USES_COGENT
   if( isNonconformal ){

     RCP<ParameterList> p = rcp(new ParameterList("Compute Basis Functions"));

     //Input
     p->set<std::string>("Gradient QP Variable Name", "Displacement Gradient");

     p->set< RCP<Cogent::Integrator> >("Cubature",     projector);
     p->set<std::string>("Coordinate Vector Name",   "Coord Vec");
     p->set<std::string>("Weights Name",               "Weights");
     p->set<std::string>("Jacobian Det Name",     "Jacobian Det");
     p->set<std::string>("Jacobian Name",             "Jacobian");
     p->set<std::string>("Jacobian Inv Name",     "Jacobian Inv");
     p->set<std::string>("BF Name",                         "BF");
     p->set<std::string>("Weighted BF Name",               "wBF");
     p->set<std::string>("Gradient BF Name",           "Grad BF");
     p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
 
     ev = rcp(new ATO::ComputeBasisFunctions<EvalT,AlbanyTraits>(*p,dl,&meshSpecs));
     fm0.template registerEvaluator<EvalT>(ev);
 
   } else
#endif
     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructComputeBasisFunctionsEvaluator(cellType, intrepidBasis, cubature));





   


  { // Time
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<std::string>("Time Name", "Time");
    p->set<std::string>("Delta Time Name", "Delta Time");
    p->set< RCP<DataLayout> >("Workset Scalar Data Layout", dl->workset_scalar);
    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    p->set<bool>("Disable Transient", true);

    ev = rcp(new LCM::Time<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
    p = stateMgr.registerStateVariable("Time",dl->workset_scalar, dl->dummy, 
                                       elementBlockName, "scalar", 0.0, true);
    ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  { // Strain
    RCP<ParameterList> p = rcp(new ParameterList(strainName));

    //Input
    p->set<std::string>("Gradient QP Variable Name", "Displacement Gradient");

    //Output
    p->set<std::string>("Strain Name", strainName);

    ev = rcp(new LCM::Strain<EvalT,AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);

    // state the strain in the state manager for ATO
    p = stateMgr.registerStateVariable(strainName, dl->qp_tensor, dl->dummy, 
                                       elementBlockName, "scalar", 0.0, false, false);
    ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

    //if(some input stuff)
    atoUtils.SaveCellStateField(fm0, stateMgr, strainName, elementBlockName, dl->qp_tensor);
  }

  if( blockType == "Body" ){

    // Linear elasticity stress
    //
    atoUtils.constructStressEvaluators( params, fm0, stateMgr, elementBlockName, stressName, strainName );
   
    
    // Body forces
    //
    atoUtils.constructBodyForceEvaluators( params, fm0, stateMgr, elementBlockName, bodyForceName );
   
    // Residual Strains
    //
    if( params->isSublist("Residual Strain") ){
      residStressName = "Residual Stress";
      atoUtils.constructResidualStressEvaluators( params, fm0, stateMgr, elementBlockName, residStressName );
    }
  } else 
  if( blockType == "Boundary" )
#ifdef ATO_USES_COGENT
  {
    // Boundary forces
    //
    atoUtils.constructBoundaryConditionEvaluators( blockSpec, fm0, stateMgr, elementBlockName, boundaryForceName );
  }
#else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                               "Cogent not enabled. 'Boundary' block type not available.");
  }
#endif
 

  /*******************************************************************************/
  /** Begin topology weighting ***************************************************/
  /*******************************************************************************/
  if(params->isType<Teuchos::RCP<ATO::TopologyArray> >("Topologies"))
  {
    Teuchos::RCP<ATO::TopologyArray> 
      topologyArray = params->get<Teuchos::RCP<ATO::TopologyArray> >("Topologies");

    Teuchos::ParameterList& wfParams = params->sublist("Apply Topology Weight Functions");
    int nfields = wfParams.get<int>("Number of Fields");
    for(int ifield=0; ifield<nfields; ifield++){
      Teuchos::ParameterList& fieldParams = wfParams.sublist(Albany::strint("Field", ifield));

      int topoIndex  = fieldParams.get<int>("Topology Index");
      std::string fieldName  = fieldParams.get<std::string>("Name");
      std::string layoutName = fieldParams.get<std::string>("Layout");
      int functionIndex      = fieldParams.get<int>("Function Index");

      std::string reqBlockType;
      if( fieldParams.isType<std::string>("Type") )
        reqBlockType = fieldParams.get<std::string>("Type");
      else
        reqBlockType = "Body";

      if( reqBlockType != blockType ) continue;

      Teuchos::RCP<PHX::DataLayout> layout;
      if( layoutName == "QP Scalar" ) layout = dl->qp_scalar;
      else
      if( layoutName == "QP Vector" ) layout = dl->qp_vector;
      else
      if( layoutName == "QP Tensor" ) layout = dl->qp_tensor;

      Teuchos::RCP<ATO::Topology> topology = (*topologyArray)[topoIndex];

      // Get distributed parameter
      if( topology->getEntityType() == "Distributed Parameter" ){
        RCP<ParameterList> p = rcp(new ParameterList("Distributed Parameter"));
        p->set<std::string>("Parameter Name", topology->getName());
        ev = rcp(new PHAL::GatherScalarNodalParameter<EvalT,AlbanyTraits>(*p, dl) );
        fm0.template registerEvaluator<EvalT>(ev);
      }
  
      RCP<ParameterList> p = rcp(new ParameterList("TopologyWeighting"));

      p->set<Teuchos::RCP<ATO::Topology> >("Topology",topology);

      p->set<std::string>("BF Name", "BF");
      p->set<std::string>("Unweighted Variable Name", fieldName);
      p->set<std::string>("Weighted Variable Name", fieldName+"_Weighted");
      p->set<std::string>("Variable Layout", layoutName);
      p->set<int>("Function Index", functionIndex);

      if( topology->getEntityType() == "Distributed Parameter" )
        ev = rcp(new ATO::TopologyFieldWeighting<EvalT,AlbanyTraits>(*p,dl));
      else
        ev = rcp(new ATO::TopologyWeighting<EvalT,AlbanyTraits>(*p,dl));

      fm0.template registerEvaluator<EvalT>(ev);

      //if(some input stuff)
      atoUtils.SaveCellStateField(fm0, stateMgr, fieldName+"_Weighted", 
                                  elementBlockName, layout);
    }
  }
  /*******************************************************************************/
  /** End topology weighting *****************************************************/
  /*******************************************************************************/

  if( blockType == "Body" )
  {
   { // Displacement Resid 
    RCP<ParameterList> p = rcp(new ParameterList("Displacement Resid"));

    p->set<bool>("Disable Transient", true);

    //Input
    if( params->isType<Teuchos::RCP<ATO::TopologyArray> > ("Topologies") )
      p->set<std::string>("Stress Name", stressName+"_Weighted");
    else 
      p->set<std::string>("Stress Name", stressName);
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);

    p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);

    //Output
    p->set<std::string>("Residual Name", resid_names[0]);
    p->set< RCP<DataLayout> >("Node Vector Data Layout", dl->node_vector);

    ev = rcp(new LCM::ElasticityResid<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
   }

   if( params->isSublist("Body Force") )
   {
    RCP<ParameterList> p = rcp(new ParameterList("Body Forces"));
    if( params->isType<Teuchos::RCP<ATO::TopologyArray> > ("Topologies") )
      p->set<std::string>("Force Name", bodyForceName+"_Weighted");
    else 
      p->set<std::string>("Force Name", bodyForceName);
    p->set< RCP<DataLayout> >("Force Data Layout", dl->qp_vector);
    p->set<std::string>("Weighted BF Name", "wBF");
    p->set< RCP<DataLayout> >("Weighted BF Data Layout", dl->node_qp_scalar);
    p->set<std::string>("In Residual Name", resid_names[0]);
    p->set< RCP<DataLayout> >("Node Vector Data Layout", dl->node_vector);
    resid_names[0] += " with Body Force";
    p->set<std::string>("Out Residual Name", resid_names[0]);
    p->set< RCP<DataLayout> >("Node Vector Data Layout", dl->node_vector);
    p->set<bool>("Negative",true);
    ev = rcp(new ATO::AddForce<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
   }

   if( params->isSublist("Residual Strain") )
   {
     {
      //Compute divergence of the residual stress
      RCP<ParameterList> p = rcp(new ParameterList("Residual Stress Divergence"));

      p->set<bool>("Disable Transient", true);

      //Input
      if( params->isType<Teuchos::RCP<ATO::TopologyArray> > ("Topologies") )
        p->set<std::string>("Stress Name", residStressName+"_Weighted");
      else 
        p->set<std::string>("Stress Name", residStressName);
      p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);
  
      p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
      p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);
  
      //Output
      p->set<std::string>("Residual Name", "Residual Force");
      p->set< RCP<DataLayout> >("Node Vector Data Layout", dl->node_vector);

      ev = rcp(new LCM::ElasticityResid<EvalT,AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
     }
     {
      RCP<ParameterList> p = rcp(new ParameterList("Add Residual Force"));
      p->set<std::string>("Force Name", "Residual Force");
      p->set< RCP<DataLayout> >("Force Data Layout", dl->node_vector);
      p->set<std::string>("In Residual Name", resid_names[0]);
      p->set< RCP<DataLayout> >("Node Vector Data Layout", dl->node_vector);
      resid_names[0] += " with Residual Force";
      p->set<std::string>("Out Residual Name", resid_names[0]);
      p->set< RCP<DataLayout> >("Node Vector Data Layout", dl->node_vector);
      ev = rcp(new ATO::AddForce<EvalT,AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
     }
   }
  } else
  if( blockType == "Boundary" )
#ifdef ATO_USES_COGENT
  {
    RCP<ParameterList> p = rcp(new ParameterList("Boundary Forces"));
    if( params->isType<Teuchos::RCP<ATO::TopologyArray> > ("Topologies") )
      p->set<std::string>("Force Name", boundaryForceName+"_Weighted");
    else 
      p->set<std::string>("Force Name", boundaryForceName);
    p->set<std::string>("Weighted BF Name", "wBF");
    resid_names[0] = "Boundary Force";
    p->set<std::string>("Out Residual Name", resid_names[0]);
    p->set< RCP<DataLayout> >("Force Data Layout", dl->qp_vector);
    p->set< RCP<DataLayout> >("Weighted BF Data Layout", dl->node_qp_scalar);
    p->set< RCP<DataLayout> >("Node Vector Data Layout", dl->node_vector);
    p->set<bool>("Negative",true);
    ev = rcp(new ATO::AddForce<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  
  }
#else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                               "Cogent not enabled. 'Boundary' block type not available.");
  }
#endif

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructScatterResidualEvaluator(/*is_vector_dof=*/ true, resid_names));

  if (fieldManagerChoice == Albany::BUILD_RESID_FM)  {
    PHX::Tag<typename EvalT::ScalarT> res_tag("Scatter", dl->dummy);
    fm0.requireField<EvalT>(res_tag);

    return res_tag.clone();
  }
  else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM) {
    Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
    return respUtils.constructResponses(fm0, *responseList, params, stateMgr, &meshSpecs);
  }

  return Teuchos::null;
}

#endif // LINEARELASTICITYPROBLEM_HPP
