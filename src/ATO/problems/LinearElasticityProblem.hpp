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

#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"
#include "PHAL_AlbanyTraits.hpp"

#ifdef ATO_USES_COGENT
#include <Cogent_Integrator.hpp>
#include <Cogent_IntegratorFactory.hpp>
#endif

// TODO:  Fix the field weighting.  Below assumes that stresses are weighted, but 
// the user can decide to weight stress or not.

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

    //! Get boolean telling code if SDBCs are utilized  
    virtual bool useSDBCs() const {return use_sdbcs_; }

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

//    Teuchos::RCP<Albany::MaterialDatabase> material_db_;

    /// Boolean marking whether SDBCs are used 
    bool use_sdbcs_; 

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

   bool blockHasBody = true;
   int numBoundaries = 0;

#ifdef ATO_USES_COGENT
   bool isNonconformal = false;
   Teuchos::ParameterList geomSpec, blockSpec;
   if(params->isSublist("Configuration")){
     if(params->sublist("Configuration").isType<bool>("Nonconformal"))
       isNonconformal = params->sublist("Configuration").get<bool>("Nonconformal");
   }
   RCP<Cogent::Integrator> bodyProjector;
   if(isNonconformal){
     // find geom spec
     Teuchos::ParameterList& blocksParams = params->sublist("Configuration").sublist("Element Blocks");
     int nBlocks = blocksParams.get<int>("Number of Element Blocks");
     bool foundBlockSpec = false;
     for(int i=0; i<nBlocks; i++){
       std::string specName_i = Albany::strint("Element Block", i);
       blockSpec = blocksParams.sublist(specName_i);
       if( blockSpec.get<std::string>("Name") == elementBlockName ){
         foundBlockSpec = true;
         geomSpec = blockSpec.sublist("Geometry Construction");
         TEUCHOS_TEST_FOR_EXCEPTION(!(geomSpec.get<bool>("Uniform Quadrature")), std::logic_error,
                                  "Nonconformal method requires 'Uniform Quadrature'");
       }
     }

     TEUCHOS_TEST_FOR_EXCEPTION(!foundBlockSpec, std::logic_error,
                               "Configuration: Element block definition not found for " << elementBlockName);

     // 'Projection Order' and 'Uniform Quadrature' are specified below the 
     // 'Geometry Construction' block.  Set the values in the 'Body' and 'Boundary X' 
     // sublists for later parsing in Cogent.
     int projectionOrder = geomSpec.get<int>("Projection Order");
     int errorChecking = geomSpec.get("Error Checking",0);
     cubatureDegree = 2*projectionOrder;
     bool uniformQuadrature = geomSpec.get<bool>("Uniform Quadrature");

     // create body projector
     blockHasBody = geomSpec.isSublist("Body");
     if(blockHasBody){
       Teuchos::ParameterList& bodySpec = geomSpec.sublist("Body");
       bodySpec.set("Projection Order", projectionOrder);
       bodySpec.set("Uniform Quadrature", uniformQuadrature);
       bodySpec.set("Error Checking", errorChecking);
       bodySpec.set("Geometry Type", "Body");
       Cogent::IntegratorFactory integratorFactory;
       bodyProjector = integratorFactory.create(cellType, intrepidBasis, bodySpec);
     }

     if( geomSpec.isType<int>("Number of Boundaries") )
       numBoundaries = geomSpec.get<int>("Number of Boundaries");
     for(int iBoundary=0; iBoundary<numBoundaries; iBoundary++){
       Teuchos::ParameterList& boundarySpec = geomSpec.sublist(Albany::strint("Boundary", iBoundary));
       boundarySpec.set("Projection Order", projectionOrder);
       boundarySpec.set("Uniform Quadrature", uniformQuadrature);
       boundarySpec.set("Error Checking", errorChecking);
       boundarySpec.set("Geometry Type", "Boundary");
     }
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
    if( blockHasBody && geomSpec.sublist("Body").isType<Teuchos::Array<std::string> >("Level Set Names") ){
      Teuchos::Array<std::string> 
        topoNames = geomSpec.sublist("Body").get<Teuchos::Array<std::string> >("Level Set Names");
      int numNames = topoNames.size();
      for(int i=0; i<numNames; i++){
        RCP<ParameterList> p = rcp(new ParameterList);
        Albany::StateStruct::MeshFieldEntity entity = Albany::StateStruct::NodalDataToElemNode;
        p = stateMgr.registerStateVariable(topoNames[i], dl->node_scalar, "all", true, &entity);
      }
    }
    for(int iBoundary=0; iBoundary<numBoundaries; iBoundary++){
      Teuchos::ParameterList boundarySpec = geomSpec.sublist(Albany::strint("Boundary", iBoundary));
      if(boundarySpec.isType<Teuchos::Array<std::string> >("Level Set Names")){
        Teuchos::Array<std::string> topoNames = boundarySpec.get<Teuchos::Array<std::string> >("Level Set Names");
        int numNames = topoNames.size();
        for(int i=0; i<numNames; i++){
          RCP<ParameterList> p = rcp(new ParameterList);
          Albany::StateStruct::MeshFieldEntity entity = Albany::StateStruct::NodalDataToElemNode;
          p = stateMgr.registerStateVariable(topoNames[i], dl->node_scalar, "all", true, &entity);
        }
      }
    }
  }
#endif

   Teuchos::ArrayRCP<std::string> dof_names(1);
   dof_names[0] = "Displacement";
   Teuchos::ArrayRCP<std::string> resid_names(1);
   resid_names[0] = "Not Set";
   

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
   if( isNonconformal && blockHasBody ){

     RCP<ParameterList> p = rcp(new ParameterList("Cogent: Compute Basis Functions"));

     //Input
     p->set< Albany::StateManager* >("State Manager Ptr", &stateMgr );
     p->set<bool>("Static Topology", true);

     p->set< RCP<Cogent::Integrator> >("Cubature", bodyProjector);
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

     atoUtils.SaveCellStateField(fm0, stateMgr, "Weights", elementBlockName, dl->qp_scalar);
 
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

  // Pre-create the GatherScalarNodalParameters evaluators for all topologies
  if(params->isType<Teuchos::RCP<ATO::TopologyArray>>("Topologies")) {
    atoUtils.constructGatherScalarParamEvaluators(params->get<Teuchos::RCP<ATO::TopologyArray> >("Topologies"),fm0);
  }

  if( blockHasBody )
  {
    std::string strainName("Strain");
    std::string stressName("Stress");
    { // Compute small strain
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
  
      atoUtils.SaveCellStateField(fm0, stateMgr, strainName, elementBlockName, dl->qp_tensor);
    }

    // Linear elasticity stress
    atoUtils.constructStressEvaluators( params, fm0, stateMgr, elementBlockName, stressName, strainName );

    // Apply user defined weighting 
    atoUtils.constructWeightedFieldEvaluators( params, fm0, stateMgr, elementBlockName, "QP Tensor", stressName );

    { // Displacement Resid (creates residual)
      RCP<ParameterList> p = rcp(new ParameterList("Displacement Resid"));

      p->set<bool>("Disable Transient", true);
      p->set<std::string>("Stress Name", stressName);
      p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);
      p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
      p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);
  
      //Output
      resid_names[0] = dof_names[0]+" Residual";
      p->set<std::string>("Residual Name", resid_names[0]);
      p->set< RCP<DataLayout> >("Node Vector Data Layout", dl->node_vector);
  
      ev = rcp(new LCM::ElasticityResid<EvalT,AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }

    if( params->isSublist("Fix Inactive Nodes") )
    { // Adds to residual
      RCP<ParameterList> p = rcp(new ParameterList("Stabilization Forces"));
      p->set<std::string>("Vector Name", dof_names[0]);
      p->set< RCP<DataLayout> >("Vector Data Layout", dl->node_vector);
      p->set<std::string>("In Residual Name", resid_names[0]);
      p->set< RCP<DataLayout> >("Node Vector Data Layout", dl->node_vector);
      resid_names[0] += " with Stabilization Force";
      p->set<std::string>("Out Residual Name", resid_names[0]);
      p->set< RCP<DataLayout> >("Node Vector Data Layout", dl->node_vector);
      ev = rcp(new ATO::AddVector<EvalT,AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }

    if( params->isSublist("Body Force") )
    {
      std::string bodyForceName("Body Force"); 
      // Create body force term
      atoUtils.constructBodyForceEvaluators( params, fm0, stateMgr, elementBlockName, bodyForceName );
      // Apply user defined weighting
      atoUtils.constructWeightedFieldEvaluators( params, fm0, stateMgr, elementBlockName, "QP Vector", bodyForceName );
      atoUtils.SaveCellStateField(fm0, stateMgr, bodyForceName, elementBlockName, dl->qp_vector);

      // Add to residual
      RCP<ParameterList> p = rcp(new ParameterList("Body Forces"));
      p->set<std::string>("Vector Name", bodyForceName);
      p->set< RCP<DataLayout> >("Vector Data Layout", dl->qp_vector);
      p->set<std::string>("Weighted BF Name", "wBF");
      p->set< RCP<DataLayout> >("Weighted BF Data Layout", dl->node_qp_scalar);
      p->set<std::string>("In Residual Name", resid_names[0]);
      p->set< RCP<DataLayout> >("Node Vector Data Layout", dl->node_vector);
      resid_names[0] += " with Body Force";
      p->set<std::string>("Out Residual Name", resid_names[0]);
      p->set< RCP<DataLayout> >("Node Vector Data Layout", dl->node_vector);
      p->set<bool>("Negative",true);
      ev = rcp(new ATO::AddVector<EvalT,AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }

    if( params->isSublist("Residual Strain") )
    {
      std::string residStressName("Residual Stress");
      // Compute residual strains and stresses
      atoUtils.constructResidualStressEvaluators( params, fm0, stateMgr, elementBlockName, residStressName );

      // Apply user defined weighting
      atoUtils.constructWeightedFieldEvaluators( params, fm0, stateMgr, elementBlockName, "QP Vector", residStressName );
      atoUtils.SaveCellStateField(fm0, stateMgr, residStressName, elementBlockName, dl->qp_vector);
        
      {
        // Compute divergence of the residual stress
        RCP<ParameterList> p = rcp(new ParameterList("Residual Stress Divergence"));

        //Input
        p->set<bool>("Disable Transient", true);
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
        // Add residual forces to residual
        RCP<ParameterList> p = rcp(new ParameterList("Add Residual Force"));
        p->set<std::string>("Vector Name", "Residual Force");
        p->set< RCP<DataLayout> >("Vector Data Layout", dl->node_vector);
        p->set<std::string>("In Residual Name", resid_names[0]);
        p->set< RCP<DataLayout> >("Node Vector Data Layout", dl->node_vector);
        resid_names[0] += " with Residual Force";
        p->set<std::string>("Out Residual Name", resid_names[0]);
        p->set< RCP<DataLayout> >("Node Vector Data Layout", dl->node_vector);
        ev = rcp(new ATO::AddVector<EvalT,AlbanyTraits>(*p));
        fm0.template registerEvaluator<EvalT>(ev);
      }
    }
  }


  if( numBoundaries )
#ifdef ATO_USES_COGENT
  {
    Cogent::IntegratorFactory integratorFactory;
    
    for(int iBoundary=0; iBoundary<numBoundaries; iBoundary++){

      // Create Boundary integrating quadrature
      Teuchos::ParameterList boundarySpec = geomSpec.sublist(Albany::strint("Boundary", iBoundary));
      RCP<Cogent::Integrator>
        boundaryProjector = integratorFactory.create(cellType, intrepidBasis, boundarySpec);
      std::string boundaryName = boundarySpec.get<std::string>("Boundary Name");

      RCP<ParameterList> p = rcp(new ParameterList("Cogent: Compute Basis Functions"));
      p->set< Albany::StateManager* >("State Manager Ptr", &stateMgr );
      p->set<bool>("Static Topology", true);

      std::string weightsName = Albany::strint("Weight", iBoundary);
      std::string wBFName = Albany::strint("wBF", iBoundary);
 
      p->set< RCP<Cogent::Integrator> >("Cubature", boundaryProjector);
      p->set<std::string>("Coordinate Vector Name",   "Coord Vec");
      p->set<std::string>("Weights Name",              weightsName);
      p->set<std::string>("Jacobian Det Name",         Albany::strint("Jacobian Det", iBoundary));
      p->set<std::string>("Jacobian Name",             Albany::strint("Jacobian",     iBoundary));
      p->set<std::string>("Jacobian Inv Name",         Albany::strint("Jacobian Inv", iBoundary));
      p->set<std::string>("BF Name",                   Albany::strint("BF",           iBoundary));
      p->set<std::string>("Weighted BF Name",          wBFName);
      p->set<std::string>("Gradient BF Name",          Albany::strint("Grad BF",      iBoundary));
      p->set<std::string>("Weighted Gradient BF Name", Albany::strint("wGrad BF",     iBoundary));
  
      ev = rcp(new ATO::ComputeBasisFunctions<EvalT,AlbanyTraits>(*p,dl,&meshSpecs));
      fm0.template registerEvaluator<EvalT>(ev);
 
      atoUtils.SaveCellStateField(fm0, stateMgr, weightsName, elementBlockName, dl->qp_scalar);
    
      // Compute boundary forces
      std::string boundaryForceName = Albany::strint("Boundary Force", iBoundary);
      if(params->isSublist("Implicit Boundary Conditions")){
        Teuchos::ParameterList& bcSpec = params->sublist("Implicit Boundary Conditions");
        atoUtils.constructBoundaryConditionEvaluators( bcSpec, fm0, stateMgr, boundaryName, boundaryForceName );
      }

      // Apply user defined weighting
      atoUtils.constructWeightedFieldEvaluators( params, fm0, stateMgr, elementBlockName, "QP Vector", boundaryForceName );
      atoUtils.SaveCellStateField(fm0, stateMgr, boundaryForceName, elementBlockName, dl->qp_vector);

      {
        RCP<ParameterList> p = rcp(new ParameterList("Boundary Forces"));

        if(resid_names[0] == "Not Set"){
          // Create residual
          resid_names[0] = boundaryForceName;
        } else {
          // Add to residual
          p->set<std::string>("In Residual Name", resid_names[0]);
          resid_names[0] += " with " + boundaryForceName;
        }

        p->set<std::string>("Vector Name", boundaryForceName);
        p->set<std::string>("Weighted BF Name", wBFName);
        p->set<std::string>("Out Residual Name", resid_names[0]);
        p->set< RCP<DataLayout> >("Vector Data Layout", dl->qp_vector);
        p->set< RCP<DataLayout> >("Weighted BF Data Layout", dl->node_qp_scalar);
        p->set< RCP<DataLayout> >("Node Vector Data Layout", dl->node_vector);
        p->set<bool>("Negative",true);
        ev = rcp(new ATO::AddVector<EvalT,AlbanyTraits>(*p));
        fm0.template registerEvaluator<EvalT>(ev);
      }
    }
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
