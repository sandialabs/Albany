//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LINEARELASTICITYMODALPROBLEM_HPP
#define LINEARELASTICITYMODALPROBLEM_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_AbstractProblem.hpp"
#include "ATO_OptimizationProblem.hpp"
#include "Albany_BCUtils.hpp"

#include "Phalanx.hpp"
#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_GatherEigenData.hpp"

namespace Albany {

  /*!
   * \brief Abstract interface for representing a 2-D finite element
   * problem.
   */
  class LinearElasticityModalProblem : 
    public ATO::OptimizationProblem ,
    public virtual Albany::AbstractProblem {
  public:
  
    //! Default constructor
    LinearElasticityModalProblem(
		      const Teuchos::RCP<Teuchos::ParameterList>& params_,
		      const Teuchos::RCP<ParamLib>& paramLib_,
		      const int numDim_);

    //! Destructor
    virtual ~LinearElasticityModalProblem();

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
    LinearElasticityModalProblem(const LinearElasticityModalProblem&);
    
    //! Private to prohibit copying
    LinearElasticityModalProblem& operator=(const LinearElasticityModalProblem&);

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
#include "Albany_ProblemUtils.hpp"
#include "Albany_ResponseUtilities.hpp"
#include "Albany_EvaluatorUtils.hpp"

#include "PHAL_GatherScalarNodalParameter.hpp"
#include "PHAL_Source.hpp"
#include "Strain.hpp"
#include "ATO_Stress.hpp"
#include "ATO_TopologyFieldWeighting.hpp"
#include "ATO_TopologyWeighting.hpp"
#include "PHAL_SaveStateField.hpp"
#include "ElasticityResid.hpp"

#include "Time.hpp"

template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
Albany::LinearElasticityModalProblem::constructEvaluators(
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

   Intrepid2::DefaultCubatureFactory cubFactory;
   RCP <Intrepid2::Cubature<PHX::Device> > cubature = cubFactory.create<PHX::Device, RealType, RealType>(*cellType, meshSpecs.cubatureDegree);

   const int numDim = cubature->getDimension();
   const int numQPts = cubature->getNumPoints();
   const int numVertices = cellType->getNodeCount();

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
   bool supportsTransient=true;


   std::string stressName("Stress");
   std::string strainName("Strain");
   std::string eigenStressName("EigenStress0");
   std::string eigenStrainName("EigenStrain0");


   Teuchos::ArrayRCP<std::string> dof_names(1);
   dof_names[0] = "Displacement";
   Teuchos::ArrayRCP<std::string> dof_names_dotdot(1);
   if (supportsTransient)
     dof_names_dotdot[0] = dof_names[0]+"_dotdot";
   Teuchos::ArrayRCP<std::string> resid_names(1);
   resid_names[0] = dof_names[0]+" Residual";

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructDOFVecInterpolationEvaluator(dof_names[0]));

   if (supportsTransient) fm0.template registerEvaluator<EvalT>
       (evalUtils.constructDOFVecInterpolationEvaluator(dof_names_dotdot[0]));

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructDOFVecGradInterpolationEvaluator(dof_names[0]));

   if (supportsTransient) fm0.template registerEvaluator<EvalT>
       (evalUtils.constructGatherSolutionEvaluator_withAcceleration(true, dof_names, Teuchos::null, dof_names_dotdot));
   else fm0.template registerEvaluator<EvalT>
       (evalUtils.constructGatherSolutionEvaluator_noTransient(/*is_vector_dof=*/ true, dof_names));

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructScatterResidualEvaluator(/*is_vector_dof=*/ true, resid_names));

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructGatherCoordinateVectorEvaluator());

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructMapToPhysicalFrameEvaluator(cellType, cubature));

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructComputeBasisFunctionsEvaluator(cellType, intrepidBasis, cubature));

   // Temporary variable used numerous times below
   Teuchos::RCP<PHX::Evaluator<AlbanyTraits> > ev;

   // Register eigenvector gather evaluator
   // TEV this should probably not happen if not modal analysis
#if defined(ALBANY_EPETRA)
  { // Gather Eigenvectors
     int nEigenvectors = 1;
     RCP<ParameterList> p = rcp(new ParameterList);
     p->set<std::string>("Eigenvector name root", "Evec");
     p->set<std::string>("Eigenvalue name root", "Eval");
     p->set<int>("Number of eigenvectors", nEigenvectors);
     ev = rcp(new PHAL::GatherEigenData<EvalT,AlbanyTraits>(*p,dl));
     fm0.template registerEvaluator<EvalT>(ev);
  }
#endif

   Teuchos::ArrayRCP<std::string> evec_names(1);
   evec_names[0] = "Evec_Re0";

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructDOFVecGradInterpolationEvaluator(evec_names[0]));

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructDOFVecInterpolationEvaluator(evec_names[0]));

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
                                       elementBlockName, "scalar", 0.0, false);
    ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }


  { // EigenStrain
    RCP<ParameterList> p = rcp(new ParameterList(eigenStrainName));

    //Input
    p->set<std::string>("Gradient QP Variable Name", "Evec_Re0 Gradient");

    //Output
    p->set<std::string>("Strain Name", eigenStrainName);

    ev = rcp(new LCM::Strain<EvalT,AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);

    // state the strain in the state manager for ATO
    p = stateMgr.registerStateVariable(eigenStrainName, dl->qp_tensor, dl->dummy,
                                       elementBlockName, "scalar", 0.0, false);
    ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }


  { // Linear elasticity stress
    RCP<ParameterList> p = rcp(new ParameterList(stressName));

    //Input
    p->set<std::string>("Strain Name", strainName);
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);

    p->set<double>("Elastic Modulus", params->get<double>("Elastic Modulus"));
    p->set<double>("Poissons Ratio",  params->get<double>("Poissons Ratio"));

    //Output
    p->set<std::string>("Stress Name", stressName);

    ev = rcp(new ATO::Stress<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

    // state the stress in the state manager so for ATO
    p = stateMgr.registerStateVariable(stressName,dl->qp_tensor, dl->dummy, elementBlockName, "scalar", 0.0);
    ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }


  { // Linear elasticity eigen-stress
    RCP<ParameterList> p = rcp(new ParameterList(stressName));

    //Input
    p->set<std::string>("Strain Name", eigenStrainName);
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);

    p->set<double>("Elastic Modulus", params->get<double>("Elastic Modulus"));
    p->set<double>("Poissons Ratio",  params->get<double>("Poissons Ratio"));

    //Output
    p->set<std::string>("Stress Name", eigenStressName);

    ev = rcp(new ATO::Stress<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

    // state the stress in the state manager so for ATO
    p = stateMgr.registerStateVariable(stressName,dl->qp_tensor, dl->dummy, elementBlockName, "scalar", 0.0);
    ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // Get distributed parameter
  if(params->isType<Teuchos::RCP<ATO::Topology> >("Topology")){
    Teuchos::RCP<ATO::Topology> topology = params->get<Teuchos::RCP<ATO::Topology> >("Topology");
    if( topology->getEntityType() == "Distributed Parameter" ){
      RCP<ParameterList> p = rcp(new ParameterList("Distributed Parameter"));

      p->set<std::string>("Parameter Name", topology->getName());
      ev = rcp(new PHAL::GatherScalarNodalParameter<EvalT,AlbanyTraits>(*p, dl) );
      fm0.template registerEvaluator<EvalT>(ev);
    }
  }
  
  // ATO penalization
  if( params->isType<Teuchos::RCP<ATO::Topology> >("Topology") )
  {
    RCP<ParameterList> p = rcp(new ParameterList("TopologyWeighting"));

    Teuchos::RCP<ATO::Topology> topology = params->get<Teuchos::RCP<ATO::Topology> >("Topology");
    p->set<Teuchos::RCP<ATO::Topology> >("Topology",topology);

    p->set<std::string>("BF Name", "BF");
    p->set<std::string>("Unweighted Variable Name", stressName);
    p->set<std::string>("Weighted Variable Name", stressName+"_Weighted");
    p->set<std::string>("Variable Layout", "QP Tensor");

    if( topology->getEntityType() == "Distributed Parameter" )
      ev = rcp(new ATO::TopologyFieldWeighting<EvalT,AlbanyTraits>(*p,dl));
    else
      ev = rcp(new ATO::TopologyWeighting<EvalT,AlbanyTraits>(*p,dl));

    fm0.template registerEvaluator<EvalT>(ev);
  }

  { // Displacement Resid
    RCP<ParameterList> p = rcp(new ParameterList("Displacement Resid"));

// TEV ... why is this here. Set above and not in LCM linearelasticity...
// TEV    p->set<bool>("Disable Transient", true);

    //Input
    p->set<std::string>("Stress Name", stressName+"_Weighted");
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);

    p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);

   // extra input for time dependent term
    p->set<std::string>("Weighted BF Name", "wBF");
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", dl->node_qp_scalar);
    p->set<std::string>("Time Dependent Variable Name", "Displacement_dotdot");
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    //Output
    p->set<std::string>("Residual Name", "Displacement Residual");
    p->set< RCP<DataLayout> >("Node Vector Data Layout", dl->node_vector);

    ev = rcp(new LCM::ElasticityResid<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (fieldManagerChoice == Albany::BUILD_RESID_FM)  {
    PHX::Tag<typename EvalT::ScalarT> res_tag("Scatter", dl->dummy);
    fm0.requireField<EvalT>(res_tag);

    return res_tag.clone();
  }
  else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM) {
    Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
    return respUtils.constructResponses(fm0, *responseList, params, stateMgr);
  }

  return Teuchos::null;
}

#endif // LINEARELASTICITYMODALPROBLEM_HPP
