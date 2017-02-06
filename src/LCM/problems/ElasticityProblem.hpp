//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ELASTICITYPROBLEM_HPP
#define ELASTICITYPROBLEM_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_AbstractProblem.hpp"

#include "Phalanx.hpp"
#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"
#include "PHAL_AlbanyTraits.hpp"

#include "AAdapt_RC_Manager.hpp"

namespace Albany {

  /*!
   * \brief Abstract interface for representing a 2-D finite element
   * problem.
   */
  class ElasticityProblem : public Albany::AbstractProblem {
  public:
  
    //! Default constructor
    ElasticityProblem(
		      const Teuchos::RCP<Teuchos::ParameterList>& params_,
		      const Teuchos::RCP<ParamLib>& paramLib_,
		      const int numDim_,
                      const Teuchos::RCP<AAdapt::rc::Manager>& rc_mgr);

    //! Destructor
    virtual ~ElasticityProblem();

    //! Return number of spatial dimensions
    virtual int spatialDimension() const { return numDim; }

    //! Build the PDE instantiations, boundary conditions, and initial solution
    virtual void buildProblem(
      Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct>>  meshSpecs,
      StateManager& stateMgr);

    // Build evaluators
    virtual Teuchos::Array< Teuchos::RCP<const PHX::FieldTag>>
    buildEvaluators(
      PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
      const Albany::MeshSpecsStruct& meshSpecs,
      Albany::StateManager& stateMgr,
      Albany::FieldManagerChoice fmchoice,
      const Teuchos::RCP<Teuchos::ParameterList>& responseList);

    //! Each problem must generate it's list of valid parameters
    Teuchos::RCP<const Teuchos::ParameterList> getValidProblemParameters() const;

    void getAllocatedStates(Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Kokkos::DynRankView<RealType, PHX::Device>>>> oldState_,
			    Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Kokkos::DynRankView<RealType, PHX::Device>>>> newState_
			    ) const;

  private:

    //! Private to prohibit copying
    ElasticityProblem(const ElasticityProblem&);
    
    //! Private to prohibit copying
    ElasticityProblem& operator=(const ElasticityProblem&);

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

    //! Boundary conditions on source term
    bool haveSource;
    int numDim;

    //! Compute exact error in displacement solution
    bool computeError;

    std::string matModel; 
    Teuchos::RCP<Albany::Layouts> dl;

    Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Kokkos::DynRankView<RealType, PHX::Device>>>> oldState;
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Kokkos::DynRankView<RealType, PHX::Device>>>> newState;

    Teuchos::RCP<AAdapt::rc::Manager> rc_mgr;
  };

}

#include "Albany_SolutionAverageResponseFunction.hpp"
#include "Albany_SolutionTwoNormResponseFunction.hpp"
#include "Albany_SolutionMaxValueResponseFunction.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_ResponseUtilities.hpp"
#include "Albany_EvaluatorUtils.hpp"

#include "Density.hpp"
#include "ElasticModulus.hpp"
#include "PoissonsRatio.hpp"
#include "PHAL_Source.hpp"
#include "Strain.hpp"
#include "DefGrad.hpp"
#include "Stress.hpp"
#include "PHAL_SaveStateField.hpp"
#include "ElasticityResid.hpp"
//#include "ElasticityDispErrResid.hpp"

#include "Time.hpp"
//#include "CapExplicit.hpp"
//#include "CapImplicit.hpp"

template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
Albany::ElasticityProblem::constructEvaluators(
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
   RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>>
     intrepidBasis = Albany::getIntrepid2Basis(meshSpecs.ctd);

   const int numNodes = intrepidBasis->getCardinality();
   const int worksetSize = meshSpecs.worksetSize;

   Intrepid2::DefaultCubatureFactory cubFactory;
   RCP <Intrepid2::Cubature<PHX::Device>> cubature = cubFactory.create<PHX::Device, RealType, RealType>(*cellType, meshSpecs.cubatureDegree);

   const int numDim = cubature->getDimension();
   const int numQPts = cubature->getNumPoints();
   const int numVertices = cellType->getNodeCount();
//   const int numVertices = cellType->getVertexCount();

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

   // Determine if transient analyses should be supported
   bool supportsTransient = false;
   if(params->isParameter("Solution Method")){
     std::string solutionMethod = params->get<std::string>("Solution Method");
     if(solutionMethod == "Transient"){
       supportsTransient = true;
     }
   }

   // Displacement Fields

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
       (evalUtils.constructGatherSolutionEvaluator_noTransient(true, dof_names));

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructScatterResidualEvaluator(true, resid_names));

   // Displacment Error Fields

   if (computeError) {

     // place transient warning message here

     int offset = numDim;

     Teuchos::ArrayRCP<std::string> edof_names(1);
       edof_names[0] = "Displacement Error";
     Teuchos::ArrayRCP<std::string> eresid_names(1);
       eresid_names[0] = edof_names[0]+" Residual";

     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructDOFVecInterpolationEvaluator(edof_names[0], offset));

     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructDOFVecGradInterpolationEvaluator(edof_names[0], offset));

     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructGatherSolutionEvaluator_noTransient(true, edof_names, offset));

     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructScatterResidualEvaluator(true, eresid_names, offset, "Scatter Error"));

   }

   // Standard FEM stuff

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructGatherCoordinateVectorEvaluator());

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructMapToPhysicalFrameEvaluator(cellType, cubature));

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructComputeBasisFunctionsEvaluator(cellType, intrepidBasis, cubature));

  // Temporary variable used numerous times below
  Teuchos::RCP<PHX::Evaluator<AlbanyTraits>> ev;

  { // Time
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<std::string>("Time Name", "Time");
    p->set<std::string>("Delta Time Name", "Delta Time");
    p->set< RCP<DataLayout>>("Workset Scalar Data Layout", dl->workset_scalar);
    p->set<RCP<ParamLib>>("Parameter Library", paramLib);
    p->set<bool>("Disable Transient", true);

    ev = rcp(new LCM::Time<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
    p = stateMgr.registerStateVariable("Time",dl->workset_scalar, dl->dummy, elementBlockName, "scalar", 0.0, true);
    ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  { // Density (optional, needed only for dynamics)

    if(supportsTransient){
      RCP<ParameterList> p = rcp(new ParameterList);

      p->set<std::string>("Cell Variable Name", "Density");
      p->set< RCP<DataLayout>>("Cell Scalar Data Layout", dl->cell_scalar2);

      p->set<RCP<ParamLib>>("Parameter Library", paramLib);
      Teuchos::ParameterList& paramList = params->sublist("Density");
      p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

      ev = rcp(new LCM::Density<EvalT,AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }
  }

  { // Elastic Modulus
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<std::string>("QP Variable Name", "Elastic Modulus");
    p->set<std::string>("QP Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout>>("Node Data Layout", dl->node_scalar);
    p->set< RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout>>("QP Vector Data Layout", dl->qp_vector);

    p->set<RCP<ParamLib>>("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Elastic Modulus");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = rcp(new LCM::ElasticModulus<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  { // Poissons Ratio 
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<std::string>("QP Variable Name", "Poissons Ratio");
    p->set<std::string>("QP Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout>>("Node Data Layout", dl->node_scalar);
    p->set< RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout>>("QP Vector Data Layout", dl->qp_vector);

    p->set<RCP<ParamLib>>("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Poissons Ratio");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = rcp(new LCM::PoissonsRatio<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (haveSource) { // Source
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                       "Error!  Sources not implemented in Elasticity yet!");

    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<std::string>("Source Name", "Source");
    p->set<std::string>("Variable Name", "Displacement");
    p->set< RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);

    p->set<RCP<ParamLib>>("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Source Functions");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = rcp(new PHAL::Source<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  { // Strain
    RCP<ParameterList> p = rcp(new ParameterList("Strain"));

    //Input
    p->set<std::string>("Gradient QP Variable Name", "Displacement Gradient");

    //Output
    p->set<std::string>("Strain Name", "Strain");

    if (Teuchos::nonnull(rc_mgr))
      rc_mgr->registerField("Strain", dl->qp_tensor, AAdapt::rc::Init::zero,
                            AAdapt::rc::Transformation::none, p);

    ev = rcp(new LCM::Strain<EvalT,AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);

    if(matModel == "CapExplicit" || matModel == "GursonSD" || matModel == "CapImplicit"){
      p = stateMgr.registerStateVariable("Strain", dl->qp_tensor, dl->dummy, elementBlockName, "scalar", 0.0, true);
      ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
    	fm0.template registerEvaluator<EvalT>(ev);
    }
  }

  { // Deformation Gradient
    RCP<ParameterList> p = rcp(new ParameterList("DefGrad"));

    //Inputs: flags, weights, GradU
    const bool avgJ = params->get("avgJ", false);
    p->set<bool>("avgJ Name", avgJ);
    const bool volavgJ = params->get("volavgJ", false);
    p->set<bool>("volavgJ Name", volavgJ);
    const bool weighted_Volume_Averaged_J = params->get("weighted_Volume_Averaged_J", false);
    p->set<bool>("weighted_Volume_Averaged_J Name", weighted_Volume_Averaged_J);
    p->set<std::string>("Weights Name","Weights");
    p->set<std::string>("Gradient QP Variable Name", "Displacement Gradient");
    p->set< RCP<DataLayout>>("QP Tensor Data Layout", dl->qp_tensor);

    //Outputs: F, J
    p->set<std::string>("DefGrad Name", "Deformation Gradient"); //dl->qp_tensor also
    p->set<std::string>("DetDefGrad Name", "Determinant of Deformation Gradient"); 
    p->set< RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);

    ev = rcp(new LCM::DefGrad<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  { // Linear elasticity stress
    RCP<ParameterList> p = rcp(new ParameterList("Stress"));

      //Input
      p->set<std::string>("Strain Name", "Strain");
      p->set< RCP<DataLayout>>("QP Tensor Data Layout", dl->qp_tensor);

      p->set<std::string>("Elastic Modulus Name", "Elastic Modulus");
      p->set< RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);

      p->set<std::string>("Poissons Ratio Name", "Poissons Ratio");  // dl->qp_scalar also

      //Output
      p->set<std::string>("Stress Name", "Stress"); //dl->qp_tensor also

      ev = rcp(new LCM::Stress<EvalT,AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
      p = stateMgr.registerStateVariable("Stress",dl->qp_tensor, dl->dummy, elementBlockName, "scalar", 0.0);
      ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
  }

  { // Displacement Resid
    RCP<ParameterList> p = rcp(new ParameterList("Displacement Resid"));

    //Input
    p->set<std::string>("Stress Name", "Stress");
    p->set< RCP<DataLayout>>("QP Tensor Data Layout", dl->qp_tensor);

    // \todo Is the required?
    p->set<std::string>("DefGrad Name", "Deformation Gradient"); //dl->qp_tensor also

    p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
    p->set< RCP<DataLayout>>("Node QP Vector Data Layout", dl->node_qp_vector);

    if(supportsTransient){
      p->set<bool>("Disable Transient", false);
      p->set<std::string>("Weighted BF Name", "wBF");
      p->set< RCP<DataLayout>>("Node QP Scalar Data Layout", dl->node_qp_scalar);
      p->set<std::string>("Time Dependent Variable Name", "Displacement_dotdot");
      p->set< RCP<DataLayout>>("QP Vector Data Layout", dl->qp_vector);
      p->set<std::string>("Density Name", "Density");
      p->set< RCP<DataLayout>>("Cell Scalar Data Layout", dl->cell_scalar2);
    }
    else{
      p->set<bool>("Disable Transient", true);
    }

    //Output
    p->set<std::string>("Residual Name", "Displacement Residual");
    p->set< RCP<DataLayout>>("Node Vector Data Layout", dl->node_vector);

    ev = rcp(new LCM::ElasticityResid<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (computeError) {
  
    { // Displacement Error "Strain"
      RCP<ParameterList> p = rcp(new ParameterList("Error Strain"));

      //Input
      p->set<std::string>("Gradient QP Variable Name", "Displacement Error Gradient");

      //Output
      p->set<std::string>("Strain Name", "Error Strain");

      ev = rcp(new LCM::Strain<EvalT,AlbanyTraits>(*p,dl));
      fm0.template registerEvaluator<EvalT>(ev);
    }

    { // Displacement Error "Stress"
      RCP<ParameterList> p = rcp(new ParameterList("Error Stress"));

      //Input
      p->set<std::string>("Strain Name", "Error Strain");
      p->set< RCP<DataLayout>>("QP Tensor Data Layout", dl->qp_tensor);

      p->set<std::string>("Elastic Modulus Name", "Elastic Modulus");
      p->set< RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);

      p->set<std::string>("Poissons Ratio Name", "Poissons Ratio");  // dl->qp_scalar also

      //Output
      p->set<std::string>("Stress Name", "Error Stress"); //dl->qp_tensor also

      ev = rcp(new LCM::Stress<EvalT,AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
      p = stateMgr.registerStateVariable("Error Stress",dl->qp_tensor, dl->dummy, elementBlockName, "scalar", 0.0);
      ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }
    
  }

  if (Teuchos::nonnull(rc_mgr)) rc_mgr->createEvaluators<EvalT>(fm0, dl);

   if (fieldManagerChoice == Albany::BUILD_RESID_FM)  {
    PHX::Tag<typename EvalT::ScalarT> res_tag("Scatter", dl->dummy);
    fm0.requireField<EvalT>(res_tag);

    if (computeError) {
      PHX::Tag<typename EvalT::ScalarT> eres_tag("Scatter Error", dl->dummy);
      fm0.requireField<EvalT>(eres_tag);
    }

    return res_tag.clone();
  }
  else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM) {
    Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
    return respUtils.constructResponses(fm0, *responseList, stateMgr);
  }

  return Teuchos::null;
}

#endif // ALBANY_ELASTICITYPROBLEM_HPP
