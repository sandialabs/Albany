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

#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"
#include "Albany_GeneralPurposeFieldsNames.hpp"

namespace Tsunami {

  /*!
   * \brief Abstract interface for 2D Boussinesq equations
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

    Teuchos::RCP<Albany::Layouts> dl;

    int numDim;
    
    double h; //water depth

    double zAlpha;

    double a, k, h0; //input scalars: a = wave amplitude, h0 = typical water depth, k = typical wave number

    double muSqr, epsilon; //dimensionless paraleters used by code computed from input scalars  
    //IKT, Question from Irina: are we running dimensionally  or non-dimensionally??
  
    bool haveSource;   //! have source term in heat equation

    /// Boolean marking whether SDBCs are used 
    bool use_sdbcs_;
 
    bool use_params_on_mesh; //boolean to indicate whether to use parameters (water depth) on mesh 
    
    int neq; 

    std::string elementBlockName;
  };

}

#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"

#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_EvaluatorUtils.hpp"
#include "Albany_ResponseUtilities.hpp"

#include "PHAL_DOFVecGradInterpolation.hpp"

#include "Tsunami_BoussinesqResid.hpp"
#include "Tsunami_BoussinesqBodyForce.hpp"
#include "Tsunami_BoussinesqParameters.hpp"


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
  
  const int numNodes = intrepidBasis->getCardinality();
  const int worksetSize = meshSpecs.worksetSize;
  
  //The following, when set to true, will load parameters (water depth) only once at the beginning
  //of the simulation to save time/computation 
  const bool enableMemoizer = this->params->get<bool>("Use MDField Memoization", true);
  
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
  

   RCP<Albany::Layouts> dl = rcp(new Albany::Layouts(worksetSize,numVertices,numNodes,numQPts,numDim, vecDim));
   Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
   bool supportsTransient = false;
   if(number_of_time_deriv > 0) 
      supportsTransient = true;
   int offset=0;

   // This problem appears to be only defined as a transient problem, throw exception if it is not
   TEUCHOS_TEST_FOR_EXCEPTION(
      number_of_time_deriv == 0,
      std::logic_error,
      "Tsunami::Boussinesq must be defined as an unsteady calculation.");

   // Temporary variable used numerous times below
   Teuchos::RCP<PHX::Evaluator<AlbanyTraits> > ev;

   // Define Field Names

     Teuchos::ArrayRCP<string> dof_names(1);
     Teuchos::ArrayRCP<string> dof_names_dot(1);
     Teuchos::ArrayRCP<string> resid_names(1);
     dof_names[0] = "EtaUE";
     if (supportsTransient)
       dof_names_dot[0] = dof_names[0]+"_dot";
     resid_names[0] = "EtaUE Residual";

     if (supportsTransient) fm0.template registerEvaluator<EvalT>
           (evalUtils.constructGatherSolutionEvaluator(true, dof_names, dof_names_dot, offset));
     else fm0.template registerEvaluator<EvalT>
           (evalUtils.constructGatherSolutionEvaluator_noTransient(true, dof_names, offset));

     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructDOFVecInterpolationEvaluator(dof_names[0], offset));

     if (supportsTransient) fm0.template registerEvaluator<EvalT>
       (evalUtils.constructDOFVecInterpolationEvaluator(dof_names_dot[0], offset));

     //     fm0.template registerEvaluator<EvalT>
     //  (evalUtils.constructDOFVecGradInterpolationEvaluator(dof_names[0], offset));

     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructScatterResidualEvaluator(true, resid_names, offset, "Scatter Boussinesq"));

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructGatherCoordinateVectorEvaluator());

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructMapToPhysicalFrameEvaluator(cellType, cubature));

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructComputeBasisFunctionsEvaluator(cellType, intrepidBasis, cubature));
  
   //Declare water depth as nodal field 
   Albany::StateStruct::MeshFieldEntity entity = Albany::StateStruct::NodalDataToElemNode;
   {
     std::string stateName("water_depth");
     std::string fieldName = "water_depth";
     RCP<ParameterList> p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName,true, &entity);
     p->set<std::string>("Field Name", fieldName);
     ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
   }
   // Intepolate water depth from nodes to QPs
   ev = evalUtils.getPSTUtils().constructDOFInterpolationEvaluator("water_depth", -1, enableMemoizer);
   fm0.template registerEvaluator<EvalT> (ev);
  
   // Intepolate surface height gradient
   ev = evalUtils.getPSTUtils().constructDOFGradInterpolationEvaluator("water_depth", -1, enableMemoizer);
   fm0.template registerEvaluator<EvalT> (ev);


   //Declare z_alpha as nodal field 
   {
     std::string stateName("z_alpha");
     std::string fieldName = "z_alpha";
     RCP<ParameterList> p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName,true, &entity);
     p->set<std::string>("Field Name", fieldName);
     ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
   }
   // Intepolate z_alpha from nodes to QPs
   ev = evalUtils.getPSTUtils().constructDOFInterpolationEvaluator("z_alpha", -1, enableMemoizer);
   fm0.template registerEvaluator<EvalT> (ev);

   { // Specialized DofVecGrad Interpolation for this problem
    
     RCP<ParameterList> p = rcp(new ParameterList("DOFVecGrad Interpolation "+dof_names[0]));
     // Input
     p->set<string>("Variable Name", dof_names[0]);
     p->set<string>("Gradient BF Name", "Grad BF");
     p->set<int>("Offset of First DOF", offset);
     
     // Output (assumes same Name as input)
     p->set<string>("Gradient Variable Name", dof_names[0]+" Gradient");
     
     ev = rcp(new PHAL::DOFVecGradInterpolation<EvalT,AlbanyTraits>(*p,dl));
     fm0.template registerEvaluator<EvalT>(ev);
   }

   { // Specialized DofVecDotGrad Interpolation for this problem
    
     RCP<ParameterList> p = rcp(new ParameterList("DOFVecDotGrad Interpolation "+dof_names_dot[0]));
     // Input
     p->set<string>("Variable Name", dof_names_dot[0]);
     p->set<string>("Gradient BF Name", "Grad BF");
     p->set<int>("Offset of First DOF", offset);
     
     // Output (assumes same Name as input)
     p->set<string>("Gradient Variable Name", dof_names_dot[0]+" Gradient");
     
     ev = rcp(new PHAL::DOFVecGradInterpolation<EvalT,AlbanyTraits>(*p,dl));
     fm0.template registerEvaluator<EvalT>(ev);
   }
  
  { //Parameters
    RCP<ParameterList> p = rcp(new ParameterList("Parameters"));
    //Input
    p->set<std::string>("Water Depth In QP Name", "water_depth");
    p->set<std::string>("z_alpha In QP Name", "z_alpha");
    p->set<double>("Water Depth", h); 
    p->set<double>("Z_alpha", zAlpha); 
    p->set<bool>("Use Parameters on Mesh", use_params_on_mesh); 
    p->set<bool>("Enable Memoizer", enableMemoizer);

    //Output
    p->set<std::string>("Water Depth QP Name", "Water Depth Field");
    p->set<std::string>("z_alpha QP Name", "z_alpha Field");
    p->set<std::string>("Beta QP Name", "Beta Field");

    ev = rcp(new Tsunami::BoussinesqParameters<EvalT,AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }
  
  // Body Force
  {
    RCP<ParameterList> p = rcp(new ParameterList("Body Force"));

    //Input
    p->set<std::string>("Coordinate Vector Name", "Coord Vec");
    p->set<std::string>("Water Depth QP Name", "Water Depth Field");
    p->set<std::string>("z_alpha QP Name", "z_alpha Field");
    p->set<std::string>("Beta QP Name", "Beta Field");
    p->set<double>("Mu Squared", muSqr);
    p->set<double>("Epsilon", epsilon);  

    Teuchos::ParameterList& paramList = params->sublist("Body Force");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    //Output
    p->set<std::string>("Body Force Name", "Body Force");

    ev = rcp(new Tsunami::BoussinesqBodyForce<EvalT,AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }


  { // Boussinesq Resid
    RCP<ParameterList> p = rcp(new ParameterList("Boussinesq Resid"));

    //Input
    p->set<std::string>("Weighted BF Name", Albany::weighted_bf_name);
    p->set<std::string>("Weighted Gradient BF Name", Albany::weighted_grad_bf_name);
    p->set<std::string>("EtaUE QP Variable Name", "EtaUE");
    p->set<std::string>("EtaUE Gradient QP Variable Name", "EtaUE Gradient");
    p->set<std::string>("EtaUE Dot Gradient QP Variable Name", "EtaUE_dot Gradient");
    p->set<std::string>("EtaUE Dot QP Variable Name", "EtaUE_dot");
    

    p->set<std::string>("Water Depth QP Name", "Water Depth Field");
    p->set<std::string>("z_alpha QP Name", "z_alpha Field");
    p->set<std::string>("Water Depth Gradient Name", "water_depth Gradient");
    p->set<std::string>("Beta QP Name", "Beta Field");
    p->set<double>("Mu Squared", muSqr);
    p->set<double>("Epsilon", epsilon);  
    p->set<std::string>("Body Force Name", "Body Force");
 
    p->set<RCP<ParamLib> >("Parameter Library", paramLib);

    //Output
    p->set<std::string>("Residual Name", "EtaUE Residual");
    p->set< RCP<DataLayout> >("Node Vector Data Layout", dl->node_vector);

    ev = rcp(new Tsunami::BoussinesqResid<EvalT,AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }


  if (fieldManagerChoice == Albany::BUILD_RESID_FM)  {
    PHX::Tag<typename EvalT::ScalarT> res_tag("Scatter Boussinesq", dl->dummy);
    fm0.requireField<EvalT>(res_tag);
  }
  else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM) {
    Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
    return respUtils.constructResponses(fm0, *responseList, Teuchos::null, stateMgr);
  }

  return Teuchos::null;
}
#endif // TSUNAMI_BOUSSINESQ_HPP
