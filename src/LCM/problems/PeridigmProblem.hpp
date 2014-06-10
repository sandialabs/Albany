//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PERIDIGMPROBLEM_HPP
#define PERIDIGMPROBLEM_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Albany_AbstractProblem.hpp"
#include "Phalanx.hpp"
#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"
#include "PHAL_AlbanyTraits.hpp"

// Interface to Peridigm peridynamics code
#include "PeridigmManager.hpp"

namespace Albany {

  /*!
   * \brief Interface to Peridigm peridynamics code.
   */
  class PeridigmProblem : public Albany::AbstractProblem {
  public:
  
    //! Default constructor
    PeridigmProblem(const Teuchos::RCP<Teuchos::ParameterList>& params,
                    const Teuchos::RCP<ParamLib>& paramLib,
                    const int numEqm,
                    const Teuchos::RCP<const Epetra_Comm>& comm);

    //! Destructor
    virtual ~PeridigmProblem();

    //! Return number of spatial dimensions
    virtual int spatialDimension() const { return numDim; }

    //! Build the PDE instantiations, boundary conditions, and initial solution
    virtual void buildProblem(Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs, StateManager& stateMgr);

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

  private:

    //! Private to prohibit copying
    PeridigmProblem(const PeridigmProblem&);
    
    //! Private to prohibit copying
    PeridigmProblem& operator=(const PeridigmProblem&);

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

  protected:

    //! Boundary conditions on source term
    bool haveSource;
    int numDim;
    bool haveMatDB;
    std::string mtrlDbFilename;
    Teuchos::RCP<QCAD::MaterialDatabase> materialDataBase;
    Teuchos::RCP<Teuchos::ParameterList> peridigmParams;
  };

}

#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_ResponseUtilities.hpp"
#include "Albany_EvaluatorUtils.hpp"

#include "PHAL_Source.hpp"
#include "CurrentCoords.hpp"
#include "GatherSphereVolume.hpp"
#include "PeridigmForce.hpp"
#include "PeridigmPartialStress.hpp"
#include "PHAL_SaveStateField.hpp"

#include "ElasticModulus.hpp"
#include "PoissonsRatio.hpp"
#include "Strain.hpp"
#include "DefGrad.hpp"
#include "Stress.hpp"
#include "PHAL_SaveStateField.hpp"
#include "ElasticityResid.hpp"

template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
Albany::PeridigmProblem::constructEvaluators(
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
   using PHAL::AlbanyTraits;

   // get the name of the current element block
   string elementBlockName = meshSpecs.ebName;

   // WHAT'S IN meshSpecs Albany::MeshSpecsStruct?
   
   RCP<shards::CellTopology> cellType = rcp(new shards::CellTopology(&meshSpecs.ctd));

   const int worksetSize = meshSpecs.worksetSize;

   // Read information from material database

   // If the user did not supply a material data base, assume that the simulation
   // is a pure peridynamics simulation
   std::string materialModelName("Peridynamics");
   // If the user did supply a material data base, query the material model name
   if(!materialDataBase.is_null()){
     materialModelName = materialDataBase->getElementBlockSublist(elementBlockName, "Material Model").get<std::string>("Model Name");
     TEUCHOS_TEST_FOR_EXCEPTION(materialModelName.length() == 0,
				std::logic_error,
				"Could not find material model for block: " + elementBlockName + " in material model data base.");
   }

   // Construct evaluators

   Teuchos::ArrayRCP<std::string> dof_name(1), dof_name_dotdot(1), residual_name(1);
   dof_name[0] = "Displacement";
   dof_name_dotdot[0] = "Displacement_dotdot";
   residual_name[0] = "Residual";

   Teuchos::RCP<PHX::Evaluator<AlbanyTraits> > ev;

   bool supportsTransient = true;

   // --------- Option 1: Peridynamics ---------

   if(materialModelName == "Peridynamics"){

     *out << "PeridigmProblem::constructEvaluators(), Creating evaluators for peridynamics material." << std::endl;

     const int numNodes = 1;
     const int numVertices = numNodes;
     const int numQuadraturePoints = 1;

     RCP<Albany::Layouts> dataLayout = rcp(new Albany::Layouts(worksetSize, numVertices, numNodes, numQuadraturePoints, numDim));
     TEUCHOS_TEST_FOR_EXCEPTION(dataLayout->vectorAndGradientLayoutsAreEquivalent==false, std::logic_error,
				"Data Layout Usage in Peridigm problems assume vecDim = numDim");
     Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dataLayout);

     { // Solution vector, which is the nodal displacements
       if(!supportsTransient)
	 fm0.template registerEvaluator<EvalT>(evalUtils.constructGatherSolutionEvaluator_noTransient(true, dof_name));
       else
	 fm0.template registerEvaluator<EvalT>(evalUtils.constructGatherSolutionEvaluator(true, dof_name, dof_name_dotdot));
     }

     { // Gather Coord Vec
       fm0.template registerEvaluator<EvalT>(evalUtils.constructGatherCoordinateVectorEvaluator());
     }

     if (haveSource) { // Source
       TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
				  "Error!  Sources not available for Peridigm!");
     }

     { // Time
       RCP<ParameterList> p = rcp(new ParameterList("Time"));
       p->set<std::string>("Time Name", "Time");
       p->set<std::string>("Delta Time Name", "Delta Time");
       p->set<RCP<DataLayout> >("Workset Scalar Data Layout", dataLayout->workset_scalar);
       // p->set<RCP<ParamLib> >("Parameter Library", paramLib);
       p->set<bool>("Disable Transient", true);
       ev = rcp(new LCM::Time<EvalT, AlbanyTraits>(*p));
       fm0.template registerEvaluator<EvalT>(ev);
       p = stateMgr.registerStateVariable("Time",
					  dataLayout->workset_scalar,
					  dataLayout->dummy,
					  elementBlockName,
					  "scalar",
					  0.0,
					  true);
       ev = rcp(new PHAL::SaveStateField<EvalT, AlbanyTraits>(*p));
       fm0.template registerEvaluator<EvalT>(ev);
     }
  
     { // Current Coordinates
       RCP<ParameterList> p = rcp(new ParameterList("Current Coordinates"));
       p->set<std::string>("Reference Coordinates Name", "Coord Vec");
       p->set<std::string>("Displacement Name", "Displacement");
       p->set<std::string>("Current Coordinates Name", "Current Coordinates");
       ev = rcp(new LCM::CurrentCoords<EvalT, AlbanyTraits>(*p, dataLayout));
       fm0.template registerEvaluator<EvalT>(ev);
     }

     { // Sphere Volume
       RCP<ParameterList> p = rcp(new ParameterList("Sphere Volume"));
       p->set<std::string>("Sphere Volume Name", "Sphere Volume");
       ev = rcp(new LCM::GatherSphereVolume<EvalT, AlbanyTraits>(*p, dataLayout));
       fm0.template registerEvaluator<EvalT>(ev);
     }

     { // Save Variables to Exodus
       const Teuchos::ParameterList& outputVariables = peridigmParams->sublist("Output").sublist("Output Variables");     
       LCM::PeridigmManager& peridigmManager = LCM::PeridigmManager::self();
       // peridigmManger::setOutputVariableList() records the variables that will be output to Exodus, determines
       // if they are node, element, or global variables, and determines if they are scalar, vector, etc.
       peridigmManager.setOutputFields(outputVariables);
       std::vector<LCM::PeridigmManager::OutputField> outputFields = peridigmManager.getOutputFields();
       for(unsigned int i=0 ; i<outputFields.size() ; ++i){
	 std::string albanyName = outputFields[i].albanyName;       
	 std::string initType = outputFields[i].initType;
	 std::string relation = outputFields[i].relation;
	 int length = outputFields[i].length;

	 Teuchos::RCP<PHX::DataLayout> layout;
	 if(relation == "node" && length == 1)
	   layout = dataLayout->node_scalar;
	 else if(relation == "node" && length == 3)
	   layout = dataLayout->node_vector;
	 else if(relation == "node" && length == 9)
	   layout = dataLayout->node_tensor;
	 else if(relation == "element" && length == 1)
	   layout = dataLayout->qp_scalar;
	 else if(relation == "element" && length == 3)
	   layout = dataLayout->qp_vector;
	 else if(relation == "element" && length == 9)
	   layout = dataLayout->qp_tensor;
	 else
	   TEUCHOS_TEST_FOR_EXCEPT_MSG(true, "\n\n**** Error in PeridigmManager::constructEvaluators(), invalid output varialble type.\n");

	 RCP<ParameterList> p = rcp(new ParameterList("Save " + albanyName));
	 p = stateMgr.registerStateVariable(albanyName,
					    layout,
					    dataLayout->dummy,
					    elementBlockName,
					    initType,
					    0.0,
					    false,
					    true);
	 ev = rcp(new PHAL::SaveStateField<EvalT, AlbanyTraits>(*p));
	 fm0.template registerEvaluator<EvalT>(ev);
       }
     }

     { // Peridigm Force
       RCP<ParameterList> p = rcp(new ParameterList("Force"));

       // Parameter list to be passed to Peridigm object
       Teuchos::ParameterList& peridigmParameterList = p->sublist("Peridigm Parameters");
       peridigmParameterList = *peridigmParams;

       // Required data layouts
       p->set< RCP<DataLayout> >("Node Vector Data Layout", dataLayout->node_vector);    

       // Input
       p->set<string>("Reference Coordinates Name", "Coord Vec");
       p->set<string>("Current Coordinates Name", "Current Coordinates");
       p->set<string>("Sphere Volume Name", "Sphere Volume");

       // Output
       p->set<string>("Force Name", "Force");
       p->set<string>("Residual Name", "Residual");

       ev = rcp(new LCM::PeridigmForce<EvalT, AlbanyTraits>(*p, dataLayout));
       fm0.template registerEvaluator<EvalT>(ev);
     }

     { // Scatter Residual
       fm0.template registerEvaluator<EvalT>(evalUtils.constructScatterResidualEvaluator(true, residual_name));
     }

     if (fieldManagerChoice == Albany::BUILD_RESID_FM)  {
       PHX::Tag<typename EvalT::ScalarT> res_tag("Scatter", dataLayout->dummy);
       fm0.requireField<EvalT>(res_tag);
       return res_tag.clone();
     }
     else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM) {
       Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dataLayout);
       return respUtils.constructResponses(fm0, *responseList, stateMgr);
     }

   } // ---- End Peridynamics Evaluators ----

   // --------- Option 2:  Classical Elasticity ---------   

   else if(materialModelName == "Peridynamic Partial Stress"){

     *out << "PeridigmProblem::constructEvaluators(), Creating evaluators for peridynamic partial stress." << std::endl;

     RCP<shards::CellTopology> cellType = rcp(new shards::CellTopology (&meshSpecs.ctd));
     RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > intrepidBasis = Albany::getIntrepidBasis(meshSpecs.ctd);

     const int numNodes = intrepidBasis->getCardinality();
     const int worksetSize = meshSpecs.worksetSize;

     Intrepid::DefaultCubatureFactory<RealType> cubFactory;
     RCP <Intrepid::Cubature<RealType> > cubature = cubFactory.create(*cellType, meshSpecs.cubatureDegree);

     const int numDim = cubature->getDimension();
     const int numQPts = cubature->getNumPoints();
     const int numVertices = cellType->getNodeCount();

     *out << "Field Dimensions: Workset=" << worksetSize 
	  << ", Vertices= " << numVertices
	  << ", Nodes= " << numNodes
	  << ", QuadPts= " << numQPts
	  << ", Dim= " << numDim << std::endl;

     // Construct standard FEM evaluators with standard field names                              
     RCP<Albany::Layouts> dataLayout = rcp(new Albany::Layouts(worksetSize, numVertices, numNodes, numQPts, numDim));
     TEUCHOS_TEST_FOR_EXCEPTION(dataLayout->vectorAndGradientLayoutsAreEquivalent==false, std::logic_error,
				"Data Layout Usage in Peridigm problems assume vecDim = numDim");
     Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dataLayout);

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

     if(supportsTransient){
       fm0.template registerEvaluator<EvalT>
	 (evalUtils.constructDOFVecInterpolationEvaluator(dof_names_dotdot[0]));
     }

     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructDOFVecGradInterpolationEvaluator(dof_names[0]));

     if(supportsTransient){
       fm0.template registerEvaluator<EvalT>
	 (evalUtils.constructGatherSolutionEvaluator_withAcceleration(true, dof_names, Teuchos::null, dof_names_dotdot));
     }
     else{
       fm0.template registerEvaluator<EvalT>
	 (evalUtils.constructGatherSolutionEvaluator_noTransient(true, dof_names));
     }

     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructScatterResidualEvaluator(true, resid_names));

     // Standard FEM stuff

     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructGatherCoordinateVectorEvaluator());

     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructMapToPhysicalFrameEvaluator(cellType, cubature));

     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructComputeBasisFunctionsEvaluator(cellType, intrepidBasis, cubature));

     // Temporary variable used numerous times below
     Teuchos::RCP<PHX::Evaluator<AlbanyTraits> > ev;

     { // Time
       RCP<ParameterList> p = rcp(new ParameterList);

       p->set<std::string>("Time Name", "Time");
       p->set<std::string>("Delta Time Name", "Delta Time");
       p->set< RCP<DataLayout> >("Workset Scalar Data Layout", dataLayout->workset_scalar);
       p->set<RCP<ParamLib> >("Parameter Library", paramLib);
       p->set<bool>("Disable Transient", true);

       ev = rcp(new LCM::Time<EvalT,AlbanyTraits>(*p));
       fm0.template registerEvaluator<EvalT>(ev);
       p = stateMgr.registerStateVariable("Time",dataLayout->workset_scalar, dataLayout->dummy, elementBlockName, "scalar", 0.0, true);
       ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
       fm0.template registerEvaluator<EvalT>(ev);
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
       p->set< RCP<DataLayout> >("QP Tensor Data Layout", dataLayout->qp_tensor);

       //Outputs: F, J
       p->set<std::string>("DefGrad Name", "Deformation Gradient");
       p->set<std::string>("DetDefGrad Name", "Determinant of Deformation Gradient"); 
       p->set< RCP<DataLayout> >("QP Scalar Data Layout", dataLayout->qp_scalar);

       ev = rcp(new LCM::DefGrad<EvalT,AlbanyTraits>(*p));
       fm0.template registerEvaluator<EvalT>(ev);
     }

     { // Peridigm Partial Stress
       RCP<ParameterList> p = rcp(new ParameterList("Partial Stress"));

       // Parameter list to be passed to Peridigm object
       Teuchos::ParameterList& peridigmParameterList = p->sublist("Peridigm Parameters");
       peridigmParameterList = *peridigmParams;

       // Required data layouts
       p->set< RCP<DataLayout> >("QP Scalar Data Layout", dataLayout->qp_scalar);
       p->set< RCP<DataLayout> >("QP Tensor Data Layout", dataLayout->qp_tensor);

       // Input
       p->set<std::string>("DetDefGrad Name", "Determinant of Deformation Gradient"); 
       p->set<std::string>("DefGrad Name", "Deformation Gradient");

       // Output
       p->set<std::string>("Stress Name", "Stress");

       ev = rcp(new LCM::PeridigmPartialStress<EvalT, AlbanyTraits>(*p));
       fm0.template registerEvaluator<EvalT>(ev);
     }

     { // Displacement Resid
       RCP<ParameterList> p = rcp(new ParameterList("Displacement Resid"));

       //Input
       p->set<std::string>("Stress Name", "Stress");
       p->set< RCP<DataLayout> >("QP Tensor Data Layout", dataLayout->qp_tensor);

       // \todo Is the required?
       p->set<std::string>("DefGrad Name", "Deformation Gradient"); //dataLayout->qp_tensor also

       p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
       p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dataLayout->node_qp_vector);

       // extra input for time dependent term
       p->set<std::string>("Weighted BF Name", "wBF");
       p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", dataLayout->node_qp_scalar);
       p->set<std::string>("Time Dependent Variable Name", "Displacement_dotdot");
       p->set< RCP<DataLayout> >("QP Vector Data Layout", dataLayout->qp_vector);

       //Output
       p->set<std::string>("Residual Name", "Displacement Residual");
       p->set< RCP<DataLayout> >("Node Vector Data Layout", dataLayout->node_vector);

       ev = rcp(new LCM::ElasticityResid<EvalT,AlbanyTraits>(*p));
       fm0.template registerEvaluator<EvalT>(ev);
     }

     if (fieldManagerChoice == Albany::BUILD_RESID_FM)  {
       PHX::Tag<typename EvalT::ScalarT> res_tag("Scatter", dataLayout->dummy);
       fm0.requireField<EvalT>(res_tag);
       return res_tag.clone();
     }
     else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM) {
       Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dataLayout);
       return respUtils.constructResponses(fm0, *responseList, stateMgr);
     }

   }  // ---- End Partial Stress Evaluators ----

   // --------- Option 3:  Classical Elasticity ---------   

   else if(materialModelName != "Peridynamics" && materialModelName != "Peridynamics Partial Stress"){

     *out << "PeridigmProblem::constructEvaluators(), Creating evaluators for classical elasticity, material model = " << materialModelName << std::endl;

     RCP<shards::CellTopology> cellType = rcp(new shards::CellTopology (&meshSpecs.ctd));
     RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > intrepidBasis = Albany::getIntrepidBasis(meshSpecs.ctd);

     const int numNodes = intrepidBasis->getCardinality();
     const int worksetSize = meshSpecs.worksetSize;

     Intrepid::DefaultCubatureFactory<RealType> cubFactory;
     RCP <Intrepid::Cubature<RealType> > cubature = cubFactory.create(*cellType, meshSpecs.cubatureDegree);

     const int numDim = cubature->getDimension();
     const int numQPts = cubature->getNumPoints();
     const int numVertices = cellType->getNodeCount();

     *out << "Field Dimensions: Workset=" << worksetSize 
	  << ", Vertices= " << numVertices
	  << ", Nodes= " << numNodes
	  << ", QuadPts= " << numQPts
	  << ", Dim= " << numDim << std::endl;

     // Construct standard FEM evaluators with standard field names                              
     RCP<Albany::Layouts> dataLayout = rcp(new Albany::Layouts(worksetSize, numVertices, numNodes, numQPts, numDim));
     TEUCHOS_TEST_FOR_EXCEPTION(dataLayout->vectorAndGradientLayoutsAreEquivalent==false, std::logic_error,
				"Data Layout Usage in Peridigm problems assume vecDim = numDim");
     Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dataLayout);

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

     if(supportsTransient){
       fm0.template registerEvaluator<EvalT>
	 (evalUtils.constructDOFVecInterpolationEvaluator(dof_names_dotdot[0]));
     }

     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructDOFVecGradInterpolationEvaluator(dof_names[0]));

     if(supportsTransient){
       fm0.template registerEvaluator<EvalT>
	 (evalUtils.constructGatherSolutionEvaluator_withAcceleration(true, dof_names, Teuchos::null, dof_names_dotdot));
     }
     else{
       fm0.template registerEvaluator<EvalT>
	 (evalUtils.constructGatherSolutionEvaluator_noTransient(true, dof_names));
     }

     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructScatterResidualEvaluator(true, resid_names));

     // Standard FEM stuff

     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructGatherCoordinateVectorEvaluator());

     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructMapToPhysicalFrameEvaluator(cellType, cubature));

     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructComputeBasisFunctionsEvaluator(cellType, intrepidBasis, cubature));

     // Temporary variable used numerous times below
     Teuchos::RCP<PHX::Evaluator<AlbanyTraits> > ev;

     { // Time
       RCP<ParameterList> p = rcp(new ParameterList);

       p->set<std::string>("Time Name", "Time");
       p->set<std::string>("Delta Time Name", "Delta Time");
       p->set< RCP<DataLayout> >("Workset Scalar Data Layout", dataLayout->workset_scalar);
       p->set<RCP<ParamLib> >("Parameter Library", paramLib);
       p->set<bool>("Disable Transient", true);

       ev = rcp(new LCM::Time<EvalT,AlbanyTraits>(*p));
       fm0.template registerEvaluator<EvalT>(ev);
       p = stateMgr.registerStateVariable("Time",dataLayout->workset_scalar, dataLayout->dummy, elementBlockName, "scalar", 0.0, true);
       ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
       fm0.template registerEvaluator<EvalT>(ev);
     }

     { // Elastic Modulus
       RCP<ParameterList> p = rcp(new ParameterList);

       p->set<std::string>("QP Variable Name", "Elastic Modulus");
       p->set<std::string>("QP Coordinate Vector Name", "Coord Vec");
       p->set< RCP<DataLayout> >("Node Data Layout", dataLayout->node_scalar);
       p->set< RCP<DataLayout> >("QP Scalar Data Layout", dataLayout->qp_scalar);
       p->set< RCP<DataLayout> >("QP Vector Data Layout", dataLayout->qp_vector);

       p->set<RCP<ParamLib> >("Parameter Library", paramLib);
       Teuchos::ParameterList& paramList = materialDataBase->getElementBlockSublist(elementBlockName, "Elastic Modulus");
       p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

       ev = rcp(new LCM::ElasticModulus<EvalT,AlbanyTraits>(*p));
       fm0.template registerEvaluator<EvalT>(ev);
     }

     { // Poissons Ratio 
       RCP<ParameterList> p = rcp(new ParameterList);

       p->set<std::string>("QP Variable Name", "Poissons Ratio");
       p->set<std::string>("QP Coordinate Vector Name", "Coord Vec");
       p->set< RCP<DataLayout> >("Node Data Layout", dataLayout->node_scalar);
       p->set< RCP<DataLayout> >("QP Scalar Data Layout", dataLayout->qp_scalar);
       p->set< RCP<DataLayout> >("QP Vector Data Layout", dataLayout->qp_vector);

       p->set<RCP<ParamLib> >("Parameter Library", paramLib);
       Teuchos::ParameterList& paramList = materialDataBase->getElementBlockSublist(elementBlockName, "Poissons Ratio");
       p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

       ev = rcp(new LCM::PoissonsRatio<EvalT,AlbanyTraits>(*p));
       fm0.template registerEvaluator<EvalT>(ev);
     }

     { // Strain
       RCP<ParameterList> p = rcp(new ParameterList("Strain"));

       //Input
       p->set<std::string>("Gradient QP Variable Name", "Displacement Gradient");

       //Output
       p->set<std::string>("Strain Name", "Strain");

       ev = rcp(new LCM::Strain<EvalT,AlbanyTraits>(*p,dataLayout));
       fm0.template registerEvaluator<EvalT>(ev);
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
       p->set< RCP<DataLayout> >("QP Tensor Data Layout", dataLayout->qp_tensor);

       //Outputs: F, J
       p->set<std::string>("DefGrad Name", "Deformation Gradient"); //dataLayout->qp_tensor also
       p->set<std::string>("DetDefGrad Name", "Determinant of Deformation Gradient"); 
       p->set< RCP<DataLayout> >("QP Scalar Data Layout", dataLayout->qp_scalar);

       ev = rcp(new LCM::DefGrad<EvalT,AlbanyTraits>(*p));
       fm0.template registerEvaluator<EvalT>(ev);
     }

     { // Linear elasticity stress
       RCP<ParameterList> p = rcp(new ParameterList("Stress"));

       //Input
       p->set<std::string>("Strain Name", "Strain");
       p->set< RCP<DataLayout> >("QP Tensor Data Layout", dataLayout->qp_tensor);

       p->set<std::string>("Elastic Modulus Name", "Elastic Modulus");
       p->set< RCP<DataLayout> >("QP Scalar Data Layout", dataLayout->qp_scalar);
       
       p->set<std::string>("Poissons Ratio Name", "Poissons Ratio");  // dataLayout->qp_scalar also

       //Output
       p->set<std::string>("Stress Name", "Stress"); //dataLayout->qp_tensor also

       ev = rcp(new LCM::Stress<EvalT,AlbanyTraits>(*p));
       fm0.template registerEvaluator<EvalT>(ev);
       p = stateMgr.registerStateVariable("Stress",dataLayout->qp_tensor, dataLayout->dummy, elementBlockName, "scalar", 0.0);
       ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
       fm0.template registerEvaluator<EvalT>(ev);
     }

     { // Displacement Resid
       RCP<ParameterList> p = rcp(new ParameterList("Displacement Resid"));

       //Input
       p->set<std::string>("Stress Name", "Stress");
       p->set< RCP<DataLayout> >("QP Tensor Data Layout", dataLayout->qp_tensor);

       // \todo Is the required?
       p->set<std::string>("DefGrad Name", "Deformation Gradient"); //dataLayout->qp_tensor also

       p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
       p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dataLayout->node_qp_vector);

       // extra input for time dependent term
       p->set<std::string>("Weighted BF Name", "wBF");
       p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", dataLayout->node_qp_scalar);
       p->set<std::string>("Time Dependent Variable Name", "Displacement_dotdot");
       p->set< RCP<DataLayout> >("QP Vector Data Layout", dataLayout->qp_vector);

       //Output
       p->set<std::string>("Residual Name", "Displacement Residual");
       p->set< RCP<DataLayout> >("Node Vector Data Layout", dataLayout->node_vector);

       ev = rcp(new LCM::ElasticityResid<EvalT,AlbanyTraits>(*p));
       fm0.template registerEvaluator<EvalT>(ev);
     }

     if (fieldManagerChoice == Albany::BUILD_RESID_FM)  {
       PHX::Tag<typename EvalT::ScalarT> res_tag("Scatter", dataLayout->dummy);
       fm0.requireField<EvalT>(res_tag);
       return res_tag.clone();
     }
     else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM) {
       Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dataLayout);
       return respUtils.constructResponses(fm0, *responseList, stateMgr);
     }
   }  // ---- End Elasticity Evaluators ----

   return Teuchos::null;
}

#endif // PERIDIGMPROBLEM_HPP
