/********************************************************************\
 *            Albany, Copyright (2010) Sandia Corporation             *
 *                                                                    *
 * Notice: This computer software was prepared by Sandia Corporation, *
 * hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
 * the Department of Energy (DOE). All rights in the computer software*
 * are reserved by DOE on behalf of the United States Government and  *
 * the Contractor as provided in the Contract. You are authorized to  *
 * use this computer software for Governmental purposes but it is not *
 * to be released or distributed to the public. NEITHER THE GOVERNMENT*
 * NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
 * ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
 * including this sentence must appear on any copies of this software.*
 *    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/

#include "LCM/LCM_FactoryTraits.hpp"
#include "NonlinearElasticityProblem.hpp"
#include "Albany_SolutionAverageResponseFunction.hpp"
#include "Albany_SolutionTwoNormResponseFunction.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Shards_BasicTopologies.hpp"
#include "Shards_CellTopology.hpp"

Albany::NonlinearElasticityProblem::
NonlinearElasticityProblem(const Teuchos::RCP<Teuchos::ParameterList>& params_,
			   const Teuchos::RCP<ParamLib>& paramLib_,
			   const int numDim_) :
  Albany::AbstractProblem(params_, paramLib_, numDim_),
  haveSource(false),
  numDim(numDim_)
{
 
  std::string& method = params->get("Name", "NonlinearElasticity ");
  *out << "Problem Name = " << method << std::endl;
  
  haveSource =  params->isSublist("Source Functions");

  matModel = params->sublist("Material Model").get("Model Name","NeoHookean");
}

Albany::NonlinearElasticityProblem::
~NonlinearElasticityProblem()
{
}

void
Albany::NonlinearElasticityProblem::
buildProblem(Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
	     Albany::StateManager& stateMgr,
	     std::vector< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses)
{
  /* Construct All Phalanx Evaluators */
  TEST_FOR_EXCEPTION(meshSpecs.size()!=1,std::logic_error,"Problem supports one Material Block");
  fm.resize(1); rfm.resize(1);
  constructEvaluators(*meshSpecs[0], stateMgr, responses);
  constructDirichletEvaluators(*meshSpecs[0]);
}

void
Albany::NonlinearElasticityProblem::constructEvaluators(
	const Albany::MeshSpecsStruct& meshSpecs,
	Albany::StateManager& stateMgr,
	std::vector< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses)
{
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::ParameterList;
  using PHX::DataLayout;
  using PHX::MDALayout;
  using std::vector;
  using LCM::FactoryTraits;
  using PHAL::AlbanyTraits;

  const bool composite = params->get("Use Composite Tet 10", false);
  RCP<shards::CellTopology> comp_cellType = rcp(new shards::CellTopology( shards::getCellTopologyData<shards::Tetrahedron<11> >() ) );
  RCP<shards::CellTopology> cellType = rcp(new shards::CellTopology (&meshSpecs.ctd));

  RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > >
    intrepidBasis = Albany::getIntrepidBasis(meshSpecs.ctd, composite);

  if (composite && meshSpecs.ctd.dimension==3 && meshSpecs.ctd.node_count==10) cellType = comp_cellType;

  numNodes = intrepidBasis->getCardinality();
  const int worksetSize = meshSpecs.worksetSize;

  Intrepid::DefaultCubatureFactory<RealType> cubFactory;
  RCP <Intrepid::Cubature<RealType> > cubature = cubFactory.create(*cellType, meshSpecs.cubatureDegree);

  numDim = cubature->getDimension();
  numQPts = cubature->getNumPoints();
  //numVertices = cellType->getNodeCount();
  numVertices = numNodes;

  *out << "Field Dimensions: Workset=" << worksetSize 
       << ", Vertices= " << numVertices
       << ", Nodes= " << numNodes
       << ", QuadPts= " << numQPts
       << ", Dim= " << numDim << endl;

   // Construct standard FEM evaluators with standard field names                              
   std::map<string, RCP<ParameterList> > evaluators_to_build;
   RCP<Albany::Layouts> dl = rcp(new Albany::Layouts(worksetSize,numVertices,numNodes,numQPts,numDim));
   Albany::ProblemUtils probUtils(dl,"LCM");
   bool supportsTransient=true;

   // Define Field Names
   Teuchos::ArrayRCP<string> dof_names(1);
     dof_names[0] = "Displacement";
   Teuchos::ArrayRCP<string> dof_names_dotdot(1);
   if (supportsTransient)
     dof_names_dotdot[0] = dof_names[0]+"_dotdot";
   Teuchos::ArrayRCP<string> resid_names(1);
     resid_names[0] = dof_names[0]+" Residual";

   evaluators_to_build["DOF "+dof_names[0]] =
     probUtils.constructDOFVecInterpolationEvaluator(dof_names[0]);

   if (supportsTransient)
     evaluators_to_build["DOF "+dof_names_dotdot[0]] =
       probUtils.constructDOFVecInterpolationEvaluator(dof_names_dotdot[0]);

   evaluators_to_build["DOF Grad "+dof_names[0]] =
     probUtils.constructDOFVecGradInterpolationEvaluator(dof_names[0]);

   if (supportsTransient) evaluators_to_build["Gather Solution"] =
       probUtils.constructGatherSolutionEvaluator(true, dof_names, dof_names_dotdot);
   else  evaluators_to_build["Gather Solution"] =
       probUtils.constructGatherSolutionEvaluator_noTransient(true, dof_names);

   evaluators_to_build["Scatter Residual"] =
     probUtils.constructScatterResidualEvaluator(true, resid_names);

   evaluators_to_build["Gather Coordinate Vector"] =
     probUtils.constructGatherCoordinateVectorEvaluator();

   evaluators_to_build["Map To Physical Frame"] =
     probUtils.constructMapToPhysicalFrameEvaluator(cellType, cubature);

   evaluators_to_build["Compute Basis Functions"] =
     probUtils.constructComputeBasisFunctionsEvaluator(cellType, intrepidBasis, cubature);

  { // Elastic Modulus
    RCP<ParameterList> p = rcp(new ParameterList);

    int type = FactoryTraits<AlbanyTraits>::id_elastic_modulus;
    p->set<int>("Type", type);

    p->set<string>("QP Variable Name", "Elastic Modulus");
    p->set<string>("QP Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Node Data Layout", dl->node_scalar);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Elastic Modulus");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    evaluators_to_build["Elastic Modulus"] = p;
  }

  { // Poissons Ratio 
    RCP<ParameterList> p = rcp(new ParameterList);

    int type = FactoryTraits<AlbanyTraits>::id_poissons_ratio;
    p->set<int>("Type", type);

    p->set<string>("QP Variable Name", "Poissons Ratio");
    p->set<string>("QP Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Node Data Layout", dl->node_scalar);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Poissons Ratio");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    evaluators_to_build["Poissons Ratio"] = p;
  }

  if (haveSource) { // Source
    TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                       "Error!  Sources not implemented in Elasticity yet!");

    RCP<ParameterList> p = rcp(new ParameterList);

    int type = FactoryTraits<AlbanyTraits>::id_source;
    p->set<int>("Type", type);

    p->set<string>("Source Name", "Source");
    p->set<string>("Variable Name", "Displacement");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Source Functions");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    evaluators_to_build["Source"] = p;
  }

  { // Deformation Gradient
    RCP<ParameterList> p = rcp(new ParameterList("DefGrad"));

    int type = FactoryTraits<AlbanyTraits>::id_defgrad;
    p->set<int>("Type", type);

    //Inputs: flags, weights, GradU
    const bool avgJ = params->get("avgJ", false);
    p->set<bool>("avgJ Name", avgJ);
    const bool volavgJ = params->get("volavgJ", false);
    p->set<bool>("volavgJ Name", volavgJ);
    p->set<string>("Weights Name","Weights");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set<string>("Gradient QP Variable Name", "Displacement Gradient");
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);

    //Outputs: F, J
    p->set<string>("DefGrad Name", "Deformation Gradient"); //dl->qp_tensor also
    p->set<string>("DetDefGrad Name", "Determinant of Deformation Gradient"); 
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

    evaluators_to_build["DefGrad"] = p;
  }

  if (matModel == "NeoHookean")
  {
    { // LCG
      RCP<ParameterList> p = rcp(new ParameterList("LCG"));

      int type = FactoryTraits<AlbanyTraits>::id_lcg;
      p->set<int>("Type", type);
     
      //Input
      p->set<string>("DefGrad Name", "Deformation Gradient");
      p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);

      //Output
      p->set<string>("LCG Name", "LCG"); //dl->qp_tensor also

      evaluators_to_build["LCG"] = p;
    }

    { // Stress
      RCP<ParameterList> p = rcp(new ParameterList("Stress"));

      int type = FactoryTraits<AlbanyTraits>::id_neohookean_stress;
      p->set<int>("Type", type);

      //Input
      p->set<string>("LCG Name", "LCG");
      p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);

      p->set<string>("Elastic Modulus Name", "Elastic Modulus");
      p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

      p->set<string>("Poissons Ratio Name", "Poissons Ratio");  // dl->qp_scalar also
      p->set<string>("DetDefGrad Name", "Determinant of Deformation Gradient");  // dl->qp_scalar also

      //Output
      p->set<string>("Stress Name", matModel); //dl->qp_tensor also

      evaluators_to_build["Stress"] = p;
      evaluators_to_build["Save Stress"] =
	stateMgr.registerStateVariable(matModel,dl->qp_tensor,
				       dl->dummy, FactoryTraits<AlbanyTraits>::id_savestatefield,"zero");
    }
  }
  else if (matModel == "NeoHookean AD")
  {
    RCP<ParameterList> p = rcp(new ParameterList("Stress"));

    int type = FactoryTraits<AlbanyTraits>::id_pisdwdf_stress;
    p->set<int>("Type", type);

    //Input
    p->set<string>("Elastic Modulus Name", "Elastic Modulus");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set<string>("Poissons Ratio Name", "Poissons Ratio");  // dl->qp_scalar also

    p->set<string>("DefGrad Name", "Deformation Gradient"); 
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);

    //Output
    p->set<string>("Stress Name", matModel); //dl->qp_tensor also

    evaluators_to_build["Stress"] = p;
    evaluators_to_build["Save Stress"] =
      stateMgr.registerStateVariable(matModel,dl->qp_tensor,
				     dl->dummy, FactoryTraits<AlbanyTraits>::id_savestatefield,"zero");
  }
  else if (matModel == "J2")
  { 
    { // Hardening Modulus
      RCP<ParameterList> p = rcp(new ParameterList);

      int type = FactoryTraits<AlbanyTraits>::id_hardening_modulus;
      p->set<int>("Type", type);

      p->set<string>("QP Variable Name", "Hardening Modulus");
      p->set<string>("QP Coordinate Vector Name", "Coord Vec");
      p->set< RCP<DataLayout> >("Node Data Layout", dl->node_scalar);
      p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
      p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

      p->set<RCP<ParamLib> >("Parameter Library", paramLib);
      Teuchos::ParameterList& paramList = params->sublist("Hardening Modulus");
      p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

      evaluators_to_build["Hardening Modulus"] = p;
    }

    { // Yield Strength
      RCP<ParameterList> p = rcp(new ParameterList);

      int type = FactoryTraits<AlbanyTraits>::id_yield_strength;
      p->set<int>("Type", type);

      p->set<string>("QP Variable Name", "Yield Strength");
      p->set<string>("QP Coordinate Vector Name", "Coord Vec");
      p->set< RCP<DataLayout> >("Node Data Layout", dl->node_scalar);
      p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
      p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

      p->set<RCP<ParamLib> >("Parameter Library", paramLib);
      Teuchos::ParameterList& paramList = params->sublist("Yield Strength");
      p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

      evaluators_to_build["Yield Strength"] = p;
    }

    { // Saturation Modulus
      RCP<ParameterList> p = rcp(new ParameterList);

      int type = FactoryTraits<AlbanyTraits>::id_sat_mod;
      p->set<int>("Type", type);

      p->set<string>("QP Variable Name", "Saturation Modulus");
      p->set<string>("QP Coordinate Vector Name", "Coord Vec");
      p->set< RCP<DataLayout> >("Node Data Layout", dl->node_scalar);
      p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
      p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

      p->set<RCP<ParamLib> >("Parameter Library", paramLib);
      Teuchos::ParameterList& paramList = params->sublist("Saturation Modulus");
      p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

      evaluators_to_build["Saturation Modulus"] = p;
    }

    { // Saturation Exponent
      RCP<ParameterList> p = rcp(new ParameterList);

      int type = FactoryTraits<AlbanyTraits>::id_sat_exp;
      p->set<int>("Type", type);

      p->set<string>("QP Variable Name", "Saturation Exponent");
      p->set<string>("QP Coordinate Vector Name", "Coord Vec");
      p->set< RCP<DataLayout> >("Node Data Layout", dl->node_scalar);
      p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
      p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

      p->set<RCP<ParamLib> >("Parameter Library", paramLib);
      Teuchos::ParameterList& paramList = params->sublist("Saturation Exponent");
      p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

      evaluators_to_build["Saturation Exponent"] = p;
    }

    if ( numDim == 3 && params->get("Compute Dislocation Density Tensor", false) )
    { // Dislocation Density Tensor
      RCP<ParameterList> p = rcp(new ParameterList("Dislocation Density"));
      
      int type = FactoryTraits<AlbanyTraits>::id_dislocation_density;
      p->set<int>("Type", type);
    
      //Input
      p->set<string>("Fp Name", "Fp");
      p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);
      p->set<string>("BF Name", "BF");
      p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", dl->node_qp_scalar);
      p->set<string>("Gradient BF Name", "Grad BF");
      p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);

      //Output
      p->set<string>("Dislocation Density Name", "G"); //dl->qp_tensor also
 
      //Declare what state data will need to be saved (name, layout, init_type)
      evaluators_to_build["Save DislocationDensity"] =
	stateMgr.registerStateVariable("G",
				       dl->qp_tensor,
				       dl->dummy, 
				       FactoryTraits<AlbanyTraits>::id_savestatefield,
				       "zero");

      evaluators_to_build["G"] = p;
    }

    {// Stress
      RCP<ParameterList> p = rcp(new ParameterList("Stress"));

      int type = FactoryTraits<AlbanyTraits>::id_j2_stress;
      p->set<int>("Type", type);

      //Input
      p->set<string>("DefGrad Name", "Deformation Gradient");
      p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);

      p->set<string>("Elastic Modulus Name", "Elastic Modulus");
      p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

      p->set<string>("Poissons Ratio Name", "Poissons Ratio");  // dl->qp_scalar also
      p->set<string>("Hardening Modulus Name", "Hardening Modulus"); // dl->qp_scalar also
      p->set<string>("Saturation Modulus Name", "Saturation Modulus"); // dl->qp_scalar also
      p->set<string>("Saturation Exponent Name", "Saturation Exponent"); // dl->qp_scalar also
      p->set<string>("Yield Strength Name", "Yield Strength"); // dl->qp_scalar also
      p->set<string>("DetDefGrad Name", "Determinant of Deformation Gradient");  // dl->qp_scalar also

      //Output
      p->set<string>("Stress Name", matModel); //dl->qp_tensor also
      p->set<string>("Fp Name", "Fp");  // dl->qp_tensor also
      p->set<string>("Eqps Name", "eqps");  // dl->qp_scalar also
 
      //Declare what state data will need to be saved (name, layout, init_type)
      evaluators_to_build["Save Stress"] =
	stateMgr.registerStateVariable(matModel,dl->qp_tensor,
				       dl->dummy, FactoryTraits<AlbanyTraits>::id_savestatefield,"zero");
      evaluators_to_build["Save Fp"] =
	stateMgr.registerStateVariable("Fp",dl->qp_tensor,
				       dl->dummy, FactoryTraits<AlbanyTraits>::id_savestatefield,"identity",true);
      evaluators_to_build["Save Eqps"] =
	stateMgr.registerStateVariable("eqps",dl->qp_scalar,
				       dl->dummy, FactoryTraits<AlbanyTraits>::id_savestatefield,"zero",true);

      evaluators_to_build["Stress"] = p;
    }
  }
  else
    TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
		       "Unrecognized Material Name: " << matModel 
		       << "  Recognized names are : NeoHookean and J2");
    

  { // Residual
    RCP<ParameterList> p = rcp(new ParameterList("Residual"));

    int type = FactoryTraits<AlbanyTraits>::id_tl_elas_resid;
    p->set<int>("Type", type);

    //Input
    p->set<string>("Stress Name", matModel);
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);

    p->set<string>("DefGrad Name", "Deformation Gradient"); //dl->qp_tensor also

    p->set<string>("DetDefGrad Name", "Determinant of Deformation Gradient");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

    p->set<string>("Weighted Gradient BF Name", "wGrad BF");
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);

    p->set<string>("Weighted BF Name", "wBF");
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", dl->node_qp_scalar);
    p->set<RCP<ParamLib> >("Parameter Library", paramLib);

    //Output
    p->set<string>("Residual Name", "Displacement Residual");
    p->set< RCP<DataLayout> >("Node Vector Data Layout", dl->node_vector);

    evaluators_to_build["Residual"] = p;
  }

  // Build Field Evaluators for each evaluation type
  PHX::EvaluatorFactory<AlbanyTraits,FactoryTraits<AlbanyTraits> > factory;
  RCP< vector< RCP<PHX::Evaluator_TemplateManager<AlbanyTraits> > > >
    evaluators;
  evaluators = factory.buildEvaluators(evaluators_to_build);

  // Create a FieldManager
  fm[0] = Teuchos::rcp(new PHX::FieldManager<AlbanyTraits>);

  // Register all Evaluators
  PHX::registerEvaluators(evaluators, *fm[0]);

  PHX::Tag<AlbanyTraits::Residual::ScalarT> res_tag("Scatter", dl->dummy);
  fm[0]->requireField<AlbanyTraits::Residual>(res_tag);
  PHX::Tag<AlbanyTraits::Jacobian::ScalarT> jac_tag("Scatter", dl->dummy);
  fm[0]->requireField<AlbanyTraits::Jacobian>(jac_tag);
  PHX::Tag<AlbanyTraits::Tangent::ScalarT> tan_tag("Scatter", dl->dummy);
  fm[0]->requireField<AlbanyTraits::Tangent>(tan_tag);
  PHX::Tag<AlbanyTraits::SGResidual::ScalarT> sgres_tag("Scatter", dl->dummy);
  fm[0]->requireField<AlbanyTraits::SGResidual>(sgres_tag);
  PHX::Tag<AlbanyTraits::SGJacobian::ScalarT> sgjac_tag("Scatter", dl->dummy);
  fm[0]->requireField<AlbanyTraits::SGJacobian>(sgjac_tag);
  PHX::Tag<AlbanyTraits::SGTangent::ScalarT> sgtan_tag("Scatter", dl->dummy);
   fm[0]->requireField<AlbanyTraits::SGTangent>(sgtan_tag);
  PHX::Tag<AlbanyTraits::MPResidual::ScalarT> mpres_tag("Scatter", dl->dummy);
  fm[0]->requireField<AlbanyTraits::MPResidual>(mpres_tag);
  PHX::Tag<AlbanyTraits::MPJacobian::ScalarT> mpjac_tag("Scatter", dl->dummy);
  fm[0]->requireField<AlbanyTraits::MPJacobian>(mpjac_tag);
  PHX::Tag<AlbanyTraits::MPTangent::ScalarT> mptan_tag("Scatter", dl->dummy);
   fm[0]->requireField<AlbanyTraits::MPTangent>(mptan_tag);

   //Construct Responses
   Teuchos::ParameterList& responseList = params->sublist("Response Functions");
   Albany::ResponseUtils respUtils(dl,"LCM");
   rfm[0] = respUtils.constructResponses(responses, responseList, evaluators_to_build, stateMgr);
}

void
Albany::NonlinearElasticityProblem::constructDirichletEvaluators(
        const Albany::MeshSpecsStruct& meshSpecs)
{

   // Construct Dirichlet evaluators for all nodesets and names
   vector<string> dirichletNames(neq);
   dirichletNames[0] = "X";
   if (neq>1) dirichletNames[1] = "Y";
   if (neq>2) dirichletNames[2] = "Z";
   Albany::DirichletUtils dirUtils;
   dfm = dirUtils.constructDirichletEvaluators(meshSpecs.nsNames, dirichletNames,
                                          this->params, this->paramLib);
}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::NonlinearElasticityProblem::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidNonlinearElasticityProblemParams");

  validPL->sublist("Elastic Modulus", false, "");
  validPL->sublist("Poissons Ratio", false, "");
  validPL->sublist("Shear Modulus", false, "");
  validPL->sublist("Material Model", false, "");
  validPL->set<bool>("avgJ", false, "Flag to indicate the J should be averaged");
  validPL->set<bool>("volavgJ", false, "Flag to indicate the J should be volume averaged");
  validPL->set<bool>("Use Composite Tet 10", false, "Flag to use the compostie tet 10 basis in Intrepid");
  if (matModel == "J2")
  {
    validPL->set<bool>("Compute Dislocation Density Tensor", false, "Flag to compute the dislocaiton density tensor (only for 3D)");
    validPL->sublist("Hardening Modulus", false, "");
    validPL->sublist("Yield Strength", false, "");
    validPL->sublist("Saturation Modulus", false, "");
    validPL->sublist("Saturation Exponent", false, "");
  }

  return validPL;
}

void
Albany::NonlinearElasticityProblem::getAllocatedStates(
   Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid::FieldContainer<RealType> > > > oldState_,
   Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid::FieldContainer<RealType> > > > newState_
   ) const
{
  oldState_ = oldState;
  newState_ = newState;
}

