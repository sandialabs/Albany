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
#include "GradientDamageProblem.hpp"
#include "Albany_SolutionAverageResponseFunction.hpp"
#include "Albany_SolutionTwoNormResponseFunction.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"

Albany::GradientDamageProblem::
GradientDamageProblem(
		      const Teuchos::RCP<Teuchos::ParameterList>& params_,
		      const Teuchos::RCP<ParamLib>& paramLib_,
		      const int numDim_) :
  Albany::AbstractProblem(params_, paramLib_, numDim_ + 1),
  haveSource(false),
  numDim(numDim_)
{
 
  std::string& method = params->get("Name", "GradientDamage ");
  *out << "Problem Name = " << method << std::endl;
  
  haveSource =  params->isSublist("Source Functions");

// Changing this ifdef changes ordering from  (X,Y,D) to (D,X,Y)
//#define NUMBER_D_FIRST
#ifdef NUMBER_D_FIRST
  D_offset=0;
  X_offset=1;
#else
  X_offset=0;
  D_offset=numDim;
#endif
}

Albany::GradientDamageProblem::
~GradientDamageProblem()
{
}

void
Albany::GradientDamageProblem::
buildProblem(
    Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
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
Albany::GradientDamageProblem::constructEvaluators(
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
   using std::map;
   using LCM::FactoryTraits;
   using PHAL::AlbanyTraits;

   RCP<shards::CellTopology> cellType = rcp(new shards::CellTopology (&meshSpecs.ctd));
   RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > >
     intrepidBasis = Albany::getIntrepidBasis(meshSpecs.ctd);

   numNodes = intrepidBasis->getCardinality();
   const int worksetSize = meshSpecs.worksetSize;

   Intrepid::DefaultCubatureFactory<RealType> cubFactory;
   RCP <Intrepid::Cubature<RealType> > cubature = cubFactory.create(*cellType, meshSpecs.cubatureDegree);

   const int numQPts = cubature->getNumPoints();
   const int numVertices = cellType->getVertexCount();

   *out << "Field Dimensions: Workset=" << worksetSize 
        << ", Vertices= " << numVertices
        << ", Nodes= " << numNodes
        << ", QuadPts= " << numQPts
        << ", Dim= " << numDim << endl;


   // Construct standard FEM evaluators with standard field names                              
   std::map<string, RCP<ParameterList> > evaluators_to_build;
   RCP<Albany::Layouts> dl = rcp(new Albany::Layouts(worksetSize,numVertices,numNodes,numQPts,numDim));
   Albany::ProblemUtils probUtils(dl,"LCM");
   string scatterName="Scatter Damage";

   // Displacement Variable
   Teuchos::ArrayRCP<string> dof_names(1);
     dof_names[0] = "Displacement";
   Teuchos::ArrayRCP<string> resid_names(1);
     resid_names[0] = "Mechanical Residual";

   evaluators_to_build["DOF "+dof_names[0]] =
     probUtils.constructDOFVecInterpolationEvaluator(dof_names[0]);

   evaluators_to_build["DOF Grad "+dof_names[0]] =
     probUtils.constructDOFVecGradInterpolationEvaluator(dof_names[0]);

   evaluators_to_build["Gather Solution"] =
     probUtils.constructGatherSolutionEvaluator_noTransient(true, dof_names, X_offset); 
   evaluators_to_build["Scatter Residual"] =
     probUtils.constructScatterResidualEvaluator(true, resid_names, X_offset);

   // Damage Variable
   Teuchos::ArrayRCP<string> ddof_names(1);
     ddof_names[0] = "Damage";
   Teuchos::ArrayRCP<string> ddof_names_dot(1);
     ddof_names_dot[0] = ddof_names[0]+"_dot";
   Teuchos::ArrayRCP<string> dresid_names(1);
     dresid_names[0] = ddof_names[0]+" Residual";

   evaluators_to_build["DOF "+ddof_names[0]] =
     probUtils.constructDOFInterpolationEvaluator(ddof_names[0]);

   evaluators_to_build["DOF "+ddof_names_dot[0]] =
     probUtils.constructDOFInterpolationEvaluator(ddof_names_dot[0]);

   evaluators_to_build["DOF Grad "+ddof_names[0]] =
     probUtils.constructDOFGradInterpolationEvaluator(ddof_names[0]);

   evaluators_to_build["Gather Damage Solution"] =
     probUtils.constructGatherSolutionEvaluator(false, ddof_names, ddof_names_dot, D_offset);

   evaluators_to_build["Scatter Damage Residual"] =
     probUtils.constructScatterResidualEvaluator(false, dresid_names, D_offset, scatterName);

   // General FEM Stuff
   evaluators_to_build["Gather Coordinate Vector"] =
     probUtils.constructGatherCoordinateVectorEvaluator();

   evaluators_to_build["Map To Physical Frame"] =
     probUtils.constructMapToPhysicalFrameEvaluator(cellType, cubature);

   evaluators_to_build["Compute Basis Functions"] =
     probUtils.constructComputeBasisFunctionsEvaluator(cellType, intrepidBasis, cubature);

  { // Bulk Modulus
    RCP<ParameterList> p = rcp(new ParameterList);

    int type = FactoryTraits<AlbanyTraits>::id_bulk_modulus;
    p->set<int>("Type", type);

    p->set<string>("QP Variable Name", "Bulk Modulus");
    p->set<string>("QP Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Node Data Layout", dl->node_scalar);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Bulk Modulus");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    evaluators_to_build["Bulk Modulus"] = p;
  }

  { // Shear Modulus 
    RCP<ParameterList> p = rcp(new ParameterList);

    int type = FactoryTraits<AlbanyTraits>::id_shear_modulus;
    p->set<int>("Type", type);

    p->set<string>("QP Variable Name", "Shear Modulus");
    p->set<string>("QP Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Node Data Layout", dl->node_scalar);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Shear Modulus");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    evaluators_to_build["Shear Modulus"] = p;
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
    p->set<string>("Gradient QP Variable Name", "Displacement Gradient");
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);

    //Outputs: F, J
    p->set<string>("DefGrad Name", "Deformation Gradient"); //dl->qp_tensor also
    p->set<string>("DetDefGrad Name", "Determinant of Deformation Gradient"); 
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

    evaluators_to_build["DefGrad"] = p;
  }
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

  {// Stress
    RCP<ParameterList> p = rcp(new ParameterList("Stress"));

    int type = FactoryTraits<AlbanyTraits>::id_j2_damage;
    p->set<int>("Type", type);

    //Input
    p->set<string>("DefGrad Name", "Deformation Gradient");
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);

    p->set<string>("Bulk Modulus Name", "Bulk Modulus");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

    p->set<string>("Shear Modulus Name", "Shear Modulus");  // dl->qp_scalar also
    p->set<string>("Hardening Modulus Name", "Hardening Modulus"); // dl->qp_scalar also
    p->set<string>("Yield Strength Name", "Yield Strength"); // dl->qp_scalar also
    p->set<string>("Saturation Modulus Name", "Saturation Modulus"); // dl->qp_scalar also
    p->set<string>("Saturation Exponent Name", "Saturation Exponent"); // dl->qp_scalar also
    p->set<string>("DetDefGrad Name", "Determinant of Deformation Gradient");  // dl->qp_scalar also
    p->set<string>("Damage Name", "Damage");

    //Output
    p->set<string>("Stress Name", "Stress"); //dl->qp_tensor also
    p->set<string>("DP Name", "DP"); // dl->qp_scalar also
    p->set<string>("Effective Stress Name", "Effective Stress"); // dl->qp_scalar also
    p->set<string>("Energy Name", "Energy"); // dl->qp_scalar also

    p->set<string>("Fp Name", "Fp");  // dl->qp_tensor also
    p->set<string>("Eqps Name", "eqps");  // dl->qp_scalar also

 
    //Declare what state data will need to be saved (name, layout, init_type)
    // A :true: as 5th argument declares that the previous state needs to be saved

    int issf = FactoryTraits<AlbanyTraits>::id_savestatefield;
    evaluators_to_build["Save Stress"] =
      stateMgr.registerStateVariable("Stress",dl->qp_tensor, dl->dummy, issf,"zero");
    evaluators_to_build["Save Fp"] =
      stateMgr.registerStateVariable("Fp",dl->qp_tensor, dl->dummy, issf,"identity",true);
    evaluators_to_build["Save Eqps"] =
      stateMgr.registerStateVariable("eqps",dl->qp_scalar, dl->dummy, issf,"zero",true);

    evaluators_to_build["Stress"] = p;
  }

  { // Displacement Resid
    RCP<ParameterList> p = rcp(new ParameterList("Mechanical Residual"));

    int type = FactoryTraits<AlbanyTraits>::id_tl_elas_resid;
    p->set<int>("Type", type);

    //Input
    p->set<string>("Stress Name", "Stress");
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
    p->set<string>("Residual Name", "Mechanical Residual");
    p->set< RCP<DataLayout> >("Node Vector Data Layout", dl->node_vector);

    evaluators_to_build["Mechanical Residual"] = p;
  }

  { // Damage length scale
    RCP<ParameterList> p = rcp(new ParameterList);

    int type = FactoryTraits<AlbanyTraits>::id_damage_ls;
    p->set<int>("Type", type);

    p->set<string>("QP Variable Name", "Damage Length Scale");
    p->set<string>("QP Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Node Data Layout", dl->node_scalar);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Damage Length Scale");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    evaluators_to_build["Damage Length Scale"] = p;
  }

  { // Damage Source
    RCP<ParameterList> p = rcp(new ParameterList("Damage Source"));

    int type = FactoryTraits<AlbanyTraits>::id_damage_source;
    p->set<int>("Type", type);

    //Input
    RealType gc = params->get("gc", 1.0);
    p->set<RealType>("gc Name", gc);
    p->set<string>("Bulk Modulus Name", "Bulk Modulus");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set<string>("Damage Name", "Damage");
    p->set<string>("DP Name", "DP");
    p->set<string>("Effective Stress Name", "Effective Stress");
    p->set<string>("Energy Name", "Energy");
    p->set<string>("DetDefGrad Name", "Determinant of Deformation Gradient");
    p->set<string>("Damage Length Scale Name", "Damage Length Scale");

    //Output
    p->set<string>("Damage Source Name", "Damage Source");

    int issf = FactoryTraits<AlbanyTraits>::id_savestatefield;
    evaluators_to_build["Save Damage Source"] =
      stateMgr.registerStateVariable("Damage Source",dl->qp_scalar, dl->dummy, issf,"zero",true);
    evaluators_to_build["Save Damage"] =
      stateMgr.registerStateVariable("Damage",dl->qp_scalar, dl->dummy, issf,"zero", true);
    evaluators_to_build["Damage Source"] = p;
  }

  { // Damage Resid
    RCP<ParameterList> p = rcp(new ParameterList("Damage Resid"));

    int type = FactoryTraits<AlbanyTraits>::id_damage_resid;
    p->set<int>("Type", type);

    //Input
    RealType gc = params->get("gc", 0.0);
    p->set<RealType>("gc Name", gc);
    p->set<string>("Weighted BF Name", "wBF");
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", dl->node_qp_scalar);
    p->set<string>("QP Variable Name", "Damage");

    p->set<string>("QP Time Derivative Variable Name", "Damage_dot");

    p->set<string>("Damage Source Name", "Damage Source");  //dl->qp_scalar

    p->set<string>("Damage Length Scale Name", "Damage Length Scale");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

    p->set<string>("Gradient QP Variable Name", "Damage Gradient");
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    p->set<string>("Weighted Gradient BF Name", "wGrad BF");
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);

    //Output
    p->set<string>("Residual Name", "Damage Residual");
    p->set< RCP<DataLayout> >("Node Scalar Data Layout", dl->node_scalar);

    evaluators_to_build["Damage Residual"] = p;
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

   PHX::Tag<AlbanyTraits::Residual::ScalarT> res_tag2(scatterName, dl->dummy);
   fm[0]->requireField<AlbanyTraits::Residual>(res_tag2);
   PHX::Tag<AlbanyTraits::Jacobian::ScalarT> jac_tag2(scatterName, dl->dummy);
   fm[0]->requireField<AlbanyTraits::Jacobian>(jac_tag2);
   PHX::Tag<AlbanyTraits::Tangent::ScalarT> tan_tag2(scatterName, dl->dummy);
   fm[0]->requireField<AlbanyTraits::Tangent>(tan_tag2);
   PHX::Tag<AlbanyTraits::SGResidual::ScalarT> sgres_tag2(scatterName, dl->dummy);
   fm[0]->requireField<AlbanyTraits::SGResidual>(sgres_tag2);
   PHX::Tag<AlbanyTraits::SGJacobian::ScalarT> sgjac_tag2(scatterName, dl->dummy);
   fm[0]->requireField<AlbanyTraits::SGJacobian>(sgjac_tag2);
   PHX::Tag<AlbanyTraits::SGTangent::ScalarT> sgtan_tag2(scatterName, dl->dummy);
   fm[0]->requireField<AlbanyTraits::SGTangent>(sgtan_tag2);
   PHX::Tag<AlbanyTraits::MPResidual::ScalarT> mpres_tag2(scatterName, dl->dummy);
   fm[0]->requireField<AlbanyTraits::MPResidual>(mpres_tag2);
   PHX::Tag<AlbanyTraits::MPJacobian::ScalarT> mpjac_tag2(scatterName, dl->dummy);
   fm[0]->requireField<AlbanyTraits::MPJacobian>(mpjac_tag2);
   PHX::Tag<AlbanyTraits::MPTangent::ScalarT> mptan_tag2(scatterName, dl->dummy);
   fm[0]->requireField<AlbanyTraits::MPTangent>(mptan_tag2);

   //Construct Rsponses
   Teuchos::ParameterList& responseList = params->sublist("Response Functions");
   Albany::ResponseUtils respUtils(dl,"LCM");
   rfm[0] = respUtils.constructResponses(responses, responseList, evaluators_to_build, stateMgr);
}

void
Albany::GradientDamageProblem::constructDirichletEvaluators(
        const Albany::MeshSpecsStruct& meshSpecs)
{
   // Construct Dirichlet evaluators for all nodesets and names
   vector<string> dirichletNames(neq);
   dirichletNames[X_offset] = "X";
   if (numDim>1) dirichletNames[X_offset+1] = "Y";
   if (numDim>2) dirichletNames[X_offset+2] = "Z";
   dirichletNames[D_offset] = "D";
   Albany::DirichletUtils dirUtils;
   dfm = dirUtils.constructDirichletEvaluators(meshSpecs.nsNames, dirichletNames,
                                          this->params, this->paramLib);
}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::GradientDamageProblem::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidGradientDamageProblemParams");

  validPL->sublist("Bulk Modulus", false, "");
  validPL->sublist("Shear Modulus", false, "");
  validPL->sublist("Poissons Ratio", false, "");
  validPL->sublist("Damage Length Scale", false, "");
  validPL->sublist("Hardening Modulus", false, "");
  validPL->sublist("Yield Strength", false, "");
  validPL->sublist("Saturation Modulus", false, "");
  validPL->sublist("Saturation Exponent", false, "");
  validPL->set<double>("gc", false, "");
  validPL->set<bool>("avgJ", false, "Flag to indicate the J should be averaged");
  validPL->set<bool>("volavgJ", false, "Flag to indicate the J should be volume averaged");
  validPL->set<bool>("Use Composite Tet 10", false, "Flag to use the compostie tet 10 basis in Intrepid");

  return validPL;
}

void
Albany::GradientDamageProblem::getAllocatedStates(
   Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid::FieldContainer<RealType> > > > oldState_,
   Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid::FieldContainer<RealType> > > > newState_
   ) const
{
  oldState_ = oldState;
  newState_ = newState;
}
