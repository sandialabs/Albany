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
#include "ThermoElasticityProblem.hpp"
#include "Albany_SolutionAverageResponseFunction.hpp"
#include "Albany_SolutionTwoNormResponseFunction.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"

Albany::ThermoElasticityProblem::
ThermoElasticityProblem(const Teuchos::RCP<Teuchos::ParameterList>& params_,
			const Teuchos::RCP<ParamLib>& paramLib_,
			const int numDim_) :
  Albany::AbstractProblem(params_, paramLib_, numDim_ + 1),
  haveSource(false),
  numDim(numDim_)
{
 
  std::string& method = params->get("Name", "ThermoElasticity ");
  *out << "Problem Name = " << method << std::endl;
  
  haveSource =  params->isSublist("Source Functions");

// Changing this ifdef changes ordering from  (X,Y,T) to (T,X,Y)
//#define NUMBER_T_FIRST
#ifdef NUMBER_T_FIRST
  T_offset=0;
  X_offset=1;
#else
  X_offset=0;
  T_offset=numDim;
#endif
}

Albany::ThermoElasticityProblem::
~ThermoElasticityProblem()
{
}

void
Albany::ThermoElasticityProblem::
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
Albany::ThermoElasticityProblem::constructEvaluators(
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

   RCP<shards::CellTopology> cellType = rcp(new shards::CellTopology (&meshSpecs.ctd));
   RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > >
     intrepidBasis = Albany::getIntrepidBasis(meshSpecs.ctd);

   const int numNodes = intrepidBasis->getCardinality();
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
   string scatterName="Scatter Heat";

   // Displacement Variable
   Teuchos::ArrayRCP<string> dof_names(1);
     dof_names[0] = "Displacement";
   Teuchos::ArrayRCP<string> resid_names(1);
     resid_names[0] = dof_names[0]+" Residual";

   evaluators_to_build["DOF "+dof_names[0]] =
     probUtils.constructDOFVecInterpolationEvaluator(dof_names[0]);

   evaluators_to_build["DOF Grad "+dof_names[0]] =
     probUtils.constructDOFVecGradInterpolationEvaluator(dof_names[0]);

   evaluators_to_build["Gather Solution"] =
     probUtils.constructGatherSolutionEvaluator_noTransient(true, dof_names, X_offset);

   evaluators_to_build["Scatter Residual"] =
     probUtils.constructScatterResidualEvaluator(true, resid_names, X_offset);

  // Temperature Variable
   Teuchos::ArrayRCP<string> tdof_names(1);
     tdof_names[0] = "Temperature";
   Teuchos::ArrayRCP<string> tdof_names_dot(1);
     tdof_names_dot[0] = tdof_names[0]+"_dot";
   Teuchos::ArrayRCP<string> tresid_names(1);
     tresid_names[0] = tdof_names[0]+" Residual";

   evaluators_to_build["DOF "+tdof_names[0]] =
     probUtils.constructDOFInterpolationEvaluator(tdof_names[0]);

   evaluators_to_build["DOF "+tdof_names_dot[0]] =
     probUtils.constructDOFInterpolationEvaluator(tdof_names_dot[0]);

   evaluators_to_build["DOF Grad "+tdof_names[0]] =
     probUtils.constructDOFGradInterpolationEvaluator(tdof_names[0]);

   evaluators_to_build["Gather Temperature Solution"] =
     probUtils.constructGatherSolutionEvaluator(false, tdof_names, tdof_names_dot, T_offset);

   evaluators_to_build["Scatter Temperature Residual"] =
     probUtils.constructScatterResidualEvaluator(false, tresid_names, T_offset, scatterName);

   // General FEM stuff
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

    // Setting this turns on linear dependence of E on T, E = E_ + dEdT*T)
    p->set<string>("QP Temperature Name", "Temperature");

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

    // Setting this turns on linear dependence of nu on T, nu = nu_ + dnudT*T)
    p->set<string>("QP Temperature Name", "Temperature");

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

  { // Strain
    RCP<ParameterList> p = rcp(new ParameterList("Strain"));

    int type = FactoryTraits<AlbanyTraits>::id_strain;
    p->set<int>("Type", type);

    //Input
    p->set<string>("Gradient QP Variable Name", "Displacement Gradient");
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);

    //Output
    p->set<string>("Strain Name", "Strain"); //dl->qp_tensor also

    evaluators_to_build["Strain"] = p;
  }

  { // Stress
    RCP<ParameterList> p = rcp(new ParameterList("Stress"));

    int type = FactoryTraits<AlbanyTraits>::id_stress;
    p->set<int>("Type", type);

    //Input
    p->set<string>("Strain Name", "Strain");
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);

    p->set<string>("Elastic Modulus Name", "Elastic Modulus");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

    p->set<string>("Poissons Ratio Name", "Poissons Ratio");  // dl->qp_scalar also

    //Output
    p->set<string>("Stress Name", "Stress"); //dl->qp_tensor also

    evaluators_to_build["Stress"] = p;
    evaluators_to_build["Save Stress"] =
      stateMgr.registerStateVariable("Stress",dl->qp_tensor,
            dl->dummy, FactoryTraits<AlbanyTraits>::id_savestatefield,"zero");
  }

  { // Displacement Resid
    RCP<ParameterList> p = rcp(new ParameterList("Displacement Resid"));

    int type = FactoryTraits<AlbanyTraits>::id_elasticityresid;
    p->set<int>("Type", type);

    //Input
    p->set<string>("Stress Name", "Stress");
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);

    p->set<string>("Weighted Gradient BF Name", "wGrad BF");
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);

    p->set<bool>("Disable Transient", true);

    //Output
    p->set<string>("Residual Name", "Displacement Residual");
    p->set< RCP<DataLayout> >("Node Vector Data Layout", dl->node_vector);

    evaluators_to_build["Elasticity Resid"] = p;
  }

  { // Thermal conductivity
    RCP<ParameterList> p = rcp(new ParameterList);

    int type = FactoryTraits<AlbanyTraits>::id_thermal_conductivity;
    p->set<int>("Type", type);

    p->set<string>("QP Variable Name", "Thermal Conductivity");
    p->set<string>("QP Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Node Data Layout", dl->node_scalar);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Thermal Conductivity");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    evaluators_to_build["Thermal Conductivity"] = p;
  }

  if (haveSource) { // Source
    RCP<ParameterList> p = rcp(new ParameterList);

    int type = FactoryTraits<AlbanyTraits>::id_source;
    p->set<int>("Type", type);

    p->set<string>("Source Name", "Source");
    p->set<string>("Variable Name", "Temperature");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Source Functions");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    evaluators_to_build["Source"] = p;
  }
  { // Temperature Resid
    RCP<ParameterList> p = rcp(new ParameterList("Temperature Resid"));

    int type = FactoryTraits<AlbanyTraits>::id_heateqresid;
    p->set<int>("Type", type);

    //Input
    p->set<string>("Weighted BF Name", "wBF");
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", dl->node_qp_scalar);
    p->set<string>("QP Variable Name", "Temperature");

    p->set<string>("QP Time Derivative Variable Name", "Temperature_dot");

    p->set<bool>("Have Source", haveSource);
    p->set<string>("Source Name", "Source");

    p->set<bool>("Have Absorption", false);

    p->set<string>("Thermal Conductivity Name", "Thermal Conductivity");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

    p->set<string>("Gradient QP Variable Name", "Temperature Gradient");
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    p->set<string>("Weighted Gradient BF Name", "wGrad BF");
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);

    //Output
    p->set<string>("Residual Name", "Temperature Residual");
    p->set< RCP<DataLayout> >("Node Scalar Data Layout", dl->node_scalar);

    evaluators_to_build["Heat2D Resid"] = p;
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
Albany::ThermoElasticityProblem::constructDirichletEvaluators(
        const Albany::MeshSpecsStruct& meshSpecs)
{
   // Construct Dirichlet evaluators for all nodesets and names
   vector<string> dirichletNames(neq);
   dirichletNames[X_offset] = "X";
   if (numDim>1) dirichletNames[X_offset+1] = "Y";
   if (numDim>2) dirichletNames[X_offset+2] = "Z";
   dirichletNames[T_offset] = "T";
   Albany::DirichletUtils dirUtils;
   dfm = dirUtils.constructDirichletEvaluators(meshSpecs.nsNames, dirichletNames,
                                          this->params, this->paramLib);
}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::ThermoElasticityProblem::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidThermoElasticityProblemParams");

  validPL->sublist("Elastic Modulus", false, "");
  validPL->sublist("Poissons Ratio", false, "");

  return validPL;
}

void
Albany::ThermoElasticityProblem::getAllocatedStates(
   Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid::FieldContainer<RealType> > > > oldState_,
   Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid::FieldContainer<RealType> > > > newState_
   ) const
{
  oldState_ = oldState;
  newState_ = newState;
}

