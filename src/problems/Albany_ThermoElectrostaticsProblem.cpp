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


#include "Albany_ThermoElectrostaticsProblem.hpp"
#include "Albany_SolutionAverageResponseFunction.hpp"
#include "Albany_SolutionTwoNormResponseFunction.hpp"
#include "Albany_SolutionMaxValueResponseFunction.hpp"
#include "Albany_SolutionFileL2ResponseFunction.hpp"
#include "Albany_InitialCondition.hpp"

#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "PHAL_FactoryTraits.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"


Albany::ThermoElectrostaticsProblem::
ThermoElectrostaticsProblem( const Teuchos::RCP<Teuchos::ParameterList>& params_,
             const Teuchos::RCP<ParamLib>& paramLib_,
             const int numDim_) :
  Albany::AbstractProblem(params_, paramLib_, 2),
  numDim(numDim_)
{
}

Albany::ThermoElectrostaticsProblem::
~ThermoElectrostaticsProblem()
{
}

void
Albany::ThermoElectrostaticsProblem::
buildProblem(
    const Albany::MeshSpecsStruct& meshSpecs,
    Albany::StateManager& stateMgr,
    std::vector< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses)
{
  /* Construct All Phalanx Evaluators */
  constructEvaluators(meshSpecs, stateMgr, responses);
  constructDirichletEvaluators(meshSpecs);
}


void
Albany::ThermoElectrostaticsProblem::constructEvaluators(
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
   using PHAL::FactoryTraits;
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
   Albany::ProblemUtils probUtils(dl);
   bool supportsTransient=false;

   // Define Field Names

   Teuchos::ArrayRCP<string> dof_names(neq);
     dof_names[0] = "Potential";
     dof_names[1] = "Temperature";

   Teuchos::ArrayRCP<string> dof_names_dot(neq);
   if (supportsTransient) {
     for (int i=0; i<neq; i++) dof_names_dot[i] = dof_names[i]+"_dot";
   }

   Teuchos::ArrayRCP<string> resid_names(neq);
     for (int i=0; i<neq; i++) resid_names[i] = dof_names[i]+" Residual";

   if (supportsTransient) evaluators_to_build["Gather Solution"] =
       probUtils.constructGatherSolutionEvaluator(false, dof_names, dof_names_dot);
   else  evaluators_to_build["Gather Solution"] =
       probUtils.constructGatherSolutionEvaluator_noTransient(false, dof_names);

   evaluators_to_build["Scatter Residual"] =
     probUtils.constructScatterResidualEvaluator(false, resid_names);

   evaluators_to_build["Gather Coordinate Vector"] =
     probUtils.constructGatherCoordinateVectorEvaluator();

   evaluators_to_build["Map To Physical Frame"] =
     probUtils.constructMapToPhysicalFrameEvaluator(cellType, cubature);

   evaluators_to_build["Compute Basis Functions"] =
     probUtils.constructComputeBasisFunctionsEvaluator(cellType, intrepidBasis, cubature);

   for (int i=0; i<neq; i++) {
     evaluators_to_build["DOF "+dof_names[i]] =
       probUtils.constructDOFInterpolationEvaluator(dof_names[i]);

     if (supportsTransient)
       evaluators_to_build["DOF "+dof_names_dot[i]] =
         probUtils.constructDOFInterpolationEvaluator(dof_names_dot[i]);

     evaluators_to_build["DOF Grad "+dof_names[i]] =
       probUtils.constructDOFGradInterpolationEvaluator(dof_names[i]);
  }


  { // Thermal conductivity
    RCP<ParameterList> p = rcp(new ParameterList);

    int type = FactoryTraits<AlbanyTraits>::id_teprop;
    p->set<int>("Type", type);

    p->set<string>("QP Variable Name", "Thermal Conductivity");
    p->set<string>("QP Variable Name 2", "Permittivity");  // really electrical conductivity
    p->set<string>("QP Variable Name 3", "Rho Cp"); 
    p->set<string>("Coordinate Vector Name", "Coord Vec");
    p->set<string>("Temperature Variable Name", "Temperature");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("TE Properties");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    evaluators_to_build["TE Properties"] = p;
  }

  {
    RCP<ParameterList> p = rcp(new ParameterList);

    int type = FactoryTraits<AlbanyTraits>::id_jouleheating;
    p->set<int>("Type", type);

    //Input
    p->set<string>("Gradient Variable Name", "Potential Gradient");
    p->set<string>("Flux Variable Name", "Potential Flux");
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);

    //Output
    p->set<string>("Source Name", "Joule");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

    evaluators_to_build["Joule Heating"] = p;
  }

  { // Potential Resid
    RCP<ParameterList> p = rcp(new ParameterList("Potential Resid"));

    int type = FactoryTraits<AlbanyTraits>::id_qcad_poisson_resid;
    p->set<int>("Type", type);

    //Input
    p->set<string>("Weighted BF Name", "wBF");
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", dl->node_qp_scalar);
    p->set<string>("QP Variable Name", "Potential");

    p->set<string>("Permittivity Name", "Permittivity");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

    p->set<string>("Gradient QP Variable Name", "Potential Gradient");
    p->set<string>("Flux QP Variable Name", "Potential Flux");
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    p->set<string>("Weighted Gradient BF Name", "wGrad BF");
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);

    p->set<bool>("Have Source", false);
    p->set<string>("Source Name", "None");

    //Output
    p->set<string>("Residual Name", "Potential Residual");
    p->set< RCP<DataLayout> >("Node Scalar Data Layout", dl->node_scalar);

    evaluators_to_build["Poisson Resid"] = p;
  }

  { // Temperature Resid
    RCP<ParameterList> p = rcp(new ParameterList("Temperature Resid"));

    int type = FactoryTraits<AlbanyTraits>::id_heateqresid;
    p->set<int>("Type", type);

    //Input
    p->set<string>("Weighted BF Name", "wBF");
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", dl->node_qp_scalar);
    p->set<string>("QP Variable Name", "Temperature");

    p->set<bool>("Have Source", true);
    p->set<string>("Source Name", "Joule");

    p->set<bool>("Have Absorption", false);

    p->set<string>("Thermal Conductivity Name", "Thermal Conductivity");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

    p->set<string>("Gradient QP Variable Name", "Temperature Gradient");
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    p->set<string>("Weighted Gradient BF Name", "wGrad BF");
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);
 
    // Poisson solve does not have transient terms
    p->set<bool>("Disable Transient", true);
    p->set<string>("QP Time Derivative Variable Name", "Temperature_dot");

    if (params->isType<string>("Convection Velocity")) {
      p->set<string>("Convection Velocity",params->get<string>("Convection Velocity"));
      p->set<string>("Rho Cp Name", "Rho Cp"); 
    }

    //Output
    p->set<string>("Residual Name", "Temperature Residual");
    p->set< RCP<DataLayout> >("Node Scalar Data Layout", dl->node_scalar);

    evaluators_to_build["ThermoElectrostatics Resid"] = p;
  }

   // Build Field Evaluators for each evaluation type
   PHX::EvaluatorFactory<AlbanyTraits,FactoryTraits<AlbanyTraits> > factory;
   RCP< vector< RCP<PHX::Evaluator_TemplateManager<AlbanyTraits> > > >
     evaluators;
   evaluators = factory.buildEvaluators(evaluators_to_build);

   // Create a FieldManager
   fm = Teuchos::rcp(new PHX::FieldManager<AlbanyTraits>);

   // Register all Evaluators
   PHX::registerEvaluators(evaluators, *fm);

   PHX::Tag<AlbanyTraits::Residual::ScalarT> res_tag("Scatter", dl->dummy);
   fm->requireField<AlbanyTraits::Residual>(res_tag);
   PHX::Tag<AlbanyTraits::Jacobian::ScalarT> jac_tag("Scatter", dl->dummy);
   fm->requireField<AlbanyTraits::Jacobian>(jac_tag);
   PHX::Tag<AlbanyTraits::Tangent::ScalarT> tan_tag("Scatter", dl->dummy);
   fm->requireField<AlbanyTraits::Tangent>(tan_tag);
   PHX::Tag<AlbanyTraits::SGResidual::ScalarT> sgres_tag("Scatter", dl->dummy);
   fm->requireField<AlbanyTraits::SGResidual>(sgres_tag);
   PHX::Tag<AlbanyTraits::SGJacobian::ScalarT> sgjac_tag("Scatter", dl->dummy);
   fm->requireField<AlbanyTraits::SGJacobian>(sgjac_tag);
   PHX::Tag<AlbanyTraits::SGTangent::ScalarT> sgtan_tag("Scatter", dl->dummy);
   fm->requireField<AlbanyTraits::SGTangent>(sgtan_tag);
   PHX::Tag<AlbanyTraits::MPResidual::ScalarT> mpres_tag("Scatter", dl->dummy);
   fm->requireField<AlbanyTraits::MPResidual>(mpres_tag);
   PHX::Tag<AlbanyTraits::MPJacobian::ScalarT> mpjac_tag("Scatter", dl->dummy);
   fm->requireField<AlbanyTraits::MPJacobian>(mpjac_tag);
   PHX::Tag<AlbanyTraits::MPTangent::ScalarT> mptan_tag("Scatter", dl->dummy);
   fm->requireField<AlbanyTraits::MPTangent>(mptan_tag);

   //Construct Rsponses
   Teuchos::ParameterList& responseList = params->sublist("Response Functions");
   Albany::ResponseUtils respUtils(dl);
   rfm = respUtils.constructResponses(responses, responseList, evaluators_to_build, stateMgr);
}

void
Albany::ThermoElectrostaticsProblem::constructDirichletEvaluators(
        const Albany::MeshSpecsStruct& meshSpecs)
{

   // Construct Dirichlet evaluators for all nodesets and names
   vector<string> dirichletNames(neq);
   dirichletNames[0] = "Phi";
   dirichletNames[1] = "T";
   Albany::DirichletUtils dirUtils;
   dfm = dirUtils.constructDirichletEvaluators(meshSpecs.nsNames, dirichletNames,
                                          this->params, this->paramLib);
}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::ThermoElectrostaticsProblem::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidThermoElectrostaticsProblemParams");

  validPL->sublist("TE Properties", false, "");
  validPL->set("Convection Velocity", "{0,0,0}", "");

  return validPL;
}
