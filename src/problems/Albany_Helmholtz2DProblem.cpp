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


#include "Albany_Helmholtz2DProblem.hpp"
#include "Albany_SolutionAverageResponseFunction.hpp"
#include "Albany_SolutionTwoNormResponseFunction.hpp"
#include "Albany_InitialCondition.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"

Albany::Helmholtz2DProblem::
Helmholtz2DProblem(
                         const Teuchos::RCP<Teuchos::ParameterList>& params_,
                         const Teuchos::RCP<ParamLib>& paramLib_) :
  Albany::AbstractProblem(params_, paramLib_, 2)
{

  std::string& method = params->get("Name", "Helmholtz 2D Problem");
  *out << "Problem Name = " << method << std::endl;
  
  ksqr = params->get<double>("Ksqr",1.0);

  haveSource =  params->isSublist("Source Functions");
}

Albany::Helmholtz2DProblem::
~Helmholtz2DProblem()
{
}

void
Albany::Helmholtz2DProblem::
buildProblem(
    Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
    Albany::StateManager& stateMgr,
    std::vector< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses)
{
  /* Construct All Phalanx Evaluators */
  TEST_FOR_EXCEPTION(meshSpecs.size()!=1,std::logic_error,"Problem supports one Material Block");
  constructEvaluators(*meshSpecs[0], stateMgr, responses);
  constructDirichletEvaluators(*meshSpecs[0]);
}

void
Albany::Helmholtz2DProblem::constructEvaluators(
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

   RCP<shards::CellTopology> cellType = rcp(new shards::CellTopology(&meshSpecs.ctd)); 
   RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > >
     intrepidBasis = Albany::getIntrepidBasis(meshSpecs.ctd);

   const int numNodes = intrepidBasis->getCardinality();
   const int worksetSize = meshSpecs.worksetSize;

   Intrepid::DefaultCubatureFactory<RealType> cubFactory;

   RCP <Intrepid::Cubature<RealType> > cubature = cubFactory.create(*cellType, meshSpecs.cubatureDegree);

   const int numDim = cubature->getDimension();
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
     dof_names[0] = "U";
     dof_names[1] = "V";

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

  if (haveSource) { // Source on U (Real) equation
    RCP<ParameterList> p = rcp(new ParameterList("Helmholtz Source U"));

    int type = FactoryTraits<AlbanyTraits>::id_source;
    p->set<int>("Type", type);

    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    p->set<string>("Pressure Source Name", "U GaussMonotone");
    p->set<string>("QP Coordinate Vector Name", "Coord Vec");


    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Source Functions");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    evaluators_to_build["Source U"] = p;
  }

  if (haveSource) { // Source on V (Imag) equation
    RCP<ParameterList> p = rcp(new ParameterList("Helmholtz Source V"));

    int type = FactoryTraits<AlbanyTraits>::id_source;
    p->set<int>("Type", type);

    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    p->set<string>("Pressure Source Name", "V GaussMonotone");
    p->set<string>("QP Coordinate Vector Name", "Coord Vec");


    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Source Functions");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    evaluators_to_build["Source V"] = p;
  }

  { // Helmholtz Resid
    RCP<ParameterList> p = rcp(new ParameterList("Helmholtz Resid"));

    int type = FactoryTraits<AlbanyTraits>::id_helmholtzresid;
    p->set<int>("Type", type);

    //Input
    p->set<string>("Weighted BF Name", "wBF");
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", dl->node_qp_scalar);

    p->set<string>("U Variable Name", "U");
    p->set<string>("V Variable Name", "V");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

    p->set<string>("Weighted Gradient BF Name", "wGrad BF");
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);

    p->set<string>("U Gradient Variable Name", "U Gradient");
    p->set<string>("V Gradient Variable Name", "V Gradient");
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    p->set<bool>("Have Source", haveSource);
    p->set<string>("U Pressure Source Name", "U GaussMonotone");
    p->set<string>("V Pressure Source Name", "V GaussMonotone");

    p->set<double>("Ksqr", ksqr);
    p->set<RCP<ParamLib> >("Parameter Library", paramLib);

    //Output
    p->set<string>("U Residual Name", "U Residual");
    p->set<string>("V Residual Name", "V Residual");
    p->set< RCP<DataLayout> >("Node Scalar Data Layout", dl->node_scalar);

    evaluators_to_build["Helmholtz Resid"] = p;
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
   PHX::Tag<AlbanyTraits::MPResidual::ScalarT> mpres_tag("Scatter", dl->dummy);
   fm->requireField<AlbanyTraits::MPResidual>(mpres_tag);
   PHX::Tag<AlbanyTraits::MPJacobian::ScalarT> mpjac_tag("Scatter", dl->dummy);
   fm->requireField<AlbanyTraits::MPJacobian>(mpjac_tag);

   // Construct Rsponses

   Teuchos::ParameterList& responseList = params->sublist("Response Functions");
   Albany::ResponseUtils respUtils(dl);
   rfm = respUtils.constructResponses(responses, responseList, evaluators_to_build, stateMgr);
}

void
Albany::Helmholtz2DProblem::constructDirichletEvaluators(
        const Albany::MeshSpecsStruct& meshSpecs)
{
   // Construct Dirichlet evaluators for all nodesets and names
   vector<string> dirichletNames(neq);
   dirichletNames[0] = "U";
   dirichletNames[1] = "V";
   Albany::DirichletUtils dirUtils;
   dfm = dirUtils.constructDirichletEvaluators(meshSpecs.nsNames, dirichletNames,
                                          this->params, this->paramLib);
}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::Helmholtz2DProblem::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidHelmhotz2DProblemParams");
  validPL->set<double>("Left BC", 0.0, "Value of Left BC [required]");
  validPL->set<double>("Right BC", 0.0, "Value to Right BC [required]");
  validPL->set<double>("Top BC", 0.0, "Value of Top BC [required]");
  validPL->set<double>("Bottom BC", 0.0, "Value to Bottom BC [required]");
  validPL->set<double>("Ksqr", 1.0, "Value of wavelength-squared [required]");

  return validPL;
}
