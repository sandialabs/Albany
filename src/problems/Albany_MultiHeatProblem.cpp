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


#include "Albany_MultiHeatProblem.hpp"
#include "Albany_SolutionAverageResponseFunction.hpp"
#include "Albany_SolutionTwoNormResponseFunction.hpp"
#include "Albany_SolutionMaxValueResponseFunction.hpp"
#include "Albany_InitialCondition.hpp"

#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "PHAL_FactoryTraits.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"


Albany::MultiHeatProblem::
MultiHeatProblem( const Teuchos::RCP<Teuchos::ParameterList>& params_,
             const Teuchos::RCP<ParamLib>& paramLib_,
             const int numDim_,
             const Teuchos::RCP<const Epetra_Comm>& comm_) :
  Albany::AbstractProblem(params_, paramLib_),
  haveSource(false),
  haveAbsorption(false),
  haveMatDB(false),
  numDim(numDim_),
  comm(comm_)
{
  this->setNumEquations(1);

  if (numDim==1) periodic = params->get("Periodic BC", false);
  else           periodic = false;
  if (periodic) *out <<" Periodic Boundary Conditions being used." <<std::endl;

  haveSource =  params->isSublist("Source Functions");
  haveAbsorption =  params->isSublist("Absorption");

  if(params->isType<string>("MaterialDB Filename")){
	haveMatDB = true;
    mtrlDbFilename = params->get<string>("MaterialDB Filename");
 // Create Material Database
    materialDB = Teuchos::rcp(new QCAD::MaterialDatabase(mtrlDbFilename, comm));
  }

}

Albany::MultiHeatProblem::
~MultiHeatProblem()
{
}

void
Albany::MultiHeatProblem::
buildProblem(
    Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
    Albany::StateManager& stateMgr,
    std::vector< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses)
{
  /* Construct All Phalanx Evaluators */
  int physSets = meshSpecs.size();
  cout << "MULTIHeat Num MeshSpecs: " << physSets << endl;
  fm.resize(physSets); rfm.resize(physSets);
  for (int ps=0; ps<physSets; ps++)
    constructEvaluators(*meshSpecs[ps], stateMgr, responses, ps);
  constructDirichletEvaluators(*meshSpecs[0]);
}

void
Albany::MultiHeatProblem::constructEvaluators(
       const Albany::MeshSpecsStruct& meshSpecs,
       Albany::StateManager& stateMgr,
       std::vector< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses,
       int ps)
{
   using Teuchos::RCP;
   using Teuchos::rcp;
   using Teuchos::ParameterList;
   using PHX::DataLayout;
   using PHX::MDALayout;
   using std::vector;
   using PHAL::FactoryTraits;
   using PHAL::AlbanyTraits;

   RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > >
     intrepidBasis = Albany::getIntrepidBasis(meshSpecs.ctd);
   RCP<shards::CellTopology> cellType = rcp(new shards::CellTopology (&meshSpecs.ctd));

   const int numNodes = intrepidBasis->getCardinality();
   const int worksetSize = meshSpecs.worksetSize;

   Intrepid::DefaultCubatureFactory<RealType> cubFactory;
   RCP <Intrepid::Cubature<RealType> > cubature = cubFactory.create(*cellType, meshSpecs.cubatureDegree);

   const int numQPts = cubature->getNumPoints();
   const int numVertices = cellType->getNodeCount();

   *out << "Field Dimensions: Workset=" << worksetSize 
        << ", Vertices= " << numVertices
        << ", Nodes= " << numNodes
        << ", QuadPts= " << numQPts
        << ", Dim= " << numDim << endl;

   // Parser will build parameter list that determines the field
   // evaluators to build
   std::map<string, RCP<ParameterList> > evaluators_to_build;

   RCP<Albany::Layouts> dl = rcp(new Albany::Layouts(worksetSize,numVertices,numNodes,numQPts,numDim));
   Albany::ProblemUtils probUtils(dl);

   Teuchos::ArrayRCP<string> dof_names(neq);
     dof_names[0] = "Temperature";
   Teuchos::ArrayRCP<string> dof_names_dot(neq);
     dof_names_dot[0] = "Temperature_dot";
   Teuchos::ArrayRCP<string> resid_names(neq);
     resid_names[0] = "Temperature Residual";

   evaluators_to_build["Gather Solution"] = 
     probUtils.constructGatherSolutionEvaluator(false, dof_names, dof_names_dot);

   evaluators_to_build["Scatter Residual"] = 
     probUtils.constructScatterResidualEvaluator(false, resid_names);

  evaluators_to_build["Gather Coordinate Vector"] = 
    probUtils.constructGatherCoordinateVectorEvaluator();

  evaluators_to_build["Map To Physical Frame"] = 
    probUtils.constructMapToPhysicalFrameEvaluator( cellType, cubature);

  evaluators_to_build["Compute Basis Functions"] =
    probUtils.constructComputeBasisFunctionsEvaluator(cellType, intrepidBasis, cubature);

  for (int i=0; i<neq; i++) {
    evaluators_to_build["DOF "+dof_names[i]] =
      probUtils.constructDOFInterpolationEvaluator(dof_names[i]);

    evaluators_to_build["DOF "+dof_names_dot[i]] =
      probUtils.constructDOFInterpolationEvaluator(dof_names_dot[i]);

    evaluators_to_build["DOF Grad "+dof_names[i]] =
      probUtils.constructDOFGradInterpolationEvaluator(dof_names[i]);
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

    if(haveMatDB)
     
      p->set< RCP<QCAD::MaterialDatabase> >("MaterialDB", materialDB);

    evaluators_to_build["Thermal Conductivity"] = p;
  }

  if (haveAbsorption) { // Absorption
    RCP<ParameterList> p = rcp(new ParameterList);

    int type = FactoryTraits<AlbanyTraits>::id_absorption;
    p->set<int>("Type", type);

    p->set<string>("QP Variable Name", "Absorption");
    p->set<string>("QP Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Node Data Layout", dl->node_scalar);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Absorption");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    evaluators_to_build["Absorption"] = p;
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
    p->set<bool>("Have Absorption", haveAbsorption);
    p->set<string>("Source Name", "Source");

    p->set<string>("Thermal Conductivity Name", "Thermal Conductivity");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

    p->set<string>("Absorption Name", "Thermal Conductivity");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    
    p->set<string>("Gradient QP Variable Name", "Temperature Gradient");
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    p->set<string>("Weighted Gradient BF Name", "wGrad BF");
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);
    if (params->isType<string>("Convection Velocity"))
    	p->set<string>("Convection Velocity",
                       params->get<string>("Convection Velocity"));
    if (params->isType<bool>("Have Rho Cp"))
    	p->set<bool>("Have Rho Cp", params->get<bool>("Have Rho Cp"));

    //Output
    p->set<string>("Residual Name", "Temperature Residual");
    p->set< RCP<DataLayout> >("Node Scalar Data Layout", dl->node_scalar);

    evaluators_to_build["Heat Resid"] = p;
  }

   // Build Field Evaluators for each evaluation type
   PHX::EvaluatorFactory<AlbanyTraits,FactoryTraits<AlbanyTraits> > factory;
   RCP< vector< RCP<PHX::Evaluator_TemplateManager<AlbanyTraits> > > >
     evaluators;
   evaluators = factory.buildEvaluators(evaluators_to_build);

   // Create a FieldManager
   fm[ps] = Teuchos::rcp(new PHX::FieldManager<AlbanyTraits>);

   // Register all Evaluators
   PHX::registerEvaluators(evaluators, *fm[ps]);

   PHX::Tag<AlbanyTraits::Residual::ScalarT> res_tag("Scatter", dl->dummy);
   fm[ps]->requireField<AlbanyTraits::Residual>(res_tag);
   PHX::Tag<AlbanyTraits::Jacobian::ScalarT> jac_tag("Scatter", dl->dummy);
   fm[ps]->requireField<AlbanyTraits::Jacobian>(jac_tag);
   PHX::Tag<AlbanyTraits::Tangent::ScalarT> tan_tag("Scatter", dl->dummy);
   fm[ps]->requireField<AlbanyTraits::Tangent>(tan_tag);
   PHX::Tag<AlbanyTraits::SGResidual::ScalarT> sgres_tag("Scatter", dl->dummy);
   fm[ps]->requireField<AlbanyTraits::SGResidual>(sgres_tag);
   PHX::Tag<AlbanyTraits::SGJacobian::ScalarT> sgjac_tag("Scatter", dl->dummy);
   fm[ps]->requireField<AlbanyTraits::SGJacobian>(sgjac_tag);
   PHX::Tag<AlbanyTraits::SGTangent::ScalarT> sgtan_tag("Scatter", dl->dummy);
   fm[ps]->requireField<AlbanyTraits::SGTangent>(sgtan_tag);
   PHX::Tag<AlbanyTraits::MPResidual::ScalarT> mpres_tag("Scatter", dl->dummy);
   fm[ps]->requireField<AlbanyTraits::MPResidual>(mpres_tag);
   PHX::Tag<AlbanyTraits::MPJacobian::ScalarT> mpjac_tag("Scatter", dl->dummy);
   fm[ps]->requireField<AlbanyTraits::MPJacobian>(mpjac_tag);
   PHX::Tag<AlbanyTraits::MPTangent::ScalarT> mptan_tag("Scatter", dl->dummy);
   fm[ps]->requireField<AlbanyTraits::MPTangent>(mptan_tag);

   //Construct Rsponses
   Teuchos::ParameterList& responseList = params->sublist("Response Functions");
   Albany::ResponseUtils respUtils(dl);
   rfm[ps] = respUtils.constructResponses(responses, responseList, evaluators_to_build, stateMgr);
}

void
Albany::MultiHeatProblem::constructDirichletEvaluators(
        const Albany::MeshSpecsStruct& meshSpecs)
{
   // Construct Dirichlet evaluators for all nodesets and names
   vector<string> dirichletNames(neq);
   dirichletNames[0] = "T";
   Albany::DirichletUtils dirUtils;
   dfm = dirUtils.constructDirichletEvaluators(meshSpecs.nsNames, dirichletNames,
                                          this->params, this->paramLib);
}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::MultiHeatProblem::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidMultiHeatProblemParams");

  if (numDim==1)
    validPL->set<bool>("Periodic BC", false, "Flag to indicate periodic BC for 1D problems");
  validPL->sublist("Thermal Conductivity", false, "");
  validPL->set("Convection Velocity", "{0,0,0}", "");
  validPL->set<bool>("Have Rho Cp", false, "Flag to indicate if rhoCp is used");
  validPL->set<string>("MaterialDB Filename","materials.xml","Filename of material database xml file");

  return validPL;
}

