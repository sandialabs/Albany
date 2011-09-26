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
#include "PoroElasticityProblem.hpp"
#include "Albany_SolutionAverageResponseFunction.hpp"
#include "Albany_SolutionTwoNormResponseFunction.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"

Albany::PoroElasticityProblem::
PoroElasticityProblem(const Teuchos::RCP<Teuchos::ParameterList>& params_,
			const Teuchos::RCP<ParamLib>& paramLib_,
			const int numDim_) :
  Albany::AbstractProblem(params_, paramLib_, numDim_ + 1), // additional DOF for pore pressure
  haveSource(false),
  numDim(numDim_)
{
 
  std::string& method = params->get("Name", "PoroElasticity ");
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

Albany::PoroElasticityProblem::
~PoroElasticityProblem()
{
}

void
Albany::PoroElasticityProblem::
buildProblem(
    const Albany::MeshSpecsStruct& meshSpecs,
    Albany::StateManager& stateMgr,
    std::vector< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses)
{
  /* Construct All Phalanx Evaluators */
  constructEvaluators(meshSpecs, stateMgr);
  constructDirichletEvaluators(meshSpecs);

  // Build response functions
  Teuchos::ParameterList& responseList = params->sublist("Response Functions");
  int num_responses = responseList.get("Number", 0);
  responses.resize(num_responses);
  for (int i=0; i<num_responses; i++) {
     std::string name = responseList.get(Albany::strint("Response",i), "??");

     if (name == "Solution Average")
       responses[i] = Teuchos::rcp(new SolutionAverageResponseFunction());

     else if (name == "Solution Two Norm")
       responses[i] = Teuchos::rcp(new SolutionTwoNormResponseFunction());

     else {
       TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                          std::endl <<
                          "Error!  Unknown response function " << name <<
                          "!" << std::endl << "Supplied parameter list is " <<
                          std::endl << responseList);
     }

  }
}


void
Albany::PoroElasticityProblem::constructEvaluators(
        const Albany::MeshSpecsStruct& meshSpecs,
        Albany::StateManager& stateMgr)
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
   string scatterName="Scatter PoreFluid";


   // ----------------------setup the solution field ---------------//

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

  // Pore Pressure Variable
   Teuchos::ArrayRCP<string> tdof_names(1);
     tdof_names[0] = "Pore Pressure";
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

   evaluators_to_build["Gather Pore Pressure Solution"] =
     probUtils.constructGatherSolutionEvaluator(false, tdof_names, tdof_names_dot, T_offset);

   evaluators_to_build["Scatter Pore Pressure Residual"] =
     probUtils.constructScatterResidualEvaluator(false, tresid_names, T_offset, scatterName);

   // ----------------------setup the solution field ---------------//



   // General FEM stuff
   evaluators_to_build["Gather Coordinate Vector"] =
     probUtils.constructGatherCoordinateVectorEvaluator();

   evaluators_to_build["Map To Physical Frame"] =
     probUtils.constructMapToPhysicalFrameEvaluator(cellType, cubature);

   evaluators_to_build["Compute Basis Functions"] =
     probUtils.constructComputeBasisFunctionsEvaluator(cellType, intrepidBasis, cubature);

   // Poroelasticity parameter

   {  // Porosity
      RCP<ParameterList> p = rcp(new ParameterList);

      int type = FactoryTraits<AlbanyTraits>::id_porosity;
      p->set<int>("Type", type);

	  p->set<string>("Porosity Name", "Porosity");
	  p->set<string>("QP Coordinate Vector Name", "Coord Vec");
	  p->set< RCP<DataLayout> >("Node Data Layout", dl->node_scalar);
	  p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
	  p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

	  p->set<RCP<ParamLib> >("Parameter Library", paramLib);
	  Teuchos::ParameterList& paramList = params->sublist("Porosity");
	  p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

	  // Setting this turns on linear dependence of E on T, E = E_ + dEdT*T)
	  p->set<string>("Strain Name", "Strain");
	  p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);
	  p->set<string>("QP Pore Pressure Name", "Pore Pressure");
	  p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

	  evaluators_to_build["Porosity"] = p;
	  evaluators_to_build["Save Porosity"] =
	  stateMgr.registerStateVariable("Porosity",dl->qp_scalar,
	              dl->dummy, FactoryTraits<AlbanyTraits>::id_savestatefield,"zero");
     }



   { // Biot Coefficient
      RCP<ParameterList> p = rcp(new ParameterList);

      int type = FactoryTraits<AlbanyTraits>::id_biotcoefficient;
      p->set<int>("Type", type);

	  p->set<string>("Biot Coefficient Name", "Biot Coefficient");
	  p->set<string>("QP Coordinate Vector Name", "Coord Vec");
	  p->set< RCP<DataLayout> >("Node Data Layout", dl->node_scalar);
	  p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
	  p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

	  p->set<RCP<ParamLib> >("Parameter Library", paramLib);
	  Teuchos::ParameterList& paramList = params->sublist("Biot Coefficient");
	  p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

	  // Setting this turns on linear dependence on porosity
	  p->set<string>("Porosity Name", "Porosity");

	  evaluators_to_build["Biot Coefficient"] = p;
	  evaluators_to_build["Save Biot Coefficient"] =
	  	  stateMgr.registerStateVariable("Biot Coefficient",dl->qp_scalar,
	  	              dl->dummy, FactoryTraits<AlbanyTraits>::id_savestatefield,"zero");
  }

   { // Biot Modulus
         RCP<ParameterList> p = rcp(new ParameterList);

         int type = FactoryTraits<AlbanyTraits>::id_biotmodulus;
         p->set<int>("Type", type);

   	  p->set<string>("Biot Modulus Name", "Biot Modulus");
   	  p->set<string>("QP Coordinate Vector Name", "Coord Vec");
   	  p->set< RCP<DataLayout> >("Node Data Layout", dl->node_scalar);
   	  p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
   	  p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

   	  p->set<RCP<ParamLib> >("Parameter Library", paramLib);
   	  Teuchos::ParameterList& paramList = params->sublist("Biot Modulus");
   	  p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

   	  // Setting this turns on linear dependence on porosity and Biot's coeffcient
   	  p->set<string>("Porosity Name", "Porosity");
      p->set<string>("Biot Coefficient Name", "Biot Coefficient");

   	  evaluators_to_build["Biot Modulus"] = p;
   	  evaluators_to_build["Save Biot Modulus"] =
   	  stateMgr.registerStateVariable("Biot Modulus",dl->qp_scalar,
   	   dl->dummy, FactoryTraits<AlbanyTraits>::id_savestatefield,"zero");
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

  // Skeleton parameter

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


    p->set<string>("Porosity Name", "Porosity"); // porosity is defined at Cubature points
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);


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
    //p->set<string>("QP Pore Pressure Name", "Pore Pressure");

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

  { // Total Stress
    RCP<ParameterList> p = rcp(new ParameterList("Total Stress"));

    int type = FactoryTraits<AlbanyTraits>::id_total_stress;
    p->set<int>("Type", type);

    //Input
    p->set<string>("Strain Name", "Strain");
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);

    p->set<string>("Elastic Modulus Name", "Elastic Modulus");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

    p->set<string>("Poissons Ratio Name", "Poissons Ratio");  // dl->qp_scalar also

    p->set<string>("Biot Coefficient Name", "Biot Coefficient");  // dl->qp_scalar also

    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

    p->set<string>("Pore Pressure Name", "Pore Pressure");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

    //Output
    p->set<string>("Total Stress Name", "Total Stress"); //dl->qp_tensor also

    evaluators_to_build["Total Stress"] = p;
    evaluators_to_build["Save Total Stress"] =
      stateMgr.registerStateVariable("Total Stress",dl->qp_tensor,
            dl->dummy, FactoryTraits<AlbanyTraits>::id_savestatefield,"zero");
  }

  { // Displacement Resid
    RCP<ParameterList> p = rcp(new ParameterList("Displacement Resid"));

    int type = FactoryTraits<AlbanyTraits>::id_poroelasticityresidmomentum;
    p->set<int>("Type", type);

    //Input
    p->set<string>("Total Stress Name", "Total Stress");
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);

    p->set<string>("Weighted Gradient BF Name", "wGrad BF");
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);

    p->set<bool>("Disable Transient", true);

    //Output
    p->set<string>("Residual Name", "Displacement Residual");
    p->set< RCP<DataLayout> >("Node Vector Data Layout", dl->node_vector);

    evaluators_to_build["PoroElasticity Momentum Resid"] = p;
  }


  if (haveSource) { // Source
    RCP<ParameterList> p = rcp(new ParameterList);

    int type = FactoryTraits<AlbanyTraits>::id_source;
    p->set<int>("Type", type);

    p->set<string>("Source Name", "Source");
    p->set<string>("QP Variable Name", "Pore Pressure");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Source Functions");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    evaluators_to_build["Source"] = p;
  }
  { // Pore Pressure Resid
    RCP<ParameterList> p = rcp(new ParameterList("Pore Pressure Resid"));

    int type = FactoryTraits<AlbanyTraits>::id_poroelasticityresidmass;
    p->set<int>("Type", type);

    //Input

    // Input from nodal points
    p->set<string>("Weighted BF Name", "wBF");
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", dl->node_qp_scalar);
    p->set<string>("QP Variable Name", "Pore Pressure"); // NOTE: QP and nodal vaue shares same name
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

    p->set<string>("QP Time Derivative Variable Name", "Pore Pressure_dot");

    p->set<bool>("Have Source", haveSource);
    p->set<string>("Source Name", "Source");

    p->set<bool>("Have Absorption", false);

    // Input from cubature points
    p->set<string>("Thermal Conductivity Name", "Thermal Conductivity");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

    p->set<string>("Porosity Name", "Porosity");
    p->set<string>("Biot Coefficient Name", "Biot Coefficient");
    p->set<string>("Biot Modulus Name", "Biot Modulus");

    p->set<string>("Gradient QP Variable Name", "Pore Pressure Gradient");
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    p->set<string>("Weighted Gradient BF Name", "wGrad BF");
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);

    p->set<string>("Strain Name", "Strain");
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);


    //Output
    p->set<string>("Residual Name", "Pore Pressure Residual");
    p->set< RCP<DataLayout> >("Node Scalar Data Layout", dl->node_scalar);

    evaluators_to_build["Poroelasticity Mass Resid"] = p;
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

   PHX::Tag<AlbanyTraits::Residual::ScalarT> res_tag2(scatterName, dl->dummy);
   fm->requireField<AlbanyTraits::Residual>(res_tag2);
   PHX::Tag<AlbanyTraits::Jacobian::ScalarT> jac_tag2(scatterName, dl->dummy);
   fm->requireField<AlbanyTraits::Jacobian>(jac_tag2);
   PHX::Tag<AlbanyTraits::Tangent::ScalarT> tan_tag2(scatterName, dl->dummy);
   fm->requireField<AlbanyTraits::Tangent>(tan_tag2);
   PHX::Tag<AlbanyTraits::SGResidual::ScalarT> sgres_tag2(scatterName, dl->dummy);
   fm->requireField<AlbanyTraits::SGResidual>(sgres_tag2);
   PHX::Tag<AlbanyTraits::SGJacobian::ScalarT> sgjac_tag2(scatterName, dl->dummy);
   fm->requireField<AlbanyTraits::SGJacobian>(sgjac_tag2);
   PHX::Tag<AlbanyTraits::SGTangent::ScalarT> sgtan_tag2(scatterName, dl->dummy);
   fm->requireField<AlbanyTraits::SGTangent>(sgtan_tag2);
   PHX::Tag<AlbanyTraits::MPResidual::ScalarT> mpres_tag2(scatterName, dl->dummy);
   fm->requireField<AlbanyTraits::MPResidual>(mpres_tag2);
   PHX::Tag<AlbanyTraits::MPJacobian::ScalarT> mpjac_tag2(scatterName, dl->dummy);
   fm->requireField<AlbanyTraits::MPJacobian>(mpjac_tag2);
   PHX::Tag<AlbanyTraits::MPTangent::ScalarT> mptan_tag2(scatterName, dl->dummy);
   fm->requireField<AlbanyTraits::MPTangent>(mptan_tag2);

   const Albany::StateManager::RegisteredStates& reg = stateMgr.getRegisteredStates();
   Albany::StateManager::RegisteredStates::const_iterator st = reg.begin();
   while (st != reg.end()) {
     PHX::Tag<AlbanyTraits::Residual::ScalarT> res_out_tag(st->first, dl->dummy);
     fm->requireField<AlbanyTraits::Residual>(res_out_tag);
     st++;
   }
}

void
Albany::PoroElasticityProblem::constructDirichletEvaluators(
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
Albany::PoroElasticityProblem::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidPoroElasticityProblemParams");
  validPL->sublist("Porosity", false, "");
  validPL->sublist("Biot Coefficient", false, "");
  validPL->sublist("Biot Modulus", false, "");
  validPL->sublist("Thermal Conductivity", false, "");
  validPL->sublist("Elastic Modulus", false, "");
  validPL->sublist("Poissons Ratio", false, "");

  return validPL;
}

void
Albany::PoroElasticityProblem::getAllocatedStates(
   Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid::FieldContainer<RealType> > > > oldState_,
   Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid::FieldContainer<RealType> > > > newState_
   ) const
{
  oldState_ = oldState;
  newState_ = newState;
}

