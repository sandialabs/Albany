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

  dofNames.resize(neq);
// Changing this ifdef changes ordering from  (X,Y,T) to (T,X,Y)
//#define NUMBER_T_FIRST
#ifdef NUMBER_T_FIRST
  T_offset=0;
  X_offset=1;
#else
  X_offset=0;
  T_offset=numDim;
#endif
  dofNames[X_offset] = "X";
  if (numDim>1) dofNames[X_offset+1] = "Y";
  if (numDim>2) dofNames[X_offset+2] = "Z";
  dofNames[T_offset] = "T";
}

Albany::ThermoElasticityProblem::
~ThermoElasticityProblem()
{
}

void
Albany::ThermoElasticityProblem::
buildProblem(
    const Albany::MeshSpecsStruct& meshSpecs,
    Albany::StateManager& stateMgr,
    std::vector< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses)
{
  /* Construct All Phalanx Evaluators */
  constructEvaluators(meshSpecs, stateMgr);
  constructDirichletEvaluators(meshSpecs.nsNames);

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
Albany::ThermoElasticityProblem::constructEvaluators(
        const Albany::MeshSpecsStruct& meshSpecs,
        Albany::StateManager& stateMgr)
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
     intrepidBasis = this->getIntrepidBasis(meshSpecs.ctd);

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

   // Parser will build parameter list that determines the field
   // evaluators to build
   map<string, RCP<ParameterList> > evaluators_to_build;

   RCP<DataLayout> node_scalar = rcp(new MDALayout<Cell,Node>(worksetSize,numNodes));
   RCP<DataLayout> qp_scalar = rcp(new MDALayout<Cell,QuadPoint>(worksetSize,numQPts));

   RCP<DataLayout> node_vector = rcp(new MDALayout<Cell,Node,Dim>(worksetSize,numNodes,numDim));
   RCP<DataLayout> qp_vector = rcp(new MDALayout<Cell,QuadPoint,Dim>(worksetSize,numQPts,numDim));
   RCP<DataLayout> qp_tensor = rcp(new MDALayout<Cell,QuadPoint,Dim,Dim>(worksetSize,numQPts,numDim,numDim));

   RCP<DataLayout> vertices_vector = 
     rcp(new MDALayout<Cell,Vertex, Dim>(worksetSize,numVertices,numDim));
   // Basis functions, Basis function gradient
   RCP<DataLayout> node_qp_scalar =
     rcp(new MDALayout<Cell,Node,QuadPoint>(worksetSize,numNodes, numQPts));
   RCP<DataLayout> node_qp_vector =
     rcp(new MDALayout<Cell,Node,QuadPoint,Dim>(worksetSize,numNodes, numQPts,numDim));

   RCP<DataLayout> dummy = rcp(new MDALayout<Dummy>(0));

  { // Gather Solution X,Y,Z
   RCP< vector<string> > dof_names = rcp(new vector<string>(1));
     (*dof_names)[0] = "Displacement";

    RCP<ParameterList> p = rcp(new ParameterList);
    int type = FactoryTraits<AlbanyTraits>::id_gather_solution;
    p->set<int>("Type", type);
    p->set< RCP< vector<string> > >("Solution Names", dof_names);
    p->set<bool>("Vector Field", true);
    p->set< RCP<DataLayout> >("Data Layout", node_vector);

    p->set<int>("Offset of First DOF", X_offset);

    // Can;t do mixture of dot and dotdot: disabling elasticity time dependence
    p->set<bool>("Disable Transient", true);

    evaluators_to_build["Gather Displacement Solution"] = p;
  }
  { // Gather Solution T
   RCP< vector<string> > dof_names = rcp(new vector<string>(1));
     (*dof_names)[0] = "Temperature";

    RCP<ParameterList> p = rcp(new ParameterList);
    int type = FactoryTraits<AlbanyTraits>::id_gather_solution;
    p->set<int>("Type", type);
    p->set< RCP< vector<string> > >("Solution Names", dof_names);
    p->set< RCP<DataLayout> >("Data Layout", node_scalar);

    p->set<int>("Offset of First DOF", T_offset);

    RCP< vector<string> > dof_names_dot = rcp(new vector<string>(1));
      (*dof_names_dot)[0] = "Temperature_dot";

    p->set< RCP< vector<string> > >("Time Dependent Solution Names", dof_names_dot);

    evaluators_to_build["Gather T Solution"] = p;
  }

  { // Gather Coordinate Vector
    RCP<ParameterList> p = rcp(new ParameterList("Elasticity Gather Coordinate Vector"));
    int type = FactoryTraits<AlbanyTraits>::id_gather_coordinate_vector;
    p->set<int>                ("Type", type);

    // Output:: Coordindate Vector at vertices
    p->set< RCP<DataLayout> >  ("Coordinate Data Layout",  vertices_vector);
    p->set< string >("Coordinate Vector Name", "Coord Vec");
    evaluators_to_build["Gather Coordinate Vector"] = p;
  }

  // UNUSED FOR NOW
  { // Map To Physical Frame: Interpolate X, Y to QuadPoints
    RCP<ParameterList> p = rcp(new ParameterList("Elasticity Map To Physical Frame"));

    int type = FactoryTraits<AlbanyTraits>::id_map_to_physical_frame;
    p->set<int>   ("Type", type);

    // Input: X, Y at vertices
    p->set< string >("Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Coordinate Data Layout", vertices_vector);

    p->set<RCP <Intrepid::Cubature<RealType> > >("Cubature", cubature);
    p->set<RCP<shards::CellTopology> >("Cell Type", cellType);

    // Output: X, Y at Quad Points (same name as input)
    p->set< RCP<DataLayout> >("QP Vector Data Layout", qp_vector);

    evaluators_to_build["Map To Physical Frame"] = p;
  }

  { // Compute Basis Functions
    RCP<ParameterList> p = rcp(new ParameterList("Elasticity Compute Basis Functions"));

    int type = FactoryTraits<AlbanyTraits>::id_compute_basis_functions;
    p->set<int>   ("Type", type);

    // Inputs: X, Y at nodes, Cubature, and Basis
    p->set<string>("Coordinate Vector Name","Coord Vec");
    p->set< RCP<DataLayout> >("Coordinate Data Layout", vertices_vector);
    p->set< RCP<Intrepid::Cubature<RealType> > >("Cubature", cubature);

    p->set< RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >
        ("Intrepid Basis", intrepidBasis);

    p->set<RCP<shards::CellTopology> >("Cell Type", cellType);

    // Outputs: BF, weightBF, Grad BF, weighted-Grad BF, all in physical space
    p->set<string>("Weights Name",          "Weights");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);
    p->set<string>("BF Name",          "BF");
    p->set<string>("Weighted BF Name", "wBF");
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", node_qp_scalar);

    p->set<string>("Gradient BF Name",          "Grad BF");
    p->set<string>("Weighted Gradient BF Name", "wGrad BF");
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", node_qp_vector);

    evaluators_to_build["Compute Basis Functions"] = p;
  }


  { // Elastic Modulus
    RCP<ParameterList> p = rcp(new ParameterList);

    int type = FactoryTraits<AlbanyTraits>::id_elastic_modulus;
    p->set<int>("Type", type);

    p->set<string>("QP Variable Name", "Elastic Modulus");
    p->set<string>("QP Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Node Data Layout", node_scalar);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", qp_vector);

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
    p->set< RCP<DataLayout> >("Node Data Layout", node_scalar);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Poissons Ratio");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    // Setting this turns on linear dependence of nu on T, nu = nu_ + dnudT*T)
    p->set<string>("QP Temperature Name", "Temperature");

    evaluators_to_build["Poissons Ratio"] = p;
  }

  { // DOFVec: Interpolate nodal Displacement values to quad points
    RCP<ParameterList> p = rcp(new ParameterList("Elasticity DOFVecInterpolation Displacement"));

    int type = FactoryTraits<AlbanyTraits>::id_dofvec_interpolation;
    p->set<int>   ("Type", type);

    // Input
    p->set<string>("Variable Name", "Displacement");
    p->set< RCP<DataLayout> >("Node Vector Data Layout",      node_vector);

    p->set<string>("BF Name", "BF");
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", node_qp_scalar);

    // Output (assumes same Name as input)
    p->set< RCP<DataLayout> >("QP Vector Data Layout", qp_vector);

    evaluators_to_build["DOFVec Displacement"] = p;
  }

  { // DOFVecGrad: Interpolate nodal Displacement gradients to quad points
    RCP<ParameterList> p = rcp(new ParameterList("Elasticity DOFVecInterpolation Displacement Grad"));

    int type = FactoryTraits<AlbanyTraits>::id_dofvec_grad_interpolation;
    p->set<int>   ("Type", type);
    // Input
    p->set<string>("Variable Name", "Displacement");
    p->set< RCP<DataLayout> >("Node Vector Data Layout",      node_vector);

    p->set<string>("Gradient BF Name", "Grad BF");
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", node_qp_vector);

    // Output
    p->set<string>("Gradient Variable Name", "Displacement Gradient");
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", qp_tensor);

    evaluators_to_build["DOFVec Grad Displacement"] = p;
  }

  if (haveSource) { // Source
    TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                       "Error!  Sources not implemented in Elasticity yet!");

    RCP<ParameterList> p = rcp(new ParameterList);

    int type = FactoryTraits<AlbanyTraits>::id_source;
    p->set<int>("Type", type);

    p->set<string>("Source Name", "Source");
    p->set<string>("Variable Name", "Displacement");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);

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
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", qp_tensor);

    //Output
    p->set<string>("Strain Name", "Strain"); //qp_tensor also

    evaluators_to_build["Strain"] = p;
  }

  { // Stress
    RCP<ParameterList> p = rcp(new ParameterList("Stress"));

    int type = FactoryTraits<AlbanyTraits>::id_stress;
    p->set<int>("Type", type);

    //Input
    p->set<string>("Strain Name", "Strain");
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", qp_tensor);

    p->set<string>("Elastic Modulus Name", "Elastic Modulus");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);

    p->set<string>("Poissons Ratio Name", "Poissons Ratio");  // qp_scalar also

    //Output
    p->set<string>("Stress Name", "Stress"); //qp_tensor also

    evaluators_to_build["Stress"] = p;
    evaluators_to_build["Save Stress"] =
      stateMgr.registerStateVariable("Stress",qp_tensor,
            dummy, FactoryTraits<AlbanyTraits>::id_savestatefield,"zero");
  }

  { // Displacement Resid
    RCP<ParameterList> p = rcp(new ParameterList("Displacement Resid"));

    int type = FactoryTraits<AlbanyTraits>::id_elasticityresid;
    p->set<int>("Type", type);

    //Input
    p->set<string>("Stress Name", "Stress");
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", qp_tensor);

    p->set<string>("Weighted Gradient BF Name", "wGrad BF");
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", node_qp_vector);

    p->set<bool>("Disable Transient", true);

    //Output
    p->set<string>("Residual Name", "Displacement Residual");
    p->set< RCP<DataLayout> >("Node Vector Data Layout", node_vector);

    evaluators_to_build["Elasticity Resid"] = p;
  }

  { // Scatter Residual
   RCP< vector<string> > resid_names = rcp(new vector<string>(1));
     (*resid_names)[0] = "Displacement Residual";

    RCP<ParameterList> p = rcp(new ParameterList);
    int type = FactoryTraits<AlbanyTraits>::id_scatter_residual;
    p->set<int>("Type", type);
    p->set<bool>("Vector Field", true);
    p->set< RCP< vector<string> > >("Residual Names", resid_names);

    p->set< RCP<DataLayout> >("Dummy Data Layout", dummy);
    p->set< RCP<DataLayout> >("Data Layout", node_vector);

    p->set<int>("Offset of First DOF", X_offset);

    evaluators_to_build["Scatter Displacement Residual"] = p;
  }
  { // Thermal conductivity
    RCP<ParameterList> p = rcp(new ParameterList);

    int type = FactoryTraits<AlbanyTraits>::id_thermal_conductivity;
    p->set<int>("Type", type);

    p->set<string>("QP Variable Name", "Thermal Conductivity");
    p->set<string>("QP Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Node Data Layout", node_scalar);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Thermal Conductivity");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    evaluators_to_build["Thermal Conductivity"] = p;
  }

  { // DOF: Interpolate nodal Temperature values to quad points
    RCP<ParameterList> p = rcp(new ParameterList("Heat2D DOFInterpolation Temperature"));

    int type = FactoryTraits<AlbanyTraits>::id_dof_interpolation;
    p->set<int>   ("Type", type);

    // Input
    p->set<string>("Variable Name", "Temperature");
    p->set< RCP<DataLayout> >("Node Data Layout",      node_scalar);

    p->set<string>("BF Name", "BF");
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", node_qp_scalar);

    // Output (assumes same Name as input)
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);

    evaluators_to_build["DOF Temperature"] = p;
  }

  {
   // DOF: Interpolate nodal Temperature Dot  values to quad points
    RCP<ParameterList> p = rcp(new ParameterList("Heat2D DOFInterpolation Temperature Dot"));

    int type = FactoryTraits<AlbanyTraits>::id_dof_interpolation;
    p->set<int>   ("Type", type);

    // Input
    p->set<string>("Variable Name", "Temperature_dot");
    p->set< RCP<DataLayout> >("Node Data Layout",      node_scalar);

    p->set<string>("BF Name", "BF");
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", node_qp_scalar);

    // Output (assumes same Name as input)
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);

    evaluators_to_build["DOF Temperature_dot"] = p;
}
  { // DOF: Interpolate nodal Temperature gradients to quad points
    RCP<ParameterList> p = rcp(new ParameterList("Heat2D DOFInterpolation Temperature Grad"));

    int type = FactoryTraits<AlbanyTraits>::id_dof_grad_interpolation;
    p->set<int>   ("Type", type);

    // Input
    p->set<string>("Variable Name", "Temperature");
    p->set< RCP<DataLayout> >("Node Data Layout",      node_scalar);

    p->set<string>("Gradient BF Name", "Grad BF");
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", node_qp_vector);

    // Output
    p->set<string>("Gradient Variable Name", "Temperature Gradient");
    p->set< RCP<DataLayout> >("QP Vector Data Layout", qp_vector);

    evaluators_to_build["DOF Grad Temperature"] = p;
  }

  if (haveSource) { // Source
    RCP<ParameterList> p = rcp(new ParameterList);

    int type = FactoryTraits<AlbanyTraits>::id_source;
    p->set<int>("Type", type);

    p->set<string>("Source Name", "Source");
    p->set<string>("Variable Name", "Temperature");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);

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
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", node_qp_scalar);
    p->set<string>("QP Variable Name", "Temperature");

    p->set<string>("QP Time Derivative Variable Name", "Temperature_dot");

    p->set<bool>("Have Source", haveSource);
    p->set<string>("Source Name", "Source");

    p->set<bool>("Have Absorption", false);

    p->set<string>("Thermal Conductivity Name", "Thermal Conductivity");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);

    p->set<string>("Gradient QP Variable Name", "Temperature Gradient");
    p->set< RCP<DataLayout> >("QP Vector Data Layout", qp_vector);

    p->set<string>("Weighted Gradient BF Name", "wGrad BF");
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", node_qp_vector);

    //Output
    p->set<string>("Residual Name", "Temperature Residual");
    p->set< RCP<DataLayout> >("Node Scalar Data Layout", node_scalar);

    evaluators_to_build["Heat2D Resid"] = p;
  }

  string fieldName="Scatter Heat";
  { // Scatter Residual
   RCP< vector<string> > resid_names = rcp(new vector<string>(1));
     (*resid_names)[0] = "Temperature Residual";

    RCP<ParameterList> p = rcp(new ParameterList);
    int type = FactoryTraits<AlbanyTraits>::id_scatter_residual;
    p->set<int>("Type", type);
    p->set< RCP< vector<string> > >("Residual Names", resid_names);

    p->set< RCP<DataLayout> >("Dummy Data Layout", dummy);
    p->set< RCP<DataLayout> >("Data Layout", node_scalar);

    p->set<int>("Offset of First DOF", T_offset);

    // Give this Scatter evaluator a different evaluatedField then the default
    p->set<string>("Scatter Field Name", fieldName);

    evaluators_to_build["Scatter Temperature Residual"] = p;
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

   PHX::Tag<AlbanyTraits::Residual::ScalarT> res_tag("Scatter", dummy);
   fm->requireField<AlbanyTraits::Residual>(res_tag);
   PHX::Tag<AlbanyTraits::Jacobian::ScalarT> jac_tag("Scatter", dummy);
   fm->requireField<AlbanyTraits::Jacobian>(jac_tag);
   PHX::Tag<AlbanyTraits::Tangent::ScalarT> tan_tag("Scatter", dummy);
   fm->requireField<AlbanyTraits::Tangent>(tan_tag);
   PHX::Tag<AlbanyTraits::SGResidual::ScalarT> sgres_tag("Scatter", dummy);
   fm->requireField<AlbanyTraits::SGResidual>(sgres_tag);
   PHX::Tag<AlbanyTraits::SGJacobian::ScalarT> sgjac_tag("Scatter", dummy);
   fm->requireField<AlbanyTraits::SGJacobian>(sgjac_tag);
   PHX::Tag<AlbanyTraits::MPResidual::ScalarT> mpres_tag("Scatter", dummy);
   fm->requireField<AlbanyTraits::MPResidual>(mpres_tag);
   PHX::Tag<AlbanyTraits::MPJacobian::ScalarT> mpjac_tag("Scatter", dummy);
   fm->requireField<AlbanyTraits::MPJacobian>(mpjac_tag);

   PHX::Tag<AlbanyTraits::Residual::ScalarT> res_tag2(fieldName, dummy);
   fm->requireField<AlbanyTraits::Residual>(res_tag2);
   PHX::Tag<AlbanyTraits::Jacobian::ScalarT> jac_tag2(fieldName, dummy);
   fm->requireField<AlbanyTraits::Jacobian>(jac_tag2);
   PHX::Tag<AlbanyTraits::Tangent::ScalarT> tan_tag2(fieldName, dummy);
   fm->requireField<AlbanyTraits::Tangent>(tan_tag2);
   PHX::Tag<AlbanyTraits::SGResidual::ScalarT> sgres_tag2(fieldName, dummy);
   fm->requireField<AlbanyTraits::SGResidual>(sgres_tag2);
   PHX::Tag<AlbanyTraits::SGJacobian::ScalarT> sgjac_tag2(fieldName, dummy);
   fm->requireField<AlbanyTraits::SGJacobian>(sgjac_tag2);
   PHX::Tag<AlbanyTraits::MPResidual::ScalarT> mpres_tag2(fieldName, dummy);
   fm->requireField<AlbanyTraits::MPResidual>(mpres_tag2);
   PHX::Tag<AlbanyTraits::MPJacobian::ScalarT> mpjac_tag2(fieldName, dummy);
   fm->requireField<AlbanyTraits::MPJacobian>(mpjac_tag2);

   const Albany::StateManager::RegisteredStates& reg = stateMgr.getRegisteredStates();
   Albany::StateManager::RegisteredStates::const_iterator st = reg.begin();
   while (st != reg.end()) {
     PHX::Tag<AlbanyTraits::Residual::ScalarT> res_out_tag(st->first, dummy);
     fm->requireField<AlbanyTraits::Residual>(res_out_tag);
     st++;
   }
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

