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


#include "Albany_NonlinearElasticityProblem.hpp"
#include "Albany_BoundaryFlux1DResponseFunction.hpp"
#include "Albany_SolutionAverageResponseFunction.hpp"
#include "Albany_SolutionTwoNormResponseFunction.hpp"
#include "Albany_InitialCondition.hpp"

#include "Intrepid_HGRAD_LINE_C1_FEM.hpp"
#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "PHAL_FactoryTraits.hpp"


Albany::NonlinearElasticityProblem::
NonlinearElasticityProblem(
                         const Teuchos::RCP<Teuchos::ParameterList>& params_,
                         const Teuchos::RCP<ParamLib>& paramLib_,
                         const int numDim_) :
  Albany::AbstractProblem(params_, paramLib_, numDim_),
  haveIC(false),
  haveSource(false)
{
 
  std::string& method = params->get("Name", "NonlinearElasticity ");
  *out << "Problem Name = " << method << std::endl;
  
  haveIC     =  params->isSublist("Initial Condition");
  haveSource =  params->isSublist("Source Functions");

  matModel = params->sublist("Material Model").get("Model Name","NeoHookean");

  dofNames.resize(neq);
  dofNames[0] = "X";
  if (neq>1) dofNames[1] = "Y";
  if (neq>2) dofNames[2] = "Z";
}

Albany::NonlinearElasticityProblem::
~NonlinearElasticityProblem()
{
}

void
Albany::NonlinearElasticityProblem::
buildProblem(
    const int worksetSize,
    Albany::StateManager& stateMgr,
    const Albany::AbstractDiscretization& disc,
    std::vector< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses,
    const Teuchos::RCP<Epetra_Vector>& u)
{
  /* Construct All Phalanx Evaluators */
  constructEvaluators(worksetSize, disc.getCubatureDegree(), stateMgr);
  constructDirichletEvaluators(disc.getNodeSetIDs());

  const Epetra_Map& dofMap = *(disc.getMap());
  int left_node = dofMap.MinAllGID();
  int right_node = dofMap.MaxAllGID();

  // Build response functions
  Teuchos::ParameterList& responseList = params->sublist("Response Functions");
  int num_responses = responseList.get("Number", 0);
  responses.resize(num_responses);
  for (int i=0; i<num_responses; i++) {
     std::ostringstream ss;
     ss << "Response " << i;
     std::string name = responseList.get(ss.str(), "??");

     if (name == "Boundary Flux 1D") {
       // Need real size, not 1.0
       double h =  1.0 / (dofMap.NumGlobalElements() - 1);
       responses[i] =
         Teuchos::rcp(new BoundaryFlux1DResponseFunction(left_node,
                                                         right_node,
                                                         0, 1, h,
                                                         dofMap));
     }

     else if (name == "Solution Average")
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

  // Build initial solution
  if (haveIC) 
    Albany::InitialCondition(u, 1, 1, params->sublist("Initial Condition"));
}


void
Albany::NonlinearElasticityProblem::constructEvaluators(
       const int worksetSize, const int cubDegree, Albany::StateManager& stateMgr)
{
   using Teuchos::RCP;
   using Teuchos::rcp;
   using Teuchos::ParameterList;
   using PHX::DataLayout;
   using PHX::MDALayout;
   using std::vector;
   using std::map;
   using PHAL::FactoryTraits;
   using PHAL::AlbanyTraits;

   RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > intrepidBasis;
   RCP<shards::CellTopology> cellType;
   switch (neq) {
     case 1:
       intrepidBasis = rcp(new Intrepid::Basis_HGRAD_LINE_C1_FEM<RealType, Intrepid::FieldContainer<RealType> >() );
       cellType = rcp(new shards::CellTopology (shards::getCellTopologyData< shards::Line<2> >()));
       break;
     case 2:
       intrepidBasis = rcp(new Intrepid::Basis_HGRAD_QUAD_C1_FEM<RealType, Intrepid::FieldContainer<RealType> >() );
       cellType = rcp(new shards::CellTopology (shards::getCellTopologyData< shards::Quadrilateral<4> >()));
       break;
     case 3:
       intrepidBasis = rcp(new Intrepid::Basis_HGRAD_HEX_C1_FEM<RealType, Intrepid::FieldContainer<RealType> >() );
       cellType = rcp(new shards::CellTopology (shards::getCellTopologyData< shards::Hexahedron<8> >()));
       break;
   }

   numNodes = intrepidBasis->getCardinality();

   Intrepid::DefaultCubatureFactory<RealType> cubFactory;
   RCP <Intrepid::Cubature<RealType> > cubature = cubFactory.create(*cellType, cubDegree);

   numDim = cubature->getDimension();
   numQPts = cubature->getNumPoints();
   numVertices = cellType->getVertexCount();

   *out << "Field Dimensions: Workset=" << worksetSize 
        << ", Vertices= " << numVertices
        << ", Nodes= " << numNodes
        << ", QuadPts= " << numQPts
        << ", Dim= " << numDim << endl;

   const bool transient = params->get("Transient", false);

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

  { // Gather Solution
   RCP< vector<string> > dof_names = rcp(new vector<string>(1));
     (*dof_names)[0] = "Displacement";

    RCP<ParameterList> p = rcp(new ParameterList);
    int type = FactoryTraits<AlbanyTraits>::id_gather_solution;
    p->set<int>("Type", type);
    p->set< RCP< vector<string> > >("Solution Names", dof_names);
    p->set<bool>("Vector Field", true);
    p->set< RCP<DataLayout> >("Data Layout", node_vector);
    p->set<bool>("Is Transient", transient);

   if (transient) {
     RCP< vector<string> > dof_names_dot = rcp(new vector<string>(1));
       (*dof_names_dot)[0] = "Displacement_dot";

     p->set< RCP< vector<string> > >("Time Dependent Solution Names", dof_names_dot);
   }

    evaluators_to_build["Gather Solution"] = p;
  }

  { // Gather Coordinate Vector
    RCP<ParameterList> p = rcp(new ParameterList("NonlinearElasticity Gather Coordinate Vector"));
    int type = FactoryTraits<AlbanyTraits>::id_gather_coordinate_vector;
    p->set<int>                ("Type", type);

    // Output:: Coordindate Vector at vertices
    p->set< RCP<DataLayout> >  ("Coordinate Data Layout",  vertices_vector);
    p->set< string >("Coordinate Vector Name", "Coord Vec");
    evaluators_to_build["Gather Coordinate Vector"] = p;
  }

  // UNUSED FOR NOW
  { // Map To Physical Frame: Interpolate X, Y to QuadPoints
    RCP<ParameterList> p = rcp(new ParameterList("NonlinearElasticity Map To Physical Frame"));

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
    RCP<ParameterList> p = rcp(new ParameterList("NonlinearElasticity Compute Basis Functions"));

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

    evaluators_to_build["Poissons Ratio"] = p;
  }

  { // DOFVec: Interpolate nodal Displacement values to quad points
    RCP<ParameterList> p = rcp(new ParameterList("NonlinearElasticity DOFVecInterpolation Displacement"));

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

  if (transient) {
   // DOF: Interpolate nodal Displacement Dot  values to quad points
    RCP<ParameterList> p = rcp(new ParameterList("NonlinearElasticity DOFVecInterpolation Displacement Dot"));

    int type = FactoryTraits<AlbanyTraits>::id_dofvec_interpolation;
    p->set<int>   ("Type", type);

    // Input
    p->set<string>("Variable Name", "Displacement_dot");
    p->set< RCP<DataLayout> >("Node Vector Data Layout",      node_vector);

    p->set<string>("BF Name", "BF");
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", node_qp_scalar);

    // Output (assumes same Name as input)
    p->set< RCP<DataLayout> >("QP Vector Data Layout", qp_vector);

    evaluators_to_build["DOFVec Displacement_dot"] = p;
  }

  { // DOFVecGrad: Interpolate nodal Displacement gradients to quad points
    RCP<ParameterList> p = rcp(new ParameterList("NonlinearElasticity DOFVecInterpolation Displacement Grad"));

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

  { // Deformation Gradient
    RCP<ParameterList> p = rcp(new ParameterList("DefGrad"));

    int type = FactoryTraits<AlbanyTraits>::id_defgrad;
    p->set<int>("Type", type);

    //Input
    p->set<string>("Gradient QP Variable Name", "Displacement Gradient");
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", qp_tensor);

    //Output
    p->set<string>("DefGrad Name", "Deformation Gradient"); //qp_tensor also

    evaluators_to_build["DefGrad"] = p;
  }

  { // Determinant of Deformation Gradient
    RCP<ParameterList> p = rcp(new ParameterList("Determinant of Deformation Gradient"));

    int type = FactoryTraits<AlbanyTraits>::id_detdefgrad;
    p->set<int>("Type", type);

    //Input
    p->set<string>("DefGrad Name", "Deformation Gradient");
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", qp_tensor);

    //Output
    p->set<string>("DetDefGrad Name", "Determinant of Deformation Gradient"); 
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);

    evaluators_to_build["DetDefGrad"] = p;
  }

  { // LCG
    RCP<ParameterList> p = rcp(new ParameterList("LCG"));

    int type = FactoryTraits<AlbanyTraits>::id_lcg;
    p->set<int>("Type", type);

    //Input
    p->set<string>("DefGrad Name", "Deformation Gradient");
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", qp_tensor);

    //Output
    p->set<string>("LCG Name", "LCG"); //qp_tensor also

    evaluators_to_build["LCG"] = p;
  }

  
 if (matModel == "NeoHookean")
  { // Stress
    RCP<ParameterList> p = rcp(new ParameterList("Stress"));

    int type = FactoryTraits<AlbanyTraits>::id_neohookean_stress;
    p->set<int>("Type", type);

    //Input
    p->set<string>("LCG Name", "LCG");
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", qp_tensor);

    p->set<string>("Elastic Modulus Name", "Elastic Modulus");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);

    p->set<string>("Poissons Ratio Name", "Poissons Ratio");  // qp_scalar also
    p->set<string>("DetDefGrad Name", "Determinant of Deformation Gradient");  // qp_scalar also

    //Output
    p->set<string>("Stress Name", "Stress"); //qp_tensor also

    evaluators_to_build["Stress"] = p;
  }
  else if (matModel == "J2")
  { // Stress
    RCP<ParameterList> p = rcp(new ParameterList("Stress"));

    int type = FactoryTraits<AlbanyTraits>::id_j2_stress;
    p->set<int>("Type", type);

    //Input
    p->set<string>("LCG Name", "LCG");
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", qp_tensor);

    p->set<string>("Elastic Modulus Name", "Elastic Modulus");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);

    p->set<string>("Poissons Ratio Name", "Poissons Ratio");  // qp_scalar also
    p->set<string>("DetDefGrad Name", "Determinant of Deformation Gradient");  // qp_scalar also

    //Output
    p->set<string>("Stress Name", "Stress"); //qp_tensor also
 
    //Declare what state data will need to be saved (name, layout)
    stateMgr.registerStateVariable("J2StressState",qp_tensor);
    stateMgr.registerStateVariable("junkTest",qp_scalar);

    evaluators_to_build["Stress"] = p;
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
    p->set<string>("Stress Name", "Stress");
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", qp_tensor);

    p->set<string>("DefGrad Name", "Deformation Gradient"); //qp_tensor also

    p->set<string>("DetDefGrad Name", "Determinant of Deformation Gradient");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);

/*
    p->set<bool>("Is Transient", transient);
    p->set<string>("QP Time Derivative Variable Name", "Displacement_dot");
*/

    p->set<string>("Weighted Gradient BF Name", "wGrad BF");
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", node_qp_vector);

    //Output
    p->set<string>("Residual Name", "Residual");
    p->set< RCP<DataLayout> >("Node Vector Data Layout", node_vector);

    evaluators_to_build["Residual"] = p;
  }

  { // Scatter Residual
   RCP< vector<string> > resid_names = rcp(new vector<string>(1));
     (*resid_names)[0] = "Residual";

    RCP<ParameterList> p = rcp(new ParameterList);
    int type = FactoryTraits<AlbanyTraits>::id_scatter_residual;
    p->set<int>("Type", type);
    p->set<bool>("Vector Field", true);
    p->set< RCP< vector<string> > >("Residual Names", resid_names);

    p->set< RCP<DataLayout> >("Dummy Data Layout", dummy);
    p->set< RCP<DataLayout> >("Data Layout", node_vector);

    evaluators_to_build["Scatter Residual"] = p;
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
}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::NonlinearElasticityProblem::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidNonlinearElasticityProblemParams");

  validPL->sublist("Elastic Modulus", false, "");
  validPL->sublist("Poissons Ratio", false, "");
  validPL->sublist("Material Model", false, "");

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

