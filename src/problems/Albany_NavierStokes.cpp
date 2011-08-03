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


#include "Albany_NavierStokes.hpp"
#include "Albany_SolutionAverageResponseFunction.hpp"
#include "Albany_SolutionTwoNormResponseFunction.hpp"
#include "Albany_SolutionMaxValueResponseFunction.hpp"
#include "Albany_InitialCondition.hpp"

#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "PHAL_FactoryTraits.hpp"
#include "Albany_Utils.hpp"


Albany::NavierStokes::
NavierStokes( const Teuchos::RCP<Teuchos::ParameterList>& params_,
             const Teuchos::RCP<ParamLib>& paramLib_,
             const int numDim_) :
  Albany::AbstractProblem(params_, paramLib_),
  haveSource(false),
  numDim(numDim_)
{
  if (numDim==1) periodic = params->get("Periodic BC", false);
  else           periodic = false;
  if (periodic) *out <<" Periodic Boundary Conditions being used." <<std::endl;

  haveFlow = params->get("Have Flow Equations", true);
  haveHeat = params->get("Have Heat Equation", true);
  haveNeut = params->get("Have Neutron Equation", true);
  havePSPG = params->get("Have Pressure Stabilization", true);
  haveSUPG = params->get("Have SUPG Stabilization", true);
  haveSource =  params->isSublist("Source Functions");
  haveNeutSource =  params->isSublist("Neutron Source Functions");

  // Compute number of equations
  int num_eq = 0;
  if (haveFlow) num_eq += numDim+1;
  if (haveHeat) num_eq += 1;
  if (haveNeut) num_eq += 1;
  this->setNumEquations(num_eq);

  // Setup DOF names
  dofNames.resize(num_eq);
  int index = 0;
  if (haveFlow) {
    dofNames[index++] = "ux";
    if (numDim>=2)
      dofNames[index++] = "uy";
    if (numDim==3)
      dofNames[index++] = "uz";
    dofNames[index++] = "p";
  }
  if (haveHeat)
    dofNames[index++] = "T";
  if (haveNeut)
    dofNames[index++] = "phi";
}

Albany::NavierStokes::
~NavierStokes()
{
}

void
Albany::NavierStokes::
buildProblem(
    const Albany::MeshSpecsStruct& meshSpecs,
    Albany::StateManager& stateMgr,
    std::vector< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses)
{
 /* Construct All Phalanx Evaluators */
  constructEvaluators(meshSpecs);
  constructDirichletEvaluators(meshSpecs.nsNames);
 
  // Build response functions
  Teuchos::ParameterList& responseList = params->sublist("Response Functions");
  int num_responses = responseList.get("Number", 0);
  int eq = responseList.get("Equation", 0);
  bool inor =  meshSpecs.interleavedOrdering;
  responses.resize(num_responses);
  for (int i=0; i<num_responses; i++) {
     std::string name = responseList.get(Albany::strint("Response",i), "??");

     if (name == "Solution Average")
       responses[i] = Teuchos::rcp(new SolutionAverageResponseFunction());

     else if (name == "Solution Two Norm")
       responses[i] = Teuchos::rcp(new SolutionTwoNormResponseFunction());

     else if (name == "Solution Max Value")
       responses[i] = Teuchos::rcp(new SolutionMaxValueResponseFunction(neq, eq, inor));

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
Albany::NavierStokes::constructEvaluators(const Albany::MeshSpecsStruct& meshSpecs)
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
  
  RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > >
    intrepidBasis = this->getIntrepidBasis(meshSpecs.ctd);
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

  if (haveFlow) { // Gather Solution Velocity, x, y, z
    RCP< vector<string> > dof_names = rcp(new vector<string>(1));
    (*dof_names)[0] = "Velocity";
     
    RCP<ParameterList> p = rcp(new ParameterList);
    int type = FactoryTraits<AlbanyTraits>::id_gather_solution;
    p->set<int>("Type", type);
    p->set<bool>("Vector Field", true);
    p->set< RCP< vector<string> > >("Solution Names", dof_names);
    p->set< RCP<DataLayout> >("Data Layout", node_vector);
     
    RCP< vector<string> > dof_names_dot = rcp(new vector<string>(1));
    (*dof_names_dot)[0] = "Velocity_dot";
     
    p->set< RCP< vector<string> > >("Time Dependent Solution Names", 
				    dof_names_dot);
     
    p->set<int>("Offset of First DOF", 0);

    evaluators_to_build["Gather Velocity Solution"] = p;
  }

  if (haveFlow) { // Gather Solution Pressure
    RCP< vector<string> > dof_names = rcp(new vector<string>(1));
    (*dof_names)[0] = "Pressure";
     
    RCP<ParameterList> p = rcp(new ParameterList);
    int type = FactoryTraits<AlbanyTraits>::id_gather_solution;
    p->set<int>("Type", type);
    p->set< RCP< vector<string> > >("Solution Names", dof_names);
    p->set< RCP<DataLayout> >("Data Layout", node_scalar);
     
    RCP< vector<string> > dof_names_dot = rcp(new vector<string>(1));
    (*dof_names_dot)[0] = "Pressure_dot";
     
    p->set< RCP< vector<string> > >("Time Dependent Solution Names", 
				    dof_names_dot);
     
    p->set<int>("Offset of First DOF", numDim);
     
    evaluators_to_build["Gather Pressure Solution"] = p;
  }

  if (haveHeat) { // Gather Solution Temperature
    RCP< vector<string> > dof_names = rcp(new vector<string>(1));
    (*dof_names)[0] = "Temperature";
     
    RCP<ParameterList> p = rcp(new ParameterList);
    int type = FactoryTraits<AlbanyTraits>::id_gather_solution;
    p->set<int>("Type", type);
    p->set< RCP< vector<string> > >("Solution Names", dof_names);
    p->set< RCP<DataLayout> >("Data Layout", node_scalar);
     
    RCP< vector<string> > dof_names_dot = rcp(new vector<string>(1));
    (*dof_names_dot)[0] = "Temperature_dot";
     
    p->set< RCP< vector<string> > >("Time Dependent Solution Names", 
				    dof_names_dot);
     
    if (haveFlow)
      p->set<int>("Offset of First DOF", numDim+1);
    else
      p->set<int>("Offset of First DOF", 0);
     
    evaluators_to_build["Gather Temperature Solution"] = p;
  }

  if (haveNeut) { // Gather neutron solution
    RCP< vector<string> > dof_names = rcp(new vector<string>(1));
    (*dof_names)[0] = "Neutron";
     
    RCP<ParameterList> p = rcp(new ParameterList);
    int type = FactoryTraits<AlbanyTraits>::id_gather_solution;
    p->set<int>("Type", type);
    p->set< RCP< vector<string> > >("Solution Names", dof_names);
    p->set< RCP<DataLayout> >("Data Layout", node_scalar);
     
    RCP< vector<string> > dof_names_dot = rcp(new vector<string>(1));
    (*dof_names_dot)[0] = "Neutron_dot";
     
    p->set< RCP< vector<string> > >("Time Dependent Solution Names", 
				    dof_names_dot);
     
    if (haveFlow) {
      if (haveHeat)
         p->set<int>("Offset of First DOF", numDim+2);
      else
         p->set<int>("Offset of First DOF", numDim+1);
    }
    else {
      if (haveHeat)
         p->set<int>("Offset of First DOF", 1);
      else
         p->set<int>("Offset of First DOF", 0);
    }
    p->set<int>("Number of DOF per Node", neq);
     
    evaluators_to_build["Gather Neutron Solution"] = p;
  }

  { // Gather Coordinate Vector
    RCP<ParameterList> p = 
      rcp(new ParameterList("Navier-Stokes Gather Coordinate Vector"));
    int type = FactoryTraits<AlbanyTraits>::id_gather_coordinate_vector;
    p->set<int> ("Type", type);

    // Input: Periodic BC flag
    p->set<bool>("Periodic BC", false);

    // Output:: Coordindate Vector at vertices
    p->set< RCP<DataLayout> >  ("Coordinate Data Layout",  vertices_vector);
    p->set< string >("Coordinate Vector Name", "Coord Vec");
    evaluators_to_build["Gather Coordinate Vector"] = p;
  }

  { // Map To Physical Frame: Interpolate X, Y to QuadPoints
    RCP<ParameterList> p = 
      rcp(new ParameterList("Navier-Stokes Map To Physical Frame"));

    int type = FactoryTraits<AlbanyTraits>::id_map_to_physical_frame;
    p->set<int>("Type", type);

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
    RCP<ParameterList> p = 
      rcp(new ParameterList("Navier-Stokes Compute Basis Functions"));

    int type = FactoryTraits<AlbanyTraits>::id_compute_basis_functions;
    p->set<int>("Type", type);

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

  if (havePSPG || haveSUPG) { // Compute Contravarient Metric Tensor
    RCP<ParameterList> p = 
      rcp(new ParameterList("Contravarient Metric Tensor"));

    int type = FactoryTraits<AlbanyTraits>::id_nsgctensor;
    p->set<int>("Type", type);

    // Inputs: X, Y at nodes, Cubature, and Basis
    p->set<string>("Coordinate Vector Name","Coord Vec");
    p->set< RCP<DataLayout> >("Coordinate Data Layout", vertices_vector);
    p->set< RCP<Intrepid::Cubature<RealType> > >("Cubature", cubature);

    p->set<RCP<shards::CellTopology> >("Cell Type", cellType);

    // Outputs: BF, weightBF, Grad BF, weighted-Grad BF, all in physical space
    p->set<string>("Contravarient Metric Tensor Name", "Gc");
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", qp_tensor);

    evaluators_to_build["Contravarient Metric Tensor"] = p;
  }

  { // Density
    RCP<ParameterList> p = rcp(new ParameterList);

    int type = FactoryTraits<AlbanyTraits>::id_nsmatprop;
    p->set<int>("Type", type);

    p->set<string>("Material Property Name", "Density");
    p->set<string>("QP Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Node Data Layout", node_scalar);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Density");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    evaluators_to_build["Density"] = p;
  }

  if (haveFlow) { // Viscosity
    RCP<ParameterList> p = rcp(new ParameterList);

    int type = FactoryTraits<AlbanyTraits>::id_nsmatprop;
    p->set<int>("Type", type);

    p->set<string>("Material Property Name", "Viscosity");
    p->set<string>("QP Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Node Data Layout", node_scalar);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Viscosity");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    evaluators_to_build["Viscosity"] = p;
  }

  if (haveHeat) { // Specific Heat
    RCP<ParameterList> p = rcp(new ParameterList);

    int type = FactoryTraits<AlbanyTraits>::id_nsmatprop;
    p->set<int>("Type", type);

    p->set<string>("Material Property Name", "Specific Heat");
    p->set<string>("QP Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Node Data Layout", node_scalar);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Specific Heat");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    evaluators_to_build["Specific Heat"] = p;
  }

  if (haveHeat) { // Thermal conductivity
    RCP<ParameterList> p = rcp(new ParameterList);

    int type = FactoryTraits<AlbanyTraits>::id_nsmatprop;
    p->set<int>("Type", type);

    p->set<string>("Material Property Name", "Thermal Conductivity");
    p->set<string>("QP Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Node Data Layout", node_scalar);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Thermal Conductivity");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    evaluators_to_build["Thermal Conductivity"] = p;
  }

  if (haveNeut) { // Neutron diffusion
    RCP<ParameterList> p = rcp(new ParameterList);

    int type = FactoryTraits<AlbanyTraits>::id_nsmatprop;
    p->set<int>("Type", type);

    p->set<string>("Material Property Name", "Neutron Diffusion");
    p->set<string>("QP Coordinate Vector Name", "Coord Vec");
    p->set<string>("Temperature QP Variable Name", "Temperature");
    p->set< RCP<DataLayout> >("Node Data Layout", node_scalar);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Neutron Diffusion");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    evaluators_to_build["Neutron Diffusion"] = p;
  }

  if (haveNeut) { // Reference temperature
    RCP<ParameterList> p = rcp(new ParameterList);

    int type = FactoryTraits<AlbanyTraits>::id_nsmatprop;
    p->set<int>("Type", type);

    p->set<string>("Material Property Name", "Reference Temperature");
    p->set<string>("QP Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Node Data Layout", node_scalar);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Reference Temperature");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    evaluators_to_build["Reference Temperature"] = p;
  }

  if (haveNeut) { // Neutron absorption cross section
    RCP<ParameterList> p = rcp(new ParameterList);

    int type = FactoryTraits<AlbanyTraits>::id_nsmatprop;
    p->set<int>("Type", type);

    p->set<string>("Material Property Name", "Neutron Absorption");
    p->set<string>("QP Coordinate Vector Name", "Coord Vec");
    p->set<string>("Temperature QP Variable Name", "Temperature");
    p->set< RCP<DataLayout> >("Node Data Layout", node_scalar);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Neutron Absorption");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    evaluators_to_build["Neutron Absorption"] = p;
  }

  if (haveNeut) { // Neutron fission cross section
    RCP<ParameterList> p = rcp(new ParameterList);

    int type = FactoryTraits<AlbanyTraits>::id_nsmatprop;
    p->set<int>("Type", type);

    p->set<string>("Material Property Name", "Neutron Fission");
    p->set<string>("QP Coordinate Vector Name", "Coord Vec");
    p->set<string>("Temperature QP Variable Name", "Temperature");
    p->set< RCP<DataLayout> >("Node Data Layout", node_scalar);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Neutron Fission");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    evaluators_to_build["Neutron Fission"] = p;
  }

  if (haveNeut && haveHeat) { // Proportionality constant
    RCP<ParameterList> p = rcp(new ParameterList);

    int type = FactoryTraits<AlbanyTraits>::id_nsmatprop;
    p->set<int>("Type", type);

    p->set<string>("Material Property Name", "Proportionality Constant");
    p->set<string>("QP Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Node Data Layout", node_scalar);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Proportionality Constant");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    evaluators_to_build["Proportionality Constant"] = p;
  }

  if (haveFlow && haveHeat) { // Volumetric Expansion Coefficient
    RCP<ParameterList> p = rcp(new ParameterList);

    int type = FactoryTraits<AlbanyTraits>::id_nsmatprop;
    p->set<int>("Type", type);

    p->set<string>("Material Property Name", 
		   "Volumetric Expansion Coefficient");
    p->set<string>("QP Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Node Data Layout", node_scalar);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = 
      params->sublist("Volumetric Expansion Coefficient");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    evaluators_to_build["Volumetric Expansion Coefficient"] = p;
  }

  if (haveHeat) { // DOF: Interpolate nodal Temperature values to quad points
    RCP<ParameterList> p = 
      rcp(new ParameterList("Navier-Stokes DOFInterpolation Temperature"));

    int type = FactoryTraits<AlbanyTraits>::id_dof_interpolation;
    p->set<int>("Type", type);

    // Input
    p->set<string>("Variable Name", "Temperature");
    p->set<string>("BF Name", "BF");
    p->set< RCP<DataLayout> >("Node Data Layout", node_scalar);
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", node_qp_scalar);

    // Output (assumes same Name as input)
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);

    evaluators_to_build["DOF Temperature"] = p;
  }

  if (haveHeat) {
    // DOF: Interpolate nodal Temperature Dot  values to quad points
    RCP<ParameterList> p = 
      rcp(new ParameterList("Navier-Stokes DOFInterpolation Temperature Dot"));

    int type = FactoryTraits<AlbanyTraits>::id_dof_interpolation;
    p->set<int>("Type", type);

    // Input
    p->set<string>("Variable Name", "Temperature_dot");
    p->set<string>("BF Name", "BF");
    p->set< RCP<DataLayout> >("Node Data Layout", node_scalar);
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", node_qp_scalar);

    // Output (assumes same Name as input)
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);

    evaluators_to_build["DOF Temperature_dot"] = p;
  }

  if (haveHeat) { 
    // DOF: Interpolate nodal Temperature gradients to quad points
    RCP<ParameterList> p = 
      rcp(new ParameterList("Navier-Stokes DOFInterpolation Temperature Grad"));

    int type = FactoryTraits<AlbanyTraits>::id_dof_grad_interpolation;
    p->set<int>("Type", type);

    // Input
    p->set<string>("Variable Name", "Temperature");
    p->set<string>("Gradient BF Name", "Grad BF");
    p->set< RCP<DataLayout> >("Node Data Layout", node_scalar);
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", node_qp_vector);

    // Output
    p->set<string>("Gradient Variable Name", "Temperature Gradient");
    p->set< RCP<DataLayout> >("QP Vector Data Layout", qp_vector);

    evaluators_to_build["DOF Grad Temperature"] = p;
  }

  if (haveNeut) { // DOF: Interpolate nodal neutron flux values to quad points
    RCP<ParameterList> p = 
      rcp(new ParameterList("Navier-Stokes DOFInterpolation Neutron"));

    int type = FactoryTraits<AlbanyTraits>::id_dof_interpolation;
    p->set<int>("Type", type);

    // Input
    p->set<string>("Variable Name", "Neutron");
    p->set<string>("BF Name", "BF");
    p->set< RCP<DataLayout> >("Node Data Layout", node_scalar);
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", node_qp_scalar);

    // Output (assumes same Name as input)
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);

    evaluators_to_build["DOF Neutron"] = p;
  }

  if (haveNeut) { 
    // DOF: Interpolate nodal neutron flux gradients to quad points
    RCP<ParameterList> p = 
      rcp(new ParameterList("Navier-Stokes DOFInterpolation Neutron Grad"));

    int type = FactoryTraits<AlbanyTraits>::id_dof_grad_interpolation;
    p->set<int>("Type", type);

    // Input
    p->set<string>("Variable Name", "Neutron");
    p->set<string>("Gradient BF Name", "Grad BF");
    p->set< RCP<DataLayout> >("Node Data Layout", node_scalar);
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", node_qp_vector);

    // Output
    p->set<string>("Gradient Variable Name", "Neutron Gradient");
    p->set< RCP<DataLayout> >("QP Vector Data Layout", qp_vector);

    evaluators_to_build["DOF Grad Neutron"] = p;
  }

  if (haveFlow) { 
    // DOF: Interpolate nodal Pressure values to quad points
    RCP<ParameterList> p = 
      rcp(new ParameterList("Navier-Stokes DOFInterpolation Pressure"));

    int type = FactoryTraits<AlbanyTraits>::id_dof_interpolation;
    p->set<int>("Type", type);

    // Input
    p->set<string>("Variable Name", "Pressure");
    p->set<string>("BF Name", "BF");
    p->set< RCP<DataLayout> >("Node Data Layout", node_scalar);
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", node_qp_scalar);

    // Output (assumes same Name as input)
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);

    evaluators_to_build["DOF Pressure"] = p;
  }

  if (haveFlow) {
    // DOF: Interpolate nodal Pressure Dot  values to quad points
    RCP<ParameterList> p = 
      rcp(new ParameterList("Navier-Stokes DOFInterpolation Pressure Dot"));

    int type = FactoryTraits<AlbanyTraits>::id_dof_interpolation;
    p->set<int>("Type", type);

    // Input
    p->set<string>("Variable Name", "Pressure_dot");
    p->set<string>("BF Name", "BF");
    p->set< RCP<DataLayout> >("Node Data Layout", node_scalar);
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", node_qp_scalar);

    // Output (assumes same Name as input)
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);

    evaluators_to_build["DOF Pressure_dot"] = p;
  }

  if (haveFlow) { // DOF: Interpolate nodal Pressure gradients to quad points
    RCP<ParameterList> p = 
      rcp(new ParameterList("Navier-Stokes DOFInterpolation Pressure Grad"));

    int type = FactoryTraits<AlbanyTraits>::id_dof_grad_interpolation;
    p->set<int>("Type", type);

    // Input
    p->set<string>("Variable Name", "Pressure");
    p->set<string>("Gradient BF Name", "Grad BF");
    p->set< RCP<DataLayout> >("Node Data Layout", node_scalar);
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", node_qp_vector);

    // Output
    p->set<string>("Gradient Variable Name", "Pressure Gradient");
    p->set< RCP<DataLayout> >("QP Vector Data Layout", qp_vector);

    evaluators_to_build["DOF Grad Pressure"] = p;
  }

  if (haveFlow) { // DOF: Interpolate nodal Velocity values to quad points
    RCP<ParameterList> p = 
      rcp(new ParameterList("Navier-Stokes DOFInterpolation Velocity"));

    int type = FactoryTraits<AlbanyTraits>::id_dofvec_interpolation;
    p->set<int>("Type", type);

    // Input
    p->set<string>("Variable Name", "Velocity");
    p->set<string>("BF Name", "BF");
    p->set< RCP<DataLayout> >("Node Vector Data Layout", node_vector);
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", node_qp_scalar);

    // Output (assumes same Name as input)
    p->set< RCP<DataLayout> >("QP Vector Data Layout", qp_vector);

    evaluators_to_build["DOF Velocity"] = p;
  }

  if (haveFlow) {
    // DOF: Interpolate nodal Velocity Dot  values to quad points
    RCP<ParameterList> p = 
      rcp(new ParameterList("Navier-Stokes DOFInterpolation Velocity Dot"));

    int type = FactoryTraits<AlbanyTraits>::id_dofvec_interpolation;
    p->set<int>("Type", type);

    // Input
    p->set<string>("Variable Name", "Velocity_dot");
    p->set<string>("BF Name", "BF");
    p->set< RCP<DataLayout> >("Node Vector Data Layout", node_vector);
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", node_qp_scalar);

    // Output (assumes same Name as input)
    p->set< RCP<DataLayout> >("QP Vector Data Layout", qp_vector);

    evaluators_to_build["DOF Velocity_dot"] = p;
  }

  if (haveFlow) { // DOF: Interpolate nodal Velocity gradients to quad points
    RCP<ParameterList> p = 
      rcp(new ParameterList("Navier-Stokes DOFInterpolation Velocity Grad"));

    int type = FactoryTraits<AlbanyTraits>::id_dofvec_grad_interpolation;
    p->set<int>("Type", type);

    // Input
    p->set<string>("Variable Name", "Velocity");
    p->set<string>("Gradient BF Name", "Grad BF");
    p->set< RCP<DataLayout> >("Node Vector Data Layout", node_vector);
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", node_qp_vector);

    // Output
    p->set<string>("Gradient Variable Name", "Velocity Gradient");
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", qp_tensor);

    evaluators_to_build["DOF Grad Velocity"] = p;
  }

   
  if (haveHeat && haveSource) { // Source
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

  if (haveNeut && haveNeutSource) { // Source
    RCP<ParameterList> p = rcp(new ParameterList);

    int type = FactoryTraits<AlbanyTraits>::id_source;
    p->set<int>("Type", type);

    p->set<string>("Source Name", "Source");
    p->set<string>("Variable Name", "Neutron");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Source Functions");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    evaluators_to_build["Neutron Source"] = p;
  }

  if (haveFlow) { // Body Force
    RCP<ParameterList> p = rcp(new ParameterList("Body Force"));

    int type = FactoryTraits<AlbanyTraits>::id_nsbodyforce;
    p->set<int>("Type", type);

    //Input
    p->set<bool>("Have Heat", haveHeat);
    p->set<string>("Temperature QP Variable Name", "Temperature");
    p->set<string>("Density QP Variable Name", "Density");
    p->set<string>("Volumetric Expansion Coefficient QP Variable Name", "Volumetric Expansion Coefficient");

    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", qp_vector); 

    Teuchos::ParameterList& paramList = params->sublist("Body Force");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);
  
    //Output
    p->set<string>("Body Force Name", "Body Force");

    evaluators_to_build["Body Force"] = p;
  } 

  if (haveFlow) { // Rm
    RCP<ParameterList> p = rcp(new ParameterList("Rm"));

    int type = FactoryTraits<AlbanyTraits>::id_nsrm;
    p->set<int>("Type", type);

    //Input
    p->set<string>("Velocity QP Variable Name", "Velocity");
    p->set<string>("Velocity Dot QP Variable Name", "Velocity_dot");
    p->set<string>("Velocity Gradient QP Variable Name", "Velocity Gradient");
    p->set<string>("Pressure Gradient QP Variable Name", "Pressure Gradient");
    p->set<string>("Density QP Variable Name", "Density");
    p->set<string>("Body Force QP Variable Name", "Body Force");

    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", qp_vector);
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", qp_tensor);
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", node_qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
  
    //Output
    p->set<string>("Rm Name", "Rm");

    evaluators_to_build["Rm"] = p;
  }

  if (haveFlow && (haveSUPG || havePSPG)) { // Tau M
    RCP<ParameterList> p = rcp(new ParameterList("Tau M"));

    int type = FactoryTraits<AlbanyTraits>::id_nstaum;
    p->set<int>("Type", type);

    //Input
    p->set<string>("Velocity QP Variable Name", "Velocity");
    p->set<std::string>("Contravarient Metric Tensor Name", "Gc"); 
    p->set<string>("Density QP Variable Name", "Density");
    p->set<string>("Viscosity QP Variable Name", "Viscosity");

    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", qp_vector);
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", qp_tensor);

    //Output
    p->set<string>("Tau M Name", "Tau M");

    evaluators_to_build["Tau M"] = p;
  }

  if (haveHeat && haveFlow && haveSUPG) { // Tau T
    RCP<ParameterList> p = rcp(new ParameterList("Tau T"));

    int type = FactoryTraits<AlbanyTraits>::id_nstaut;
    p->set<int>("Type", type);

    //Input
    p->set<string>("Velocity QP Variable Name", "Velocity");
    p->set<std::string>("Contravarient Metric Tensor Name", "Gc"); 
    p->set<string>("Thermal Conductivity Name", "Thermal Conductivity");
    p->set<string>("Density QP Variable Name", "Density");
    p->set<string>("Specific Heat QP Variable Name", "Specific Heat");

    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", qp_vector);
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", qp_tensor);

    //Output
    p->set<string>("Tau T Name", "Tau T");

    evaluators_to_build["Tau T"] = p;
  }

  if (haveFlow) { // Momentum Resid
    RCP<ParameterList> p = rcp(new ParameterList("Momentum Resid"));

    int type = FactoryTraits<AlbanyTraits>::id_nsmomentumeqresid;
    p->set<int>("Type", type);

    //Input
    p->set<string>("Weighted BF Name", "wBF");
    p->set<string>("Weighted Gradient BF Name", "wGrad BF");
    p->set<string>("Velocity Gradient QP Variable Name", "Velocity Gradient");
    p->set<string>("Pressure QP Variable Name", "Pressure");
    p->set<string>("Pressure Gradient QP Variable Name", "Pressure Gradient");
    p->set<string>("Viscosity QP Variable Name", "Viscosity");
    p->set<string>("Rm Name", "Rm");

    p->set<bool>("Have SUPG", haveSUPG);
    p->set<string>("Velocity QP Variable Name", "Velocity");
    p->set<string>("Density QP Variable Name", "Density");
    p->set<string> ("Tau M Name", "Tau M");
 
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", qp_vector);    
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", qp_tensor);
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", node_qp_scalar);
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", node_qp_vector);
    
    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
  
    //Output
    p->set<string>("Residual Name", "Momentum Residual");
    p->set< RCP<DataLayout> >("Node Vector Data Layout", node_vector);

    evaluators_to_build["Momentum Resid"] = p;
  }

  if (haveFlow) { // Continuity Resid
    RCP<ParameterList> p = rcp(new ParameterList("Continuity Resid"));

    int type = FactoryTraits<AlbanyTraits>::id_nscontinuityeqresid;
    p->set<int>("Type", type);

    //Input
    p->set<string>("Weighted BF Name", "wBF");
    p->set<string>("Gradient QP Variable Name", "Velocity Gradient");
    p->set<string>("Density QP Variable Name", "Density");

    p->set<bool>("Have PSPG", havePSPG);
    p->set<string>("Weighted Gradient BF Name", "wGrad BF");
    p->set<std::string> ("Tau M Name", "Tau M");
    p->set<std::string> ("Rm Name", "Rm");

    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", node_qp_scalar);  
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", node_qp_vector);
    p->set< RCP<PHX::DataLayout> >("QP Scalar Data Layout", qp_scalar);
    p->set< RCP<PHX::DataLayout> >("QP Vector Data Layout", qp_vector);
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", qp_tensor);

    //Output
    p->set<string>("Residual Name", "Continuity Residual");
    p->set< RCP<DataLayout> >("Node Scalar Data Layout", node_scalar);

    evaluators_to_build["Continuity Resid"] = p;
  }

   if (haveHeat) { // Temperature Resid
    RCP<ParameterList> p = rcp(new ParameterList("Temperature Resid"));

    int type = FactoryTraits<AlbanyTraits>::id_nsthermaleqresid;
    p->set<int>("Type", type);

    //Input
    p->set<string>("Weighted BF Name", "wBF");
    p->set<string>("Weighted Gradient BF Name", "wGrad BF");
    p->set<string>("QP Variable Name", "Temperature");
    p->set<string>("QP Time Derivative Variable Name", "Temperature_dot");
    p->set<string>("Gradient QP Variable Name", "Temperature Gradient");
    p->set<string>("Density QP Variable Name", "Density");
    p->set<string>("Specific Heat QP Variable Name", "Specific Heat");
    p->set<string>("Thermal Conductivity Name", "Thermal Conductivity");
    p->set<string>("Proportionality Constant Name", "Proportionality Constant");
    p->set<string>("Neutron Fission Name", "Neutron Fission");
    
    p->set<bool>("Have Source", haveSource);
    p->set<string>("Source Name", "Source");

    p->set<bool>("Have Flow", haveFlow);
    p->set<string>("Velocity QP Variable Name", "Velocity");

    p->set<bool>("Have Neutron", haveNeut);
    p->set<string>("Neutron QP Variable Name", "Neutron");
    
    p->set<bool>("Have SUPG", haveSUPG);
    p->set<string> ("Tau T Name", "Tau T");
 
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", qp_vector);
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", node_qp_scalar);
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", node_qp_vector);

    //Output
    p->set<string>("Residual Name", "Temperature Residual");
    p->set< RCP<DataLayout> >("Node Scalar Data Layout", node_scalar);

    evaluators_to_build["Heat Resid"] = p;
  }

   if (haveNeut) { // Neutron Resid
    RCP<ParameterList> p = rcp(new ParameterList("Neutron Resid"));

    int type = FactoryTraits<AlbanyTraits>::id_nsneutroneqresid;
    p->set<int>("Type", type);

    //Input
    p->set<string>("Weighted BF Name", "wBF");
    p->set<string>("Weighted Gradient BF Name", "wGrad BF");
    p->set<string>("QP Variable Name", "Neutron");
    p->set<string>("Gradient QP Variable Name", "Neutron Gradient");
    p->set<string>("Neutron Diffusion Name", "Neutron Diffusion");
    p->set<string>("Neutron Absorption Name", "Neutron Absorption");
    p->set<string>("Neutron Fission Name", "Neutron Fission");
    p->set<string>("Reference Temperature Name", "Reference Temperature");
    
    p->set<bool>("Have Neutron Source", haveNeutSource);
    p->set<string>("Source Name", "Neutron Source");

    p->set<bool>("Have Flow", haveFlow);
    p->set<string>("Velocity QP Variable Name", "Velocity");

    p->set<bool>("Have Heat", haveHeat);
    p->set<string> ("Temperature QP Variable Name", "Temperature");
 
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", qp_vector);
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", node_qp_scalar);
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", node_qp_vector);

    //Output
    p->set<string>("Residual Name", "Neutron Residual");
    p->set< RCP<DataLayout> >("Node Scalar Data Layout", node_scalar);

    evaluators_to_build["Neutron Resid"] = p;
  }

  if (haveFlow) { // Momentum Scatter Residual
    RCP< vector<string> > resid_names = rcp(new vector<string>(1));
    (*resid_names)[0] = "Momentum Residual";
   
    RCP<ParameterList> p = rcp(new ParameterList);
    int type = FactoryTraits<AlbanyTraits>::id_scatter_residual;
    p->set<int>("Type", type);
    p->set<bool>("Vector Field", true);
    p->set< RCP< vector<string> > >("Residual Names", resid_names);

    p->set< RCP<DataLayout> >("Dummy Data Layout", dummy);
    p->set< RCP<DataLayout> >("Data Layout", node_vector);

    p->set<int>("Offset of First DOF", 0);
    p->set<string>("Scatter Field Name", "Scatter Momentum");

    evaluators_to_build["Scatter Momentum Residual"] = p;
  }

  if (haveFlow) { // Continuity Scatter Residual
    RCP< vector<string> > resid_names = rcp(new vector<string>(1));
    (*resid_names)[0] = "Continuity Residual";
   
    RCP<ParameterList> p = rcp(new ParameterList);
    int type = FactoryTraits<AlbanyTraits>::id_scatter_residual;
    p->set<int>("Type", type);
    p->set< RCP< vector<string> > >("Residual Names", resid_names);

    p->set< RCP<DataLayout> >("Dummy Data Layout", dummy);
    p->set< RCP<DataLayout> >("Data Layout", node_scalar);

    p->set<int>("Offset of First DOF", numDim);
    p->set<string>("Scatter Field Name", "Scatter Continuity");

    evaluators_to_build["Scatter Continuity Residual"] = p;
  }

  if (haveHeat) { // Temperature Scatter Residual
    RCP< vector<string> > resid_names = rcp(new vector<string>(1));
    (*resid_names)[0] = "Temperature Residual";
   
    RCP<ParameterList> p = rcp(new ParameterList);
    int type = FactoryTraits<AlbanyTraits>::id_scatter_residual;
    p->set<int>("Type", type);
    p->set< RCP< vector<string> > >("Residual Names", resid_names);

    p->set< RCP<DataLayout> >("Dummy Data Layout", dummy);
    p->set< RCP<DataLayout> >("Data Layout", node_scalar);

    if (haveFlow)
      p->set<int>("Offset of First DOF", numDim+1);
    else
      p->set<int>("Offset of First DOF", 0);
    p->set<string>("Scatter Field Name", "Scatter Temperature");

    evaluators_to_build["Scatter Temperature Residual"] = p;
  }

  if (haveNeut) { // Neutron Scatter Residual
    RCP< vector<string> > resid_names = rcp(new vector<string>(1));
    (*resid_names)[0] = "Neutron Residual";
   
    RCP<ParameterList> p = rcp(new ParameterList);
    int type = FactoryTraits<AlbanyTraits>::id_scatter_residual;
    p->set<int>("Type", type);
    p->set< RCP< vector<string> > >("Residual Names", resid_names);

    p->set< RCP<DataLayout> >("Dummy Data Layout", dummy);
    p->set< RCP<DataLayout> >("Data Layout", node_scalar);

    if (haveFlow) {
       if (haveHeat)
          p->set<int>("Offset of First DOF", numDim+2);
       else
          p->set<int>("Offset of First DOF", numDim+1);
    }
    else {
       if (haveHeat) 
          p->set<int>("Offset of First DOF", 1);
       else
          p->set<int>("Offset of First DOF", 0);
    }
    p->set<int>("Number of DOF per Node", neq);
    p->set<string>("Scatter Field Name", "Scatter Neutron");

    evaluators_to_build["Scatter Neutron Residual"] = p;
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

  if (haveFlow) {
    PHX::Tag<AlbanyTraits::Residual::ScalarT> res_tag("Scatter Momentum", 
						      dummy);
    fm->requireField<AlbanyTraits::Residual>(res_tag);
    PHX::Tag<AlbanyTraits::Jacobian::ScalarT> jac_tag("Scatter Momentum", 
						      dummy);
    fm->requireField<AlbanyTraits::Jacobian>(jac_tag);
    PHX::Tag<AlbanyTraits::Tangent::ScalarT> tan_tag("Scatter Momentum", 
						     dummy);
    fm->requireField<AlbanyTraits::Tangent>(tan_tag);
    PHX::Tag<AlbanyTraits::SGResidual::ScalarT> sgres_tag("Scatter Momentum", 
							  dummy);
    fm->requireField<AlbanyTraits::SGResidual>(sgres_tag);
    PHX::Tag<AlbanyTraits::SGJacobian::ScalarT> sgjac_tag("Scatter Momentum", 
							  dummy);
    fm->requireField<AlbanyTraits::SGJacobian>(sgjac_tag);
    PHX::Tag<AlbanyTraits::MPResidual::ScalarT> mpres_tag("Scatter Momentum", 
							  dummy);
    fm->requireField<AlbanyTraits::MPResidual>(mpres_tag);
    PHX::Tag<AlbanyTraits::MPJacobian::ScalarT> mpjac_tag("Scatter Momentum", 
							  dummy);
    fm->requireField<AlbanyTraits::MPJacobian>(mpjac_tag);

    PHX::Tag<AlbanyTraits::Residual::ScalarT> res_tag2("Scatter Continuity", 
						       dummy);
    fm->requireField<AlbanyTraits::Residual>(res_tag2);
    PHX::Tag<AlbanyTraits::Jacobian::ScalarT> jac_tag2("Scatter Continuity", 
						       dummy);
    fm->requireField<AlbanyTraits::Jacobian>(jac_tag2);
    PHX::Tag<AlbanyTraits::Tangent::ScalarT> tan_tag2("Scatter Continuity", 
						      dummy);
    fm->requireField<AlbanyTraits::Tangent>(tan_tag2);
    PHX::Tag<AlbanyTraits::SGResidual::ScalarT> sgres_tag2("Scatter Continuity",
							   dummy);
    fm->requireField<AlbanyTraits::SGResidual>(sgres_tag2);
    PHX::Tag<AlbanyTraits::SGJacobian::ScalarT> sgjac_tag2("Scatter Continuity",
							   dummy);
    fm->requireField<AlbanyTraits::SGJacobian>(sgjac_tag2);
    PHX::Tag<AlbanyTraits::MPResidual::ScalarT> mpres_tag2("Scatter Continuity",
							   dummy);
    fm->requireField<AlbanyTraits::MPResidual>(mpres_tag2);
    PHX::Tag<AlbanyTraits::MPJacobian::ScalarT> mpjac_tag2("Scatter Continuity",
							   dummy);
    fm->requireField<AlbanyTraits::MPJacobian>(mpjac_tag2);
  }

  if (haveHeat) {
    PHX::Tag<AlbanyTraits::Residual::ScalarT> res_tag("Scatter Temperature", 
						       dummy);
    fm->requireField<AlbanyTraits::Residual>(res_tag);
    PHX::Tag<AlbanyTraits::Jacobian::ScalarT> jac_tag("Scatter Temperature", 
						       dummy);
    fm->requireField<AlbanyTraits::Jacobian>(jac_tag);
    PHX::Tag<AlbanyTraits::Tangent::ScalarT> tan_tag("Scatter Temperature", 
						      dummy);
    fm->requireField<AlbanyTraits::Tangent>(tan_tag);
    PHX::Tag<AlbanyTraits::SGResidual::ScalarT> sgres_tag("Scatter Temperature",
							   dummy);
    fm->requireField<AlbanyTraits::SGResidual>(sgres_tag);
    PHX::Tag<AlbanyTraits::SGJacobian::ScalarT> sgjac_tag("Scatter Temperature",
							   dummy);
    fm->requireField<AlbanyTraits::SGJacobian>(sgjac_tag);
    PHX::Tag<AlbanyTraits::MPResidual::ScalarT> mpres_tag("Scatter Temperature",
							   dummy);
    fm->requireField<AlbanyTraits::MPResidual>(mpres_tag);
    PHX::Tag<AlbanyTraits::MPJacobian::ScalarT> mpjac_tag("Scatter Temperature",
							   dummy);
    fm->requireField<AlbanyTraits::MPJacobian>(mpjac_tag);
  }

  if (haveNeut) {
    PHX::Tag<AlbanyTraits::Residual::ScalarT> res_tag("Scatter Neutron", 
						       dummy);
    fm->requireField<AlbanyTraits::Residual>(res_tag);
    PHX::Tag<AlbanyTraits::Jacobian::ScalarT> jac_tag("Scatter Neutron", 
						       dummy);
    fm->requireField<AlbanyTraits::Jacobian>(jac_tag);
    PHX::Tag<AlbanyTraits::Tangent::ScalarT> tan_tag("Scatter Neutron", 
						      dummy);
    fm->requireField<AlbanyTraits::Tangent>(tan_tag);
    PHX::Tag<AlbanyTraits::SGResidual::ScalarT> sgres_tag("Scatter Neutron",
							   dummy);
    fm->requireField<AlbanyTraits::SGResidual>(sgres_tag);
    PHX::Tag<AlbanyTraits::SGJacobian::ScalarT> sgjac_tag("Scatter Neutron",
							   dummy);
    fm->requireField<AlbanyTraits::SGJacobian>(sgjac_tag);
    PHX::Tag<AlbanyTraits::MPResidual::ScalarT> mpres_tag("Scatter Neutron",
							   dummy);
    fm->requireField<AlbanyTraits::MPResidual>(mpres_tag);
    PHX::Tag<AlbanyTraits::MPJacobian::ScalarT> mpjac_tag("Scatter Neutron",
							   dummy);
    fm->requireField<AlbanyTraits::MPJacobian>(mpjac_tag);
  }
}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::NavierStokes::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidNavierStokesParams");

  if (numDim==1)
    validPL->set<bool>("Periodic BC", false, "Flag to indicate periodic BC for 1D problems");
  validPL->set<bool>("Have Flow Equations", true);
  validPL->set<bool>("Have Heat Equation", true);
  validPL->set<bool>("Have Neutron Equation", true);
  validPL->set<bool>("Have Pressure Stabilization", true);
  validPL->set<bool>("Have SUPG Stabilization", true);
  validPL->sublist("Thermal Conductivity", false, "");
  validPL->sublist("Density", false, "");
  validPL->sublist("Viscosity", false, "");
  validPL->sublist("Volumetric Expansion Coefficient", false, "");
  validPL->sublist("Specific Heat", false, "");
  validPL->sublist("Body Force", false, "");
  validPL->sublist("Neutron Source", false, "");
  validPL->sublist("Neutron Diffusion", false, "");
  validPL->sublist("Neutron Absorption", false, "");
  validPL->sublist("Neutron Fission", false, "");
  validPL->sublist("Proportionality Constant", false, "");

  return validPL;
}

