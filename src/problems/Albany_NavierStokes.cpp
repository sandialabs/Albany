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
#include "Albany_ProblemUtils.hpp"


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
  haveHeat = params->get("Have Heat Equation", false);
  haveNeut = params->get("Have Neutron Equation", false);
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
  constructDirichletEvaluators(meshSpecs);
 
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
  

   // Construct standard FEM evaluators with standard field names                              
   std::map<string, RCP<ParameterList> > evaluators_to_build;
   RCP<Albany::Layouts> dl = rcp(new Albany::Layouts(worksetSize,numVertices,numNodes,numQPts,numDim));
   Albany::ProblemUtils probUtils(dl);
   bool supportsTransient=true;
   int offset=0;

   // Define Field Names

   if (haveFlow) {

     Teuchos::ArrayRCP<string> dof_names(1);
     Teuchos::ArrayRCP<string> dof_names_dot(1);
     Teuchos::ArrayRCP<string> resid_names(1);
     dof_names[0] = "Velocity";
     dof_names_dot[0] = dof_names[0]+"_dot";
     resid_names[0] = "Momentum Residual";
     evaluators_to_build["Gather Velocity Solution "] =
       probUtils.constructGatherSolutionEvaluator(true, dof_names, dof_names_dot, offset);

     evaluators_to_build["DOF "+dof_names[0]] =
       probUtils.constructDOFVecInterpolationEvaluator(dof_names[0]);

     evaluators_to_build["DOF "+dof_names_dot[0]] =
       probUtils.constructDOFVecInterpolationEvaluator(dof_names_dot[0]);

     evaluators_to_build["DOF Grad "+dof_names[0]] =
       probUtils.constructDOFVecGradInterpolationEvaluator(dof_names[0]);

     evaluators_to_build["Scatter Momentum Residual"] =
       probUtils.constructScatterResidualEvaluator(true, resid_names,offset, "Scatter Momentum");
     offset += numDim;
   }

   if (haveFlow) {
     Teuchos::ArrayRCP<string> dof_names(1);
     Teuchos::ArrayRCP<string> dof_names_dot(1);
     Teuchos::ArrayRCP<string> resid_names(1);
     dof_names[0] = "Pressure";
     dof_names_dot[0] = dof_names[0]+"_dot";
     resid_names[0] = "Continuity Residual";
     evaluators_to_build["Gather Pressure Solution "] =
       probUtils.constructGatherSolutionEvaluator(false, dof_names, dof_names_dot, offset);

     evaluators_to_build["DOF "+dof_names[0]] =
       probUtils.constructDOFInterpolationEvaluator(dof_names[0]);

     evaluators_to_build["DOF "+dof_names_dot[0]] =
       probUtils.constructDOFInterpolationEvaluator(dof_names_dot[0]);

     evaluators_to_build["DOF Grad "+dof_names[0]] =
       probUtils.constructDOFGradInterpolationEvaluator(dof_names[0]);

     evaluators_to_build["Scatter Continuity Residual"] =
       probUtils.constructScatterResidualEvaluator(false, resid_names,offset, "Scatter Continuity");
     offset ++;
   }

   if (haveHeat) { // Gather Solution Temperature
     Teuchos::ArrayRCP<string> dof_names(1);
     Teuchos::ArrayRCP<string> dof_names_dot(1);
     Teuchos::ArrayRCP<string> resid_names(1);
     dof_names[0] = "Temperature";
     dof_names_dot[0] = dof_names[0]+"_dot";
     resid_names[0] = dof_names[0]+" Residual";
     evaluators_to_build["Gather Temperature Solution "] =
       probUtils.constructGatherSolutionEvaluator(false, dof_names, dof_names_dot, offset);

     evaluators_to_build["DOF "+dof_names[0]] =
       probUtils.constructDOFInterpolationEvaluator(dof_names[0]);

     evaluators_to_build["DOF "+dof_names_dot[0]] =
       probUtils.constructDOFInterpolationEvaluator(dof_names_dot[0]);

     evaluators_to_build["DOF Grad "+dof_names[0]] =
       probUtils.constructDOFGradInterpolationEvaluator(dof_names[0]);

     evaluators_to_build["Scatter Temperature Residual"] =
       probUtils.constructScatterResidualEvaluator(false, resid_names, offset, "Scatter Temperature");
     offset ++;

   }
   if (haveNeut) { // Gather Solution Neutron
     Teuchos::ArrayRCP<string> dof_names(1);
     Teuchos::ArrayRCP<string> dof_names_dot(1);
     Teuchos::ArrayRCP<string> resid_names(1);
     dof_names[0] = "Neutron";
     dof_names_dot[0] = dof_names[0]+"_dot";
     resid_names[0] = dof_names[0]+" Residual";
     evaluators_to_build["Gather Neutron Solution "] =
       probUtils.constructGatherSolutionEvaluator(false, dof_names, dof_names_dot, offset);

     evaluators_to_build["DOF "+dof_names[0]] =
       probUtils.constructDOFInterpolationEvaluator(dof_names[0]);

     evaluators_to_build["DOF "+dof_names_dot[0]] =
       probUtils.constructDOFInterpolationEvaluator(dof_names_dot[0]);

     evaluators_to_build["DOF Grad "+dof_names[0]] =
       probUtils.constructDOFGradInterpolationEvaluator(dof_names[0]);

     evaluators_to_build["Scatter Neutron Residual"] =
       probUtils.constructScatterResidualEvaluator(false, resid_names, offset, "Scatter Neutron");
     offset ++;
   }

   evaluators_to_build["Gather Coordinate Vector"] =
     probUtils.constructGatherCoordinateVectorEvaluator();

   evaluators_to_build["Map To Physical Frame"] =
     probUtils.constructMapToPhysicalFrameEvaluator(cellType, cubature);

   evaluators_to_build["Compute Basis Functions"] =
     probUtils.constructComputeBasisFunctionsEvaluator(cellType, intrepidBasis, cubature); 

  if (havePSPG || haveSUPG) { // Compute Contravarient Metric Tensor
    RCP<ParameterList> p = 
      rcp(new ParameterList("Contravarient Metric Tensor"));

    int type = FactoryTraits<AlbanyTraits>::id_nsgctensor;
    p->set<int>("Type", type);

    // Inputs: X, Y at nodes, Cubature, and Basis
    p->set<string>("Coordinate Vector Name","Coord Vec");
    p->set< RCP<DataLayout> >("Coordinate Data Layout", dl->vertices_vector);
    p->set< RCP<Intrepid::Cubature<RealType> > >("Cubature", cubature);

    p->set<RCP<shards::CellTopology> >("Cell Type", cellType);

    // Outputs: BF, weightBF, Grad BF, weighted-Grad BF, all in physical space
    p->set<string>("Contravarient Metric Tensor Name", "Gc");
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);

    evaluators_to_build["Contravarient Metric Tensor"] = p;
  }

  { // Density
    RCP<ParameterList> p = rcp(new ParameterList);

    int type = FactoryTraits<AlbanyTraits>::id_nsmatprop;
    p->set<int>("Type", type);

    p->set<string>("Material Property Name", "Density");
    p->set<string>("QP Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Node Data Layout", dl->node_scalar);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

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
    p->set< RCP<DataLayout> >("Node Data Layout", dl->node_scalar);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

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
    p->set< RCP<DataLayout> >("Node Data Layout", dl->node_scalar);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

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
    p->set< RCP<DataLayout> >("Node Data Layout", dl->node_scalar);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

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
    p->set< RCP<DataLayout> >("Node Data Layout", dl->node_scalar);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

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
    p->set< RCP<DataLayout> >("Node Data Layout", dl->node_scalar);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

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
    p->set< RCP<DataLayout> >("Node Data Layout", dl->node_scalar);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

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
    p->set< RCP<DataLayout> >("Node Data Layout", dl->node_scalar);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

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
    p->set< RCP<DataLayout> >("Node Data Layout", dl->node_scalar);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

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
    p->set< RCP<DataLayout> >("Node Data Layout", dl->node_scalar);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = 
      params->sublist("Volumetric Expansion Coefficient");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    evaluators_to_build["Volumetric Expansion Coefficient"] = p;
  }

  if (haveHeat && haveSource) { // Source
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

  if (haveNeut && haveNeutSource) { // Source
    RCP<ParameterList> p = rcp(new ParameterList);

    int type = FactoryTraits<AlbanyTraits>::id_source;
    p->set<int>("Type", type);

    p->set<string>("Source Name", "Source");
    p->set<string>("Variable Name", "Neutron");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

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

    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector); 

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

    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);

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

    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);

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

    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);

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
 
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);    
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", dl->node_qp_scalar);
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);
    
    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
  
    //Output
    p->set<string>("Residual Name", "Momentum Residual");
    p->set< RCP<DataLayout> >("Node Vector Data Layout", dl->node_vector);

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

    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", dl->node_qp_scalar);  
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);
    p->set< RCP<PHX::DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<PHX::DataLayout> >("QP Vector Data Layout", dl->qp_vector);
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);

    //Output
    p->set<string>("Residual Name", "Continuity Residual");
    p->set< RCP<DataLayout> >("Node Scalar Data Layout", dl->node_scalar);

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
 
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", dl->node_qp_scalar);
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);

    //Output
    p->set<string>("Residual Name", "Temperature Residual");
    p->set< RCP<DataLayout> >("Node Scalar Data Layout", dl->node_scalar);

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
 
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", dl->node_qp_scalar);
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);

    //Output
    p->set<string>("Residual Name", "Neutron Residual");
    p->set< RCP<DataLayout> >("Node Scalar Data Layout", dl->node_scalar);

    evaluators_to_build["Neutron Resid"] = p;
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
						      dl->dummy);
    fm->requireField<AlbanyTraits::Residual>(res_tag);
    PHX::Tag<AlbanyTraits::Jacobian::ScalarT> jac_tag("Scatter Momentum", 
						      dl->dummy);
    fm->requireField<AlbanyTraits::Jacobian>(jac_tag);
    PHX::Tag<AlbanyTraits::Tangent::ScalarT> tan_tag("Scatter Momentum", 
						     dl->dummy);
    fm->requireField<AlbanyTraits::Tangent>(tan_tag);
    PHX::Tag<AlbanyTraits::SGResidual::ScalarT> sgres_tag("Scatter Momentum", 
							  dl->dummy);
    fm->requireField<AlbanyTraits::SGResidual>(sgres_tag);
    PHX::Tag<AlbanyTraits::SGJacobian::ScalarT> sgjac_tag("Scatter Momentum", 
							  dl->dummy);
    fm->requireField<AlbanyTraits::SGJacobian>(sgjac_tag);
    PHX::Tag<AlbanyTraits::MPResidual::ScalarT> mpres_tag("Scatter Momentum", 
							  dl->dummy);
    fm->requireField<AlbanyTraits::MPResidual>(mpres_tag);
    PHX::Tag<AlbanyTraits::MPJacobian::ScalarT> mpjac_tag("Scatter Momentum", 
							  dl->dummy);
    fm->requireField<AlbanyTraits::MPJacobian>(mpjac_tag);

    PHX::Tag<AlbanyTraits::Residual::ScalarT> res_tag2("Scatter Continuity", 
						       dl->dummy);
    fm->requireField<AlbanyTraits::Residual>(res_tag2);
    PHX::Tag<AlbanyTraits::Jacobian::ScalarT> jac_tag2("Scatter Continuity", 
						       dl->dummy);
    fm->requireField<AlbanyTraits::Jacobian>(jac_tag2);
    PHX::Tag<AlbanyTraits::Tangent::ScalarT> tan_tag2("Scatter Continuity", 
						      dl->dummy);
    fm->requireField<AlbanyTraits::Tangent>(tan_tag2);
    PHX::Tag<AlbanyTraits::SGResidual::ScalarT> sgres_tag2("Scatter Continuity",
							   dl->dummy);
    fm->requireField<AlbanyTraits::SGResidual>(sgres_tag2);
    PHX::Tag<AlbanyTraits::SGJacobian::ScalarT> sgjac_tag2("Scatter Continuity",
							   dl->dummy);
    fm->requireField<AlbanyTraits::SGJacobian>(sgjac_tag2);
    PHX::Tag<AlbanyTraits::MPResidual::ScalarT> mpres_tag2("Scatter Continuity",
							   dl->dummy);
    fm->requireField<AlbanyTraits::MPResidual>(mpres_tag2);
    PHX::Tag<AlbanyTraits::MPJacobian::ScalarT> mpjac_tag2("Scatter Continuity",
							   dl->dummy);
    fm->requireField<AlbanyTraits::MPJacobian>(mpjac_tag2);
  }

  if (haveHeat) {
    PHX::Tag<AlbanyTraits::Residual::ScalarT> res_tag("Scatter Temperature", 
						       dl->dummy);
    fm->requireField<AlbanyTraits::Residual>(res_tag);
    PHX::Tag<AlbanyTraits::Jacobian::ScalarT> jac_tag("Scatter Temperature", 
						       dl->dummy);
    fm->requireField<AlbanyTraits::Jacobian>(jac_tag);
    PHX::Tag<AlbanyTraits::Tangent::ScalarT> tan_tag("Scatter Temperature", 
						      dl->dummy);
    fm->requireField<AlbanyTraits::Tangent>(tan_tag);
    PHX::Tag<AlbanyTraits::SGResidual::ScalarT> sgres_tag("Scatter Temperature",
							   dl->dummy);
    fm->requireField<AlbanyTraits::SGResidual>(sgres_tag);
    PHX::Tag<AlbanyTraits::SGJacobian::ScalarT> sgjac_tag("Scatter Temperature",
							   dl->dummy);
    fm->requireField<AlbanyTraits::SGJacobian>(sgjac_tag);
    PHX::Tag<AlbanyTraits::MPResidual::ScalarT> mpres_tag("Scatter Temperature",
							   dl->dummy);
    fm->requireField<AlbanyTraits::MPResidual>(mpres_tag);
    PHX::Tag<AlbanyTraits::MPJacobian::ScalarT> mpjac_tag("Scatter Temperature",
							   dl->dummy);
    fm->requireField<AlbanyTraits::MPJacobian>(mpjac_tag);
  }

  if (haveNeut) {
    PHX::Tag<AlbanyTraits::Residual::ScalarT> res_tag("Scatter Neutron", 
						       dl->dummy);
    fm->requireField<AlbanyTraits::Residual>(res_tag);
    PHX::Tag<AlbanyTraits::Jacobian::ScalarT> jac_tag("Scatter Neutron", 
						       dl->dummy);
    fm->requireField<AlbanyTraits::Jacobian>(jac_tag);
    PHX::Tag<AlbanyTraits::Tangent::ScalarT> tan_tag("Scatter Neutron", 
						      dl->dummy);
    fm->requireField<AlbanyTraits::Tangent>(tan_tag);
    PHX::Tag<AlbanyTraits::SGResidual::ScalarT> sgres_tag("Scatter Neutron",
							   dl->dummy);
    fm->requireField<AlbanyTraits::SGResidual>(sgres_tag);
    PHX::Tag<AlbanyTraits::SGJacobian::ScalarT> sgjac_tag("Scatter Neutron",
							   dl->dummy);
    fm->requireField<AlbanyTraits::SGJacobian>(sgjac_tag);
    PHX::Tag<AlbanyTraits::MPResidual::ScalarT> mpres_tag("Scatter Neutron",
							   dl->dummy);
    fm->requireField<AlbanyTraits::MPResidual>(mpres_tag);
    PHX::Tag<AlbanyTraits::MPJacobian::ScalarT> mpjac_tag("Scatter Neutron",
							   dl->dummy);
    fm->requireField<AlbanyTraits::MPJacobian>(mpjac_tag);
  }
}

void
Albany::NavierStokes::constructDirichletEvaluators(
        const Albany::MeshSpecsStruct& meshSpecs)
{
   // Construct Dirichlet evaluators for all nodesets and names
   vector<string> dirichletNames(neq);
   int index = 0;
   if (haveFlow) {
     dirichletNames[index++] = "ux";
     if (numDim>=2) dirichletNames[index++] = "uy";
     if (numDim==3) dirichletNames[index++] = "uz";
     dirichletNames[index++] = "p";
   }
   if (haveHeat) dirichletNames[index++] = "T";
   if (haveNeut) dirichletNames[index++] = "phi";
   Albany::DirichletUtils dirUtils;
   dfm = dirUtils.constructDirichletEvaluators(meshSpecs.nsNames, dirichletNames,
                                          this->params, this->paramLib);
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

