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
#include "NonlinearElasticityProblem.hpp"
#include "Albany_BoundaryFlux1DResponseFunction.hpp"
#include "Albany_SolutionAverageResponseFunction.hpp"
#include "Albany_SolutionTwoNormResponseFunction.hpp"
#include "Albany_Utils.hpp"
#include "Shards_BasicTopologies.hpp"
#include "Shards_CellTopology.hpp"

Albany::NonlinearElasticityProblem::
NonlinearElasticityProblem(
                         const Teuchos::RCP<Teuchos::ParameterList>& params_,
                         const Teuchos::RCP<ParamLib>& paramLib_,
                         const int numDim_) :
  Albany::AbstractProblem(params_, paramLib_, numDim_),
  haveSource(false),
  numDim(numDim_)
{
 
  std::string& method = params->get("Name", "NonlinearElasticity ");
  *out << "Problem Name = " << method << std::endl;
  
  haveSource =  params->isSublist("Source Functions");

  matModel = params->sublist("Material Model").get("Model Name","NeoHookean");

  dofNames.resize(neq);
  dofNames[0] = "X";
  if (neq>1) dofNames[1] = "Y";
  if (neq>2) dofNames[2] = "Z";

  if (matModel == "NeoHookean" || matModel == "NeoHookean AD")
   this->nstates=numDim*numDim;
  else if (matModel == "J2")  this->nstates=2*numDim*numDim+1;
  *out << "Num States to Store: " << this->nstates << std::endl;
  
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
    std::vector< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses)
{
  /* Construct All Phalanx Evaluators */
  constructEvaluators(worksetSize, disc.getCubatureDegree(), disc.getCellTopologyData(), stateMgr);
  constructDirichletEvaluators(disc.getNodeSetIDs());

  const Epetra_Map& dofMap = *(disc.getMap());
  int left_node = dofMap.MinAllGID();
  int right_node = dofMap.MaxAllGID();

  // Build response functions
  Teuchos::ParameterList& responseList = params->sublist("Response Functions");
  int num_responses = responseList.get("Number", 0);
  responses.resize(num_responses);
  for (int i=0; i<num_responses; i++) {
     std::string name = responseList.get(Albany::strint("Response",i), "??");

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
}


void
Albany::NonlinearElasticityProblem::constructEvaluators(
       const int worksetSize, const int cubDegree,
       const CellTopologyData& ctd, Albany::StateManager& stateMgr)
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

   const bool composite = params->get("Use Composite Tet 10", false);
   RCP<shards::CellTopology> comp_cellType = rcp(new shards::CellTopology( shards::getCellTopologyData<shards::Tetrahedron<11> >() ) );
   RCP<shards::CellTopology> cellType = rcp(new shards::CellTopology (&ctd));

   RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > >
     intrepidBasis = this->getIntrepidBasis(ctd, composite);

   if (composite && ctd.dimension==3 && ctd.node_count==10) cellType = comp_cellType;

   numNodes = intrepidBasis->getCardinality();

   Intrepid::DefaultCubatureFactory<RealType> cubFactory;
   RCP <Intrepid::Cubature<RealType> > cubature = cubFactory.create(*cellType, cubDegree);

   numDim = cubature->getDimension();
   numQPts = cubature->getNumPoints();
   //numVertices = cellType->getNodeCount();
   numVertices = numNodes;

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

  { // Gather Solution
   RCP< vector<string> > dof_names = rcp(new vector<string>(1));
     (*dof_names)[0] = "Displacement";

    RCP<ParameterList> p = rcp(new ParameterList);
    int type = FactoryTraits<AlbanyTraits>::id_gather_solution;
    p->set<int>("Type", type);
    p->set< RCP< vector<string> > >("Solution Names", dof_names);
    p->set<bool>("Vector Field", true);
    p->set< RCP<DataLayout> >("Data Layout", node_vector);

    RCP< vector<string> > dof_names_dot = rcp(new vector<string>(1));
     (*dof_names_dot)[0] = "Displacement_dot";

    p->set< RCP< vector<string> > >("Time Dependent Solution Names", dof_names_dot);

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

  {
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

    //Inputs: flags, weights, GradU
    const bool avgJ = params->get("avgJ", false);
    p->set<bool>("avgJ Name", avgJ);
    const bool volavgJ = params->get("volavgJ", false);
    p->set<bool>("volavgJ Name", volavgJ);
    p->set<string>("Weights Name","Weights");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);
    p->set<string>("Gradient QP Variable Name", "Displacement Gradient");
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", qp_tensor);

    //Outputs: F, J
    p->set<string>("DefGrad Name", "Deformation Gradient"); //qp_tensor also
    p->set<string>("DetDefGrad Name", "Determinant of Deformation Gradient"); 
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);

    evaluators_to_build["DefGrad"] = p;
  }

 if (matModel == "NeoHookean")
 {
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
     p->set<string>("Stress Name", matModel); //qp_tensor also

     evaluators_to_build["Stress"] = p;
     evaluators_to_build["Save Stress"] =
      stateMgr.registerStateVariable(matModel,qp_tensor,
            dummy, FactoryTraits<AlbanyTraits>::id_savestatefield,"zero");
   }
  }
  else if (matModel == "NeoHookean AD")
  {
     RCP<ParameterList> p = rcp(new ParameterList("Stress"));

     int type = FactoryTraits<AlbanyTraits>::id_pisdwdf_stress;
     p->set<int>("Type", type);

     //Input
     p->set<string>("Elastic Modulus Name", "Elastic Modulus");
     p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);
     p->set<string>("Poissons Ratio Name", "Poissons Ratio");  // qp_scalar also

     p->set<string>("DefGrad Name", "Deformation Gradient"); 
     p->set< RCP<DataLayout> >("QP Tensor Data Layout", qp_tensor);

     //Output
     p->set<string>("Stress Name", matModel); //qp_tensor also

     evaluators_to_build["Stress"] = p;
     evaluators_to_build["Save Stress"] =
      stateMgr.registerStateVariable(matModel,qp_tensor,
            dummy, FactoryTraits<AlbanyTraits>::id_savestatefield,"zero");
  }
  else if (matModel == "J2")
  { 
    { // Hardening Modulus
      RCP<ParameterList> p = rcp(new ParameterList);

      int type = FactoryTraits<AlbanyTraits>::id_hardening_modulus;
      p->set<int>("Type", type);

      p->set<string>("QP Variable Name", "Hardening Modulus");
      p->set<string>("QP Coordinate Vector Name", "Coord Vec");
      p->set< RCP<DataLayout> >("Node Data Layout", node_scalar);
      p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);
      p->set< RCP<DataLayout> >("QP Vector Data Layout", qp_vector);

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
      p->set< RCP<DataLayout> >("Node Data Layout", node_scalar);
      p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);
      p->set< RCP<DataLayout> >("QP Vector Data Layout", qp_vector);

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
      p->set< RCP<DataLayout> >("Node Data Layout", node_scalar);
      p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);
      p->set< RCP<DataLayout> >("QP Vector Data Layout", qp_vector);

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
      p->set< RCP<DataLayout> >("Node Data Layout", node_scalar);
      p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);
      p->set< RCP<DataLayout> >("QP Vector Data Layout", qp_vector);

      p->set<RCP<ParamLib> >("Parameter Library", paramLib);
      Teuchos::ParameterList& paramList = params->sublist("Saturation Exponent");
      p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

      evaluators_to_build["Saturation Exponent"] = p;
    }

    {// Stress
      RCP<ParameterList> p = rcp(new ParameterList("Stress"));

      int type = FactoryTraits<AlbanyTraits>::id_j2_stress;
      p->set<int>("Type", type);

      //Input
      p->set<string>("DefGrad Name", "Deformation Gradient");
      p->set< RCP<DataLayout> >("QP Tensor Data Layout", qp_tensor);

      p->set<string>("Elastic Modulus Name", "Elastic Modulus");
      p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);

      p->set<string>("Poissons Ratio Name", "Poissons Ratio");  // qp_scalar also
      p->set<string>("Hardening Modulus Name", "Hardening Modulus"); // qp_scalar also
      p->set<string>("Saturation Modulus Name", "Saturation Modulus"); // qp_scalar also
      p->set<string>("Saturation Exponent Name", "Saturation Exponent"); // qp_scalar also
      p->set<string>("Yield Strength Name", "Yield Strength"); // qp_scalar also
      p->set<string>("DetDefGrad Name", "Determinant of Deformation Gradient");  // qp_scalar also
      p->set<string>("Fp Name", "Fp");  // qp_tensor also
      p->set<string>("Eqps Name", "eqps");  // qp_scalar also

      //Output
      p->set<string>("Stress Name", matModel); //qp_tensor also
 
      //Declare what state data will need to be saved (name, layout, init_type)
//      stateMgr.registerStateVariable("stress",qp_tensor,"zero");
//      stateMgr.registerStateVariable("Fp",qp_tensor,"identity");
//      stateMgr.registerStateVariable("eqps",qp_scalar,"zero");
     evaluators_to_build["Save Stress"] =
      stateMgr.registerStateVariable(matModel,qp_tensor,
            dummy, FactoryTraits<AlbanyTraits>::id_savestatefield,"zero");
     evaluators_to_build["Save Fp"] =
      stateMgr.registerStateVariable("Fp",qp_tensor,
            dummy, FactoryTraits<AlbanyTraits>::id_savestatefield,"identity");
     evaluators_to_build["Save Eqps"] =
      stateMgr.registerStateVariable("eqps",qp_scalar,
            dummy, FactoryTraits<AlbanyTraits>::id_savestatefield,"zero");

      evaluators_to_build["Stress"] = p;
    }
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
    p->set<string>("Stress Name", matModel);
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", qp_tensor);

    p->set<string>("DefGrad Name", "Deformation Gradient"); //qp_tensor also

    p->set<string>("DetDefGrad Name", "Determinant of Deformation Gradient");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);

    p->set<string>("Weighted Gradient BF Name", "wGrad BF");
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", node_qp_vector);

    p->set<string>("Weighted BF Name", "wBF");
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", node_qp_scalar);
    p->set<RCP<ParamLib> >("Parameter Library", paramLib);

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
   PHX::Tag<AlbanyTraits::MPResidual::ScalarT> mpres_tag("Scatter", dummy);
   fm->requireField<AlbanyTraits::MPResidual>(mpres_tag);
   PHX::Tag<AlbanyTraits::MPJacobian::ScalarT> mpjac_tag("Scatter", dummy);
   fm->requireField<AlbanyTraits::MPJacobian>(mpjac_tag);

   // Output states as part of Residual Evaluation
   const Albany::StateManager::RegisteredStates& reg = stateMgr.getRegisteredStates();
   Albany::StateManager::RegisteredStates::const_iterator st = reg.begin();
   while (st != reg.end()) {
     PHX::Tag<AlbanyTraits::Residual::ScalarT> res_out_tag(st->first, dummy);
     fm->requireField<AlbanyTraits::Residual>(res_out_tag);
     st++;
   }
}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::NonlinearElasticityProblem::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidNonlinearElasticityProblemParams");

  validPL->sublist("Elastic Modulus", false, "");
  validPL->sublist("Poissons Ratio", false, "");
  validPL->sublist("Shear Modulus", false, "");
  validPL->sublist("Material Model", false, "");
  validPL->set<bool>("avgJ", false, "Flag to indicate the J should be averaged");
  validPL->set<bool>("volavgJ", false, "Flag to indicate the J should be volume averaged");
  validPL->set<bool>("Use Composite Tet 10", false, "Flag to use the compostie tet 10 basis in Intrepid");
  if (matModel == "J2")
  {
    validPL->sublist("Hardening Modulus", false, "");
    validPL->sublist("Yield Strength", false, "");
    validPL->sublist("Saturation Modulus", false, "");
    validPL->sublist("Saturation Exponent", false, "");
  }

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

