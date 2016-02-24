//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Aeras_HydrostaticProblem.hpp"

#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "PHAL_FactoryTraits.hpp"
#include <string>
#include <sstream>


Aeras::HydrostaticProblem::
HydrostaticProblem( const Teuchos::RCP<Teuchos::ParameterList>& params_,
             const Teuchos::RCP<ParamLib>& paramLib_,
             const int numDim_) :
  Albany::AbstractProblem(params_, paramLib_),
  dof_names_tracers(arcpFromArray(params_->sublist("Hydrostatic Problem").
        get<Teuchos::Array<std::string> >("Tracers",
            Teuchos::Array<std::string>()))),
  numDim(numDim_),
  numLevels (params_->sublist("Hydrostatic Problem").get<int>("Number of Vertical Levels", 10)),
  numTracers(dof_names_tracers.size())
{
  // Set number of scalar equation per node, neq,  based on numDim
  std::cout << "Number of Vertical Levels: " << numLevels << std::endl;
  std::cout << "Number of Tracers        : " << numTracers << std::endl;
  std::cout << "Names of Tracers         : ";
  for (int i=0; i<numTracers; ++i) std::cout <<dof_names_tracers[i]<<"  ";
  std::cout << std::endl;

  neq       = 1 + (3*numLevels) + (numTracers*numLevels);

  // Set the num PDEs for the null space object to pass to ML
  this->rigidBodyModes->setNumPDEs(neq);
}

Aeras::HydrostaticProblem::
~HydrostaticProblem()
{
}

void
Aeras::HydrostaticProblem::
buildProblem(
  Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
  Albany::StateManager& stateMgr)
{
  using Teuchos::rcp;

 /* Construct All Phalanx Evaluators */
  TEUCHOS_TEST_FOR_EXCEPTION(
      meshSpecs.size()!=1,
      std::logic_error,
      "Problem supports one Material Block");
  fm.resize(1);
  fm[0] = rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
  buildEvaluators(*fm[0],
                  *meshSpecs[0],
                  stateMgr,
                  Albany::BUILD_RESID_FM, 
		  Teuchos::null);
  constructDirichletEvaluators(*meshSpecs[0]);
  
  // Build a sideset evaluator if sidesets are present
  if(meshSpecs[0]->ssNames.size() > 0)
     constructNeumannEvaluators(meshSpecs[0]);
}

Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
Aeras::HydrostaticProblem::
buildEvaluators(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fmchoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructeEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  Albany::ConstructEvaluatorsOp<HydrostaticProblem>
    op(*this,
       fm0,
       meshSpecs,
       stateMgr,
       fmchoice,
       responseList);
  Sacado::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes> fe(op);
  return *op.tags;
}

void
Aeras::HydrostaticProblem::constructDirichletEvaluators(
        const Albany::MeshSpecsStruct& meshSpecs)
{
   // Construct Dirichlet evaluators for all nodesets and names
   std::vector<std::string> dirichletNames(1 + 3*numLevels + numTracers*numLevels);

   int dbc=0;
   std::ostringstream s;
   s << "SPressure";
   dirichletNames[dbc++] = s.str();

   for (int i=0; i<numLevels; ++i) {
     {
       s.str(std::string());
       s << "Velx_"<<i;
       dirichletNames[dbc++] = s.str();
     }{
       s.str(std::string());
       s << "Vely_"<<i;
       dirichletNames[dbc++] = s.str();
     }{
       s.str(std::string());
       s << "Temperature_"<<i;
       dirichletNames[dbc++] = s.str();
     }
   }

   for (int t=0; t<numTracers; ++t) {
     for (int i=0; i<numLevels; ++i) {
       s.str(std::string());
       s << dof_names_tracers[t]<<"_"<<i;
       dirichletNames[dbc++] = s.str();
     }
   }

   Albany::BCUtils<Albany::DirichletTraits> dirUtils;
   dfm = dirUtils.constructBCEvaluators(meshSpecs.nsNames,
                                        dirichletNames,
                                        this->params,
                                        this->paramLib);
   offsets_ = dirUtils.getOffsets(); 
}

// Neumann BCs
void
Aeras::HydrostaticProblem::
constructNeumannEvaluators(const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs)
{

   // Note: we only enter this function if sidesets are defined in the mesh file
   // i.e. meshSpecs.ssNames.size() > 0

   Albany::BCUtils<Albany::NeumannTraits> nbcUtils;

   // Check to make sure that Neumann BCs are given in the input file

   if(!nbcUtils.haveBCSpecified(this->params)) {
      return;
   }


   // Construct BC evaluators for all side sets and names
   // Note that the string index sets up the equation offset, so ordering is important

   std::vector<std::string> neumannNames(1 + 1);
   Teuchos::Array<Teuchos::Array<int> > offsets;
   offsets.resize(1 + 1);

   neumannNames[0] = "rho";
   offsets[0].resize(1);
   offsets[0][0] = 0;
   offsets[1].resize(1);
   offsets[1][0] = 0;

   neumannNames[1] = "all";

   // Construct BC evaluators for all possible names of conditions
   // Should only specify flux vector components (dUdx, dUdy, dUdz)
   std::vector<std::string> condNames(1); //(dUdx, dUdy, dUdz)
   Teuchos::ArrayRCP<std::string> dof_names(1);
     dof_names[0] = "rho";

   // Note that sidesets are only supported for two and 3D currently
   if(numDim == 1)
    condNames[0] = "(dFluxdx)";
   else if(numDim == 2)
    condNames[0] = "(dFluxdx, dFluxdy)";
   else if(numDim == 3)
    condNames[0] = "(dFluxdx, dFluxdy, dFluxdz)";
   else
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
       std::endl << "Error: Sidesets only supported in 2 and 3D." << std::endl);

//   condNames[1] = "dFluxdn";
//   condNames[2] = "basal";
//   condNames[3] = "P";
//   condNames[4] = "lateral";

   nfm.resize(1); // Aeras X scalar advection problem only has one
                  // element block

   nfm[0] = nbcUtils.constructBCEvaluators(meshSpecs,
                                           neumannNames,
                                           dof_names,
                                           true,
                                           0,
                                           condNames,
                                           offsets,
                                           dl,
                                           this->params,
                                           this->paramLib);


}

Teuchos::RCP<const Teuchos::ParameterList>
Aeras::HydrostaticProblem::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidHydrostaticProblemParams");

  validPL->sublist("Hydrostatic Problem", false, "");
  return validPL;
}

namespace Aeras{
template <>
Teuchos::RCP<const PHX::FieldTag>
HydrostaticProblem::constructEvaluators<PHAL::AlbanyTraits::Jacobian>(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fieldManagerChoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  //IKT, FIXME, 2/12/16: once Aeras::ComputeAndScatterJac is ready, 
  //need to delete everything between ---- in this function.
  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());
  *out << "Aeras::HydrostaticProblem Jacobian specialization of constructEvaluators" << std::endl; 
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::ParameterList;
  using PHX::DataLayout;
  using PHX::MDALayout;
  using std::vector;
  using std::string;
  using std::map;
  using PHAL::AlbanyTraits;
  typedef PHAL::AlbanyTraits::Jacobian EvalT; 
  {
    Teuchos::ParameterList& xzhydrostatic_params = params->sublist("Hydrostatic Problem");
    const typename EvalT::ScalarT Ptop = xzhydrostatic_params.get<double>("Ptop", 101.325); 
    const typename EvalT::ScalarT P0   = xzhydrostatic_params.get<double>("P0",   101325.0); 
    Eta<EvalT>::self(Ptop, P0, numLevels);
  }


  RCP<Intrepid2::Basis<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > >
    intrepidBasis = Albany::getIntrepid2Basis(meshSpecs.ctd);
  RCP<shards::CellTopology> cellType = rcp(new shards::CellTopology (&meshSpecs.ctd));

  //Get element name
  const CellTopologyData *ctd = cellType->getCellTopologyData();
  std::string name     = ctd->name;
  size_t      len      = name.find("_");
  if (len != std::string::npos) name = name.substr(0,len);
  if (name == "Quadrilateral" || name == "ShellQuadrilateral") 
		TEUCHOS_TEST_FOR_EXCEPTION(true,
		Teuchos::Exceptions::InvalidParameter,"Aeras::Hydrostatic no longer works with isoparameteric " <<
		"Quads/ShellQuads! Please re-run with spectral elements (IKT, 1/12/2016).");
  
  const int numNodes = intrepidBasis->getCardinality();
  const int worksetSize = meshSpecs.worksetSize;
  
  RCP <Intrepid2::CubaturePolylib<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > > polylib = rcp(new Intrepid2::CubaturePolylib<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> >(meshSpecs.cubatureDegree, meshSpecs.cubatureRule));
  std::vector< Teuchos::RCP<Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout,PHX::Device> > > > cubatures(2, polylib); 
  RCP <Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout,PHX::Device> > > cubature = rcp( new Intrepid2::CubatureTensor<RealType,Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout,PHX::Device> >(cubatures));
  
  const int numQPts = cubature->getNumPoints();
  const int numVertices = cellType->getNodeCount();
  
  const int vecDim = 3;
  *out << "Field Dimensions: Workset=" << worksetSize 
       << ", Vertices  = " << numVertices
       << ", Nodes     = " << numNodes
       << ", QuadPts   = " << numQPts
       << ", Dim       = " << numDim 
       << ", Neq       = " << neq 
       << ", VecDim    = " << vecDim       
       << ", numLevels = " << numLevels 
       << ", numTracers= " << numTracers << std::endl;
  
  if (numNodes != numQPts) { 
    TEUCHOS_TEST_FOR_EXCEPTION(true,
         Teuchos::Exceptions::InvalidParameter, "Aeras::HydrostaticProblem must be run such that nNodes == numQPts!  " 
         <<  "This does not hold: numNodes = " <<  numNodes << ", numQPts = " << numQPts << "."); 
  }

  //Evaluators for DOFs that depend on levels
  
  //Evaluators for DOFs that depend on levels
  dl = rcp(new Aeras::Layouts(worksetSize,numVertices,numNodes,numQPts,numDim, vecDim, numLevels));
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);

  // Temporary variable used numerous times below
  Teuchos::RCP<PHX::Evaluator<AlbanyTraits> > ev;

  // Define Node Field Names
  Teuchos::ArrayRCP<std::string> dof_names_nodes(1);
  Teuchos::ArrayRCP<std::string> dof_names_nodes_dot(1);
  Teuchos::ArrayRCP<std::string> dof_names_nodes_gradient(1);
  Teuchos::ArrayRCP<std::string> dof_names_nodes_resid(1);
  dof_names_nodes[0]          = "SPressure";
  dof_names_nodes_dot[0]      = dof_names_nodes[0]+"_dot";
  dof_names_nodes_gradient[0] = dof_names_nodes[0]+"_gradient";
  dof_names_nodes_resid[0]    = dof_names_nodes[0]+"_residual";

  // Define Level Field Names
  int numLevelDOF = 2;
  Teuchos::ArrayRCP<std::string> dof_names_levels(2);
  Teuchos::ArrayRCP<std::string> dof_names_levels_dot(2);
  Teuchos::ArrayRCP<std::string> dof_names_levels_gradient(2);
  Teuchos::ArrayRCP<std::string> dof_names_levels_src(2);
  Teuchos::ArrayRCP<std::string> dof_names_levels_resid(2);
  dof_names_levels[0]          = "Velx";
  dof_names_levels[1]          = "Temperature";

  for (int i=0; i<2; ++i) {
    dof_names_levels_dot[i]      = dof_names_levels[i]+"_dot";
    dof_names_levels_gradient[i] = dof_names_levels[i]+"_gradient";
    dof_names_levels_src  [i]    = dof_names_levels[i]+"_source";
    dof_names_levels_resid[i]    = dof_names_levels[i]+"_residual";
  }

  // Define Tracer Field Names
  Teuchos::ArrayRCP<std::string> dof_names_tracers_dot(numTracers);
  Teuchos::ArrayRCP<std::string> dof_names_tracers_gradient(numTracers);
  Teuchos::ArrayRCP<std::string> dof_names_tracers_src(numTracers);
  Teuchos::ArrayRCP<std::string> dof_names_tracers_resid(numTracers);
  Teuchos::ArrayRCP<std::string> dof_names_tracers_deta(numTracers);

  for (int t=0; t<numTracers; ++t) {
    dof_names_tracers_dot     [t] = dof_names_tracers[t]+"_dot";
    dof_names_tracers_gradient[t] = dof_names_tracers[t]+"_gradient";
    dof_names_tracers_src     [t] = dof_names_tracers[t]+"_source";
    dof_names_tracers_resid   [t] = dof_names_tracers[t]+"_residual";
    dof_names_tracers_deta    [t] = dof_names_tracers[t]+"_deta";
  }
 
  {
    RCP<ParameterList> p = rcp(new ParameterList("DOF Interpolation "+dof_names_nodes[0]));
    p->set<string>("Variable Name",                                   dof_names_nodes[0]);
    p->set<Teuchos::RCP<PHX::DataLayout> >("Nodal Variable Layout",     dl->node_scalar);
    p->set<Teuchos::RCP<PHX::DataLayout> >("Quadpoint Variable Layout", dl->qp_scalar);
    p->set<string>("BF Name", "BF");

    ev = rcp(new Aeras::DOFInterpolation<EvalT,AlbanyTraits>(*p,dl));
    fm0.registerEvaluator<EvalT>(ev);
  }

  {
    RCP<ParameterList> p = rcp(new ParameterList("DOF Interpolation "+dof_names_nodes_dot[0]));
    p->set<string>("Variable Name", dof_names_nodes_dot[0]);
    p->set<Teuchos::RCP<PHX::DataLayout> >("Nodal Variable Layout",     dl->node_scalar);
    p->set<Teuchos::RCP<PHX::DataLayout> >("Quadpoint Variable Layout", dl->qp_scalar);
    p->set<string>("BF Name", "BF");

    ev = rcp(new Aeras::DOFInterpolation<EvalT,AlbanyTraits>(*p,dl));
    fm0.registerEvaluator<EvalT>(ev);
  }

  for (int t=0; t<numTracers; ++t) {
    RCP<ParameterList> p = rcp(new ParameterList("Tracer Interpolation "+dof_names_tracers[t]));
    p->set<string>("Variable Name", dof_names_tracers[t]);
    p->set<string>("BF Name", "BF");

    ev = rcp(new Aeras::DOFInterpolation<EvalT,AlbanyTraits>(*p,dl));
    fm0.registerEvaluator<EvalT>(ev);
  }

  for (int t=0; t<numTracers; ++t) {
    RCP<ParameterList> p = rcp(new ParameterList("Tracer Interpolation "+dof_names_tracers_dot[t]));
    p->set<string>("Variable Name", dof_names_tracers_dot[t]);
    p->set<string>("BF Name", "BF");

    ev = rcp(new Aeras::DOFInterpolation<EvalT,AlbanyTraits>(*p,dl));
    fm0.registerEvaluator<EvalT>(ev);
  }

  {
    RCP<ParameterList> p = rcp(new ParameterList("DOF Grad Interpolation "+dof_names_nodes_gradient[0]));
    // Input
    p->set<string>("Variable Name",          dof_names_nodes[0]);
    p->set<string>("Gradient BF Name",       "Grad BF");
    p->set<string>("Gradient Variable Name", dof_names_nodes_gradient[0]);

    ev = rcp(new Aeras::DOFGradInterpolation<EvalT,AlbanyTraits>(*p,dl));
    fm0.registerEvaluator<EvalT>(ev);
  }

  for (int t=0; t<numTracers; ++t) {
    RCP<ParameterList> p = rcp(new ParameterList("Tracer Grad Interpolation "+dof_names_tracers_gradient[t]));
    // Input
    p->set<string>("Variable Name", dof_names_tracers[t]);
    p->set<string>("Gradient BF Name", "Grad BF");
    p->set<string>("Gradient Variable Name", dof_names_tracers_gradient[t]);

    ev = rcp(new Aeras::DOFGradInterpolationLevels<EvalT,AlbanyTraits>(*p,dl));
    fm0.registerEvaluator<EvalT>(ev);
  }

  if (numDim == 2) {
    RCP<ParameterList> p = rcp(new ParameterList("Gather Coordinate Vector"));
    // Input:
    
    // Output:: Coordindate Vector at vertices
    p->set<string>("Coordinate Vector Name", "Coord Vec");
    
    ev = rcp(new Aeras::GatherCoordinateVector<EvalT,AlbanyTraits>(*p,dl));
    fm0.registerEvaluator<EvalT>(ev);
  }
  //Planar case: 
  else {
    fm0.registerEvaluator<EvalT>
      (evalUtils.constructGatherCoordinateVectorEvaluator());
  }

  if (numDim == 2)
  {
    RCP<ParameterList> p = rcp(new ParameterList("Compute Basis Functions"));

    // Inputs: X, Y at nodes, Cubature, and Basis
    p->set< RCP<Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout,PHX::Device> > > >("Cubature", cubature);
 
    p->set< RCP<Intrepid2::Basis<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > > > 
        ("Intrepid2 Basis", intrepidBasis);
 
    p->set<RCP<shards::CellTopology> >("Cell Type", cellType);
    // Outputs: BF, weightBF, Grad BF, weighted-Grad BF, all in physical space
    p->set<string>("Spherical Coord Name",       "Lat-Long");
    p->set<std::string>("Lambda Coord Nodal Name", "Lat Nodal");
    p->set<std::string>("Theta Coord Nodal Name", "Long Nodal");
    p->set<string>("Coordinate Vector Name",     "Coord Vec");
    p->set<string>("Weights Name",               "Weights");
    p->set<string>("BF Name",                    "BF");
    p->set<string>("Weighted BF Name",           "wBF");
    p->set<string>("Gradient BF Name",           "Grad BF");
    p->set<string>("Weighted Gradient BF Name",  "wGrad BF");
    p->set<string>("Gradient Gradient BF Name",  "Grad Grad BF");
    p->set<string>("Weighted Gradient Gradient BF Name",  "wGrad Grad BF");
    p->set<string>("Jacobian Det Name",          "Jacobian Det");
    p->set<string>("Jacobian Name",              "Jacobian");
    p->set<string>("Jacobian Inv Name",          "Jacobian Inv");
    p->set<std::size_t>("spatialDim",            3);

    ev = rcp(new Aeras::ComputeBasisFunctions<EvalT,AlbanyTraits>(*p,dl));
    fm0.registerEvaluator<EvalT>(ev);
  }
  else {
    fm0.registerEvaluator<EvalT>
      (evalUtils.constructComputeBasisFunctionsEvaluator(cellType, intrepidBasis, cubature));
  }

  
  {
    RCP<ParameterList> p = rcp(new ParameterList("Compute And Scatter Jacobian"));

    p->set< Teuchos::ArrayRCP<string> >("Node Residual Names",   dof_names_nodes_resid);

    Teuchos::ArrayRCP<std::string> vector_level_names(1);
    Teuchos::ArrayRCP<std::string> scalar_level_names(1);
    vector_level_names[0]     = dof_names_levels_resid[0];
    scalar_level_names[0]     = dof_names_levels_resid[1];
    p->set< Teuchos::ArrayRCP<std::string> >("Vector Level Residual Names",        vector_level_names);
    p->set< Teuchos::ArrayRCP<std::string> >("Scalar Level Residual Names",        scalar_level_names);

    p->set< Teuchos::ArrayRCP<string> >("Tracer Residual Names", dof_names_tracers_resid);
    p->set<std::string>("Weighted BF Name",                     "wBF");
    p->set<std::string>("Weighted Gradient BF Name",            "wGrad BF");
    p->set<string>("BF Name",                    "BF");
    p->set<string>("Gradient BF Name",           "Grad BF");

    p->set<string>("Scatter Field Name", "Compute And Scatter Jacobian");

    ev = rcp(new Aeras::ComputeAndScatterJac<EvalT,AlbanyTraits>(*p,dl));
    fm0.registerEvaluator<EvalT>(ev);
  }


  if (fieldManagerChoice == Albany::BUILD_RESID_FM)  {
    PHX::Tag<typename EvalT::ScalarT> res_tag("Compute And Scatter Jacobian", dl->dummy);
    fm0.requireField<EvalT>(res_tag);
  }
  else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM) {
    Aeras::LayeredResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
    return respUtils.constructResponses(fm0, *responseList, Teuchos::null, stateMgr);
  }


  return Teuchos::null;
}
}
