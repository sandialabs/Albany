//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Aeras_ShallowWaterProblemNoAD.hpp"

#include "Shards_CellTopology.hpp"
#include "PHAL_FactoryTraits.hpp"
#include "Albany_Utils.hpp"
#include "Albany_BCUtils.hpp"
#include "Albany_ProblemUtils.hpp"
#include <string>


Aeras::ShallowWaterProblemNoAD::
ShallowWaterProblemNoAD( const Teuchos::RCP<Teuchos::ParameterList>& params_,
             const Teuchos::RCP<ParamLib>& paramLib_,
             const int spatialDim_) :
  Albany::AbstractProblem(params_, paramLib_),
  spatialDim(spatialDim_)
{
  TEUCHOS_TEST_FOR_EXCEPTION(spatialDim!=2 && spatialDim!=3,std::logic_error,"Shallow water problem is only written for 2 or 3D.");
  std::string eqnSet = params_->sublist("Equation Set").get<std::string>("Type", "Shallow Water"); 
  // Set number of scalar equation per node, neq,  based on spatialDim
  if      (spatialDim==2) { modelDim=2; neq=3; } // Planar 2D problem
  else if (spatialDim ==3 ) { //2D shells embedded in 3D
    if (eqnSet == "Scalar") { modelDim=2; neq=1; } 
    else { modelDim=2; neq=3; } 
  }

  bool useExplHyperviscosity = params_->sublist("Shallow Water Problem").get<bool>("Use Explicit Hyperviscosity", false);
  bool useImplHyperviscosity = params_->sublist("Shallow Water Problem").get<bool>("Use Implicit Hyperviscosity", false);
  bool usePrescribedVelocity = params_->sublist("Shallow Water Problem").get<bool>("Use Prescribed Velocity", false); 
  bool plotVorticity = params_->sublist("Shallow Water Problem").get<bool>("Plot Vorticity", false); 

  TEUCHOS_TEST_FOR_EXCEPTION( useExplHyperviscosity || useImplHyperviscosity ,std::logic_error,"Shallow Water No AD 3D problem " <<
                              "does not work with hyperviscosity!\n");


  if (useImplHyperviscosity) {
    if (usePrescribedVelocity) //TC1 case: only 1 extra hyperviscosity dof 
      neq = 4; 
    //If we're using hyperviscosity for Shallow water equations, we have double the # of dofs. 
    else  
      neq = 2*neq; 
  }

//No need to plot vorticity when prescrVel == 1.
  if (plotVorticity) {
     if (!usePrescribedVelocity) {
       //one extra stationary equation for vorticity
       neq++;
     }
     else {
       std::cout << "Prescribed Velocity is ON, in this case option PlotVorticity=true is ignored." << std::endl; 
     }
  }


  std::cout << "eqnSet, modelDim, neq: " << eqnSet << ", " << modelDim << ", " << neq << std::endl; 
  // Set the num PDEs for the null space object to pass to ML
  this->rigidBodyModes->setNumPDEs(neq);
}

Aeras::ShallowWaterProblemNoAD::
~ShallowWaterProblemNoAD()
{
}

void
Aeras::ShallowWaterProblemNoAD::
buildProblem(
  Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
  Albany::StateManager& stateMgr)
{
  using Teuchos::rcp;

 /* Construct All Phalanx Evaluators */
  TEUCHOS_TEST_FOR_EXCEPTION(meshSpecs.size()!=1,std::logic_error,"Problem supports one Material Block");
  fm.resize(1);
  fm[0]  = rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
  buildEvaluators(*fm[0], *meshSpecs[0], stateMgr, Albany::BUILD_RESID_FM, 
		  Teuchos::null);
}

Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
Aeras::ShallowWaterProblemNoAD::
buildEvaluators(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fmchoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructeEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  Albany::ConstructEvaluatorsOp<ShallowWaterProblemNoAD> op(
    *this, fm0, meshSpecs, stateMgr, fmchoice, responseList);
  Sacado::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes> fe(op);
  return *op.tags;
}


Teuchos::RCP<const Teuchos::ParameterList>
Aeras::ShallowWaterProblemNoAD::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidShallowWaterProblemNoADParams");

  validPL->sublist("Shallow Water Problem", false, "");
  validPL->sublist("Aeras Surface Height", false, "");
  validPL->sublist("Aeras Shallow Water Source", false, "");
  validPL->sublist("Equation Set", false, "");
  return validPL;
}

namespace Aeras{
template <>
Teuchos::RCP<const PHX::FieldTag>
ShallowWaterProblemNoAD::constructEvaluators<PHAL::AlbanyTraits::DistParamDeriv>(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fieldManagerChoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  //Do nothing -- DistParamDerivs are not meant to work for Aeras and with AlbanyT 
	return Teuchos::null;
}

template <>
Teuchos::RCP<const PHX::FieldTag>
ShallowWaterProblemNoAD::constructEvaluators<PHAL::AlbanyTraits::Jacobian>(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fieldManagerChoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
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
  
//  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());
//  *out << "Aeras::ShallowWaterProblemNoAD Jacobian specialization of constructEvaluators" << std::endl;
  
  RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> >
    intrepidBasis = Albany::getIntrepid2Basis(meshSpecs.ctd);
 
  RCP<shards::CellTopology> cellType = rcp(new shards::CellTopology (&meshSpecs.ctd));
  
  const int numNodes = intrepidBasis->getCardinality();
  const int worksetSize = meshSpecs.worksetSize;
  
//  RCP <Intrepid2::Polylib<RealType, Kokkos::DynRankView<RealType, PHX::Device> > > polylib = rcp(new Intrepid2::Polylib<RealType, Kokkos::DynRankView<RealType, PHX::Device> >(meshSpecs.cubatureDegree, meshSpecs.cubatureRule));
//  std::vector< Teuchos::RCP<Intrepid2::Cubature<PHX::Device> > > cubatures(2, polylib); 
//  RCP <Intrepid2::Cubature<PHX::Device> > cubature = rcp( new Intrepid2::CubatureTensor<RealType,Kokkos::DynRankView<RealType, PHX::Device> >(cubatures));
//  Regular Gauss Quadrature.

  Intrepid2::DefaultCubatureFactory cubFactory;
  RCP <Intrepid2::Cubature<PHX::Device> > cubature = cubFactory.create<PHX::Device, RealType, RealType>(*cellType, meshSpecs.cubatureDegree, meshSpecs.cubatureRule);


  const int numQPts     = cubature->getNumPoints();
  const int numVertices = meshSpecs.ctd.node_count;
  int vecDim = neq;

  /*if (neq == 1 || neq == 2) 
    vecDim = neq; 
  else 
    vecDim = spatialDim; 
  */
  
  *out << "Field Dimensions: Workset=" << worksetSize 
       << ", Vertices= " << numVertices
       << ", Nodes= "    << numNodes
       << ", QuadPts= "  << numQPts
       << ", Spatial Dim= "  << spatialDim 
       << ", Model Dim= "  << modelDim 
       << ", vecDim= "   << vecDim << std::endl;
  
   dl = rcp(new Aeras::Layouts(worksetSize,numVertices,numNodes,numQPts, modelDim, vecDim, 0));
   Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);

   // Temporary variable used numerous times below
   Teuchos::RCP<PHX::Evaluator<AlbanyTraits> > ev;

   // Define Field Names

  Teuchos::ArrayRCP<std::string> dof_names(1);
  Teuchos::ArrayRCP<std::string> dof_names_dot(1);
  Teuchos::ArrayRCP<std::string> dof_names_dotdot(1);
  Teuchos::ArrayRCP<std::string> resid_names(1);
  dof_names[0] = "Flow State";
  dof_names_dot[0] = dof_names[0]+"_dot";
  dof_names_dotdot[0] = dof_names[0]+"_dotdot";
  resid_names[0] = "ShallowWater Residual";

  //IKT: this is the equivalent of the supportsTransient flag in LCM. 
  //It tells the code to build 2nd derivative terms, which we need for 
  //explicit integration of the system with hyperviscosity.  
  //TODO? set this to off when it's not needed (i.e., when no hyperviscosity 
  //and/or implicit time stepping).
  bool explicitHV = true;

  // Construct Standard FEM evaluators for Vector equation
  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructGatherSolutionEvaluator(true, dof_names, dof_names_dot));
  if (explicitHV) fm0.template registerEvaluator<EvalT>
     (evalUtils.constructGatherSolutionEvaluator_withAcceleration(true, dof_names, Teuchos::null, dof_names_dotdot));
   
  if (explicitHV) fm0.template registerEvaluator<EvalT>
    (evalUtils.constructDOFVecInterpolationEvaluator(dof_names_dotdot[0]));

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructDOFVecInterpolationEvaluator(dof_names[0]));

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructDOFVecInterpolationEvaluator(dof_names_dot[0]));

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructDOFVecGradInterpolationEvaluator(dof_names[0]));

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructScatterResidualEvaluator(true, resid_names, 0, "Scatter ShallowWater"));

  // Shells: 3 coords for 2D topology
  if (spatialDim != modelDim) {
    RCP<ParameterList> p = rcp(new ParameterList("Gather Coordinate Vector"));
    // Input:
    
    // Output:: Coordindate Vector at vertices
    p->set<string>("Coordinate Vector Name", "Coord Vec");
    
    ev = rcp(new Aeras::GatherCoordinateVector<EvalT,AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }
  //Planar case: 
  else {
  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructGatherCoordinateVectorEvaluator());
  }

//  fm0.template registerEvaluator<EvalT>
//    (evalUtils.constructMapToPhysicalFrameEvaluator(cellType, cubature));

  // Shells: 3 coords for 2D topology
//  if (spatialDim != modelDim)
  if(1)
  {
    RCP<ParameterList> p = rcp(new ParameterList("Compute Basis Functions"));

    // Inputs: X, Y at nodes, Cubature, and Basis
    p->set< RCP<Intrepid2::Cubature<PHX::Device> > >("Cubature", cubature);
 
    p->set< RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> > > 
        ("Intrepid2 Basis", intrepidBasis);

    p->set<RCP<shards::CellTopology> >("Cell Type", cellType);
    // Outputs: BF, weightBF, Grad BF, weighted-Grad BF, all in physical space
    p->set<string>("Spherical Coord Name",       "Lat-Long");
    p->set<std::string>("Lambda Coord Nodal Name", "Lat Nodal");
    p->set<std::string>("Theta Coord Nodal Name", "Long Nodal");
    p->set<string>("Coordinate Vector Name",          "Coord Vec");
    p->set<string>("Weights Name",          "Weights");
    p->set<string>("BF Name",          "BF");
    p->set<string>("Weighted BF Name", "wBF");
    p->set<string>("Gradient BF Name",          "Grad BF");
    p->set<string>("Weighted Gradient BF Name", "wGrad BF");
    p->set<string>("Jacobian Det Name",          "Jacobian Det");
    p->set<string>("Jacobian Name",          "Jacobian");
    p->set<string>("Jacobian Inv Name",          "Jacobian Inv");
    p->set<std::size_t>("spatialDim", spatialDim);

    ev = rcp(new Aeras::ComputeBasisFunctions<EvalT,AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }
  //Planar case:
  //IK, 2/11/15: Planar case is obsolete I believe.  Should it be removed?  
  else {
    fm0.template registerEvaluator<EvalT>
      (evalUtils.constructComputeBasisFunctionsEvaluator(cellType, intrepidBasis, cubature));
  }

  
  {
    RCP<ParameterList> p = rcp(new ParameterList("SW Compute And Scatter Jacobian"));

    p->set< Teuchos::ArrayRCP<string> >("Node Residual Names",   dof_names);

    p->set<std::string>("Weighted BF Name",                     "wBF");
    p->set<std::string>("Weighted Gradient BF Name",            "wGrad BF");
    p->set<string>("BF Name",                    "BF");
    p->set<string>("Gradient BF Name",           "Grad BF");
    p->set<std::string>("Lambda Coord Nodal Name", "Lat Nodal");
    p->set<std::string>("Theta Coord Nodal Name", "Long Nodal");

    p->set<string>("Scatter Field Name", "SW Compute And Scatter Jacobian");

    ev = rcp(new Aeras::SW_ComputeAndScatterJac<EvalT,AlbanyTraits>(*p,dl));
    fm0.registerEvaluator<EvalT>(ev);
  }


  if (fieldManagerChoice == Albany::BUILD_RESID_FM)  {
    PHX::Tag<typename EvalT::MeshScalarT> res_tag("SW Compute And Scatter Jacobian", dl->dummy);
    fm0.requireField<EvalT>(res_tag);
  }
  else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM) {
    Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
    return respUtils.constructResponses(fm0, *responseList, Teuchos::null, stateMgr);
  }

  return Teuchos::null;
}
} 
