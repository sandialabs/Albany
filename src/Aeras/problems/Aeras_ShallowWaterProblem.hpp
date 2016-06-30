//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AERAS_SHALLOWWATERPROBLEM_HPP
#define AERAS_SHALLOWWATERPROBLEM_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_AbstractProblem.hpp"

#include "Phalanx.hpp"
#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"

#include "Aeras_Layouts.hpp"

namespace Aeras {

  /*!
   * \brief Abstract interface for representing a 1-D finite element
   * problem.
   */
  class ShallowWaterProblem : public Albany::AbstractProblem {
  public:
  
    //! Default constructor
    ShallowWaterProblem(const Teuchos::RCP<Teuchos::ParameterList>& params,
		 const Teuchos::RCP<ParamLib>& paramLib,
		 const int spatialDim_);

    //! Destructor
    ~ShallowWaterProblem();

    //! Return number of spatial dimensions
    virtual int spatialDimension() const { return spatialDim; }

    //! Build the PDE instantiations, boundary conditions, and initial solution
    virtual void buildProblem(
      Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
      Albany::StateManager& stateMgr);

    // Build evaluators
    virtual Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
    buildEvaluators(
      PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
      const Albany::MeshSpecsStruct& meshSpecs,
      Albany::StateManager& stateMgr,
      Albany::FieldManagerChoice fmchoice,
      const Teuchos::RCP<Teuchos::ParameterList>& responseList);

    //! Each problem must generate it's list of valide parameters
    Teuchos::RCP<const Teuchos::ParameterList> getValidProblemParameters() const;

  private:

    //! Private to prohibit copying
    ShallowWaterProblem(const ShallowWaterProblem&);
    
    //! Private to prohibit copying
    ShallowWaterProblem& operator=(const ShallowWaterProblem&);

  public:

    //! Main problem setup routine. Not directly called, but indirectly by following functions
    template <typename EvalT> Teuchos::RCP<const PHX::FieldTag>
    constructEvaluators(
      PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
      const Albany::MeshSpecsStruct& meshSpecs,
      Albany::StateManager& stateMgr,
      Albany::FieldManagerChoice fmchoice,
      const Teuchos::RCP<Teuchos::ParameterList>& responseList);

    void constructDirichletEvaluators(const Albany::MeshSpecsStruct& meshSpecs);
    void constructNeumannEvaluators(const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs);

  protected:
    int spatialDim; // 3 for shells
    int modelDim;   // 2 for shells
    Teuchos::RCP<Aeras::Layouts> dl;

  };

}

#include "Intrepid2_CubaturePolylib.hpp"
#include "Intrepid2_CubatureTensor.hpp"

#include "Shards_CellTopology.hpp"

#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_EvaluatorUtils.hpp"
#include "Albany_ResponseUtilities.hpp"
#include "PHAL_Neumann.hpp"

#include "Aeras_ShallowWaterResid.hpp"
#include "Aeras_ShallowWaterSource.hpp"
#include "Aeras_ShallowWaterHyperViscosity.hpp"
#include "Aeras_SurfaceHeight.hpp"

#include "Aeras_ComputeBasisFunctions.hpp"
#include "Aeras_GatherCoordinateVector.hpp"
template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
Aeras::ShallowWaterProblem::constructEvaluators(
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
  
  RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> >
    intrepidBasis = Albany::getIntrepid2Basis(meshSpecs.ctd);
 
  RCP<shards::CellTopology> cellType = rcp(new shards::CellTopology (&meshSpecs.ctd));
  
  const int numNodes = intrepidBasis->getCardinality();
  const int worksetSize = meshSpecs.worksetSize;
  
  RCP <Intrepid2::CubaturePolylib<RealType, Kokkos::DynRankView<RealType, PHX::Device> > > polylib = rcp(new Intrepid2::CubaturePolylib<RealType, Kokkos::DynRankView<RealType, PHX::Device> >(meshSpecs.cubatureDegree, meshSpecs.cubatureRule));
  std::vector< Teuchos::RCP<Intrepid2::Cubature<PHX::Device> > > cubatures(2, polylib); 
  RCP <Intrepid2::Cubature<PHX::Device> > cubature = rcp( new Intrepid2::CubatureTensor<RealType,Kokkos::DynRankView<RealType, PHX::Device> >(cubatures));
//  Regular Gauss Quadrature.
//  Intrepid2::DefaultCubatureFactory cubFactory;
//  RCP <Intrepid2::Cubature<PHX::Device> > cubature = cubFactory.create<PHX::Device, RealType, RealType>(*cellType, meshSpecs.cubatureDegree);


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

  { // ShallowWater Resid
    RCP<ParameterList> p = rcp(new ParameterList("Shallow Water Resid"));
   
    //Input
    p->set<RCP<shards::CellTopology> >("Cell Type", cellType);
    p->set<std::string>("Weighted BF Name", "wBF");
    p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
    p->set<std::string>("QP Variable Name", dof_names[0]);
    p->set<std::string>("Nodal Variable Name", dof_names[0]);
    p->set<std::string>("Time Dependent Nodal Variable Name", dof_names_dotdot[0]);
    p->set<std::string>("QP Time Derivative Variable Name", dof_names_dot[0]);
    p->set<std::string>("Time Dependent Variable Name", dof_names_dotdot[0]);
    p->set<std::string>("Gradient QP Variable Name", "Flow State Gradient");
    p->set<std::string>("Aeras Surface Height QP Variable Name", "Aeras Surface Height");
    p->set<std::string>("Shallow Water Source QP Variable Name", "Shallow Water Source");
    p->set<std::string>("Hyperviscosity Name", "Shallow Water Hyperviscosity");
    p->set<string>("Coordinate Vector Name",          "Coord Vec");
    p->set<string>("Spherical Coord Name",       "Lat-Long");
    p->set<std::string>("Lambda Coord Nodal Name", "Lat Nodal");
    p->set<std::string>("Theta Coord Nodal Name", "Long Nodal");

    p->set<string>("Gradient BF Name",          "Grad BF");
    p->set<string>("Weights Name",          "Weights");
    p->set<string>("Jacobian Name",          "Jacobian");
    p->set<string>("Jacobian Inv Name",          "Jacobian Inv");
    p->set<string>("Jacobian Det Name",          "Jacobian Det");
    p->set< RCP<Intrepid2::Cubature<PHX::Device> > >("Cubature", cubature);
    p->set< RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> > > ("Intrepid2 Basis", intrepidBasis);

    p->set<std::size_t>("spatialDim", spatialDim);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);

    Teuchos::ParameterList& paramList = params->sublist("Shallow Water Problem");
    p->set<Teuchos::ParameterList*>("Shallow Water Problem", &paramList);

    //Output
    p->set<std::string>("Residual Name",       resid_names[0]);

    ev = rcp(new Aeras::ShallowWaterResid<EvalT,AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  { // Aeras surface height for shallow water equations (hs) 

    RCP<ParameterList> p = rcp(new ParameterList("Aeras Surface Height"));

    //Input
    p->set<std::string>("Spherical Coord Name", "Lat-Long");
    
    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Aeras Surface Height");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);
  
    //Output
    p->set<std::string>("Aeras Surface Height QP Variable Name", "Aeras Surface Height");

    ev = rcp(new Aeras::SurfaceHeight<EvalT,AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
    
  }
  { // Aeras hyperviscosity for shallow water equations

    RCP<ParameterList> p = rcp(new ParameterList("Shallow Water Hyperviscosity"));

    //Input
    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Shallow Water Problem");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    //Output
    p->set<std::string>("Hyperviscosity Name", "Shallow Water Hyperviscosity");
    ev = rcp(new Aeras::ShallowWaterHyperViscosity<EvalT,AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }
 

  { // Aeras source for shallow water equations

    RCP<ParameterList> p = rcp(new ParameterList("Shallow Water Source"));

    //Input
    p->set<std::string>("Spherical Coord Name", "Lat-Long");
    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Shallow Water Problem");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    //Output
    p->set<std::string>("Shallow Water Source QP Variable Name", "Shallow Water Source");
    ev = rcp(new Aeras::ShallowWaterSource<EvalT,AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }
  
 
/*
  { // Aeras viscosity
    RCP<ParameterList> p = rcp(new ParameterList("Aeras Viscosity"));

    //Input
    p->set<std::string>("Coordinate Vector Name", "Coord Vec");
    p->set<std::string>("Gradient QP Variable Name", "Velocity Gradient");
    p->set<std::string>("Temperature Name", "Temperature");
    p->set<std::string>("Flow Factor Name", "Flow Factor");
    
    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Aeras Viscosity");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);
  
    //Output
    p->set<std::string>("Aeras Viscosity QP Variable Name", "Aeras Viscosity");

    ev = rcp(new Aeras::ViscosityFO<EvalT,AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }
*/



  if (fieldManagerChoice == Albany::BUILD_RESID_FM)  {
    PHX::Tag<typename EvalT::ScalarT> res_tag("Scatter ShallowWater", dl->dummy);
    fm0.requireField<EvalT>(res_tag);
  }
  else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM) {
    Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
    return respUtils.constructResponses(fm0, *responseList, Teuchos::null, stateMgr);
  }


  return Teuchos::null;
}
#endif // AERAS_SHALLOWWATERPROBLEM_HPP
