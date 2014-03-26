//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
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
    Teuchos::RCP<Albany::Layouts> dl;

  };

}

#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_CubaturePolylib.hpp"
#include "Intrepid_CubatureTensor.hpp"

#include "Shards_CellTopology.hpp"

#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_EvaluatorUtils.hpp"
#include "Albany_ResponseUtilities.hpp"
#include "PHAL_Neumann.hpp"

#include "Aeras_ShallowWaterResid.hpp"
#include "Aeras_SurfaceHeight.hpp"
#include "Aeras_Atmosphere.hpp"

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
  
  RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > >
    intrepidBasis = Albany::getIntrepidBasis(meshSpecs.ctd);
  RCP<shards::CellTopology> cellType = rcp(new shards::CellTopology(shards::getCellTopologyData< shards::Quadrilateral<4> >()));
  
  const int numNodes = intrepidBasis->getCardinality();
  const int worksetSize = meshSpecs.worksetSize;
  
  RCP <Intrepid::CubaturePolylib<RealType> > polylib = rcp(new Intrepid::CubaturePolylib<RealType>(meshSpecs.cubatureDegree, meshSpecs.cubatureRule));
  std::vector< Teuchos::RCP<Intrepid::Cubature<RealType> > > cubatures(2, polylib); 
  RCP <Intrepid::Cubature<RealType> > cubature = rcp( new Intrepid::CubatureTensor<RealType>(cubatures));
//  Regular Gauss Quadrature.
//  Intrepid::DefaultCubatureFactory<RealType> cubFactory;
//  RCP <Intrepid::Cubature<RealType> > cubature = cubFactory.create(*cellType, meshSpecs.cubatureDegree);


  const int numQPts     = cubature->getNumPoints();
  const int numVertices = cellType->getNodeCount();
  int vecDim = spatialDim;
  
  *out << "Field Dimensions: Workset=" << worksetSize 
       << ", Vertices= " << numVertices
       << ", Nodes= "    << numNodes
       << ", QuadPts= "  << numQPts
       << ", Spatial Dim= "  << spatialDim 
       << ", Model Dim= "  << modelDim 
       << ", vecDim= "   << vecDim << std::endl;
  
   dl = rcp(new Albany::Layouts(worksetSize,numVertices,numNodes,numQPts, modelDim, vecDim));
   Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);

   // Temporary variable used numerous times below
   Teuchos::RCP<PHX::Evaluator<AlbanyTraits> > ev;

   // Define Field Names

  Teuchos::ArrayRCP<std::string> dof_names(1);
  Teuchos::ArrayRCP<std::string> dof_names_dot(1);
  Teuchos::ArrayRCP<std::string> resid_names(1);
  dof_names[0] = "Flow State";
  dof_names_dot[0] = dof_names[0]+"_dot";
  resid_names[0] = "ShallowWater Residual";

  // Construct Standard FEM evaluators for Vector equation
  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructGatherSolutionEvaluator(true, dof_names, dof_names_dot));

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
    p->set< RCP<Intrepid::Cubature<RealType> > >("Cubature", cubature);
 
    p->set< RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > > 
        ("Intrepid Basis", intrepidBasis);
 
    p->set<RCP<shards::CellTopology> >("Cell Type", cellType);
    // Outputs: BF, weightBF, Grad BF, weighted-Grad BF, all in physical space
    p->set<string>("Spherical Coord Name",       "Lat-Long");
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
  else {
  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructComputeBasisFunctionsEvaluator(cellType, intrepidBasis, cubature));
  }

  { // ShallowWater Resid
    RCP<ParameterList> p = rcp(new ParameterList("Shallow Water Resid"));
   
    //Input
    p->set<std::string>("Weighted BF Name", "wBF");
    p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
    p->set<std::string>("QP Variable Name", dof_names[0]);
    p->set<std::string>("Nodal Variable Name", dof_names[0]);
    p->set<std::string>("QP Time Derivative Variable Name", dof_names_dot[0]);
    p->set<std::string>("Gradient QP Variable Name", "Flow State Gradient");
    p->set<std::string>("Aeras Surface Height QP Variable Name", "Aeras Surface Height");
    p->set<std::string>("Shallow Water Source QP Variable Name", "Shallow Water Source");
    p->set<string>("Coordinate Vector Name",          "Coord Vec");
    p->set<string>("Spherical Coord Name",       "Lat-Long");

    p->set<string>("Gradient BF Name",          "Grad BF");
    p->set<string>("Weights Name",          "Weights");
    p->set<string>("Jacobian Name",          "Jacobian");
    p->set<string>("Jacobian Inv Name",          "Jacobian Inv");
    p->set<string>("Jacobian Det Name",          "Jacobian Det");
    p->set< RCP<Intrepid::Cubature<RealType> > >("Cubature", cubature);
    p->set< RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > > ("Intrepid Basis", intrepidBasis);

    p->set<std::size_t>("spatialDim", spatialDim);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);

    Teuchos::ParameterList& paramList = params->sublist("Shallow Water Problem");
    p->set<Teuchos::ParameterList*>("Shallow Water Problem", &paramList);

    //Output
    p->set<std::string>("Residual Name", "Pre Atmosphere Residual");

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

  { // Aeras Atmosphere for shallow water equations 

    RCP<ParameterList> p = rcp(new ParameterList("Aeras Atmosphere"));

    p->set<int>("Number of Tracers", 5),
    p->set<int>("Number of Levels" ,30),

    //Input
    p->set<std::string>("Spherical Coord Name", "Lat-Long");
    p->set<std::string>("Coordinate Vector Name", "Coord Vec");
    p->set<std::string>("QP Variable Name", dof_names[0]);
    p->set<std::string>("Residual Name In", "Pre Atmosphere Residual");
    
    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Aeras Atmosphere");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);
  
    //Output
    p->set<std::string>("Tracer Vector Old Name", "Tracer Vector Old");
    p->set<std::string>("Tracer Vector New Name", "Tracer Vector New");
    p->set<std::string>("Residual Name",       resid_names[0]);

    ev = rcp(new Aeras::Atmosphere<EvalT,AlbanyTraits>(*p,dl));
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
