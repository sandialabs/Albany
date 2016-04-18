//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_ELLIPTIC_2D_HPP
#define FELIX_ELLIPTIC_2D_HPP 1

#include "Intrepid2_FieldContainer.hpp"
#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Phalanx.hpp"
#include "Shards_CellTopology.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_AbstractProblem.hpp"
#include "Albany_EvaluatorUtils.hpp"
#include "Albany_ResponseUtilities.hpp"

#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"
#include "PHAL_DOFCellToSide.hpp"

#include "FELIX_Elliptic2DResidual.hpp"

//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN

namespace FELIX
{

/*!
 * \brief  A 2D problem for the subglacial hydrology
 */
class Elliptic2D : public Albany::AbstractProblem
{
public:

  //! Default constructor
  Elliptic2D (const Teuchos::RCP<Teuchos::ParameterList>& params,
             const Teuchos::RCP<ParamLib>& paramLib,
             const int numDimensions);

  //! Destructor
  virtual ~Elliptic2D();

  //! Return number of spatial dimensions
  virtual int spatialDimension () const
  {
      return numDim;
  }

  //! Build the PDE instantiations, boundary conditions, and initial solution
  virtual void buildProblem (Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
                             Albany::StateManager& stateMgr);

  // Build evaluators
  virtual Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
  buildEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                   const Albany::MeshSpecsStruct& meshSpecs,
                   Albany::StateManager& stateMgr,
                   Albany::FieldManagerChoice fmchoice,
                   const Teuchos::RCP<Teuchos::ParameterList>& responseList);

  //! Each problem must generate it's list of valide parameters
  Teuchos::RCP<const Teuchos::ParameterList> getValidProblemParameters() const;

  //! Main problem setup routine. Not directly called, but indirectly by buildEvaluators
  template <typename EvalT> Teuchos::RCP<const PHX::FieldTag>
  constructEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                       const Albany::MeshSpecsStruct& meshSpecs,
                       Albany::StateManager& stateMgr,
                       Albany::FieldManagerChoice fmchoice,
                       const Teuchos::RCP<Teuchos::ParameterList>& responseList);

  template <typename EvalT> Teuchos::RCP<const PHX::FieldTag>
  constructEvaluators2D (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                       const Albany::MeshSpecsStruct& meshSpecs,
                       Albany::StateManager& stateMgr,
                       Albany::FieldManagerChoice fmchoice,
                       const Teuchos::RCP<Teuchos::ParameterList>& responseList);

  template <typename EvalT> Teuchos::RCP<const PHX::FieldTag>
  constructEvaluators3D (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                       const Albany::MeshSpecsStruct& meshSpecs,
                       Albany::StateManager& stateMgr,
                       Albany::FieldManagerChoice fmchoice,
                       const Teuchos::RCP<Teuchos::ParameterList>& responseList);

  void constructDirichletEvaluators (const Albany::MeshSpecsStruct& meshSpecs);
protected:

  int numDim;

  Teuchos::ArrayRCP<std::string> dof_names;
  Teuchos::ArrayRCP<std::string> resid_names;

  Teuchos::RCP<shards::CellTopology> cellType;
  Teuchos::RCP<shards::CellTopology> sideType;

  Teuchos::RCP<Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout,PHX::Device> > >  cellCubature;
  Teuchos::RCP<Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout,PHX::Device> > >  sideCubature;

  Teuchos::RCP<Intrepid2::Basis<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > > cellBasis;
  Teuchos::RCP<Intrepid2::Basis<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > > sideBasis;

  Teuchos::RCP<Albany::Layouts> dl, dl_side;

  std::string sideSetName;

  std::string cellEBName;
  std::string sideEBName;
  std::string basalEBName;
};

// ===================================== IMPLEMENTATION ======================================= //

template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
Elliptic2D::constructEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                 const Albany::MeshSpecsStruct& meshSpecs,
                                 Albany::StateManager& stateMgr,
                                 Albany::FieldManagerChoice fieldManagerChoice,
                                 const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  if (numDim==2)
  {
    return constructEvaluators2D<EvalT> (fm0,meshSpecs,stateMgr,fieldManagerChoice,responseList);
  }
  return constructEvaluators3D<EvalT> (fm0,meshSpecs,stateMgr,fieldManagerChoice,responseList);
}

template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
Elliptic2D::constructEvaluators2D (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                   const Albany::MeshSpecsStruct& meshSpecs,
                                   Albany::StateManager& stateMgr,
                                   Albany::FieldManagerChoice fieldManagerChoice,
                                   const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  using Teuchos::RCP;
  using Teuchos::rcp;
  using PHX::DataLayout;
  using PHX::MDALayout;
  using PHAL::AlbanyTraits;

  int offset = 0;

  // Using the utility for the common evaluators
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);

  // Service variables for registering state variables and evaluators
  Albany::StateStruct::MeshFieldEntity entity;
  RCP<PHX::Evaluator<AlbanyTraits> > ev;
  RCP<Teuchos::ParameterList> p;

  // -------------------- Starting evaluators construction and registration ------------------------ //

  ev = evalUtils.constructGatherSolutionEvaluator_noTransient(false, dof_names);
  fm0.template registerEvaluator<EvalT> (ev);

  ev = evalUtils.constructComputeBasisFunctionsEvaluator(cellType, cellBasis, cellCubature);
  fm0.template registerEvaluator<EvalT> (ev);

  ev = evalUtils.constructGatherCoordinateVectorEvaluator();
  fm0.template registerEvaluator<EvalT> (ev);

  ev = evalUtils.constructScatterResidualEvaluator(false, resid_names, offset, "Scatter Elliptic2D");
  fm0.template registerEvaluator<EvalT> (ev);

  ev = evalUtils.constructMapToPhysicalFrameEvaluator(cellType, cellCubature);
  fm0.template registerEvaluator<EvalT> (ev);

  // ------- DOF interpolations -------- //

  // Solution
  ev = evalUtils.constructDOFInterpolationEvaluator(dof_names[0]);
  fm0.template registerEvaluator<EvalT> (ev);

  // Solution Gradient
  ev = evalUtils.constructDOFGradInterpolationEvaluator(dof_names[0]);
  fm0.template registerEvaluator<EvalT> (ev);

  // ------- Elliptic2D Residual Elliptic Eqn-------- //
  p = rcp(new Teuchos::ParameterList("Elliptic2D Residual"));

  //Input
  p->set<std::string> ("Coordinate Vector Variable Name", "Coord Vec");
  p->set<std::string> ("BF Variable Name", "BF");
  p->set<std::string> ("Weighted BF Variable Name", "wBF");
  p->set<std::string> ("Weighted Gradient BF Variable Name", "wGrad BF");
  p->set<std::string> ("Solution QP Variable Name", dof_names[0]);
  p->set<std::string> ("Solution Gradient QP Variable Name", dof_names[0] + " Gradient");
  p->set<bool> ("Side Equation", false);

  //Output
  p->set<std::string> ("Residual Variable Name",resid_names[0]);

  ev = rcp(new FELIX::Elliptic2DResidual<EvalT,AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  // ----------------------------------------------------- //

  RCP<Teuchos::ParameterList> paramList = rcp(new Teuchos::ParameterList("Param List"));
  {
     // response
    RCP<const Albany::MeshSpecsStruct> meshSpecsPtr = Teuchos::rcpFromRef(meshSpecs);
    paramList->set<RCP<const Albany::MeshSpecsStruct> >("Mesh Specs Struct", meshSpecsPtr);
    paramList->set<RCP<ParamLib> >("Parameter Library", paramLib);
  }

  if (fieldManagerChoice == Albany::BUILD_RESID_FM)
  {
    PHX::Tag<typename EvalT::ScalarT> res_tag("Scatter Elliptic2D", dl->dummy);
    fm0.requireField<EvalT>(res_tag);
  }
  else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM)
  {
    Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);

    return respUtils.constructResponses(fm0, *responseList, paramList, stateMgr);
  }

  return Teuchos::null;
}

template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
Elliptic2D::constructEvaluators3D (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                   const Albany::MeshSpecsStruct& meshSpecs,
                                   Albany::StateManager& stateMgr,
                                   Albany::FieldManagerChoice fieldManagerChoice,
                                   const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  using Teuchos::RCP;
  using Teuchos::rcp;
  using PHX::DataLayout;
  using PHX::MDALayout;
  using PHAL::AlbanyTraits;

  int offset = 0;

  // Using the utility for the common evaluators
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);

  // Service variables for registering state variables and evaluators
  Albany::StateStruct::MeshFieldEntity entity;
  RCP<PHX::Evaluator<AlbanyTraits> > ev;
  RCP<Teuchos::ParameterList> p;

  // -------------------- Starting evaluators construction and registration ------------------------ //

  ev = evalUtils.constructGatherSolutionEvaluator_noTransient(false, dof_names);
  fm0.template registerEvaluator<EvalT> (ev);

  ev = evalUtils.constructComputeBasisFunctionsEvaluator(cellType, cellBasis, cellCubature);
  fm0.template registerEvaluator<EvalT> (ev);

  // 2D basis function
  ev = evalUtils.constructComputeBasisFunctionsSideEvaluator(cellType, sideBasis, sideCubature, sideSetName);
  fm0.template registerEvaluator<EvalT> (ev);

  ev = evalUtils.constructGatherCoordinateVectorEvaluator();
  fm0.template registerEvaluator<EvalT> (ev);

  ev = evalUtils.constructScatterResidualEvaluator(false, resid_names, offset, "Scatter Elliptic2D");
  fm0.template registerEvaluator<EvalT> (ev);

  ev = evalUtils.constructMapToPhysicalFrameEvaluator(cellType, cellCubature);
  fm0.template registerEvaluator<EvalT> (ev);

  // ------- DOF interpolations -------- //

  // Solution Gradient
  ev = evalUtils.constructDOFGradInterpolationEvaluator(dof_names[0]);
  fm0.template registerEvaluator<EvalT> (ev);

  // Solution Side
  ev = evalUtils.constructDOFInterpolationSideEvaluator(dof_names[0],sideSetName);
  fm0.template registerEvaluator<EvalT> (ev);

  // Solution Gradient Side
  ev = evalUtils.constructDOFGradInterpolationSideEvaluator(dof_names[0],sideSetName);
  fm0.template registerEvaluator<EvalT> (ev);

  // -------- Restriction of Solution to Side Field -------- /
  p = rcp(new Teuchos::ParameterList("Cell To Side"));

  p->set<std::string>("Cell Variable Name", dof_names[0]);
  p->set<std::string>("Side Variable Name", dof_names[0]);
  p->set<Teuchos::RCP<shards::CellTopology> >("Cell Type",cellType);
  p->set<std::string>("Side Set Name",sideSetName);

  ev = rcp(new PHAL::DOFCellToSide<EvalT,AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  // ------- Elliptic2D Residual Elliptic Eqn-------- //
  p = rcp(new Teuchos::ParameterList("Elliptic2D Residual"));

  //Input
  p->set<std::string> ("Coordinate Vector Variable Name", "Coord Vec");
  p->set<std::string> ("BF Variable Name", "BF "+sideSetName);
  p->set<std::string> ("Weighted Measure Variable Name", "Weighted Measure "+sideSetName);
  p->set<std::string> ("Inverse Metric Name", "Inv Metric "+sideSetName);
  p->set<std::string> ("Gradient BF Variable Name", "Grad BF "+sideSetName);
  p->set<std::string> ("Solution Variable Name", dof_names[0]);
  p->set<std::string> ("Solution QP Variable Name", dof_names[0]);
  p->set<std::string> ("Solution Gradient QP Variable Name", dof_names[0] + " Gradient");
  p->set<std::string> ("Side Set Name",sideSetName);
  p->set<bool> ("Side Equation", true);
  p->set<Teuchos::RCP<shards::CellTopology> >("Cell Type",cellType);

  //Output
  p->set<std::string> ("Residual Variable Name",resid_names[0]);

  ev = rcp(new FELIX::Elliptic2DResidual<EvalT,AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  // ----------------------------------------------------- //

  RCP<Teuchos::ParameterList> paramList = rcp(new Teuchos::ParameterList("Param List"));
  {
     // response
    RCP<const Albany::MeshSpecsStruct> meshSpecsPtr = Teuchos::rcpFromRef(meshSpecs);
    paramList->set<RCP<const Albany::MeshSpecsStruct> >("Mesh Specs Struct", meshSpecsPtr);
    paramList->set<RCP<ParamLib> >("Parameter Library", paramLib);
  }

  if (fieldManagerChoice == Albany::BUILD_RESID_FM)
  {
    PHX::Tag<typename EvalT::ScalarT> res_tag("Scatter Elliptic2D", dl->dummy);
    fm0.requireField<EvalT>(res_tag);
  }
  else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM)
  {
    Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);

    return respUtils.constructResponses(fm0, *responseList, paramList, stateMgr);
  }

  return Teuchos::null;
}

} // Namespace FELIX

#endif // FELIX_ELLIPTIC_2D_HPP
