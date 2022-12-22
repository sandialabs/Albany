//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_SIDE_LAPLACIAN_HPP
#define ALBANY_SIDE_LAPLACIAN_HPP 1

#include "Shards_CellTopology.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_AbstractProblem.hpp"
#include "Albany_EvaluatorUtils.hpp"
#include "Albany_ResponseUtilities.hpp"
#include "Albany_GeneralPurposeFieldsNames.hpp"

#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"
#include "PHAL_DOFCellToSide.hpp"
#include "PHAL_DummyResidual.hpp"

#include "PHAL_SideLaplacianResidual.hpp"

//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN

namespace Albany
{

/*!
 * \brief  A 2D problem for the subglacial hydrology
 */
class SideLaplacian : public Albany::AbstractProblem
{
public:

  //! Default constructor
  SideLaplacian (const Teuchos::RCP<Teuchos::ParameterList>& params,
                 const Teuchos::RCP<ParamLib>& paramLib);

  //! Destructor
  virtual ~SideLaplacian();

  //! Return number of spatial dimensions
  virtual int spatialDimension () const
  {
      return numDim;
  }
  
  //! Get boolean telling code if SDBCs are utilized  
  virtual bool useSDBCs() const {return use_sdbcs_; }

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

  //! Each problem must generate it's list of valid parameters
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

  Teuchos::RCP<shards::CellTopology>                                cellType;
  Teuchos::RCP<Intrepid2::Cubature<PHX::Device>>                    cellCubature;
  Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>>   cellBasis;

  Teuchos::RCP<shards::CellTopology>                                sideType;
  Teuchos::RCP<Intrepid2::Cubature<PHX::Device>>                    sideCubature;
  Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>>   sideBasis;

  Teuchos::RCP<Albany::Layouts> dl, dl_side;

  std::string sideSetName;

  std::string cellEBName;
  std::string sideEBName;
  
  /// Boolean marking whether SDBCs are used 
  bool use_sdbcs_; 
};

// ===================================== IMPLEMENTATION ======================================= //

template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
SideLaplacian::constructEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                    const Albany::MeshSpecsStruct& meshSpecs,
                                    Albany::StateManager& stateMgr,
                                    Albany::FieldManagerChoice fieldManagerChoice,
                                    const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  if (numDim==2) {
    return constructEvaluators2D<EvalT> (fm0,meshSpecs,stateMgr,fieldManagerChoice,responseList);
  } else {
    return constructEvaluators3D<EvalT> (fm0,meshSpecs,stateMgr,fieldManagerChoice,responseList);
  }
}

template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
SideLaplacian::constructEvaluators2D (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                      const Albany::MeshSpecsStruct& meshSpecs,
                                      Albany::StateManager& stateMgr,
                                      Albany::FieldManagerChoice fieldManagerChoice,
                                      const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Using the utility for the common evaluators
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);

  // Service variables for registering state variables and evaluators
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;
  Teuchos::RCP<Teuchos::ParameterList> p;

  // -------------------- Starting evaluators construction and registration ------------------------ //

  ev = evalUtils.constructGatherSolutionEvaluator_noTransient(false, dof_names);
  fm0.template registerEvaluator<EvalT> (ev);

  ev = evalUtils.constructComputeBasisFunctionsEvaluator(cellType, cellBasis, cellCubature);
  fm0.template registerEvaluator<EvalT> (ev);

  ev = evalUtils.constructGatherCoordinateVectorEvaluator();
  fm0.template registerEvaluator<EvalT> (ev);

  ev = evalUtils.constructScatterResidualEvaluator(false, resid_names, 0, "Scatter SideLaplacian");
  fm0.template registerEvaluator<EvalT> (ev);

  ev = evalUtils.constructMapToPhysicalFrameEvaluator(cellType, cellCubature);
  fm0.template registerEvaluator<EvalT> (ev);

  // ------- DOF interpolations -------- //

  // Solution
  ev = evalUtils.constructDOFInterpolationEvaluator(dof_names[1]);
  fm0.template registerEvaluator<EvalT> (ev);

  // Solution Gradient
  ev = evalUtils.constructDOFGradInterpolationEvaluator(dof_names[1]);
  fm0.template registerEvaluator<EvalT> (ev);

  // ------- Side Laplacian Residual -------- //
  p = Teuchos::rcp(new Teuchos::ParameterList("Side Laplacian Residual"));

  //Input
  p->set<std::string> ("Coordinate Vector Variable Name", Albany::coord_vec_name);
  p->set<std::string> ("BF Variable Name", Albany::bf_name);
  p->set<std::string> ("Weighted Measure Variable Name", Albany::weights_name);
  p->set<std::string> ("Gradient BF Variable Name", Albany::grad_bf_name);
  p->set<std::string> ("Solution QP Variable Name", dof_names[1]);
  p->set<std::string> ("Solution Gradient QP Variable Name", dof_names[1] + " Gradient");
  p->set<bool> ("Side Equation", false);

  //Output
  p->set<std::string> ("Residual Variable Name",resid_names[1]);

  ev = Teuchos::rcp(new PHAL::SideLaplacianResidual<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  // -------- Dummy Residual for 2nd equation --------- //
  p = Teuchos::rcp(new Teuchos::ParameterList("Dummy Residual"));

  //Input
  p->set<std::string> ("Solution Variable Name", dof_names[0]);
  p->set<std::string> ("Residual Variable Name",resid_names[0]);

  ev = Teuchos::rcp(new PHAL::DummyResidual<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  // ----------------------------------------------------- //

  Teuchos::RCP<Teuchos::ParameterList> paramList = Teuchos::rcp(new Teuchos::ParameterList("Param List"));
  {
     // response
    Teuchos::RCP<const Albany::MeshSpecsStruct> meshSpecsPtr = Teuchos::rcpFromRef(meshSpecs);
    paramList->set<Teuchos::RCP<const Albany::MeshSpecsStruct> >("Mesh Specs Struct", meshSpecsPtr);
    paramList->set<Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);
  }

  if (fieldManagerChoice == Albany::BUILD_RESID_FM)
  {
    PHX::Tag<typename EvalT::ScalarT> res_tag("Scatter SideLaplacian", dl->dummy);
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
SideLaplacian::constructEvaluators3D (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                      const Albany::MeshSpecsStruct& meshSpecs,
                                      Albany::StateManager& stateMgr,
                                      Albany::FieldManagerChoice fieldManagerChoice,
                                      const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  int offsetU = 1;

  // Using the utility for the common evaluators
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);

  // Service variables for registering state variables and evaluators
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;
  Teuchos::RCP<Teuchos::ParameterList> p;

  // -------------------- Starting evaluators construction and registration ------------------------ //

  ev = evalUtils.constructGatherSolutionEvaluator_noTransient(false, dof_names[0]);
  fm0.template registerEvaluator<EvalT> (ev);

  ev = evalUtils.constructGatherSolutionSideEvaluator(dof_names[1],sideSetName,cellType,offsetU);
  fm0.template registerEvaluator<EvalT> (ev);

  ev = evalUtils.constructComputeBasisFunctionsEvaluator(cellType, cellBasis, cellCubature);
  fm0.template registerEvaluator<EvalT> (ev);

  // 2D basis function
  ev = evalUtils.constructComputeBasisFunctionsSideEvaluator(cellType, sideBasis, sideCubature, sideSetName);
  fm0.template registerEvaluator<EvalT> (ev);

  ev = evalUtils.constructGatherCoordinateVectorEvaluator();
  fm0.template registerEvaluator<EvalT> (ev);

  ev = evalUtils.constructScatterSideEqnResidualEvaluator(cellType, sideSetName, false, resid_names[1], offsetU, "Scatter SideLaplacian");
  fm0.template registerEvaluator<EvalT> (ev);

  ev = evalUtils.constructScatterResidualEvaluator(false, resid_names[0], 0, "Scatter Dummy");
  fm0.template registerEvaluator<EvalT> (ev);

  ev = evalUtils.constructMapToPhysicalFrameEvaluator(cellType, cellCubature);
  fm0.template registerEvaluator<EvalT> (ev);

  // ------- DOF interpolations -------- //

  // Solution Gradient
  ev = evalUtils.constructDOFGradInterpolationEvaluator(dof_names[1]);
  fm0.template registerEvaluator<EvalT> (ev);

  // Solution Side
  ev = evalUtils.constructDOFInterpolationSideEvaluator(dof_names[1],sideSetName);
  fm0.template registerEvaluator<EvalT> (ev);

  // Solution Gradient Side
  std::string grad_side_name = dof_names[1] + "_gradient_" + sideSetName;
  ev = evalUtils.constructDOFGradInterpolationSideEvaluator(dof_names[1],sideSetName,grad_side_name);
  fm0.template registerEvaluator<EvalT> (ev);

  //---- Restrict vertex coordinates from cell-based to cell-side-based
  ev = evalUtils.getMSTUtils().constructDOFCellToSideEvaluator(Albany::coord_vec_name,sideSetName,"Vertex Vector",cellType,Albany::coord_vec_name + "_" + sideSetName);
  fm0.template registerEvaluator<EvalT> (ev);

  // ------- Side Laplacian Residual -------- //
  p = Teuchos::rcp(new Teuchos::ParameterList("Side Laplacian Residual"));

  //Input
  p->set<std::string> ("Coordinate Vector Variable Name", Albany::coord_vec_name + "_" + sideSetName);
  p->set<std::string> ("BF Variable Name", Albany::bf_name + "_" + sideSetName);
  p->set<std::string> ("Weighted Measure Variable Name", Albany::weighted_measure_name + "_" + sideSetName);
  p->set<std::string> ("Metric Name", Albany::metric_name + "_" + sideSetName);
  p->set<std::string> ("Gradient BF Variable Name", Albany::grad_bf_name + "_" + sideSetName);
  p->set<std::string> ("Solution Variable Name", dof_names[1]);
  p->set<std::string> ("Solution QP Variable Name", dof_names[1]);
  p->set<std::string> ("Solution Gradient QP Variable Name", grad_side_name);
  p->set<std::string> ("Side Set Name",sideSetName);
  p->set<bool> ("Side Equation", true);
  p->set<Teuchos::RCP<shards::CellTopology> >("Cell Type",cellType);

  //Output
  p->set<std::string> ("Residual Variable Name",resid_names[1]);

  ev = Teuchos::rcp(new PHAL::SideLaplacianResidual<EvalT,PHAL::AlbanyTraits>(*p,dl_side));
  fm0.template registerEvaluator<EvalT>(ev);

  // -------- Dummy Residual for 2nd equation --------- //
  p = Teuchos::rcp(new Teuchos::ParameterList("Dummy Residual"));

  //Input
  p->set<std::string> ("Solution Variable Name", dof_names[0]);
  p->set<std::string> ("Residual Variable Name",resid_names[0]);

  ev = Teuchos::rcp(new PHAL::DummyResidual<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  // ----------------------------------------------------- //

  Teuchos::RCP<Teuchos::ParameterList> paramList = Teuchos::rcp(new Teuchos::ParameterList("Param List"));
  {
     // response
    Teuchos::RCP<const Albany::MeshSpecsStruct> meshSpecsPtr = Teuchos::rcpFromRef(meshSpecs);
    paramList->set<Teuchos::RCP<const Albany::MeshSpecsStruct> >("Mesh Specs Struct", meshSpecsPtr);
    paramList->set<Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);
  }

  if (fieldManagerChoice == Albany::BUILD_RESID_FM)
  {
    PHX::Tag<typename EvalT::ScalarT> res1_tag("Scatter SideLaplacian", dl->dummy);
    fm0.requireField<EvalT>(res1_tag);
    PHX::Tag<typename EvalT::ScalarT> res2_tag("Scatter Dummy", dl->dummy);
    fm0.requireField<EvalT>(res2_tag);
  }
  else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM)
  {
    Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);

    return respUtils.constructResponses(fm0, *responseList, paramList, stateMgr);
  }

  return Teuchos::null;
}

} // namespace Albany

#endif // ALBANY_SIDE_LAPLACIAN_HPP
