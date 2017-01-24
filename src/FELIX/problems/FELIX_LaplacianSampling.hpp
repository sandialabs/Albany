//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_LAPLACIAN_SAMPLING_PROBLEM_HPP
#define FELIX_LAPLACIAN_SAMPLING_PROBLEM_HPP 1

#include <type_traits>

#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Phalanx.hpp"
#include "Shards_CellTopology.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_AbstractProblem.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_EvaluatorUtils.hpp"
#include "Albany_ResponseUtilities.hpp"

#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"
#include "PHAL_AlbanyTraits.hpp"

#include "PHAL_LoadStateField.hpp"
#include "PHAL_LoadSideSetStateField.hpp"

#include "FELIX_LaplacianRegularizationResidual.hpp"

//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN

namespace FELIX
{

/*!
 * \brief Abstract interface for representing a 1-D finite element
 * problem.
 */
class LaplacianSampling : public Albany::AbstractProblem
{
public:

  //! Default constructor
  LaplacianSampling (const Teuchos::RCP<Teuchos::ParameterList>& params,
            const Teuchos::RCP<Teuchos::ParameterList>& discParams,
            const Teuchos::RCP<ParamLib>& paramLib,
            const int numDim_);

  //! Destructor
  ~LaplacianSampling();

  //! Return number of spatial dimensions
  virtual int spatialDimension() const { return numDim; }

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

private:

  //! Private to prohibit copying
  LaplacianSampling(const LaplacianSampling&);

  //! Private to prohibit copying
  LaplacianSampling& operator=(const LaplacianSampling&);

public:

  //! Main problem setup routine. Not directly called, but indirectly by following functions
  template <typename EvalT> Teuchos::RCP<const PHX::FieldTag>
  constructEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                       const Albany::MeshSpecsStruct& meshSpecs,
                       Albany::StateManager& stateMgr,
                       Albany::FieldManagerChoice fmchoice,
                       const Teuchos::RCP<Teuchos::ParameterList>& responseList);

  void constructDirichletEvaluators(const Albany::MeshSpecsStruct& meshSpecs);
  void constructNeumannEvaluators(const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs);

protected:

  Teuchos::RCP<shards::CellTopology> cellType;

  Teuchos::RCP<Intrepid2::Cubature<PHX::Device> >  cellCubature;

  Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> > cellBasis;

  int numDim;
  Teuchos::RCP<Albany::Layouts> dl;

  //! Discretization parameters
  Teuchos::RCP<Teuchos::ParameterList> discParams;

  std::string elementBlockName;
};

} // Namespace FELIX

// ================================ IMPLEMENTATION ============================ //

template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
FELIX::LaplacianSampling::constructEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                      const Albany::MeshSpecsStruct& meshSpecs,
                                      Albany::StateManager& stateMgr,
                                      Albany::FieldManagerChoice fieldManagerChoice,
                                      const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);

  int offset=0;

  Albany::StateStruct::MeshFieldEntity entity;
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;
  Teuchos::RCP<Teuchos::ParameterList> p;

  // ---------------------------- Registering state variables ------------------------- //

  std::string stateName, fieldName, param_name;


  stateName = "weighted_normal_sample";
  entity = Albany::StateStruct::NodalDataToElemNode;
  p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName, true, &entity);
  p->set<std::string>("Field Name", stateName);
  ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
  fm0.template registerEvaluator<EvalT>(ev);


  // ----------  Define Field Names ----------- //
  Teuchos::ArrayRCP<std::string> dof_names(1);
  Teuchos::ArrayRCP<std::string> resid_names(1);
  dof_names[0] = "prior_sample";


  // ------------------- Interpolations and utilities ------------------ //

  // Gather solution field
  ev = evalUtils.constructGatherSolutionEvaluator_noTransient(false, dof_names, offset);
  fm0.template registerEvaluator<EvalT> (ev);

  // Interpolate solution field
  ev = evalUtils.constructDOFInterpolationEvaluator(dof_names[0]);
  fm0.template registerEvaluator<EvalT> (ev);

  // Interpolate solution gradient
  ev = evalUtils.constructDOFGradInterpolationEvaluator(dof_names[0]);
  fm0.template registerEvaluator<EvalT> (ev);

  // Scatter residual
  resid_names[0] = "Laplacian Residual";
  ev = evalUtils.constructScatterResidualEvaluator(false, resid_names, offset, "Scatter Laplacian Sampling");
  fm0.template registerEvaluator<EvalT> (ev);

  //----- Gather Coordinate Vector (general parameters)
  ev = evalUtils.constructGatherCoordinateVectorEvaluator();
  fm0.template registerEvaluator<EvalT> (ev);


  // Map to physical frame
  ev = evalUtils.constructMapToPhysicalFrameEvaluator(cellType, cellCubature);
  fm0.template registerEvaluator<EvalT> (ev);

  // Compute basis funcitons
  ev = evalUtils.constructComputeBasisFunctionsEvaluator(cellType, cellBasis, cellCubature);
  fm0.template registerEvaluator<EvalT> (ev);

  // -------------------------------- FELIX evaluators ------------------------- //

  // --- FO Stokes Resid --- //
  {
  p = Teuchos::rcp(new Teuchos::ParameterList("Laplacian Resid"));

  //Input
  p->set<std::string>("Gradient BF Name", "Grad BF");
  p->set<std::string>("Field Variable Name", "prior_sample");
  p->set<std::string>("Field Gradient Variable Name", "prior_sample Gradient");
  p->set<std::string>("Forcing Field Name", "weighted_normal_sample");
  p->set<std::string>("Weighted Measure Name", "Weights");
  p->set<double>("Mass Coefficient", params->sublist("FELIX Laplacian Regularization").get<double>("Mass Coefficient",1.0));
  p->set<double>("Laplacian Coefficient", params->sublist("FELIX Laplacian Regularization").get<double>("Laplacian Coefficient",1.0));
  p->set<Teuchos::RCP<shards::CellTopology> >("Cell Type", cellType);

    //Output
    p->set<std::string>("Laplacian Residual Name", "Laplacian Residual");

    ev = Teuchos::rcp(new FELIX::LaplacianRegularizationResidual<EvalT,PHAL::AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (fieldManagerChoice == Albany::BUILD_RESID_FM)
  {
    // Require scattering of residual
    PHX::Tag<typename EvalT::ScalarT> res_tag("Scatter Laplacian Sampling", dl->dummy);
    fm0.requireField<EvalT>(res_tag);
  }
  else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM)
  {
    // ----------------------- Responses --------------------- //
    Teuchos::RCP<Teuchos::ParameterList> paramList = Teuchos::rcp(new Teuchos::ParameterList("Param List"));
    paramList->set<Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

    Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
    return respUtils.constructResponses(fm0, *responseList, paramList, stateMgr);
  }

  return Teuchos::null;
}

#endif // FELIX_STOKES_FO_PROBLEM_HPP
