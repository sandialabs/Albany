//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ADVECTION_PROBLEM_HPP
#define ADVECTION_PROBLEM_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_TestForException.hpp"

#include "Albany_AbstractProblem.hpp"

#include "Phalanx.hpp"
#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"
#include "Albany_ProblemUtils.hpp"
#include "QCAD_MaterialDatabase.hpp"

namespace Albany {

class AdvectionProblem : public AbstractProblem {

  public:
 
    AdvectionProblem(
      const Teuchos::RCP<Teuchos::ParameterList>& params,
      const Teuchos::RCP<ParamLib>& param_lib,
      const int num_dims,
      Teuchos::RCP<const Teuchos::Comm<int> >& commT);

    ~AdvectionProblem();

    int spatialDimension() const { return num_dims_; }

    void buildProblem(
        Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
        StateManager& stateMgr);

    Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> > buildEvaluators(
        PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
        const Albany::MeshSpecsStruct& meshSpecs,
        Albany::StateManager& stateMgr,
        Albany::FieldManagerChoice fmchoice,
        const Teuchos::RCP<Teuchos::ParameterList>& responseList);

    Teuchos::RCP<const Teuchos::ParameterList>
    getValidProblemParameters() const;

  private:

    AdvectionProblem(const AdvectionProblem&);
    
    AdvectionProblem& operator=(const AdvectionProblem&);

  public:

    template <typename EvalT>
    Teuchos::RCP<const PHX::FieldTag> constructEvaluators(
        PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
        const Albany::MeshSpecsStruct& meshSpecs,
        Albany::StateManager& stateMgr,
        Albany::FieldManagerChoice fmchoice,
        const Teuchos::RCP<Teuchos::ParameterList>& responseList);

    void constructDirichletEvaluators(
        const std::vector<std::string>& nodeSetIDs);

    void constructNeumannEvaluators(
        const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs);

  protected:

    int num_dims_;

    Teuchos::RCP<QCAD::MaterialDatabase> material_db_;

    Teuchos::RCP<Albany::Layouts> dl_;

};

}

#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "Albany_Utils.hpp"
#include "Albany_BCUtils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_EvaluatorUtils.hpp"
#include "Albany_ResponseUtilities.hpp"
#include "PHAL_SaveStateField.hpp"

template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
Albany::AdvectionProblem::constructEvaluators(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fieldManagerChoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList) {

  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::ParameterList;
  using PHX::DataLayout;
  using PHX::MDALayout;
  using std::vector;
  using std::string;
  using PHAL::AlbanyTraits;

  const CellTopologyData* const elem_top = &meshSpecs.ctd;
  std::string eb_name = meshSpecs.ebName;
  std::string material_name;
  material_name = material_db_->getElementBlockParam<std::string>(
      eb_name, "material");

  RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> >
    intrepid_basis = Albany::getIntrepid2Basis(*elem_top);

  RCP<shards::CellTopology> elem_type =
    rcp(new shards::CellTopology(elem_top));

  Intrepid2::DefaultCubatureFactory cub_factory;

  RCP<Intrepid2::Cubature<PHX::Device> > elem_cubature =
    cub_factory.create<PHX::Device, RealType, RealType>(
        *elem_type, meshSpecs.cubatureDegree);


  const int workset_size = meshSpecs.worksetSize;
  const int num_vertices = elem_type->getNodeCount();
  const int num_nodes = intrepid_basis->getCardinality();
  const int num_qps = elem_cubature->getNumPoints();

  *out  << "Field Dimensions: Workset=" << workset_size
        << ", Vertices= " << num_vertices
        << ", Nodes= " << num_nodes
        << ", QuadPts= " << num_qps
        << ", Dim= " << num_dims_
        << std::endl;

  dl_ = rcp(new Albany::Layouts(
        workset_size, num_vertices, num_nodes, num_qps, num_dims_));

  Teuchos::ArrayRCP<std::string> dof_names(1);
  Teuchos::ArrayRCP<std::string> resid_names(1);
  dof_names[0] = "Phi";
  resid_names[0] = "Phi Residual";

  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> eval_utils(dl_);
  Teuchos::RCP<PHX::Evaluator<AlbanyTraits> > ev;

  fm0.template registerEvaluator<EvalT>(
      eval_utils.constructGatherSolutionEvaluator_noTransient(
        false, dof_names));

  fm0.template registerEvaluator<EvalT>(
      eval_utils.constructScatterResidualEvaluator(
        false, resid_names));

  fm0.template registerEvaluator<EvalT>(
      eval_utils.constructDOFInterpolationEvaluator(dof_names[0]));

  fm0.template registerEvaluator<EvalT>(
      eval_utils.constructDOFGradInterpolationEvaluator(dof_names[0]));

  fm0.template registerEvaluator<EvalT>(
      eval_utils.constructGatherCoordinateVectorEvaluator());

  fm0.template registerEvaluator<EvalT>(
      eval_utils.constructComputeBasisFunctionsEvaluator(
        elem_type, intrepid_basis, elem_cubature));

}

#endif
