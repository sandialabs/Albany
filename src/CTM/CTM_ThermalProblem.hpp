#ifndef CTM_THERMAL_PROBLEM_HPP
#define CTM_THERMAL_PROBLEM_HPP

#include "CTM_Teuchos.hpp"
#include <Albany_AbstractProblem.hpp>
#include <PHAL_AlbanyTraits.hpp>
#include <MaterialDatabase.h>
#include <Phalanx.hpp>

namespace CTM {

class ThermalProblem : public Albany::AbstractProblem {

  public:

    /// \brief Convenience typedef.
    typedef Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> FC;

    /// \brief Constructor.
    /// \param params The parameterlist that defines this Albany problem.
    /// \param param_lib We ignore this.
    /// \param num_dims The number of spatial dimensions of the problem.
    /// \param comm The Teuchos communicator object.
    ThermalProblem(
        const RCP<ParameterList>& params,
        RCP<ParamLib> const& param_lib,
        const int num_dims,
        RCP<const Teuchos::Comm<int> >& comm);

    /// \brief Explicitly prohibit copying.
    ThermalProblem(const ThermalProblem&) = delete;

    /// \brief Explicitly prohibit assignment.
    ThermalProblem& operator=(const ThermalProblem&) = delete;

    /// \brief Destructor.
    ~ThermalProblem();

    /// \brief Get the number of spatial dimensions for this problem.
    int spatialDimension() const {return num_dims;}

    /// \brief Build the problem.
    /// \param mesh_specs An array of mesh specs structs defining the
    /// discretization-specific parameters.
    /// \param state_mgr The state manager object.
    /// \details This constructs the Phalanx field managers and registers
    /// the appropriate Phalanx evaluators in those field managers.
    void buildProblem(
        ArrayRCP<RCP<Albany::MeshSpecsStruct> > mesh_specs,
        Albany::StateManager& state_mgr);

    /// \brief Construct the volumetric evaluators.
    /// \param fm The field manager to register evaluators into.
    /// \param mesh_specs The mesh specs struct for this subset of the mesh.
    /// \param fm_choice We assume this is always the residual choice
    /// \param response_list We ignore this.
    /// \details This constructs the Phalanx field managers responsible
    /// for volumetric contributions to the residual vector and the
    /// Jacobian matrix by calling constructEvaluators for the appropriate
    /// evaluation types.
    Teuchos::Array<RCP<const PHX::FieldTag> > buildEvaluators(
        PHX::FieldManager<PHAL::AlbanyTraits>& fm,
        const Albany::MeshSpecsStruct& mesh_specs,
        Albany::StateManager& state_mgr,
        Albany::FieldManagerChoice fm_choice,
        const RCP<ParameterList>& response_list);

    /// \brief Construct the evaluators for a specific evaluation type.
    /// \details See buildEvaluators. This performs the field manager
    /// registration.
    template <typename EvalT>
    RCP<const PHX::FieldTag> constructEvaluators(
        PHX::FieldManager<PHAL::AlbanyTraits>& fm,
        const Albany::MeshSpecsStruct& mesh_specs,
        Albany::StateManager& state_mgr,
        Albany::FieldManagerChoice fm_choice,
        const RCP<ParameterList>& response_list);

    /// \brief Construct the dirichlet evaluators.
    /// \details This constructs the Phalanx field managers responsible for
    /// contributions to the residual vector and Jacobian matrix from the
    /// Dirichlet boundary conditions.
    void constructDirichletEvaluators(
        const Albany::MeshSpecsStruct& mesh_specs);

    /// \brief Construct the nuemann evaluators.
    /// \details This constructs the Phalanx field managers responsible for
    /// contributions to the residual vector and Jacobian matrix from the
    /// Neumann boundary conditions.
    void constructNeumannEvaluators(
        const RCP<Albany::MeshSpecsStruct>& mesh_specs);

    /// \brief I don't know what this does yet.
    void getAllocatedStates(
        ArrayRCP<ArrayRCP<RCP<FC> > > old_state,
        ArrayRCP<ArrayRCP<RCP<FC> > > new_state) const;

  private:

    int num_dims;
    RCP<Albany::Layouts> dl;
    RCP<LCM::MaterialDatabase> material_db;
    ArrayRCP<ArrayRCP<RCP<FC> > > old_state;
    ArrayRCP<ArrayRCP<RCP<FC> > > new_state;

};

}

#include <Albany_EvaluatorUtils.hpp>
#include <Intrepid2_FieldContainer.hpp>
#include <Intrepid2_DefaultCubatureFactory.hpp>
#include <Shards_CellTopology.hpp>

template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag> CTM::ThermalProblem::constructEvaluators(
    PHX::FieldManager<PHAL::AlbanyTraits>& fm,
    const Albany::MeshSpecsStruct& mesh_specs,
    Albany::StateManager& state_mgr,
    Albany::FieldManagerChoice fm_choice,
    const RCP<ParameterList>& response_list) {

  // convenience typedefs
  typedef RCP<Intrepid2::Basis<RealType, Intrepid2::FieldContainer_Kokkos<
    RealType, PHX::Layout, PHX::Device> > > Intrepid2Basis;

  typedef Intrepid2::DefaultCubatureFactory<RealType, Intrepid2::FieldContainer_Kokkos<
    RealType, PHX::Layout, PHX::Device> > CubatureFactory;

  typedef RCP<Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<
    RealType, PHX::Layout, PHX::Device> > > Cubature;

  // get the name of the current element block
  auto eb_name = mesh_specs.ebName;

  // define cell topology
  RCP<shards::CellTopology> cell_type =
    rcp(new shards::CellTopology(&mesh_specs.ctd));

  // get the intrepid basis for the cell topology
  Intrepid2Basis basis = Albany::getIntrepid2Basis(mesh_specs.ctd);

  // get the cubature
  CubatureFactory cub_factory;
  Cubature cubature = cub_factory.create(
      *cell_type, mesh_specs.cubatureDegree);

  // define a layouts structure
  const int num_nodes = basis->getCardinality();
  const int ws_size = mesh_specs.worksetSize;
  const int num_qps = cubature->getNumPoints();
  const int num_vtx = cell_type->getNodeCount();
  dl = rcp(new Albany::Layouts(ws_size, num_vtx, num_nodes, num_qps, num_dims));

  *out << "Field Dimensions: Workset= " << ws_size
    << ", Vertices= " << num_vtx
    << ", Nodes= " << num_nodes
    << ", QPs= " << num_qps
    << ", Dim= " << num_dims << std::endl;

}

#endif
