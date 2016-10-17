#ifndef CTM_THERMAL_PROBLEM_HPP
#define CTM_THERMAL_PROBLEM_HPP

#include "CTM_Teuchos.hpp"
#include "Albany_ProblemUtils.hpp"
#include <Albany_AbstractProblem.hpp>
#include <PHAL_AlbanyTraits.hpp>
#include <MaterialDatabase.h>
#include <Phalanx.hpp>

namespace CTM {

    class ThermalProblem : public Albany::AbstractProblem {
    public:

        /// \brief Convenience typedef.
        //typedef Kokkos::DynRankView<RealType, PHX::Device> FC;
        //typedef Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> FC;

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

        int spatialDimension() const {
            return num_dims;
        }

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
        /// \param fm0_choice We assume this is always the residual choice
        /// \param response_list We ignore this.
        /// \details This constructs the Phalanx field managers responsible
        /// for volumetric contributions to the residual vector and the
        /// Jacobian matrix by calling constructEvaluators for the appropriate
        /// evaluation types.
        Teuchos::Array<RCP<const PHX::FieldTag> > buildEvaluators(
                PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                const Albany::MeshSpecsStruct& mesh_specs,
                Albany::StateManager& state_mgr,
                Albany::FieldManagerChoice fm_choice,
                const RCP<ParameterList>& response_list);

        /// \brief Construct the evaluators for a specific evaluation type.
        /// \details See buildEvaluators. This performs the field manager
        /// registration.
        template <typename EvalT>
        RCP<const PHX::FieldTag> constructEvaluators(
                PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                const Albany::MeshSpecsStruct& mesh_specs,
                Albany::StateManager& state_mgr,
                Albany::FieldManagerChoice fm_choice,
                const RCP<ParameterList>& response_list);

        /// \brief Construct the dirichlet evaluators.
        /// \details This constructs the Phalanx field managers responsible for
        /// contributions to the residual vector and Jacobian matrix from the
        /// Dirichlet boundary conditions.
        void constructDirichletEvaluators(
                const RCP<Albany::MeshSpecsStruct>& mesh_specs);

        /// \brief Construct the nuemann evaluators.
        /// \details This constructs the Phalanx field managers responsible for
        /// contributions to the residual vector and Jacobian matrix from the
        /// Neumann boundary conditions.
        void constructNeumannEvaluators(
                const RCP<Albany::MeshSpecsStruct>& mesh_specs);
        
        /// \brief Get valid parameters for this problem
        /// \details Each problem must generate it's list of valid parameters
        ///
        Teuchos::RCP<const Teuchos::ParameterList>
        getValidProblemParameters() const;

//        /// \brief I don't know what this does yet.
//        void getAllocatedStates(
//                ArrayRCP<ArrayRCP<RCP<FC> > > old_state,
//                ArrayRCP<ArrayRCP<RCP<FC> > > new_state) const;

    private:

        int num_dims;
        RCP<Albany::Layouts> dl;
        RCP<LCM::MaterialDatabase> material_db_;
        std::string materialFileName_;
        RCP<const Teuchos::Comm<int>> comm_;
//        ArrayRCP<ArrayRCP<RCP<FC> > > old_state;
//        ArrayRCP<ArrayRCP<RCP<FC> > > new_state;

    protected:

        // Source function type

        enum SOURCE_TYPE {
            SOURCE_TYPE_NONE, //! No source
            SOURCE_TYPE_INPUT, //! Source is specified in input file
            SOURCE_TYPE_MATERIAL //! Source is specified in material database
        };
        // have source term?
        bool have_source_;
        // Type of thermal source that is in effect
        SOURCE_TYPE thermal_source_;
        
        // Has the thermal source been evaluated in this element block?
        bool thermal_source_evaluated_;

        // is it a transient problem?
        bool isTransient_;
    };

}

#include <Albany_EvaluatorUtils.hpp>
#include <Intrepid2_FieldContainer.hpp>
#include <Intrepid2_DefaultCubatureFactory.hpp>
#include <Shards_CellTopology.hpp>
#include "PHAL_Source.hpp"

// Constitutive Model Interface and parameters
#include "Kinematics.hpp"
#include "ConstitutiveModelInterface.hpp"
#include "ConstitutiveModelParameters.hpp"

// Generic Transport Residual
#include "TransportResidual.hpp"

// Thermomechanics specific evaluators
#include "ThermoMechanicalCoefficients.hpp"

template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag> CTM::ThermalProblem::constructEvaluators(
        PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
        const Albany::MeshSpecsStruct& mesh_specs,
        Albany::StateManager& state_mgr,
        Albany::FieldManagerChoice fm_choice,
        const RCP<ParameterList>& response_list) {

    // convenience typedefs
    typedef Teuchos::RCP<
            Intrepid2::Basis<PHX::Device, RealType, RealType> >
            Intrepid2Basis;

    // get the name of the current element block
    auto eb_name = mesh_specs.ebName;

    // get the name of the material model to be used (and make sure there is one)
    std::string material_model_name =
            material_db_->
            getElementBlockSublist(eb_name, "Material Model").get<std::string>(
            "Model Name");
    TEUCHOS_TEST_FOR_EXCEPTION(
            material_model_name.length() == 0,
            std::logic_error,
            "A material model must be defined for block: "
            + eb_name);

    // define cell topology
    RCP<shards::CellTopology> cell_type =
            rcp(new shards::CellTopology(&mesh_specs.ctd));

    // get the intrepid basis for the cell topology
    Intrepid2Basis intrepidBasis = Albany::getIntrepid2Basis(mesh_specs.ctd);

    // get the cubature
    Intrepid2::DefaultCubatureFactory cubFactory;
    Teuchos::RCP<Intrepid2::Cubature<PHX::Device> > cubature =
            cubFactory.create<PHX::Device, RealType, RealType>(*cell_type, mesh_specs.cubatureDegree);

    // define a layouts structure
    const int num_nodes = intrepidBasis->getCardinality();
    const int ws_size = mesh_specs.worksetSize;
    const int num_qps = cubature->getNumPoints();
    const int num_vtx = cell_type->getNodeCount();
    dl = rcp(new Albany::Layouts(ws_size, num_vtx, num_nodes, num_qps, num_dims));

    /*
     *out << "Field Dimensions: Workset= " << ws_size
            << ", Vertices= " << num_vtx
            << ", Nodes= " << num_nodes
            << ", QPs= " << num_qps
            << ", Dim= " << num_dims << std::endl;
     */
    // evaluator utility
    Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
    // Temporary variable used numerous times below
    Teuchos::RCP<PHX::Evaluator < PHAL::AlbanyTraits>> ev;

    // register variable names
    Teuchos::ArrayRCP<std::string> dof_names(1);
    Teuchos::ArrayRCP<std::string> dof_names_dot(1);
    Teuchos::ArrayRCP<std::string> resid_names(1);

    dof_names[0] = "Temperature";
    dof_names_dot[0] = "Temperature Dot";
    resid_names[0] = dof_names[0] + " Residual";

    int offset = 0;
    if (isTransient_) {
        fm0.template registerEvaluator<EvalT>
                (evalUtils.constructGatherSolutionEvaluator(
                false,
                dof_names,
                dof_names_dot,
                offset));
        fm0.template registerEvaluator<EvalT>
                (evalUtils.constructDOFInterpolationEvaluator(dof_names_dot[0], offset));
    } else {
        fm0.template registerEvaluator<EvalT>
                (evalUtils.constructGatherSolutionEvaluator_noTransient(false,
                dof_names,
                offset));
    }

    fm0.template registerEvaluator<EvalT>
            (evalUtils.constructGatherCoordinateVectorEvaluator());

    fm0.template registerEvaluator<EvalT>
            (evalUtils.constructDOFInterpolationEvaluator(dof_names[0], offset));

    fm0.template registerEvaluator<EvalT>
            (evalUtils.constructDOFGradInterpolationEvaluator(dof_names[0], offset));

    fm0.template registerEvaluator<EvalT>
            (evalUtils.constructMapToPhysicalFrameEvaluator(cell_type,
            cubature));

    fm0.template registerEvaluator<EvalT>
            (evalUtils.constructComputeBasisFunctionsEvaluator(cell_type,
            intrepidBasis,
            cubature));

    fm0.template registerEvaluator<EvalT>
            (evalUtils.constructScatterResidualEvaluator(false,
            resid_names,
            offset,
            "Scatter Temperature"));

    // Heat Source in Heat Equation
    if (thermal_source_ != SOURCE_TYPE_NONE) {

        Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
                new Teuchos::ParameterList);

        p->set<std::string>("Source Name", "Heat Source");
        p->set<std::string>("Variable Name", "Temperature");
        p->set<Teuchos::RCP < PHX::DataLayout >> (
                "QP Scalar Data Layout",
                dl->qp_scalar);

        p->set<Teuchos::RCP < ParamLib >> ("Parameter Library", paramLib);

        if (thermal_source_ == SOURCE_TYPE_INPUT) { // Thermal source in input file

            Teuchos::ParameterList& paramList = params->sublist("Source Functions")
                    .sublist("Thermal Source");
            p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

            ev = Teuchos::rcp(new PHAL::Source<EvalT, PHAL::AlbanyTraits>(*p));
            fm0.template registerEvaluator<EvalT>(ev);

            thermal_source_evaluated_ = true;

        } else if (thermal_source_ == SOURCE_TYPE_MATERIAL) {

            // There may not be a source in every element block
            if (material_db_->isElementBlockSublist(eb_name, "Source Functions")) { // Thermal source in matDB

                Teuchos::ParameterList& srcParamList = material_db_->
                        getElementBlockSublist(eb_name, "Source Functions");

                if (srcParamList.isSublist("Thermal Source")) {

                    Teuchos::ParameterList& paramList = srcParamList.sublist(
                            "Thermal Source");
                    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

                    ev = Teuchos::rcp(new PHAL::Source<EvalT, PHAL::AlbanyTraits>(*p));
                    fm0.template registerEvaluator<EvalT>(ev);

                    thermal_source_evaluated_ = true;
                }
            } else // Do not evaluate heat source in TransportResidual
            {
                thermal_source_evaluated_ = false;
            }
        } else
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                "Unrecognized thermal source specified in input file");
    }

    { // Constitutive Model Parameters
        Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
                new Teuchos::ParameterList("Constitutive Model Parameters"));
        std::string matName = material_db_->getElementBlockParam<std::string>(
                eb_name, "material");
        Teuchos::ParameterList& param_list =
                material_db_->getElementBlockSublist(eb_name, matName);
        // for quantities that depends on temperature
        p->set<std::string>("Temperature Name", dof_names[0]);
        param_list.set<bool>("Have Temperature", true);

        // optional spatial dependence
        p->set<std::string>("QP Coordinate Vector Name", "Coord Vec");

        // pass through material properties
        p->set<Teuchos::ParameterList*>("Material Parameters", &param_list);

        Teuchos::RCP<LCM::ConstitutiveModelParameters<EvalT, PHAL::AlbanyTraits>>
                cmpEv =
                Teuchos::rcp(
                new LCM::ConstitutiveModelParameters<EvalT, PHAL::AlbanyTraits>(
                *p,
                dl));
        fm0.template registerEvaluator<EvalT>(cmpEv);
    }
    
    {
        Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
                new Teuchos::ParameterList("ThermoMechanical Coefficients"));

        std::string matName =
                material_db_->getElementBlockParam<std::string>(eb_name, "material");
        Teuchos::ParameterList& param_list =
                material_db_->getElementBlockSublist(eb_name, matName);
        p->set<Teuchos::ParameterList*>("Material Parameters", &param_list);

        // Input
        p->set<std::string>("Temperature Name", "Temperature");
        p->set<std::string>("Temperature Dot Name", "Temperature Dot");
        // next line is important
        p->set<std::string>("Solution Method Type", "No Continuation");
        p->set<std::string>("Thermal Conductivity Name", "Thermal Conductivity");
        p->set<std::string>("Thermal Transient Coefficient Name",
                "Thermal Transient Coefficient");

        // Output
        p->set<std::string>("Thermal Diffusivity Name", "Thermal Diffusivity");

        ev = Teuchos::rcp(
                new LCM::ThermoMechanicalCoefficients<EvalT, PHAL::AlbanyTraits>(
                *p,
                dl));
        fm0.template registerEvaluator<EvalT>(ev);
    }

    {
        Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
                new Teuchos::ParameterList("Temperature Residual"));

        // Input
        p->set<std::string>("Scalar Variable Name", "Temperature");
        p->set<std::string>("Scalar Gradient Variable Name",
                "Temperature Gradient");
        p->set<std::string>("Weights Name", "Weights");
        p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
        p->set<std::string>("Weighted BF Name", "wBF");

        // Transient
        if (isTransient_) {
            p->set<bool>("Have Transient", true);
        }
        p->set<std::string>("Scalar Dot Name", "Temperature Dot");
        p->set<std::string>("Transient Coefficient Name",
                "Thermal Transient Coefficient");
        p->set<std::string>("Solution Method Type", "No Continuation");

        // Diffusion
        p->set<bool>("Have Diffusion", true);
        p->set<std::string>("Diffusivity Name", "Thermal Diffusivity");

        // Thermal Source (internal energy generation)
        if (thermal_source_evaluated_) {
            p->set<bool>("Have Second Source", true);
            p->set<std::string>("Second Source Name", "Heat Source");
        }

        // Output
        p->set<std::string>("Residual Name", "Temperature Residual");

        ev = Teuchos::rcp(
                new LCM::TransportResidual<EvalT, PHAL::AlbanyTraits>(*p, dl));
        fm0.template registerEvaluator<EvalT>(ev);
    }
    
    if (fm_choice == Albany::BUILD_RESID_FM) {
        Teuchos::RCP<const PHX::FieldTag> ret_tag;

        PHX::Tag<typename EvalT::ScalarT > temperature_tag("Scatter Temperature",
                dl->dummy);
        fm0.requireField<EvalT>(temperature_tag);
        ret_tag = temperature_tag.clone();

        return ret_tag;
    }

    return Teuchos::null;
    
}

#endif
