#ifndef CTM_MECHANICS_PROBLEM_HPP
#define CTM_MECHANICS_PROBLEM_HPP

#include "CTM_Teuchos.hpp"
#include "Albany_ProblemUtils.hpp"
#include <Albany_AbstractProblem.hpp>
#include <PHAL_AlbanyTraits.hpp>
#include <MaterialDatabase.h>
#include <Phalanx.hpp>

namespace CTM {

    class MechanicsProblem : public Albany::AbstractProblem {
    public:

        /// \brief Convenience typedef.
        //typedef Kokkos::DynRankView<RealType, PHX::Device> FC;
        //typedef Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> FC;

        /// \brief Constructor.
        /// \param params The parameterlist that defines this Albany problem.
        /// \param param_lib We ignore this.
        /// \param num_dims The number of spatial dimensions of the problem.
        /// \param comm The Teuchos communicator object.
        MechanicsProblem(
                const RCP<ParameterList>& params,
                RCP<ParamLib> const& param_lib,
                const int num_dims,
                RCP<const Teuchos::Comm<int> >& comm);

        /// \brief Explicitly prohibit copying.
        MechanicsProblem(const MechanicsProblem&) = delete;

        /// \brief Explicitly prohibit assignment.
        MechanicsProblem& operator=(const MechanicsProblem&) = delete;

        /// \brief Destructor.
        ~MechanicsProblem();

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
//
#include "PHAL_NSMaterialProperty.hpp"
#include "PHAL_Source.hpp"
#include "PHAL_SaveStateField.hpp"
//
#include "FieldNameMap.hpp"
//
#include "MechanicsResidual.hpp"
#include "CurrentCoords.hpp"
//#include "TvergaardHutchinson.hpp"
#include "MeshSizeField.hpp"
//#include "SurfaceCohesiveResidual.hpp"

// Constitutive Model Interface and parameters
#include "Kinematics.hpp"
#include "ConstitutiveModelInterface.hpp"
#include "ConstitutiveModelParameters.hpp"
#include "Strain.hpp"
#include "FirstPK.hpp"

template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag> CTM::MechanicsProblem::constructEvaluators(
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

    // Define Field Names
    // generate the field name map
    LCM::FieldNameMap field_name_map(false);
    Teuchos::RCP<std::map < std::string, std::string>> fnm =
            field_name_map.getMap();
    const std::string cauchy = (*fnm)["Cauchy_Stress"];
    const std::string firstPK = (*fnm)["FirstPK"];
    const std::string temperature = (*fnm)["Temperature"];
    const std::string mech_source = (*fnm)["Mechanical_Source"];
    const std::string defgrad = (*fnm)["F"];
    const std::string J = (*fnm)["J"];


    // evaluator utility
    Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
    // Temporary variable used numerous times below
    Teuchos::RCP<PHX::Evaluator < PHAL::AlbanyTraits>> ev;

    // register variable names
    Teuchos::ArrayRCP<std::string> dof_names(1);
    Teuchos::ArrayRCP<std::string> dof_names_dot(1);
    Teuchos::ArrayRCP<std::string> dof_names_dotdot(1);
    Teuchos::ArrayRCP<std::string> resid_names(1);
    dof_names[0] = "Displacement";
    dof_names_dot[0] = "Velocity";
    dof_names_dotdot[0] = "Acceleration";
    resid_names[0] = dof_names[0] + " Residual";

    int offset = 0;
    if (isTransient_) {
        fm0.template registerEvaluator<EvalT>
                (evalUtils.constructGatherSolutionEvaluator_withAcceleration(
                true,
                dof_names,
                dof_names_dot,
                dof_names_dotdot));
        //
        fm0.template registerEvaluator<EvalT>
                (evalUtils.constructDOFVecInterpolationEvaluator(dof_names_dot[0]));
        //
        fm0.template registerEvaluator<EvalT>
                (evalUtils.constructDOFVecInterpolationEvaluator(dof_names_dotdot[0]));
        //
        fm0.template registerEvaluator<EvalT>
                (evalUtils.constructDOFVecGradInterpolationEvaluator(dof_names_dot[0]));
        //
    } else {
        fm0.template registerEvaluator<EvalT>
                (evalUtils.constructGatherSolutionEvaluator_noTransient(true,
                dof_names));
    }

    fm0.template registerEvaluator<EvalT>
            (evalUtils.constructGatherCoordinateVectorEvaluator());


    fm0.template registerEvaluator<EvalT>
            (evalUtils.constructDOFVecInterpolationEvaluator(dof_names[0]));

    fm0.template registerEvaluator<EvalT>
            (evalUtils.constructDOFVecGradInterpolationEvaluator(dof_names[0]));

    fm0.template registerEvaluator<EvalT>
            (evalUtils.constructMapToPhysicalFrameEvaluator(cell_type,
            cubature));

    fm0.template registerEvaluator<EvalT>
            (evalUtils.constructComputeBasisFunctionsEvaluator(cell_type,
            intrepidBasis,
            cubature));

    fm0.template registerEvaluator<EvalT>
            (evalUtils.constructScatterResidualEvaluator(true,
            resid_names));
    offset += num_dims;
    //

    { // Current Coordinates
        Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
                new Teuchos::ParameterList("Current Coordinates"));
        p->set<std::string>("Reference Coordinates Name", "Coord Vec");
        p->set<std::string>("Displacement Name", "Displacement");
        p->set<std::string>("Current Coordinates Name", "Current Coordinates");
        ev = Teuchos::rcp(
                new LCM::CurrentCoords<EvalT, PHAL::AlbanyTraits>(*p, dl));
        fm0.template registerEvaluator<EvalT>(ev);
    }

    {
        Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
                new Teuchos::ParameterList);

        p->set<std::string>("Material Property Name", temperature);
        p->set<Teuchos::RCP < PHX::DataLayout >> ("Data Layout", dl->qp_scalar);
        p->set<std::string>("Coordinate Vector Name", "Coord Vec");
        p->set<Teuchos::RCP < PHX::DataLayout >> (
                "Coordinate Vector Data Layout",
                dl->qp_vector);

        p->set<Teuchos::RCP < ParamLib >> ("Parameter Library", paramLib);
        Teuchos::ParameterList& paramList = params->sublist("Temperature");

        // This evaluator is called to set a constant temperature when "Variable Type"
        // is set to "Constant." It is also called when "Variable Type" is set to
        // "Time Dependent." There are two "Type" variables in the PL - "Type" and
        // "Variable Type". For the last case, lets set "Type" to "Time Dependent" to hopefully
        // make the evaluator call a little more general (GAH)
        std::string temp_type = paramList.get<std::string>("Variable Type", "None");
        if (temp_type == "Time Dependent") {

            paramList.set<std::string>("Type", temp_type);

        }

        p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

        ev = Teuchos::rcp(
                new PHAL::NSMaterialProperty<EvalT, PHAL::AlbanyTraits>(*p));
        fm0.template registerEvaluator<EvalT>(ev);
    }

    { // Constitutive Model Parameters
        Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
                new Teuchos::ParameterList("Constitutive Model Parameters"));
        std::string matName = material_db_->getElementBlockParam<std::string>(
                eb_name, "material");
        Teuchos::ParameterList& param_list =
                material_db_->getElementBlockSublist(eb_name, matName);
        p->set<std::string>("Temperature Name", temperature);
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
                new Teuchos::ParameterList("Constitutive Model Interface"));
        std::string matName = material_db_->getElementBlockParam<std::string>(
                eb_name, "material");
        Teuchos::ParameterList& param_list =
                material_db_->getElementBlockSublist(eb_name, matName);

        // FIXME: figure out how to do this better
        param_list.set<bool>("Have Temperature", false);
        p->set<std::string>("Temperature Name", temperature);
        param_list.set<bool>("Have Temperature", true);

        param_list.set<Teuchos::RCP<std::map < std::string, std::string>>>(
                "Name Map",
                fnm);
        p->set<Teuchos::ParameterList*>("Material Parameters", &param_list);

        Teuchos::RCP<LCM::ConstitutiveModelInterface<EvalT, PHAL::AlbanyTraits>>
                cmiEv =
                Teuchos::rcp(
                new LCM::ConstitutiveModelInterface<EvalT, PHAL::AlbanyTraits>(
                *p,
                dl));
        fm0.template registerEvaluator<EvalT>(cmiEv);

        // register state variables
        for (int sv = 0; sv < cmiEv->getNumStateVars(); ++sv) {
            cmiEv->fillStateVariableStruct(sv);
            p = state_mgr.registerStateVariable(cmiEv->getName(),
                    cmiEv->getLayout(),
                    dl->dummy,
                    eb_name,
                    cmiEv->getInitType(),
                    cmiEv->getInitValue(),
                    cmiEv->getStateFlag(),
                    cmiEv->getOutputFlag());
            ev = Teuchos::rcp(
                    new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
            fm0.template registerEvaluator<EvalT>(ev);
        }
    }

    { // Kinematics quantities
        Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
                new Teuchos::ParameterList("Kinematics"));

        // send in integration weights and the displacement gradient
        p->set<std::string>("Weights Name", "Weights");
        p->set<Teuchos::RCP < PHX::DataLayout >> (
                "QP Scalar Data Layout",
                dl->qp_scalar);
        p->set<std::string>("Gradient QP Variable Name", "Displacement Gradient");
        p->set<Teuchos::RCP < PHX::DataLayout >> (
                "QP Tensor Data Layout",
                dl->qp_tensor);

        //Outputs: F, J
        p->set<std::string>("DefGrad Name", defgrad); //dl_->qp_tensor also
        p->set<std::string>("DetDefGrad Name", J);
        p->set<Teuchos::RCP < PHX::DataLayout >> (
                "QP Scalar Data Layout",
                dl->qp_scalar);

        ev = Teuchos::rcp(
                new LCM::Kinematics<EvalT, PHAL::AlbanyTraits>(*p, dl));
        fm0.template registerEvaluator<EvalT>(ev);
    }

    { // Strain
        Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(new Teuchos::ParameterList("Strain"));

        //Input
        p->set<std::string>("Gradient QP Variable Name", "Displacement Gradient");

        //Output
        p->set<std::string>("Strain Name", "Strain");

        ev = rcp(new LCM::Strain<EvalT, PHAL::AlbanyTraits>(*p, dl));
        fm0.template registerEvaluator<EvalT>(ev);
    }

    {
        // convert Cauchy stress to first Piola-Kirchhoff
        Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
                new Teuchos::ParameterList("First PK Stress"));
        //Input
        p->set<std::string>("Stress Name", cauchy);
        p->set<std::string>("DefGrad Name", defgrad);

        //Output
        p->set<std::string>("First PK Stress Name", firstPK);
        p->set<bool>("Small Strain", false);

        p->set<Teuchos::RCP < ParamLib >> ("Parameter Library", paramLib);

        ev = Teuchos::rcp(new LCM::FirstPK<EvalT, PHAL::AlbanyTraits>(*p, dl));
        fm0.template registerEvaluator<EvalT>(ev);
    }


    { // Residual
        Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
                new Teuchos::ParameterList("Displacement Residual"));
        //Input
        p->set<std::string>("Stress Name", firstPK);
        p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
        p->set<std::string>("Weighted BF Name", "wBF");
        p->set<std::string>("Acceleration Name", "Acceleration");
        p->set<bool>("Disable Dynamics", true);
        p->set<Teuchos::RCP < ParamLib >> ("Parameter Library", paramLib);
        //Output
        p->set<std::string>("Residual Name", "Displacement Residual");
        ev = Teuchos::rcp(
                new LCM::MechanicsResidual<EvalT, PHAL::AlbanyTraits>(*p, dl));
        fm0.template registerEvaluator<EvalT>(ev);
    }

    if (fm_choice == Albany::BUILD_RESID_FM) {

        Teuchos::RCP<const PHX::FieldTag> ret_tag;

        PHX::Tag<typename EvalT::ScalarT > res_tag("Scatter", dl->dummy);
        fm0.requireField<EvalT>(res_tag);
        ret_tag = res_tag.clone();

        return ret_tag;
    }


    return Teuchos::null;
}
#endif
