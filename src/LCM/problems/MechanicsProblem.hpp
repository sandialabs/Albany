//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef MECHANICSPROBLEM_HPP
#define MECHANICSPROBLEM_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_AbstractProblem.hpp"

#include "Phalanx.hpp"
#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"
#include "PHAL_AlbanyTraits.hpp"

namespace Albany {

  //----------------------------------------------------------------------------
  ///
  /// \brief Definition for the Mechanics Problem
  ///
  class MechanicsProblem : public Albany::AbstractProblem {
  public:

    typedef Intrepid::FieldContainer<RealType> FC;

    ///
    /// Default constructor
    ///
    MechanicsProblem(const Teuchos::RCP<Teuchos::ParameterList>& params,
                     const Teuchos::RCP<ParamLib>& param_lib,
                     const int num_dims,
                     const Teuchos::RCP<const Epetra_Comm>& comm);
    ///
    /// Destructor
    ///
    virtual
    ~MechanicsProblem();

    ///
    Teuchos::RCP<std::map<std::string, std::string> >
    constructFieldNameMap(bool surface_flag);

    ///
    /// Return number of spatial dimensions
    ///
    virtual
    int
    spatialDimension() const { return num_dims_; }

    ///
    /// Build the PDE instantiations, boundary conditions, initial solution
    ///
    virtual
    void
    buildProblem(Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >
                 meshSpecs,
                 StateManager& stateMgr);

    ///
    /// Build evaluators
    ///
    virtual
    Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
    buildEvaluators(PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                    const Albany::MeshSpecsStruct& meshSpecs,
                    Albany::StateManager& stateMgr,
                    Albany::FieldManagerChoice fmchoice,
                    const Teuchos::RCP<Teuchos::ParameterList>& responseList);

    ///
    /// Each problem must generate it's list of valid parameters
    ///
    Teuchos::RCP<const Teuchos::ParameterList>
    getValidProblemParameters() const;

    ///
    /// Retrieve the state data
    ///
    void
    getAllocatedStates(Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<FC> > >
                       old_state,
                       Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<FC> > >
                       new_state) const;

    //----------------------------------------------------------------------------
  private:

    ///
    /// Private to prohibit copying
    ///
    MechanicsProblem(const MechanicsProblem&);

    ///
    /// Private to prohibit copying
    ///
    MechanicsProblem& operator=(const MechanicsProblem&);

    //----------------------------------------------------------------------------
  public:

    ///
    /// Main problem setup routine.
    /// Not directly called, but indirectly by following functions
    ///
    template <typename EvalT>
    Teuchos::RCP<const PHX::FieldTag>
    constructEvaluators(PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                        const Albany::MeshSpecsStruct& meshSpecs,
                        Albany::StateManager& stateMgr,
                        Albany::FieldManagerChoice fmchoice,
                        const Teuchos::RCP<Teuchos::ParameterList>&
                        responseList);

    ///
    /// Setup for the dirichlet BCs
    ///
    void
    constructDirichletEvaluators(const Albany::MeshSpecsStruct& meshSpecs);

    ///
    /// Setup for the traction BCs
    ///
    void
    constructNeumannEvaluators(const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs);

    //----------------------------------------------------------------------------
  protected:

    ///
    /// Enumerated type describing how a variable appears
    ///
    enum MECH_VAR_TYPE {
      MECH_VAR_TYPE_NONE,      //! Variable does not appear
      MECH_VAR_TYPE_CONSTANT,  //! Variable is a constant
      MECH_VAR_TYPE_DOF        //! Variable is a degree-of-freedom
    };

    ///
    /// Accessor for variable type
    ///
    void getVariableType(Teuchos::ParameterList& param_list,
                         const std::string& default_type,
                         MECH_VAR_TYPE& variable_type,
                         bool& have_variable,
                         bool& have_equation);

    ///
    /// Conversion from enum to string
    ///
    std::string variableTypeToString(const MECH_VAR_TYPE variable_type);

    ///
    /// Construct a string for consistent output with surface elements
    ///
    //std::string stateString(std::string, bool);

    ///
    /// Boundary conditions on source term
    ///
    bool have_source_;

    ///
    /// num of dimensions
    ///
    int num_dims_;

    ///
    /// number of integration points
    ///
    int num_pts_;

    ///
    /// number of element nodes
    ///
    int num_nodes_;

    ///
    /// number of element vertices
    ///
    int num_vertices_;

    ///
    /// Type of mechanics variable (disp or acc)
    ///
    MECH_VAR_TYPE mech_type_;

    ///
    /// Type of temperature variable
    ///
    MECH_VAR_TYPE temperature_type_;

    ///
    /// Type of pressure variable
    ///
    MECH_VAR_TYPE pressure_type_;

    ///
    /// Type of concentration variable
    ///
    MECH_VAR_TYPE transport_type_;

    ///
    /// Type of concentration variable
    ///
    MECH_VAR_TYPE hydrostress_type_;

    ///
    /// Have mechanics
    ///
    bool have_mech_;

    ///
    /// Have temperature
    ///
    bool have_temperature_;

    ///
    /// Have pressure
    ///
    bool have_pressure_;

    ///
    /// Have transport
    ///
    bool have_transport_;

    ///
    /// Have transport
    ///
    bool have_hydrostress_;

    ///
    /// Have mechanics equation
    ///
    bool have_mech_eq_;

    ///
    /// Have temperature equation
    ///
    bool have_temperature_eq_;

    ///
    /// Have pressure equation
    ///
    bool have_pressure_eq_;

    ///
    /// Have transport equation
    ///
    bool have_transport_eq_;

    ///
    /// Have projected hydrostatic stress term
    /// in transport equation
    ///
    bool have_hydrostress_eq_;

    ///
    /// Have a Peridynamics block
    ///
    bool have_peridynamics_;

    ///
    /// Data layouts
    ///
    Teuchos::RCP<Albany::Layouts> dl_;

    ///
    /// RCP to matDB object
    ///
    Teuchos::RCP<QCAD::MaterialDatabase> material_db_;

    ///
    /// old state data
    ///
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<FC> > > old_state_;

    ///
    /// new state data
    ///
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<FC> > > new_state_;

  };
  //----------------------------------------------------------------------------
}


#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_ResponseUtilities.hpp"
#include "Albany_EvaluatorUtils.hpp"

#include "PHAL_NSMaterialProperty.hpp"
#include "PHAL_Source.hpp"
#include "PHAL_SaveStateField.hpp"
//#include "PHAL_ThermalConductivity.hpp"

#include "FieldNameMap.hpp"

#include "ElasticModulus.hpp"
#include "PoissonsRatio.hpp"
#include "DefGrad.hpp"
#include "PisdWdF.hpp"
#include "HardeningModulus.hpp"
#include "YieldStrength.hpp"
#include "TLElasResid.hpp"
#include "MechanicsResidual.hpp"
#include "Time.hpp"
#include "MooneyRivlinDamage.hpp"
#include "MooneyRivlin_Incompressible.hpp"
#include "RecoveryModulus.hpp"
#include "SurfaceBasis.hpp"
#include "SurfaceVectorJump.hpp"
#include "SurfaceVectorGradient.hpp"
#include "SurfaceScalarJump.hpp"
// #include "SurfaceScalarGradient.hpp"
#include "SurfaceScalarGradientOperator.hpp"
#include "SurfaceVectorResidual.hpp"
#include "CurrentCoords.hpp"
#include "TvergaardHutchinson.hpp"
#include "SurfaceCohesiveResidual.hpp"

// Constitutive Model Interface and parameters
#include "Kinematics.hpp"
#include "ConstitutiveModelInterface.hpp"
#include "ConstitutiveModelParameters.hpp"

// Generic Transport Residual
#include "TransportResidual.hpp"

// Thermomechanics specific evaluators
#include "ThermoMechanicalCoefficients.hpp"

// Poromechanics specific evaluators
#include "GradientElementLength.hpp"
#include "BiotCoefficient.hpp"
#include "BiotModulus.hpp"
#include "KCPermeability.hpp"
#include "Porosity.hpp"
#include "TLPoroPlasticityResidMass.hpp"
#include "TLPoroStress.hpp"
#include "SurfaceTLPoroMassResidual.hpp"

// Thermohydromechanics specific evaluators
#include "ThermoPoroPlasticityResidMass.hpp"
#include "ThermoPoroPlasticityResidEnergy.hpp"
#include "MixtureThermalExpansion.hpp"
#include "MixtureSpecificHeat.hpp"

// Hydrogen transport specific evaluators
#include "ScalarL2ProjectionResidual.hpp"
#include "SurfaceL2ProjectionResidual.hpp"
#include "HDiffusionDeformationMatterResidual.hpp"
#include "SurfaceHDiffusionDefResidual.hpp"
#include "LatticeDefGrad.hpp"
#include "TransportCoefficients.hpp"

//------------------------------------------------------------------------------
template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
Albany::MechanicsProblem::
constructEvaluators(PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
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
  using PHAL::AlbanyTraits;
  using shards::CellTopology;
  using shards::getCellTopologyData;

  // get the name of the current element block
  std::string eb_name = meshSpecs.ebName;

  // get the name of the material model to be used (and make sure there is one)
  std::string materialModelName =
    material_db_->
    getElementBlockSublist(eb_name,"Material Model").get<std::string>("Model Name");
  TEUCHOS_TEST_FOR_EXCEPTION(materialModelName.length()==0, std::logic_error,
                             "A material model must be defined for block: "
                             +eb_name);

#ifdef ALBANY_VERBOSE
  *out << "In MechanicsProblem::constructEvaluators" << std::endl;
  *out << "element block name: " << eb_name << std::endl;
  *out << "material model name: " << materialModelName << std::endl;
#endif

  // define cell topologies
  RCP<CellTopology> comp_cellType =
    rcp(new CellTopology(getCellTopologyData<shards::Tetrahedron<11> >()));
  RCP<shards::CellTopology> cellType =
    rcp(new CellTopology (&meshSpecs.ctd));

  // Check if we are setting the composite tet flag
  bool composite = false;
  if ( material_db_->isElementBlockParam(eb_name,"Use Composite Tet 10") )
    composite =
      material_db_->getElementBlockParam<bool>(eb_name,
                                               "Use Composite Tet 10");

  // Surface element checking
  bool surface_element = false;
  bool cohesive_element = false;
  RealType thickness = 0.0;
  if ( material_db_->isElementBlockParam(eb_name,"Surface Element") ){
    surface_element =
      material_db_->getElementBlockParam<bool>(eb_name,"Surface Element");
    if ( material_db_->isElementBlockParam(eb_name,"Cohesive Element") )
      cohesive_element =
        material_db_->getElementBlockParam<bool>(eb_name,
                                               "Cohesive Element");
  }

  if (surface_element) {
    if ( material_db_->
         isElementBlockParam(eb_name,"Localization thickness parameter") ) {
      thickness =
        material_db_->
        getElementBlockParam<RealType>(eb_name,"Localization thickness parameter");
    } else {
      thickness = 0.1;
    }
  }

  std::string msg =
    "Surface elements are not yet supported with the composite tet";
  // FIXME, really need to check for WEDGE_12 topologies
  TEUCHOS_TEST_FOR_EXCEPTION(composite && surface_element,
                             std::logic_error,
                             msg);

  // get the intrepid basis for the given cell topology
  RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > >
    intrepidBasis = Albany::getIntrepidBasis(meshSpecs.ctd, composite);

  if (composite &&
      meshSpecs.ctd.dimension==3 &&
      meshSpecs.ctd.node_count==10) cellType = comp_cellType;

  Intrepid::DefaultCubatureFactory<RealType> cubFactory;
  RCP <Intrepid::Cubature<RealType> > cubature =
    cubFactory.create(*cellType, meshSpecs.cubatureDegree);


  // FIXME, this could probably go into the ProblemUtils
  // just like the call to getIntrepidBasis
  RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > >
    surfaceBasis;
  RCP<shards::CellTopology> surfaceTopology;
  RCP<Intrepid::Cubature<RealType> > surfaceCubature;
  if (surface_element)
  {
#ifdef ALBANY_VERBOSE
    *out << "In Surface Element Logic" << std::endl;
#endif

    std::string name = meshSpecs.ctd.name;
    if ( name == "Triangle_3" || name == "Quadrilateral_4" )
    {
      surfaceBasis =
        rcp(new Intrepid::Basis_HGRAD_LINE_C1_FEM<RealType,Intrepid::FieldContainer<RealType> >() );
      surfaceTopology =
        rcp(new shards::CellTopology( shards::getCellTopologyData<shards::Line<2> >()) );
      surfaceCubature =
        cubFactory.create(*surfaceTopology, meshSpecs.cubatureDegree);
    }
    else if ( name == "Wedge_6" )
    {
      surfaceBasis =
        rcp(new Intrepid::Basis_HGRAD_TRI_C1_FEM<RealType, Intrepid::FieldContainer<RealType> >() );
      surfaceTopology =
        rcp(new shards::CellTopology( shards::getCellTopologyData<shards::Triangle<3> >()) );
      surfaceCubature =
        cubFactory.create(*surfaceTopology, meshSpecs.cubatureDegree);
    }
    else if ( name == "Hexahedron_8" )
    {
      surfaceBasis =
        rcp(new Intrepid::Basis_HGRAD_QUAD_C1_FEM<RealType, Intrepid::FieldContainer<RealType> >() );
      surfaceTopology =
        rcp(new shards::CellTopology( shards::getCellTopologyData<shards::Quadrilateral<4> >()) );
      surfaceCubature =
        cubFactory.create(*surfaceTopology, meshSpecs.cubatureDegree);
    }

#ifdef ALBANY_VERBOSE
    *out << "surfaceCubature->getNumPoints(): "
         << surfaceCubature->getNumPoints() << std::endl;
    *out << "surfaceCubature->getDimension(): "
         << surfaceCubature->getDimension() << std::endl;
#endif
  }

  // Note that these are the volume element quantities
  num_nodes_ = intrepidBasis->getCardinality();
  const int workset_size = meshSpecs.worksetSize;

#ifdef ALBANY_VERBOSE
  *out << "Setting num_pts_, surface element is "
       << surface_element << std::endl;
#endif
  num_dims_ = cubature->getDimension();
  if ( !surface_element ) {
    num_pts_ = cubature->getNumPoints();
  } else {
    num_pts_ = surfaceCubature->getNumPoints();
  }
  num_vertices_ = num_nodes_;

#ifdef ALBANY_VERBOSE
  *out << "Field Dimensions: Workset=" << workset_size
       << ", Vertices= " << num_vertices_
       << ", Nodes= " << num_nodes_
       << ", QuadPts= " << num_pts_
       << ", Dim= " << num_dims_ << std::endl;
#endif

  // Construct standard FEM evaluators with standard field names
  dl_ =
    rcp(new Albany::Layouts(workset_size,num_vertices_,num_nodes_,num_pts_,num_dims_));
  msg = "Data Layout Usage in Mechanics problems assume vecDim = num_dims_";
  TEUCHOS_TEST_FOR_EXCEPTION(dl_->vectorAndGradientLayoutsAreEquivalent==false,
                             std::logic_error,
                             msg);
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl_);
  int offset = 0;
  // Temporary variable used numerous times below
  RCP<PHX::Evaluator<AlbanyTraits> > ev;

  // Define Field Names
  // generate the field name map to deal with outputing surface element info
  LCM::FieldNameMap field_name_map(surface_element);
  RCP<std::map<std::string, std::string> > fnm = field_name_map.getMap();
  std::string cauchy       = (*fnm)["Cauchy_Stress"];
  std::string Fp           = (*fnm)["Fp"];
  std::string eqps         = (*fnm)["eqps"];
  std::string temperature  = (*fnm)["Temperature"];
  std::string mech_source  = (*fnm)["Mechanical_Source"];
  // Poromechanics variables
  std::string totStress    = (*fnm)["Total_Stress"];
  std::string kcPerm       = (*fnm)["KCPermeability"];
  std::string biotModulus  = (*fnm)["Biot_Modulus"];
  std::string biotCoeff    = (*fnm)["Biot_Coefficient"];
  std::string porosity     = (*fnm)["Porosity"];
  std::string porePressure = (*fnm)["Pore_Pressure"];
  // Hydrogen diffusion variable
  std::string transport  = (*fnm)["Transport"];
  std::string hydroStress  = (*fnm)["HydroStress"];
  std::string diffusionCoefficient = (*fnm)["Diffusion_Coefficient"];
  std::string convectionCoefficient = (*fnm)["Tau_Contribution"];
  std::string trappedConcentration = (*fnm)["Trapped_Concentration"];
  std::string totalConcentration = (*fnm)["Total_Concentration"];
  std::string effectiveDiffusivity = (*fnm)["Effective_Diffusivity"];
  std::string trappedSolvent = (*fnm)["Trapped_Solvent"];
  std::string strainRateFactor = (*fnm)["Strain_Rate_Factor"];
  std::string eqilibriumParameter = (*fnm)["Concentration_Equilibrium_Parameter"];
  std::string gradientElementLength = (*fnm)["Gradient_Element_Length"];

  if (have_mech_eq_) {
    Teuchos::ArrayRCP<std::string> dof_names(1);
    Teuchos::ArrayRCP<std::string> resid_names(1);
    dof_names[0] = "Displacement";
    resid_names[0] = dof_names[0]+" Residual";

    fm0.template registerEvaluator<EvalT>
      (evalUtils.constructGatherSolutionEvaluator_noTransient(true,
                                                              dof_names));

    fm0.template registerEvaluator<EvalT>
      (evalUtils.constructGatherCoordinateVectorEvaluator());

    if ( !surface_element ) {
      fm0.template registerEvaluator<EvalT>
        (evalUtils.constructDOFVecInterpolationEvaluator(dof_names[0]));

      fm0.template registerEvaluator<EvalT>
        (evalUtils.constructDOFVecGradInterpolationEvaluator(dof_names[0]));

      fm0.template registerEvaluator<EvalT>
        (evalUtils.constructMapToPhysicalFrameEvaluator(cellType,
                                                        cubature));

      fm0.template registerEvaluator<EvalT>
        (evalUtils.constructComputeBasisFunctionsEvaluator(cellType,
                                                           intrepidBasis,
                                                           cubature));
    }

    fm0.template registerEvaluator<EvalT>
      (evalUtils.constructScatterResidualEvaluator(true,
                                                   resid_names));
    offset += num_dims_;
  }
  else if (have_mech_) { // constant configuration
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<std::string>("Material Property Name", "Displacement");
    p->set< RCP<DataLayout> >("Data Layout", dl_->qp_vector);
    p->set<std::string>("Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Coordinate Vector Data Layout", dl_->qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Displacement");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = rcp(new PHAL::NSMaterialProperty<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

  }
  if (have_temperature_eq_) { // Gather Solution Temperature
    Teuchos::ArrayRCP<std::string> dof_names(1);
    Teuchos::ArrayRCP<std::string> resid_names(1);
    dof_names[0] = "Temperature";
    resid_names[0] = dof_names[0]+" Residual";
    fm0.template registerEvaluator<EvalT>
      (evalUtils.constructGatherSolutionEvaluator_noTransient(false,
                                                              dof_names,
                                                              offset));

    fm0.template registerEvaluator<EvalT>
      (evalUtils.constructGatherCoordinateVectorEvaluator());

    if ( !surface_element ) {
      fm0.template registerEvaluator<EvalT>
        (evalUtils.constructDOFInterpolationEvaluator(dof_names[0]));

      fm0.template registerEvaluator<EvalT>
        (evalUtils.constructDOFGradInterpolationEvaluator(dof_names[0]));

      fm0.template registerEvaluator<EvalT>
        (evalUtils.constructMapToPhysicalFrameEvaluator(cellType,
                                                        cubature));

      fm0.template registerEvaluator<EvalT>
        (evalUtils.constructComputeBasisFunctionsEvaluator(cellType,
                                                           intrepidBasis,
                                                           cubature));
    }

    fm0.template registerEvaluator<EvalT>
      (evalUtils.constructScatterResidualEvaluator(false,
                                                   resid_names,
                                                   offset,
                                                   "Scatter Temperature"));
    offset++;
  }
  else if ( (!have_temperature_eq_  && have_temperature_)
		          || have_transport_eq_ || have_transport_ ) {
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<std::string>("Material Property Name", temperature);
    p->set< RCP<DataLayout> >("Data Layout", dl_->qp_scalar);
    p->set<std::string>("Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Coordinate Vector Data Layout", dl_->qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Temperature");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = rcp(new PHAL::NSMaterialProperty<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (have_pressure_eq_) {
    Teuchos::ArrayRCP<std::string> dof_names(1);
    Teuchos::ArrayRCP<std::string> resid_names(1);
    dof_names[0] = "Pore_Pressure";
    resid_names[0] = dof_names[0]+" Residual";

    fm0.template registerEvaluator<EvalT>
      (evalUtils.constructGatherSolutionEvaluator_noTransient(false,
                                                              dof_names,
                                                              offset));
    fm0.template registerEvaluator<EvalT>
      (evalUtils.constructGatherCoordinateVectorEvaluator());

    if ( !surface_element ) {
      fm0.template registerEvaluator<EvalT>
        (evalUtils.constructDOFInterpolationEvaluator(dof_names[0]));

      fm0.template registerEvaluator<EvalT>
        (evalUtils.constructDOFGradInterpolationEvaluator(dof_names[0]));

      fm0.template registerEvaluator<EvalT>
        (evalUtils.constructMapToPhysicalFrameEvaluator(cellType,
                                                        cubature));

      fm0.template registerEvaluator<EvalT>
        (evalUtils.constructComputeBasisFunctionsEvaluator(cellType,
                                                           intrepidBasis,
                                                           cubature));
    }

    fm0.template registerEvaluator<EvalT>
      (evalUtils.constructScatterResidualEvaluator(false,
                                                   resid_names,
                                                   offset,
                                                   "Scatter Pore_Pressure"));
    offset++;
  }
  else if (have_pressure_) { // constant Pressure
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<std::string>("Material Property Name", "Pressure");
    p->set< RCP<DataLayout> >("Data Layout", dl_->qp_scalar);
    p->set<std::string>("Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Coordinate Vector Data Layout", dl_->qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Pressure");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = rcp(new PHAL::NSMaterialProperty<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }


  if (have_transport_eq_) { // Gather solution for transport problem
    // Lattice Concentration
    Teuchos::ArrayRCP<std::string> dof_names(1);
    Teuchos::ArrayRCP<std::string> resid_names(1);
    dof_names[0] = "Transport";
    resid_names[0] = dof_names[0]+" Residual";
    fm0.template registerEvaluator<EvalT>
      (evalUtils.constructGatherSolutionEvaluator_noTransient(false,
                                                              dof_names,
                                                              offset));
    if ( !surface_element ) {
         fm0.template registerEvaluator<EvalT>
           (evalUtils.constructDOFInterpolationEvaluator(dof_names[0]));

         fm0.template registerEvaluator<EvalT>
           (evalUtils.constructDOFGradInterpolationEvaluator(dof_names[0]));

         fm0.template registerEvaluator<EvalT>
           (evalUtils.constructMapToPhysicalFrameEvaluator(cellType,
                                                           cubature));

         fm0.template registerEvaluator<EvalT>
           (evalUtils.constructComputeBasisFunctionsEvaluator(cellType,
                                                              intrepidBasis,
                                                              cubature));
       }

    fm0.template registerEvaluator<EvalT>
      (evalUtils.constructScatterResidualEvaluator(false,
                                                   resid_names,
                                                   offset,
                                                   "Scatter Transport"));
    offset++; // for lattice concentration
  }
  else if (have_transport_) { // Constant transport scalar value
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<std::string>("Material Property Name", "Transport");
    p->set< RCP<DataLayout> >("Data Layout", dl_->qp_scalar);
    p->set<std::string>("Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Coordinate Vector Data Layout", dl_->qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Transport");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = rcp(new PHAL::NSMaterialProperty<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (have_hydrostress_eq_) { // Gather solution for transport problem
    Teuchos::ArrayRCP<std::string> dof_names(1);
    Teuchos::ArrayRCP<std::string> resid_names(1);
    dof_names[0] = "HydroStress";
    resid_names[0] = dof_names[0]+" Residual";
    fm0.template registerEvaluator<EvalT>
      (evalUtils.constructGatherSolutionEvaluator_noTransient(false,
                                                              dof_names,
                                                              offset));

    if ( !surface_element ) {
         fm0.template registerEvaluator<EvalT>
           (evalUtils.constructDOFInterpolationEvaluator(dof_names[0]));

         fm0.template registerEvaluator<EvalT>
           (evalUtils.constructDOFGradInterpolationEvaluator(dof_names[0]));

         fm0.template registerEvaluator<EvalT>
           (evalUtils.constructMapToPhysicalFrameEvaluator(cellType,
                                                           cubature));

         fm0.template registerEvaluator<EvalT>
           (evalUtils.constructComputeBasisFunctionsEvaluator(cellType,
                                                              intrepidBasis,
                                                              cubature));
       }

    fm0.template registerEvaluator<EvalT>
      (evalUtils.constructScatterResidualEvaluator(false,
                                                   resid_names,
                                                   offset,
                                                   "Scatter HydroStress"));
    offset++; // for hydrostatic stress
  }

  { // Time
    RCP<ParameterList> p = rcp(new ParameterList("Time"));
    p->set<std::string>("Time Name", "Time");
    p->set<std::string>("Delta Time Name", "Delta Time");
    p->set< RCP<DataLayout> >("Workset Scalar Data Layout", dl_->workset_scalar);
    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    p->set<bool>("Disable Transient", true);
    ev = rcp(new LCM::Time<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
    p = stateMgr.registerStateVariable("Time",
                                       dl_->workset_scalar,
                                       dl_->dummy,
                                       eb_name,
                                       "scalar",
                                       0.0,
                                       true);
    ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (have_mech_eq_) { // Current Coordinates
    RCP<ParameterList> p = rcp(new ParameterList("Current Coordinates"));
    p->set<std::string>("Reference Coordinates Name", "Coord Vec");
    p->set<std::string>("Displacement Name", "Displacement");
    p->set<std::string>("Current Coordinates Name", "Current Coordinates");
    ev = rcp(new LCM::CurrentCoords<EvalT,AlbanyTraits>(*p,dl_));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (have_temperature_eq_ || have_temperature_) {
    RCP<ParameterList> p = rcp(new ParameterList("Save Temperature"));
    p = stateMgr.registerStateVariable(temperature,
                                       dl_->qp_scalar,
                                       dl_->dummy,
                                       eb_name,
                                       "scalar",
                                       0.0,
                                       true,
                                      false);
    ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (have_pressure_eq_ || have_pressure_) {
    RCP<ParameterList> p = rcp(new ParameterList("Save Pore Pressure"));
    p = stateMgr.registerStateVariable(porePressure,
                                       dl_->qp_scalar,
                                       dl_->dummy,
                                       eb_name,
                                       "scalar",
                                       0.0,
                                       true,
                                       false);
    ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }


  if (have_transport_eq_ || have_transport_) {
    RCP<ParameterList> p = rcp(new ParameterList("Save Transport"));
    p = stateMgr.registerStateVariable(transport,
                                       dl_->qp_scalar,
                                       dl_->dummy,
                                       eb_name,
                                       "scalar",
                                       38.7,
                                       true,
                                       true);
    ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }


  if (have_hydrostress_eq_ || have_hydrostress_) {
    RCP<ParameterList> p = rcp(new ParameterList("Save HydroStress"));
    p = stateMgr.registerStateVariable(hydroStress,
                                       dl_->qp_scalar,
                                       dl_->dummy,
                                       eb_name,
                                       "scalar",
                                       0.0,
                                       true,
                                       true);
    ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (have_source_) { // Source
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<std::string>("Source Name", "Source");
    p->set<std::string>("Variable Name", "Displacement");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl_->qp_scalar);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Source Functions");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = rcp(new PHAL::Source<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  { // Constitutive Model Parameters
    RCP<ParameterList> p = rcp(new ParameterList("Constitutive Model Parameters"));
    std::string matName = material_db_->getElementBlockParam<std::string>(eb_name,"material");
    Teuchos::ParameterList& param_list =
      material_db_->getElementBlockSublist(eb_name,matName);
    if (have_temperature_ || have_temperature_eq_) {
      param_list.set<bool>("Have Temperature", true);
    }

    p->set<Teuchos::ParameterList*>("Material Parameters", &param_list);

    RCP<LCM::ConstitutiveModelParameters<EvalT,AlbanyTraits> > cmpEv =
      rcp(new LCM::ConstitutiveModelParameters<EvalT,AlbanyTraits>(*p,dl_));
    fm0.template registerEvaluator<EvalT>(cmpEv);
  }

  if (have_mech_eq_) {
    RCP<ParameterList> p = rcp(new ParameterList("Constitutive Model Interface"));
    std::string matName = material_db_->getElementBlockParam<std::string>(eb_name,"material");
    Teuchos::ParameterList& param_list =
      material_db_->getElementBlockSublist(eb_name,matName);
    if (have_temperature_ || have_temperature_eq_) {
      param_list.set<bool>("Have Temperature", true);
    }
    param_list.set<RCP<std::map<std::string, std::string> > >("Name Map", fnm);
    p->set<Teuchos::ParameterList*>("Material Parameters", &param_list);

    RCP<LCM::ConstitutiveModelInterface<EvalT,AlbanyTraits> > cmiEv =
      rcp(new LCM::ConstitutiveModelInterface<EvalT,AlbanyTraits>(*p,dl_));
    fm0.template registerEvaluator<EvalT>(cmiEv);

    // register state variables
    for (int sv(0); sv < cmiEv->getNumStateVars(); ++sv) {
      cmiEv->fillStateVariableStruct(sv);
      p = stateMgr.registerStateVariable(cmiEv->getName(),
                                         cmiEv->getLayout(),
                                         dl_->dummy,
                                         eb_name,
                                         cmiEv->getInitType(),
                                         cmiEv->getInitValue(),
                                         cmiEv->getStateFlag(),
                                         cmiEv->getOutputFlag());
      ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }
  }

  if (have_mech_eq_ && materialModelName == "MooneyRivlinDamage") {
    RCP<ParameterList> p = rcp(new ParameterList("MooneyRivlinDamage Stress"));

    //Input
    p->set<std::string>("DefGrad Name", "F");
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl_->qp_tensor);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl_->qp_scalar);

    p->set<std::string>("DetDefGrad Name", "J");  // dl_->qp_scalar also
    p->set<std::string>("alpha Name", "alpha");

    // defaults for parameters
    RealType c1(0.0), c2(0.0), c(0.0), zeta_inf(0.0), iota(0.0);

    c1       = material_db_->getElementBlockParam<RealType>(eb_name,"c1");
    c2       = material_db_->getElementBlockParam<RealType>(eb_name,"c2");
    c        = material_db_->getElementBlockParam<RealType>(eb_name,"c");
    zeta_inf = material_db_->getElementBlockParam<RealType>(eb_name,"zeta_inf");
    iota     = material_db_->getElementBlockParam<RealType>(eb_name,"iota");

    p->set<RealType>("c1 Name", c1);
    p->set<RealType>("c2 Name", c2);
    p->set<RealType>("c Name", c);
    p->set<RealType>("zeta_inf Name", zeta_inf);
    p->set<RealType>("iota Name", iota);


    //Output
    p->set<std::string>("Stress Name", cauchy); //dl_->qp_tensor also

    ev = rcp(new LCM::MooneyRivlinDamage<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

    // optional output
    bool outputFlag(true);
    if ( material_db_->isElementBlockParam(eb_name,"Output " + cauchy) )
      outputFlag =
        material_db_->getElementBlockParam<bool>(eb_name,"Output " + cauchy);

    p = stateMgr.registerStateVariable(cauchy,
                                       dl_->qp_tensor,
                                       dl_->dummy,
                                       eb_name,
                                       "scalar",
                                       0.0,
                                       false,
                                       outputFlag);
    ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

    outputFlag = true;
    if ( material_db_->isElementBlockParam(eb_name,"Output Alpha") )
      outputFlag =
        material_db_->getElementBlockParam<bool>(eb_name,"Output Alpha");

    p = stateMgr.registerStateVariable("alpha",
                                       dl_->qp_scalar,
                                       dl_->dummy,
                                       eb_name,
                                       "scalar",
                                       1.0,
                                       false,
                                       outputFlag);
    ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if ( have_mech_eq_ && materialModelName == "MooneyRivlinIncompressible") {
    RCP<ParameterList> p =
      rcp(new ParameterList("MooneyRivlinIncompressible Stress"));

    //Input
    p->set<std::string>("DefGrad Name", "F");
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl_->qp_tensor);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl_->qp_scalar);

    p->set<std::string>("DetDefGrad Name", "J");  // dl_->qp_scalar also

    // defaults for parameters
    RealType c1(0.0), c2(0.0), c(0.0), mu(0.0);

    c1 = material_db_->getElementBlockParam<RealType>(eb_name,"c1");
    c2 = material_db_->getElementBlockParam<RealType>(eb_name,"c2");
    mu = material_db_->getElementBlockParam<RealType>(eb_name,"mu");

    p->set<RealType>("c1 Name", c1);
    p->set<RealType>("c2 Name", c2);
    p->set<RealType>("mu Name", mu);

    //Output
    p->set<std::string>("Stress Name", cauchy); //dl_->qp_tensor also

    ev = rcp(new LCM::MooneyRivlin_Incompressible<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

    // optional output
    bool outputFlag(true);
    if ( material_db_->isElementBlockParam(eb_name,"Output " + cauchy) )
      outputFlag =
        material_db_->getElementBlockParam<bool>(eb_name,"Output " + cauchy);

    p = stateMgr.registerStateVariable(cauchy,
                                       dl_->qp_tensor,
                                       dl_->dummy,
                                       eb_name,
                                       "scalar",
                                       0.0,
                                       false,
                                       outputFlag);
    ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // Surface Element Block
  if ( surface_element )
  {

    {// Surface Basis
     // SurfaceBasis_Def.hpp
      RCP<ParameterList> p = rcp(new ParameterList("Surface Basis"));

      // inputs
      p->set<std::string>("Reference Coordinates Name", "Coord Vec");
      p->set< RCP<Intrepid::Cubature<RealType> > >("Cubature", surfaceCubature);
      p->set< RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >("Intrepid Basis", surfaceBasis);
      if (have_mech_eq_) {
        p->set<std::string>("Current Coordinates Name", "Current Coordinates");
      }

      // outputs
      p->set<std::string>("Reference Basis Name", "Reference Basis");
      p->set<std::string>("Reference Area Name", "Reference Area");
      p->set<std::string>("Reference Dual Basis Name", "Reference Dual Basis");
      p->set<std::string>("Reference Normal Name", "Reference Normal");
      p->set<std::string>("Current Basis Name", "Current Basis");

      ev = rcp(new LCM::SurfaceBasis<EvalT,AlbanyTraits>(*p,dl_));
      fm0.template registerEvaluator<EvalT>(ev);
    }

    if (have_mech_eq_) { // Surface Jump
      //SurfaceVectorJump_Def.hpp
      RCP<ParameterList> p = rcp(new ParameterList("Surface Vector Jump"));

      // inputs
      p->set< RCP<Intrepid::Cubature<RealType> > >("Cubature", surfaceCubature);
      p->set< RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >("Intrepid Basis", surfaceBasis);
      p->set<std::string>("Vector Name", "Current Coordinates");

      // outputs
      p->set<std::string>("Vector Jump Name", "Vector Jump");

      ev = rcp(new LCM::SurfaceVectorJump<EvalT,AlbanyTraits>(*p,dl_));
      fm0.template registerEvaluator<EvalT>(ev);
    }

    if (have_pressure_eq_ ) { // Surface Pore Pressure Jump
      //SurfaceScalarJump_Def.hpp
      RCP<ParameterList> p = rcp(new ParameterList("Surface Scalar Jump"));

      // inputs
      p->set< RCP<Intrepid::Cubature<RealType> > >("Cubature", surfaceCubature);
      p->set< RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >("Intrepid Basis", surfaceBasis);
      p->set<std::string>("Nodal Scalar Name", "Pore_Pressure");

      // outputs
      p->set<std::string>("Scalar Jump Name", "Pore_Pressure Jump");
      p->set<std::string>("Scalar Average Name", porePressure);

      ev = rcp(new LCM::SurfaceScalarJump<EvalT,AlbanyTraits>(*p,dl_));
      fm0.template registerEvaluator<EvalT>(ev);

    }

    if (have_temperature_eq_ ) { // Surface Temperature Jump
      //SurfaceScalarJump_Def.hpp
      RCP<ParameterList> p = rcp(new ParameterList("Surface Scalar Jump"));

      // inputs
      p->set< RCP<Intrepid::Cubature<RealType> > >("Cubature", surfaceCubature);
      p->set< RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >("Intrepid Basis", surfaceBasis);
      p->set<std::string>("Nodal Scalar Name", "Temperature");

      // outputs
      p->set<std::string>("Scalar Jump Name", "Temperature Jump");
      p->set<std::string>("Scalar Average Name", temperature);

      ev = rcp(new LCM::SurfaceScalarJump<EvalT,AlbanyTraits>(*p,dl_));
      fm0.template registerEvaluator<EvalT>(ev);

    }

    if (have_transport_eq_ ) { // Surface Concentration Jump
      //SurfaceScalarJump_Def.hpp
      RCP<ParameterList> p = rcp(new ParameterList("Surface Scalar Jump"));

      // inputs
      p->set< RCP<Intrepid::Cubature<RealType> > >("Cubature", surfaceCubature);
      p->set< RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >("Intrepid Basis", surfaceBasis);
      p->set<std::string>("Nodal Scalar Name", "Transport");

      // outputs
      p->set<std::string>("Scalar Jump Name", "Transport Jump");
      p->set<std::string>("Scalar Average Name", transport);

      ev = rcp(new LCM::SurfaceScalarJump<EvalT,AlbanyTraits>(*p,dl_));
      fm0.template registerEvaluator<EvalT>(ev);

    }

    if (have_hydrostress_eq_ ) { // Surface Hydrostatic StressJump
      //SurfaceScalarJump_Def.hpp
      RCP<ParameterList> p = rcp(new ParameterList("Surface Scalar Jump"));

      // inputs
      p->set< RCP<Intrepid::Cubature<RealType> > >("Cubature", surfaceCubature);
      p->set< RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >("Intrepid Basis", surfaceBasis);
      p->set<std::string>("Nodal Scalar Name", "HydroStress");

      // outputs
      p->set<std::string>("Scalar Jump Name", "HydroStress Jump");
      p->set<std::string>("Scalar Average Name", hydroStress);

      ev = rcp(new LCM::SurfaceScalarJump<EvalT,AlbanyTraits>(*p,dl_));
      fm0.template registerEvaluator<EvalT>(ev);

    }

    if (have_mech_eq_) { // Surface Gradient
      //SurfaceVectorGradient_Def.hpp
      RCP<ParameterList> p = rcp(new ParameterList("Surface Vector Gradient"));

      // inputs
      p->set<RealType>("thickness",thickness);
      bool WeightedVolumeAverageJ(false);
      if ( material_db_->isElementBlockParam(eb_name,"Weighted Volume Average J") )
        p->set<bool>("Weighted Volume Average J Name",
                     material_db_->getElementBlockParam<bool>(eb_name,"Weighted Volume Average J") );
      if ( material_db_->isElementBlockParam(eb_name,"Average J Stabilization Parameter") )
        p->set<RealType>("Averaged J Stabilization Parameter Name",
                         material_db_->getElementBlockParam<RealType>(eb_name,"Average J Stabilization Parameter") );
      p->set< RCP<Intrepid::Cubature<RealType> > >("Cubature", surfaceCubature);
      p->set<std::string>("Weights Name","Reference Area");
      p->set<std::string>("Current Basis Name", "Current Basis");
      p->set<std::string>("Reference Dual Basis Name", "Reference Dual Basis");
      p->set<std::string>("Reference Normal Name", "Reference Normal");
      p->set<std::string>("Vector Jump Name", "Vector Jump");

      // outputs
      p->set<std::string>("Surface Vector Gradient Name", "F");
      p->set<std::string>("Surface Vector Gradient Determinant Name", "J");

      ev = rcp(new LCM::SurfaceVectorGradient<EvalT,AlbanyTraits>(*p,dl_));
      fm0.template registerEvaluator<EvalT>(ev);

    }

    // Surface Gradient Operator
    if (have_pressure_eq_ ) {
      //SurfaceScalarGradientOperator_Def.hpp
      RCP<ParameterList> p = rcp(new ParameterList("Surface Scalar Gradient Operator"));
      // inputs
      p->set<RealType>("thickness",thickness);
      p->set< RCP<Intrepid::Cubature<RealType> > >("Cubature", surfaceCubature);
      p->set< RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >("Intrepid Basis", surfaceBasis);
      p->set<std::string>("Reference Dual Basis Name", "Reference Dual Basis");
      p->set<std::string>("Reference Normal Name", "Reference Normal");

      // NOTE: NOT surf_Pore_Pressure here
      // NOTE: If you need to compute gradient for more than one scalar field, that could cause troubles
      if (have_pressure_eq_ == true) p->set<std::string>("Nodal Scalar Name", "Pore_Pressure");

      // outputs
      p->set<std::string>("Surface Scalar Gradient Operator Name", "Surface Scalar Gradient Operator");
      p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl_->node_qp_vector);
      if (have_pressure_eq_ == true) p->set<std::string>("Surface Scalar Gradient Name", "Surface Pressure Gradient");
      p->set< RCP<DataLayout> >("QP Vector Data Layout", dl_->qp_vector);

      ev = rcp(new LCM::SurfaceScalarGradientOperator<EvalT,AlbanyTraits>(*p,dl_));
      fm0.template registerEvaluator<EvalT>(ev);

     }

     if ( have_transport_eq_) {
      //SurfaceScalarGradientOperator_Def.hpp
      RCP<ParameterList> p = rcp(new ParameterList("Surface Scalar Gradient Operator"));
      // inputs
      p->set<RealType>("thickness",thickness);
      p->set< RCP<Intrepid::Cubature<RealType> > >("Cubature", surfaceCubature);
      p->set< RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >("Intrepid Basis", surfaceBasis);
      p->set<std::string>("Reference Dual Basis Name", "Reference Dual Basis");
      p->set<std::string>("Reference Normal Name", "Reference Normal");

      // NOTE: NOT surf_Pore_Pressure here
      // NOTE: If you need to compute gradient for more than one scalar field, that could cause troubles
      if (have_transport_eq_ == true) p->set<std::string>("Nodal Scalar Name", "Transport");

      // outputs
      p->set<std::string>("Surface Scalar Gradient Operator Name", "Surface Scalar Gradient Operator");
      p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl_->node_qp_vector);
      if (have_transport_eq_ == true) p->set<std::string>("Surface Scalar Gradient Name", "Surface Transport Gradient");
      p->set< RCP<DataLayout> >("QP Vector Data Layout", dl_->qp_vector);

      ev = rcp(new LCM::SurfaceScalarGradientOperator<EvalT,AlbanyTraits>(*p,dl_));
      fm0.template registerEvaluator<EvalT>(ev);

    }

     if ( have_hydrostress_eq_) {
      //SurfaceScalarGradientOperator_Def.hpp
      RCP<ParameterList> p = rcp(new ParameterList("Surface Scalar Gradient Operator"));
      // inputs
      p->set<RealType>("thickness",thickness);
      p->set< RCP<Intrepid::Cubature<RealType> > >("Cubature", surfaceCubature);
      p->set< RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >("Intrepid Basis", surfaceBasis);
      p->set<std::string>("Reference Dual Basis Name", "Reference Dual Basis");
      p->set<std::string>("Reference Normal Name", "Reference Normal");

      // NOTE: NOT surf_Pore_Pressure here
      // NOTE: If you need to compute gradient for more than one scalar field, that could cause troubles
      if (have_transport_eq_ == true) p->set<std::string>("Nodal Scalar Name", "HydroStress");

      // outputs
      p->set<std::string>("Surface Scalar Gradient Operator Name", "Surface Scalar Gradient Operator");
      p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl_->node_qp_vector);
      p->set<std::string>("Surface Scalar Gradient Name", "Surface HydroStress Gradient");
      p->set< RCP<DataLayout> >("QP Vector Data Layout", dl_->qp_vector);

      ev = rcp(new LCM::SurfaceScalarGradientOperator<EvalT,AlbanyTraits>(*p,dl_));
      fm0.template registerEvaluator<EvalT>(ev);

    }

    if(cohesive_element)
    {
      if (have_mech_eq_) { // Surface Traction based on cohesive element
        //TvergaardHutchinson_Def.hpp
        RCP<ParameterList> p = rcp(new ParameterList("Surface Cohesive Traction"));

        // inputs
        p->set< RCP<Intrepid::Cubature<RealType> > >("Cubature", surfaceCubature);
        p->set< RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >("Intrepid Basis", surfaceBasis);
        p->set<std::string>("Vector Jump Name", "Vector Jump");
        p->set<std::string>("Current Basis Name", "Current Basis");

        if ( material_db_->isElementBlockParam(eb_name,"delta_1") )
          p->set<RealType>("delta_1 Name", material_db_->getElementBlockParam<RealType>(eb_name,"delta_1"));
        else
          p->set<RealType>("delta_1 Name", 0.5);

        if ( material_db_->isElementBlockParam(eb_name,"delta_2") )
          p->set<RealType>("delta_2 Name", material_db_->getElementBlockParam<RealType>(eb_name,"delta_2"));
        else
          p->set<RealType>("delta_2 Name", 0.5);

        if ( material_db_->isElementBlockParam(eb_name,"delta_c") )
          p->set<RealType>("delta_c Name", material_db_->getElementBlockParam<RealType>(eb_name,"delta_c"));
        else
          p->set<RealType>("delta_c Name", 1.0);

        if ( material_db_->isElementBlockParam(eb_name,"sigma_c") )
          p->set<RealType>("sigma_c Name", material_db_->getElementBlockParam<RealType>(eb_name,"sigma_c"));
        else
          p->set<RealType>("sigma_c Name", 1.0);

        if ( material_db_->isElementBlockParam(eb_name,"beta_0") )
          p->set<RealType>("beta_0 Name", material_db_->getElementBlockParam<RealType>(eb_name,"beta_0"));
        else
          p->set<RealType>("beta_0 Name", 0.0);

        if ( material_db_->isElementBlockParam(eb_name,"beta_1") )
          p->set<RealType>("beta_1 Name", material_db_->getElementBlockParam<RealType>(eb_name,"beta_1"));
        else
          p->set<RealType>("beta_1 Name", 0.0);

        if ( material_db_->isElementBlockParam(eb_name,"beta_2") )
          p->set<RealType>("beta_2 Name", material_db_->getElementBlockParam<RealType>(eb_name,"beta_2"));
        else
          p->set<RealType>("beta_2 Name", 1.0);

        // outputs
        p->set<std::string>("Cohesive Traction Name","Cohesive Traction");
        ev = rcp(new LCM::TvergaardHutchinson<EvalT,AlbanyTraits>(*p,dl_));
        fm0.template registerEvaluator<EvalT>(ev);
      }

      { // Surface Cohesive Residual
        // SurfaceCohesiveResidual_Def.hpp
        RCP<ParameterList> p = rcp(new ParameterList("Surface Cohesive Residual"));

        // inputs
        p->set< RCP<Intrepid::Cubature<RealType> > >("Cubature", surfaceCubature);
        p->set< RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >("Intrepid Basis", surfaceBasis);
        p->set<std::string>("Cohesive Traction Name", "Cohesive Traction");
        p->set<std::string>("Reference Area Name", "Reference Area");

        // outputs
        p->set<std::string>("Surface Cohesive Residual Name", "Displacement Residual");

        ev = rcp(new LCM::SurfaceCohesiveResidual<EvalT,AlbanyTraits>(*p,dl_));
        fm0.template registerEvaluator<EvalT>(ev);
      }

    }
    else
    {

      if (have_mech_eq_) { // Surface Residual
        // SurfaceVectorResidual_Def.hpp
        RCP<ParameterList> p = rcp(new ParameterList("Surface Vector Residual"));

        // inputs
        p->set<RealType>("thickness",thickness);
        p->set< RCP<Intrepid::Cubature<RealType> > >("Cubature", surfaceCubature);
        p->set< RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >("Intrepid Basis", surfaceBasis);
        p->set<std::string>("DefGrad Name", "F");
        p->set<std::string>("Stress Name", cauchy);
        p->set<std::string>("Current Basis Name", "Current Basis");
        p->set<std::string>("Reference Dual Basis Name", "Reference Dual Basis");
        p->set<std::string>("Reference Normal Name", "Reference Normal");
        p->set<std::string>("Reference Area Name", "Reference Area");

        // Effective stress theory for poromechanics problem
        if (have_pressure_eq_) {
          p->set<std::string>("Pore Pressure Name", porePressure);
          p->set<std::string>("Biot Coefficient Name", biotCoeff);
        }

        // outputs
        p->set<std::string>("Surface Vector Residual Name", "Displacement Residual");

        ev = rcp(new LCM::SurfaceVectorResidual<EvalT,AlbanyTraits>(*p,dl_));
        fm0.template registerEvaluator<EvalT>(ev);
      }
    } // end of coehesive/surface element block
  } else {

    if (have_mech_eq_) { // Kinematics quantities
      RCP<ParameterList> p = rcp(new ParameterList("Kinematics"));

      // set flags to optionally volume average J with a weighted average
      if ( material_db_->
           isElementBlockParam(eb_name,"Weighted Volume Average J") ){
        p->set<bool>("Weighted Volume Average J",
                     material_db_->
                     getElementBlockParam<bool>(eb_name,
                                                "Weighted Volume Average J") );
      }

      if ( material_db_->
           isElementBlockParam(eb_name,
                               "Average J Stabilization Parameter") ){
        p->set<RealType>
          ("Average J Stabilization Parameter",
           material_db_->
           getElementBlockParam<RealType>(eb_name,
                                          "Average J Stabilization Parameter"));
      }

      // set flag for return strain and velocity gradient
      bool have_strain(false), have_velocity_gradient(false);

      if(material_db_-> isElementBlockParam(eb_name,"Strain Flag")){
        p->set<bool>("Strain Flag",
        			material_db_->
        			getElementBlockParam<bool>(eb_name,"Strain Flag"));
        have_strain = material_db_->
                    getElementBlockParam<bool>(eb_name,"Strain Flag");
        if(have_strain)
          p->set<std::string>("Strain Name", "Strain");
      }

      if(material_db_-> isElementBlockParam(eb_name,"Velocity Gradient Flag")){
        p->set<bool>("Velocity Gradient Flag",
        			material_db_->
        			getElementBlockParam<bool>(eb_name,"Velocity Gradient Flag"));
        have_velocity_gradient = material_db_->
                    getElementBlockParam<bool>(eb_name,"Velocity Gradient Flag");
        if(have_velocity_gradient)
          p->set<std::string>("Velocity Gradient Name", "Velocity Gradient");
      }

      // send in integration weights and the displacement gradient
      p->set<std::string>("Weights Name","Weights");
      p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl_->qp_scalar);
      p->set<std::string>("Gradient QP Variable Name", "Displacement Gradient");
      p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl_->qp_tensor);

      //Outputs: F, J
      p->set<std::string>("DefGrad Name", "F"); //dl_->qp_tensor also
      p->set<std::string>("DetDefGrad Name", "J");
      p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl_->qp_scalar);

      //ev = rcp(new LCM::DefGrad<EvalT,AlbanyTraits>(*p));
      ev = rcp(new LCM::Kinematics<EvalT,AlbanyTraits>(*p,dl_));
      fm0.template registerEvaluator<EvalT>(ev);


      // optional output
      bool outputFlag(false);
      if ( material_db_->isElementBlockParam(eb_name,"Output Deformation Gradient") )
        outputFlag =
          material_db_->getElementBlockParam<bool>(eb_name,"Output Deformation Gradient");

      p = stateMgr.registerStateVariable("F",
                                         dl_->qp_tensor,
                                         dl_->dummy,
                                         eb_name,
                                         "identity",
                                         1.0,
                                         outputFlag);
      ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);

      // need J and J_old to perform time integration for poromechanics problem
      outputFlag = false;
      if ( material_db_->isElementBlockParam(eb_name,"Output J") )
        outputFlag =
          material_db_->getElementBlockParam<bool>(eb_name,"Output J");
      if (have_pressure_eq_ || outputFlag) {
        p = stateMgr.registerStateVariable("J",
                                           dl_->qp_scalar,
                                           dl_->dummy,
                                           eb_name,
                                           "scalar",
                                           1.0,
                                           true);
        ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
        fm0.template registerEvaluator<EvalT>(ev);
      }

      // Optional output: strain
      if(have_strain){
        outputFlag = false;
        if(material_db_-> isElementBlockParam(eb_name,"Output Strain"))
          outputFlag =
            material_db_-> getElementBlockParam<bool>(eb_name,"Output Strain");

        p = stateMgr.registerStateVariable("Strain",
                                         dl_->qp_tensor,
                                         dl_->dummy,
                                         eb_name,
                                         "scalar",
                                         0.0,
                                         outputFlag);
        ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
        fm0.template registerEvaluator<EvalT>(ev);
      }

      // Optional output: velocity gradient
      if(have_velocity_gradient){
        outputFlag = false;
        if(material_db_-> isElementBlockParam(eb_name,
          "Output Velocity Gradient"))
          outputFlag =
            material_db_-> getElementBlockParam<bool>(eb_name,
              "Output Velocity Gradient");

        p = stateMgr.registerStateVariable("Velocity Gradient",
                                         dl_->qp_tensor,
                                         dl_->dummy,
                                         eb_name,
                                         "scalar",
                                         0.0,
                                         outputFlag);
        ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
        fm0.template registerEvaluator<EvalT>(ev);
      }

    }

    if (have_mech_eq_)
    { // Residual
      RCP<ParameterList> p = rcp(new ParameterList("Displacement Residual"));
      //Input
      p->set<std::string>("Stress Name", cauchy);
      p->set<std::string>("DefGrad Name", "F");
      p->set<std::string>("DetDefGrad Name", "J");
      p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
      p->set<std::string>("Weighted BF Name", "wBF");

      // Strain flag for small deformation problem
      if(material_db_-> isElementBlockParam(eb_name,"Strain Flag")){
        p->set<bool>("Strain Flag","Strain Flag");
      }

      // Effective stress theory for poromechanics problem
      if (have_pressure_eq_) {
        p->set<bool>("Have Pore Pressure", true);
        p->set<std::string>("Pore Pressure Name", porePressure);
        p->set<std::string>("Biot Coefficient Name", biotCoeff);
      }

      p->set<RCP<ParamLib> >("Parameter Library", paramLib);
      //Output
      p->set<std::string>("Residual Name", "Displacement Residual");
      ev = rcp(new LCM::MechanicsResidual<EvalT,AlbanyTraits>(*p,dl_));
      fm0.template registerEvaluator<EvalT>(ev);
    }
  }

  // Element length in the direction of solution gradient
  if ( (have_pressure_eq_ || have_transport_eq_) && !surface_element) {
    RCP<ParameterList> p = rcp(new ParameterList("Gradient_Element_Length"));

    //Input
    if (have_pressure_eq_){
      p->set<std::string>("Unit Gradient QP Variable Name", "Pore_Pressure Gradient");
    } else if (have_transport_eq_){
      p->set<std::string>("Unit Gradient QP Variable Name", "Transport Gradient");
    }
    p->set<std::string>("Gradient BF Name", "Grad BF");

    //Output
    p->set<std::string>("Element Length Name", gradientElementLength);

    ev = rcp(new LCM::GradientElementLength<EvalT,AlbanyTraits>(*p,dl_));
    fm0.template registerEvaluator<EvalT>(ev);

    /*
    p = stateMgr.registerStateVariable("Gradient Element Length",
                                       dl_->qp_scalar,
                                       dl_->dummy,
                                       eb_name,
                                       "scalar",
                                       1.0);
    ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

    */
  }

  if (have_pressure_eq_) {  // Porosity
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<std::string>("Porosity Name", porosity);
    p->set<std::string>("QP Coordinate Vector Name", "Coord Vec");
    // Setting this turns on dependence of strain and pore pressure)
    //p->set<std::string>("Strain Name", "Strain");
    if (have_mech_eq_) p->set<std::string>("DetDefGrad Name", "J");
    // porosity update based on Coussy's poromechanics (see p.79)
    p->set<std::string>("QP Pore Pressure Name", porePressure);
    p->set<std::string>("Biot Coefficient Name", biotCoeff);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList =
      material_db_->getElementBlockSublist(eb_name,"Porosity");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    // Output Porosity
    ev = rcp(new LCM::Porosity<EvalT,AlbanyTraits>(*p,dl_));
    fm0.template registerEvaluator<EvalT>(ev);
    p = stateMgr.registerStateVariable(porosity,
                                       dl_->qp_scalar,
                                       dl_->dummy,
                                       eb_name,
                                       "scalar",
                                       0.5);
    ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (have_pressure_eq_) { // Biot Coefficient
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<std::string>("Biot Coefficient Name", biotCoeff);
    p->set<std::string>("QP Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Node Data Layout", dl_->node_scalar);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl_->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl_->qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList =
      material_db_->getElementBlockSublist(eb_name,"Biot Coefficient");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = rcp(new LCM::BiotCoefficient<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

    /*
    p = stateMgr.registerStateVariable(biotCoeff,
                                       dl_->qp_scalar,
                                       dl_->dummy,
                                       eb_name,
                                       "scalar",
                                       1.0);
    ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

    */
  }

  if (have_pressure_eq_) { // Biot Modulus
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<std::string>("Biot Modulus Name", biotModulus);
    p->set<std::string>("QP Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Node Data Layout", dl_->node_scalar);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl_->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl_->qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList =
      material_db_->getElementBlockSublist(eb_name,"Biot Modulus");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    // Setting this turns on linear dependence on porosity and Biot's coeffcient
    p->set<std::string>("Porosity Name", porosity);
    p->set<std::string>("Biot Coefficient Name", biotCoeff);

    ev = rcp(new LCM::BiotModulus<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

    /*
    p = stateMgr.registerStateVariable(biotModulus,
                                       dl_->qp_scalar,
                                       dl_->dummy,
                                       eb_name,
                                       "scalar",
                                       1.0e20);
    // Very large value means incompressible phases
    ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

    */
  }

  if (have_pressure_eq_) { // Kozeny-Carman Permeaiblity
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<std::string>("Kozeny-Carman Permeability Name", kcPerm);
    p->set<std::string>("QP Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Node Data Layout", dl_->node_scalar);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl_->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl_->qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList =
      material_db_->getElementBlockSublist(eb_name,"Kozeny-Carman Permeability");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    // Setting this turns on Kozeny-Carman relation
    p->set<std::string>("Porosity Name", porosity);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl_->qp_scalar);

    ev = rcp(new LCM::KCPermeability<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
    p = stateMgr.registerStateVariable(kcPerm,
                                       dl_->qp_scalar,
                                       dl_->dummy,
                                       eb_name,
                                       "scalar",
                                       1.0); // Must be nonzero
    ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // Pore Pressure Residual (Bulk Element)
  if (have_pressure_eq_ && !surface_element) {
    RCP<ParameterList> p = rcp(new ParameterList("Pore_Pressure Residual"));

    //Input

    // Input from nodal points, basis function stuff
    p->set<std::string>("Weights Name","Weights");
    p->set<std::string>("Weighted BF Name", "wBF");
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout",
                              dl_->node_qp_scalar);
    p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl_->node_qp_vector);

    // Inputs: X, Y at nodes, Cubature, and Basis
    p->set<std::string>("Coordinate Vector Name","Coord Vec");
    p->set< RCP<DataLayout> >("Coordinate Data Layout", dl_->vertices_vector);
    p->set< RCP<Intrepid::Cubature<RealType> > >("Cubature", cubature);
    p->set<RCP<shards::CellTopology> >("Cell Type", cellType);

    // DT for  time integration
    p->set<std::string>("Delta Time Name", "Delta Time");
    p->set< RCP<DataLayout> >("Workset Scalar Data Layout", dl_->workset_scalar);

    p->set<bool>("Have Source", false);
    p->set<std::string>("Source Name", "Source");
    p->set<bool>("Have Absorption", false);

    // Input from cubature points
    p->set<std::string>("Element Length Name", "Gradient Element Length");
    p->set<std::string>("QP Pore Pressure Name", porePressure);
    p->set<std::string>("QP Time Derivative Variable Name", porePressure);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl_->qp_scalar);

    //p->set<std::string>("Material Property Name", "Stabilization Parameter");
    p->set<std::string>("Porosity Name", "Porosity");
    p->set<std::string>("Thermal Conductivity Name", "Thermal Conductivity");
    p->set<std::string>("Kozeny-Carman Permeability Name", kcPerm);
    p->set<std::string>("Biot Coefficient Name", biotCoeff);
    p->set<std::string>("Biot Modulus Name", biotModulus);

    p->set<std::string>("Gradient QP Variable Name", "Pore_Pressure Gradient");
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl_->qp_vector);

    if (have_mech_eq_) {
      p->set<bool>("Have Mechanics", true);
      p->set<std::string>("DefGrad Name", "F");
      p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl_->qp_tensor);
      p->set<std::string>("DetDefGrad Name", "J");
      p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl_->qp_scalar);
    }
    RealType stab_param(0.0);
    if ( material_db_->isElementBlockParam(eb_name, "Stabilization Parameter") ) {
      stab_param =
        material_db_->getElementBlockParam<RealType>(eb_name,
                                                     "Stabilization Parameter");
    }
    p->set<RealType>("Stabilization Parameter", stab_param);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);

    //Output
    p->set<std::string>("Residual Name", "Pore_Pressure Residual");
    p->set< RCP<DataLayout> >("Node Scalar Data Layout", dl_->node_scalar);

    ev = rcp(new LCM::TLPoroPlasticityResidMass<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

    // Output QP pore pressure
    p = stateMgr.registerStateVariable(porePressure,
                                       dl_->qp_scalar,
                                       dl_->dummy,
                                       eb_name,
                                       "scalar",
                                       0.0,
                                       true);
    ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (have_pressure_eq_ && surface_element) { // Pore Pressure Resid for Surface
    RCP<ParameterList> p = rcp(new ParameterList("Pore_Pressure Residual"));

    //Input
    p->set<RealType>("thickness",thickness);
    p->set< RCP<Intrepid::Cubature<RealType> > >("Cubature", surfaceCubature);
    p->set< RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >("Intrepid Basis", surfaceBasis);
    p->set<std::string>("Surface Scalar Gradient Operator Name", "Surface Scalar Gradient Operator");
    p->set<std::string>("Scalar Gradient Name", "Surface Pressure Gradient");
    p->set<std::string>("Current Basis Name", "Current Basis");
    p->set<std::string>("Reference Dual Basis Name", "Reference Dual Basis");
    p->set<std::string>("Reference Normal Name", "Reference Normal");
    p->set<std::string>("Reference Area Name", "Reference Area");
    p->set<std::string>("Pore Pressure Name", porePressure);
    p->set<std::string>("Nodal Pore Pressure Name", "Pore_Pressure"); // NOTE: NOT surf_Pore_Pressure here
    p->set<std::string>("Biot Coefficient Name", biotCoeff);
    p->set<std::string>("Biot Modulus Name", biotModulus);
    p->set<std::string>("Kozeny-Carman Permeability Name", kcPerm);
    p->set<std::string>("Delta Time Name", "Delta Time");
    if (have_mech_eq_) {
      p->set<std::string>("DefGrad Name", "F");
      p->set<std::string>("DetDefGrad Name", "J");
    }

    //Output
    p->set<std::string>("Residual Name", "Pore_Pressure Residual");
    p->set< RCP<DataLayout> >("Node Scalar Data Layout", dl_->node_scalar);

    ev = rcp(new LCM::SurfaceTLPoroMassResidual<EvalT,AlbanyTraits>(*p,dl_));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (have_transport_eq_ || have_transport_){ // Transport Coefficients
    RCP<ParameterList> p = rcp(new ParameterList("Transport Coefficients"));

    std::string matName = material_db_->getElementBlockParam<std::string>(eb_name,"material");
    Teuchos::ParameterList& param_list = material_db_->
     getElementBlockSublist(eb_name,matName).sublist("Transport Coefficients");
     p->set<Teuchos::ParameterList*>("Material Parameters", &param_list);

    //Input
    p->set<std::string>("Lattice Concentration Name", transport);
    p->set<std::string>("Temperature Name", temperature);
    if ( materialModelName == "J2" ) {
      p->set<std::string>("Equivalent Plastic Strain Name", eqps);
    }

    //Output
    p->set<std::string>("Trapped Concentration Name", trappedConcentration);
    p->set<std::string>("Total Concentration Name", totalConcentration);
    p->set<std::string>("Effective Diffusivity Name", effectiveDiffusivity);
    p->set<std::string>("Trapped Solvent Name", trappedSolvent);
    p->set<std::string>("Strain Rate Factor Name", strainRateFactor);
    p->set<std::string>("Diffusion Coefficient Name", diffusionCoefficient);
    p->set<std::string>("Tau Contribution Name", convectionCoefficient);
    p->set<std::string>("Concentration Equilibrium Parameter Name",
    	                              eqilibriumParameter);

    ev = rcp(new LCM::TransportCoefficients<EvalT,AlbanyTraits>(*p,dl_));
    fm0.template registerEvaluator<EvalT>(ev);

    bool outputFlag(true);
        if ( material_db_->isElementBlockParam(eb_name,"Output " + trappedConcentration) )
          outputFlag =
            material_db_->getElementBlockParam<bool>(eb_name,"Output " + trappedConcentration);

    p = stateMgr.registerStateVariable(trappedConcentration,dl_->qp_scalar,
                                       dl_->dummy, eb_name,
                                       "scalar", 0.0, false, outputFlag);
    ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

    p = stateMgr.registerStateVariable(strainRateFactor,dl_->qp_scalar,
                                       dl_->dummy, eb_name,
                                       "scalar", 0.0, false);
    ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

    p = stateMgr.registerStateVariable(convectionCoefficient,dl_->qp_scalar,
                                       dl_->dummy, eb_name,
                                       "scalar", 0.0, false);
    ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

    p = stateMgr.registerStateVariable(diffusionCoefficient,dl_->qp_scalar,
                                       dl_->dummy, eb_name,
                                       "scalar", 0.0, false);
    ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

    p = stateMgr.registerStateVariable(effectiveDiffusivity,dl_->qp_scalar,
                                       dl_->dummy, eb_name,
                                       "scalar", 0.0, false);
    ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // Transport of the temperature field
  if (have_temperature_eq_ && !surface_element)
  {
    RCP<ParameterList> p = rcp(new ParameterList("ThermoMechanical Coefficients"));

    std::string matName =
      material_db_->getElementBlockParam<std::string>(eb_name,"material");
    Teuchos::ParameterList& param_list =
      material_db_->getElementBlockSublist(eb_name,matName);
    p->set<Teuchos::ParameterList*>("Material Parameters", &param_list);

    // Input
    p->set<std::string>("Temperature Name", "Temperature");
    p->set<std::string>("Thermal Conductivity Name", "Thermal Conductivity");
    p->set<std::string>("Thermal Transient Coefficient Name",
                   "Thermal Transient Coefficient");

    // Output
    p->set<std::string>("Thermal Diffusivity Name", "Thermal Diffusivity");

    ev = rcp(new LCM::ThermoMechanicalCoefficients<EvalT,AlbanyTraits>(*p,dl_));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // Transport of the temperature field
  if (have_temperature_eq_ && !surface_element)
  {
    RCP<ParameterList> p = rcp(new ParameterList("Temperature Residual"));

    // Input
    p->set<std::string>("Scalar Variable Name", "Temperature");
    p->set<std::string>("Scalar Gradient Variable Name", "Temperature Gradient");
    p->set<std::string>("Weights Name", "Weights");
    p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
    p->set<std::string>("Weighted BF Name", "wBF");

    // Transient
    p->set<bool>("Have Transient", true);
    p->set<std::string>("Delta Time Name", "Delta Time");
    p->set<std::string>("Transient Coefficient Name", "Thermal Transient Coefficient");

    // Diffusion
    p->set<bool>("Have Diffusion", true);
    p->set<std::string>("Diffusivity Name", "Thermal Diffusivity");

    // Source
    if (have_mech_ || have_mech_eq_) {
      p->set<bool>("Have Source", true);
      p->set<std::string>("Source Name", mech_source);
    }

    // Output
    p->set<std::string>("Residual Name", "Temperature Residual");

    ev = rcp(new LCM::TransportResidual<EvalT,AlbanyTraits>(*p,dl_));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // Hydrogen Transport model proposed in Foulk et al 2014
  if (have_transport_eq_ && !surface_element){
    RCP<ParameterList> p = rcp(new ParameterList("Transport Residual"));

    //Input
    p->set<std::string>("Element Length Name", gradientElementLength);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl_->qp_scalar);

    p->set<std::string>("Weighted BF Name", "wBF");
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", dl_->node_qp_scalar);

    p->set<std::string>("Weights Name","Weights");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl_->qp_scalar);

    p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl_->node_qp_vector);

    p->set<std::string>("Gradient BF Name", "Grad BF");
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl_->node_qp_vector);

    p->set<std::string>("eqps Name", eqps);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl_->qp_scalar);

    p->set<std::string>("Strain Rate Factor Name", strainRateFactor);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl_->qp_scalar);

    p->set<std::string>("Trapped Concentration Name", trappedConcentration);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl_->qp_scalar);

    p->set<std::string>("Trapped Solvent Name", trappedSolvent);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl_->qp_scalar);

    p->set<std::string>("Deformation Gradient Name", "F");
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl_->qp_tensor);

    p->set<std::string>("Effective Diffusivity Name", effectiveDiffusivity);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl_->qp_scalar);

    p->set<std::string>("Diffusion Coefficient Name", diffusionCoefficient);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl_->qp_scalar);

    p->set<std::string>("Tau Contribution Name", convectionCoefficient);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl_->qp_scalar);

    p->set<std::string>("QP Variable Name", "Transport");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl_->qp_scalar);

    p->set<std::string>("Gradient QP Variable Name", "Transport Gradient");
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl_->qp_vector);

    p->set<std::string>("Gradient Hydrostatic Stress Name", "HydroStress Gradient");
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl_->qp_vector);

    p->set<std::string>("Stress Name", cauchy);
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl_->qp_tensor);

    p->set<std::string>("Delta Time Name", "Delta Time");
    p->set< RCP<DataLayout> >("Workset Scalar Data Layout", dl_->workset_scalar);

    RealType stab_param(1.0);
	if ( material_db_->isElementBlockParam(eb_name, "Stabilization Parameter") ) {
          stab_param =
            material_db_->getElementBlockParam<RealType>(eb_name,
                                                         "Stabilization Parameter");
	}
	p->set<RealType>("Stabilization Parameter", stab_param);
	p->set<RCP<ParamLib> >("Parameter Library", paramLib);

    //Output
    p->set<std::string>("Residual Name", "Transport Residual");
    p->set< RCP<DataLayout> >("Node Scalar Data Layout", dl_->node_scalar);

    ev = rcp(new LCM::HDiffusionDeformationMatterResidual<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
    p = stateMgr.registerStateVariable("Transport",dl_->qp_scalar,
                                       dl_->dummy, eb_name, "scalar", 0.0, true, true);
    ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

    p = stateMgr.registerStateVariable("Transport Gradient",
                                       dl_->qp_vector, dl_->dummy ,
                                       eb_name, "scalar" , 0.0  , true, true);
    ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

  }

  if (have_transport_eq_ && surface_element) { // Transport Resid for Surface
    RCP<ParameterList> p = rcp(new ParameterList("Transport Residual"));

    //Input
    p->set<RealType>("thickness",thickness);
    p->set< RCP<Intrepid::Cubature<RealType> > >("Cubature", surfaceCubature);
    p->set< RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >("Intrepid Basis", surfaceBasis);
    p->set<std::string>("Surface Scalar Gradient Operator Name", "Surface Scalar Gradient Operator");
    p->set<std::string>("Scalar Gradient Name", "Surface Transport Gradient");
    p->set<std::string>("Current Basis Name", "Current Basis");
    p->set<std::string>("Reference Dual Basis Name", "Reference Dual Basis");
    p->set<std::string>("Reference Normal Name", "Reference Normal");
    p->set<std::string>("Reference Area Name", "Reference Area");
    p->set<std::string>("Transport Name", transport);
    p->set<std::string>("Nodal Transport Name",  "Transport"); // NOTE: NOT surf_Transport here
    p->set<std::string>("Diffusion Coefficient Name", diffusionCoefficient);
    p->set<std::string>("Effective Diffusivity Name", effectiveDiffusivity);
    p->set<std::string>("Tau Contribution Name", convectionCoefficient);
    p->set<std::string>("Strain Rate Factor Name", strainRateFactor);
    //p->set<std::string>("Element Length Name", gradientElementLength);
    p->set<std::string>("Surface HydroStress Gradient Name", "Surface HydroStress Gradient");
    p->set<std::string>("eqps Name", eqps);
    p->set<std::string>("Delta Time Name", "Delta Time");
    if (have_mech_eq_) {
      p->set<std::string>("DefGrad Name", "F");
      p->set<std::string>("DetDefGrad Name", "J");
    }

    RealType stab_param(1.0);
	if ( material_db_->isElementBlockParam(eb_name, "Stabilization Parameter") ) {
          stab_param =
            material_db_->getElementBlockParam<RealType>(eb_name,
                                                         "Stabilization Parameter");
	}
	p->set<RealType>("Stabilization Parameter", stab_param);
	p->set<RCP<ParamLib> >("Parameter Library", paramLib);

    //Output
    p->set<std::string>("Residual Name", "Transport Residual");
    p->set< RCP<DataLayout> >("Node Scalar Data Layout", dl_->node_scalar);

    ev = rcp(new LCM::SurfaceHDiffusionDefResidual<EvalT,AlbanyTraits>(*p,dl_));
    fm0.template registerEvaluator<EvalT>(ev);

    p = stateMgr.registerStateVariable(transport,dl_->qp_scalar,
                                       dl_->dummy, eb_name, "scalar", 0.0, true, true);
    ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

    p = stateMgr.registerStateVariable("Surface Transport Gradient",
                                       dl_->qp_vector, dl_->dummy ,
                                       eb_name, "scalar" , 0.0  , true, true);
    ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (have_hydrostress_eq_ && !surface_element){ // L2 hydrostatic stress projection
    RCP<ParameterList> p = rcp(new ParameterList("HydroStress Residual"));

    //Input
    p->set<std::string>("Weighted BF Name", "wBF");
    p->set< RCP<DataLayout> >
      ("Node QP Scalar Data Layout", dl_->node_qp_scalar);

    p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
    p->set< RCP<DataLayout> >
      ("Node QP Vector Data Layout", dl_->node_qp_vector);

    p->set<bool>("Have Source", false);
    p->set<std::string>("Source Name", "Source");

    p->set<std::string>("Deformation Gradient Name", "F");
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl_->qp_tensor);

    p->set<std::string>("QP Variable Name", hydroStress);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl_->qp_scalar);

    p->set<std::string>("Stress Name", cauchy);
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl_->qp_tensor);

    //Output
    p->set<std::string>("Residual Name", "HydroStress Residual");
    p->set< RCP<DataLayout> >("Node Scalar Data Layout", dl_->node_scalar);

    ev = rcp(new LCM::ScalarL2ProjectionResidual<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
    p = stateMgr.registerStateVariable(hydroStress,dl_->qp_scalar, dl_->dummy,
                                       eb_name, "scalar", 0.0, true);
    ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }


  if (have_hydrostress_eq_ && surface_element) { // Transport Resid for Surface
    RCP<ParameterList> p = rcp(new ParameterList("HydroStress Residual"));

    //Input
    p->set<RealType>("thickness",thickness);
    p->set< RCP<Intrepid::Cubature<RealType> > >("Cubature", surfaceCubature);
    p->set< RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >("Intrepid Basis", surfaceBasis);
    p->set<std::string>("Surface Scalar Gradient Operator Name", "Surface Scalar Gradient Operator");
    p->set<std::string>("Scalar Gradient Name", "Surface Transport Gradient");
    p->set<std::string>("Current Basis Name", "Current Basis");
    p->set<std::string>("Reference Dual Basis Name", "Reference Dual Basis");
    p->set<std::string>("Reference Normal Name", "Reference Normal");
    p->set<std::string>("Reference Area Name", "Reference Area");
    p->set<std::string>("HydoStress Name", hydroStress);
    p->set<std::string>("Cauchy Stress Name", cauchy);
    p->set<std::string>("Jacobian Name", "J");

    //Output
    p->set<std::string>("Residual Name", "HydroStress Residual");
    p->set< RCP<DataLayout> >("Node Scalar Data Layout", dl_->node_scalar);

    ev = rcp(new LCM::SurfaceL2ProjectionResidual<EvalT,AlbanyTraits>(*p,dl_));
    fm0.template registerEvaluator<EvalT>(ev);
  }


  if (fieldManagerChoice == Albany::BUILD_RESID_FM)  {
    Teuchos::RCP<const PHX::FieldTag> ret_tag;
    if (have_mech_eq_) {
      PHX::Tag<typename EvalT::ScalarT> res_tag("Scatter", dl_->dummy);
      fm0.requireField<EvalT>(res_tag);
      ret_tag = res_tag.clone();
    }
    if (have_pressure_eq_) {
      PHX::Tag<typename EvalT::ScalarT> pres_tag("Scatter Pore_Pressure", dl_->dummy);
      fm0.requireField<EvalT>(pres_tag);
      ret_tag = pres_tag.clone();
    }
    if (have_temperature_eq_) {
      PHX::Tag<typename EvalT::ScalarT> temperature_tag("Scatter Temperature", dl_->dummy);
      fm0.requireField<EvalT>(temperature_tag);
      ret_tag = temperature_tag.clone();
    }
    if (have_transport_eq_) {
      PHX::Tag<typename EvalT::ScalarT> transport_tag("Scatter Transport", dl_->dummy);
      fm0.requireField<EvalT>(transport_tag);
      ret_tag = transport_tag.clone();
    }
    if (have_hydrostress_eq_) {
      PHX::Tag<typename EvalT::ScalarT> l2projection_tag("Scatter HydroStress", dl_->dummy);
      fm0.requireField<EvalT>(l2projection_tag);
      ret_tag = l2projection_tag.clone();
    }
    return ret_tag;
  }
  else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM) {
    Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl_);
    return respUtils.constructResponses(fm0, *responseList, stateMgr);
  }

  return Teuchos::null;
}

#endif
