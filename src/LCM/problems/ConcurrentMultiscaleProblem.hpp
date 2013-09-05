//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_ConcurrentMultiscaleProblem_hpp)
#define LCM_ConcurrentMultiscaleProblem_hpp

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
  /// \brief Definition for the ConcurrentMultiscale Problem
  ///
  class ConcurrentMultiscaleProblem : public Albany::AbstractProblem {
  public:

    typedef Intrepid::FieldContainer<RealType> FC;

    ///
    /// Default constructor
    ///
    ConcurrentMultiscaleProblem(const Teuchos::RCP<Teuchos::ParameterList>& params,
                     const Teuchos::RCP<ParamLib>& param_lib,
                     const int num_dims,
                     const Teuchos::RCP<const Epetra_Comm>& comm);
    ///
    /// Destructor
    ///
    virtual
    ~ConcurrentMultiscaleProblem();

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
    ConcurrentMultiscaleProblem(const ConcurrentMultiscaleProblem&);

    ///
    /// Private to prohibit copying
    ///
    ConcurrentMultiscaleProblem& operator=(const ConcurrentMultiscaleProblem&);

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

    //----------------------------------------------------------------------------
  protected:

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
    ///  Map of to indicate overlap block
    ///
    std::map< std::string, bool > coarse_overlap_map_;

    ///
    /// Flag to indicate overlap block
    ///
    std::map< std::string, bool > fine_overlap_map_;

    ///
    /// Map for the lagrange multiplier blocks
    ///
    std::map< std::string, bool > lm_overlap_map_;

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

#include "FieldNameMap.hpp"

#include "MechanicsResidual.hpp"
#include "Time.hpp"

// Constitutive Model Interface and parameters
#include "Kinematics.hpp"
#include "ConstitutiveModelInterface.hpp"
#include "ConstitutiveModelParameters.hpp"


//------------------------------------------------------------------------------
template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
Albany::ConcurrentMultiscaleProblem::
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
  *out << "In ConcurrentMultiscaleProblem::constructEvaluators" << std::endl;
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

  // get the intrepid basis for the given cell topology
  RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > >
    intrepidBasis = Albany::getIntrepidBasis(meshSpecs.ctd, composite);

  if (composite && 
      meshSpecs.ctd.dimension==3 && 
      meshSpecs.ctd.node_count==10) cellType = comp_cellType;

  Intrepid::DefaultCubatureFactory<RealType> cubFactory;
  RCP <Intrepid::Cubature<RealType> > cubature = 
    cubFactory.create(*cellType, meshSpecs.cubatureDegree);

  // Note that these are the volume element quantities
  num_nodes_ = intrepidBasis->getCardinality();
  const int workset_size = meshSpecs.worksetSize;

  num_dims_ = cubature->getDimension();
  num_pts_ = cubature->getNumPoints();
  num_vertices_ = num_nodes_;

#ifdef ALBANY_VERBOSE
  *out << "Field Dimensions: Workset=" << workset_size 
       << ", Vertices= " << num_vertices_
       << ", Nodes= " << num_nodes_
       << ", QuadPts= " << num_pts_
       << ", Dim= " << num_dims_ << std::endl;
#endif

  // Construct standard FEM evaluators with standard field names                
  RCP<Albany::Layouts> dl = 
    rcp(new Albany::Layouts(workset_size,num_vertices_,num_nodes_,num_pts_,num_dims_));
  std::string msg = "Data Layout Usage in Mechanics problems assume vecDim = num_dims_";
  TEUCHOS_TEST_FOR_EXCEPTION(dl->vectorAndGradientLayoutsAreEquivalent==false, 
                             std::logic_error,
                             msg);
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
  int offset = 0;
  // Temporary variable used numerous times below
  RCP<PHX::Evaluator<AlbanyTraits> > ev;

  // Define Field Names

  { 
    Teuchos::ArrayRCP<std::string> dof_names(1);
    Teuchos::ArrayRCP<std::string> resid_names(1);
    dof_names[0] = "Displacement";
    resid_names[0] = dof_names[0]+" Residual";

    fm0.template registerEvaluator<EvalT>
      (evalUtils.constructGatherSolutionEvaluator_noTransient(true, 
                                                              dof_names));

    fm0.template registerEvaluator<EvalT>
      (evalUtils.constructGatherCoordinateVectorEvaluator());

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

    fm0.template registerEvaluator<EvalT>
      (evalUtils.constructScatterResidualEvaluator(true, 
                                                   resid_names));
    offset += num_dims_;
  }

  if ( lm_overlap_map_[eb_name] ) { // add lagrange multiplier field
    Teuchos::ArrayRCP<std::string> dof_names(1);
    Teuchos::ArrayRCP<std::string> resid_names(1);
    dof_names[0] = "lagrange_multiplier";
    resid_names[0] = dof_names[0]+"_residual";

    fm0.template registerEvaluator<EvalT>
      (evalUtils.constructGatherSolutionEvaluator_noTransient(true, 
                                                              dof_names,
                                                              offset));

    fm0.template registerEvaluator<EvalT>
      (evalUtils.constructGatherCoordinateVectorEvaluator());

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

    offset += num_dims_;
  } 

  // generate the field name map to deal with outputing surface element info
  LCM::FieldNameMap field_name_map(false);
  RCP<std::map<std::string, std::string> > fnm = field_name_map.getMap();
  std::string cauchy       = (*fnm)["Cauchy_Stress"];
  std::string Fp           = (*fnm)["Fp"];
  std::string eqps         = (*fnm)["eqps"];
  
  { // Time
    RCP<ParameterList> p = rcp(new ParameterList("Time"));
    p->set<std::string>("Time Name", "Time");
    p->set<std::string>("Delta Time Name", "Delta Time");
    p->set< RCP<DataLayout> >("Workset Scalar Data Layout", dl->workset_scalar);
    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    p->set<bool>("Disable Transient", true);
    ev = rcp(new LCM::Time<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
    p = stateMgr.registerStateVariable("Time",
                                       dl->workset_scalar, 
                                       dl->dummy, 
                                       eb_name, 
                                       "scalar", 
                                       0.0, 
                                       true);
    ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if ( lm_overlap_map_[eb_name] ) {
    RCP<ParameterList> p = rcp(new ParameterList("Save Lagrange Multiplier"));
    p = stateMgr.registerStateVariable("Lagrange Multiplier",
                                       dl->qp_scalar, 
                                       dl->dummy, 
                                       eb_name, 
                                       "scalar", 
                                       0.0, 
                                       true,
                                       false);
    ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (have_source_) { // Source
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<std::string>("Source Name", "Source");
    p->set<std::string>("Variable Name", "Displacement");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

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

    p->set<Teuchos::ParameterList*>("Material Parameters", &param_list);

    RCP<LCM::ConstitutiveModelParameters<EvalT,AlbanyTraits> > cmpEv = 
      rcp(new LCM::ConstitutiveModelParameters<EvalT,AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(cmpEv);
  }

  {
    RCP<ParameterList> p = rcp(new ParameterList("Constitutive Model Interface"));
    std::string matName = material_db_->getElementBlockParam<std::string>(eb_name,"material");
    Teuchos::ParameterList& param_list = 
      material_db_->getElementBlockSublist(eb_name,matName);

    param_list.set<RCP<std::map<std::string, std::string> > >("Name Map", fnm);
    p->set<Teuchos::ParameterList*>("Material Parameters", &param_list);

    RCP<LCM::ConstitutiveModelInterface<EvalT,AlbanyTraits> > cmiEv =
      rcp(new LCM::ConstitutiveModelInterface<EvalT,AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(cmiEv);

    // register state variables
    for (int sv(0); sv < cmiEv->getNumStateVars(); ++sv) {
      cmiEv->fillStateVariableStruct(sv);
      p = stateMgr.registerStateVariable(cmiEv->getName(),
                                         cmiEv->getLayout(),
                                         dl->dummy, 
                                         eb_name, 
                                         cmiEv->getInitType(),
                                         cmiEv->getInitValue(),
                                         cmiEv->getStateFlag(),
                                         cmiEv->getOutputFlag());
      ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }
  }
 
  
  { // Kinematics quantities
    RCP<ParameterList> p = rcp(new ParameterList("Kinematics"));

    // set flags to optionally volume average J with a weighted average
    if ( material_db_->isElementBlockParam(eb_name,"Weighted Volume Average J") ){
      p->set<bool>("Weighted Volume Average J",
                   material_db_->
                   getElementBlockParam<bool>(eb_name,
                                              "Weighted Volume Average J") );
    }

    if ( material_db_->isElementBlockParam(eb_name,
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
    p->set<std::string>("Gradient QP Variable Name", "Displacement Gradient");

    //Outputs: F, J
    p->set<std::string>("DefGrad Name", "F"); //dl->qp_tensor also
    p->set<std::string>("DetDefGrad Name", "J"); 

    //ev = rcp(new LCM::DefGrad<EvalT,AlbanyTraits>(*p));
    ev = rcp(new LCM::Kinematics<EvalT,AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);


    // optional output
    bool outputFlag(false);
    if ( material_db_->isElementBlockParam(eb_name,"Output Deformation Gradient") )
      outputFlag = 
        material_db_->getElementBlockParam<bool>(eb_name,"Output Deformation Gradient");

    p = stateMgr.registerStateVariable("F",
                                       dl->qp_tensor, 
                                       dl->dummy, 
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
    if (outputFlag) {
      p = stateMgr.registerStateVariable("J",
                                         dl->qp_scalar,
                                         dl->dummy,
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
                                         dl->qp_tensor,
                                         dl->dummy,
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
                                         dl->qp_tensor,
                                         dl->dummy,
                                         eb_name,
                                         "scalar",
                                         0.0,
                                         outputFlag);
      ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }

  }

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

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    //Output
    p->set<std::string>("Residual Name", "Displacement Residual");
    ev = rcp(new LCM::MechanicsResidual<EvalT,AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (fieldManagerChoice == Albany::BUILD_RESID_FM)  {
    Teuchos::RCP<const PHX::FieldTag> ret_tag;
    {
      PHX::Tag<typename EvalT::ScalarT> res_tag("Scatter", dl->dummy);
      fm0.requireField<EvalT>(res_tag);
      ret_tag = res_tag.clone();
    }
    if ( lm_overlap_map_[eb_name] ) {
      PHX::Tag<typename EvalT::ScalarT> lagrange_multiplier_tag("Scatter Lagrange Multiplier", dl->dummy);
      fm0.requireField<EvalT>(lagrange_multiplier_tag);
      ret_tag = lagrange_multiplier_tag.clone();
    }
    return ret_tag;
  }
  else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM) {
    Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
    return respUtils.constructResponses(fm0, *responseList, stateMgr);
  }

  return Teuchos::null;
}

#endif
