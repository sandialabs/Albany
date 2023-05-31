//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_THERMALPROBLEM_HPP
#define ALBANY_THERMALPROBLEM_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_AbstractProblem.hpp"

#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"
#include "Albany_ProblemUtils.hpp"

#include "PHAL_ConvertFieldType.hpp"
#include "Albany_MaterialDatabase.hpp"

#include "PHAL_RandomPhysicalParameter.hpp"

#include "PHAL_IsAvailable.hpp"

namespace Albany {

  /*!
   * \brief Abstract interface for representing a 1-D finite element
   * problem.
   */
  class ThermalProblem : public AbstractProblem {
  public:

    //! Default constructor
    ThermalProblem(
      const Teuchos::RCP<Teuchos::ParameterList>& params,
      const Teuchos::RCP<ParamLib>& paramLib,
      //const Teuchos::RCP<DistributedParameterLibrary>& distParamLib,
      const int numDim_,
      const Teuchos::RCP<const Teuchos_Comm >& commT_); 

    //! Destructor
    ~ThermalProblem();

    //! Return number of spatial dimensions
    virtual int spatialDimension() const { return numDim; }
    
    //! Get boolean telling code if SDBCs are utilized  
    virtual bool useSDBCs() const {return use_sdbcs_; }

    //! Build the PDE instantiations, boundary conditions, and initial solution
    virtual void buildProblem(
      Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecs> >  meshSpecs,
      StateManager& stateMgr);

    // Build evaluators
    virtual Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
    buildEvaluators(
      PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
      const Albany::MeshSpecs& meshSpecs,
      Albany::StateManager& stateMgr,
      Albany::FieldManagerChoice fmchoice,
      const Teuchos::RCP<Teuchos::ParameterList>& responseList);

    //! Each problem must generate it's list of valid parameters
    Teuchos::RCP<const Teuchos::ParameterList> getValidProblemParameters() const;

  private:

    //! Private to prohibit copying
    ThermalProblem(const ThermalProblem&);

    //! Private to prohibit copying
    ThermalProblem& operator=(const ThermalProblem&);

  public:

    //! Main problem setup routine. Not directly called, but indirectly by following functions
    template <typename EvalT>
    Teuchos::RCP<const PHX::FieldTag>
    constructEvaluators(
      PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
      const Albany::MeshSpecs& meshSpecs,
      Albany::StateManager& stateMgr,
      Albany::FieldManagerChoice fmchoice,
      const Teuchos::RCP<Teuchos::ParameterList>& responseList);

    void constructDirichletEvaluators(const std::vector<std::string>& nodeSetIDs);
    void constructNeumannEvaluators(const Teuchos::RCP<Albany::MeshSpecs>& meshSpecs);

  protected:

   int numDim;
   Teuchos::Array<double> kappa;  // thermal conductivity
   double                 C;      // heat capacity
   double                 rho;    // density
   std::string            thermal_source; //thermal source name 
   bool                   conductivityIsDistParam;

   //! Problem PL 
   const Teuchos::RCP<Teuchos::ParameterList> params; 

   Teuchos::RCP<const Teuchos_Comm> commT; 

   Teuchos::RCP<Albany::Layouts> dl;

   /// Boolean marking whether SDBCs are used 
   bool use_sdbcs_; 

  };

} //  namespace Albany

#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_EvaluatorUtils.hpp"
#include "Albany_ResponseUtilities.hpp"
//#include "PHAL_Neumann.hpp"
#include "PHAL_ThermalResid.hpp"
#include "PHAL_SharedParameter.hpp"
#include "Albany_ParamEnum.hpp"

template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
Albany::ThermalProblem::constructEvaluators(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecs& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fieldManagerChoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  using PHAL::AlbanyTraits;
  using PHX::DataLayout;
  using std::string;
  using std::vector;
  using Teuchos::ParameterList;
  using Teuchos::RCP;
  using Teuchos::rcp;

  const CellTopologyData* const elem_top = &meshSpecs.ctd;

  RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>> intrepidBasis =
      Albany::getIntrepid2Basis(*elem_top);
  RCP<shards::CellTopology> cellType = rcp(new shards::CellTopology(elem_top));

  int const numNodes    = intrepidBasis->getCardinality();
  int const worksetSize = meshSpecs.worksetSize;
  int const cubDegree = params->get("Cubature Degree", 3);
  Intrepid2::DefaultCubatureFactory     cubFactory;
  RCP<Intrepid2::Cubature<PHX::Device>> cellCubature =
      cubFactory.create<PHX::Device, RealType, RealType>(
          *cellType, cubDegree);

  int const numQPtsCell = cellCubature->getNumPoints();
  int const numVertices = cellType->getNodeCount();

  // Get the solution method type
  SolutionMethodType SolutionType = getSolutionMethod();

  ALBANY_ASSERT(
      SolutionType != SolutionMethodType::Unknown,
      "Solution Method must be Steady, Transient, "
      "Continuation");
  
  if (SolutionType == SolutionMethodType::Transient) { // Problem is transient
    ALBANY_ASSERT(
        number_of_time_deriv == 1,
        "You are using a transient solution method in Albany::ThermalProblem but number of time derivatives != 1!"); 
  }
  else { //Problem is steady
    ALBANY_ASSERT(
        number_of_time_deriv == 0,
        "You are using a steady solution method in Albany::ThermalProblem but number of time derivatives > 0!");
  } 

  *out << "Field Dimensions: Workset=" << worksetSize
       << ", Vertices= " << numVertices << ", Nodes= " << numNodes
       << ", QuadPts= " << numQPtsCell << ", Dim= " << numDim << std::endl;

  dl = rcp(new Albany::Layouts(
      worksetSize, numVertices, numNodes, numQPtsCell, numDim));
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);

  // Temporary variable used numerous times below
  Teuchos::RCP<PHX::Evaluator<AlbanyTraits>> ev;

  Teuchos::ArrayRCP<string> dof_names(neq);
  dof_names[0] = "Temperature";
  Teuchos::ArrayRCP<string> dof_names_dot(neq);
  dof_names_dot[0] = "Temperature_dot";
  Teuchos::ArrayRCP<string> resid_names(neq);
  resid_names[0] = "Temperature Residual";

  fm0.template registerEvaluator<EvalT>(
      evalUtils.constructGatherSolutionEvaluator(
          false, dof_names, dof_names_dot));

  fm0.template registerEvaluator<EvalT>(
      evalUtils.constructScatterResidualEvaluator(false, resid_names));

  fm0.template registerEvaluator<EvalT>(
      evalUtils.constructGatherCoordinateVectorEvaluator());

  fm0.template registerEvaluator<EvalT>(
      evalUtils.constructMapToPhysicalFrameEvaluator(cellType, cellCubature));

  fm0.template registerEvaluator<EvalT>(
      evalUtils.constructComputeBasisFunctionsEvaluator(
          cellType, intrepidBasis, cellCubature));

  for (unsigned int i = 0; i < neq; i++) {
    fm0.template registerEvaluator<EvalT>(
        evalUtils.constructDOFInterpolationEvaluator(dof_names[i]));

    fm0.template registerEvaluator<EvalT>(
        evalUtils.constructDOFInterpolationEvaluator(dof_names_dot[i]));

    fm0.template registerEvaluator<EvalT>(
        evalUtils.constructDOFGradInterpolationEvaluator(dof_names[i]));
  }

  Teuchos::RCP<Albany::ScalarParameterAccessors<EvalT>> accessors = this->getAccessors()->template at<EvalT>();

  {
    RCP<ParameterList> p = rcp(new ParameterList("Theta 0"));
    p->set< RCP<ParamLib> >("Parameter Library", paramLib);
    const std::string param_name = "Theta 0";
    p->set<std::string>("Parameter Name", param_name);
    p->set<Teuchos::RCP<ScalarParameterAccessors<EvalT>>>("Accessors", accessors);
    p->set<const Teuchos::ParameterList*>("Parameters List", &params->sublist("Parameters"));
    p->set<double>("Default Nominal Value", 0.);
    RCP<PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits>> ptr_theta_0;
    ptr_theta_0 = rcp(new PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ptr_theta_0);
  }
  {
    RCP<ParameterList> p = rcp(new ParameterList("Theta 1"));
    p->set< RCP<ParamLib> >("Parameter Library", paramLib);
    const std::string param_name = "Theta 1";
    p->set<std::string>("Parameter Name", param_name);
    p->set<Teuchos::RCP<ScalarParameterAccessors<EvalT>>>("Accessors", accessors);
    p->set<const Teuchos::ParameterList*>("Parameters List", &params->sublist("Parameters"));
    p->set<double>("Default Nominal Value", 0.);
    RCP<PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits>> ptr_theta_1;
    ptr_theta_1 = rcp(new PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ptr_theta_1);
  }

  if (params->isSublist("Random Parameters")) {
    auto rparams = params->sublist("Random Parameters");
    int nrparams = rparams.get<int>("Number Of Parameters");
    for (int i_rparams=0; i_rparams<nrparams; ++i_rparams)
    {
      auto rparams_i = rparams.sublist(util::strint("Parameter",i_rparams));

      RCP<ParameterList> p = rcp(new ParameterList("Theta 1"));
      p->set< RCP<ParamLib> >("Parameter Library", paramLib);
      const std::string param_name = rparams_i.get<std::string>("Name");
      p->set<std::string>("Parameter Name", param_name); //output name
      const std::string rparam_name = rparams_i.get<std::string>("Standard Normal Parameter");
      p->set<std::string>("Random Parameter Name", rparam_name); //input name
      p->set<const Teuchos::ParameterList*>("Parameters List", &params->sublist("Parameters"));
      p->set<const Teuchos::ParameterList*>("Distribution", &rparams_i.sublist("Distribution"));
      RCP<PHAL::RandomPhysicalParameter<EvalT,PHAL::AlbanyTraits>> ptr_rparam;
      ptr_rparam = rcp(new PHAL::RandomPhysicalParameter<EvalT,PHAL::AlbanyTraits>(*p,dl));
      fm0.template registerEvaluator<EvalT>(ptr_rparam);
    }
  }

  if (!conductivityIsDistParam) {  
    //Shared parameter for sensitivity analysis: kappa_x
    const std::string param_name = "kappa_x Parameter";

    if(!PHAL::is_field_evaluated<EvalT>(fm0, param_name, dl->shared_param)) {
      RCP<ParameterList> p = rcp(new ParameterList("Thermal Conductivity: kappa_x"));
      p->set< RCP<ParamLib> >("Parameter Library", paramLib);
      p->set<std::string>("Parameter Name", param_name);
      p->set<Teuchos::RCP<ScalarParameterAccessors<EvalT>>>("Accessors", accessors);
      p->set<const Teuchos::ParameterList*>("Parameters List", &params->sublist("Parameters"));
      p->set<double>("Default Nominal Value", kappa[0]);
      RCP<PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits>> ptr_kappa_x;
      ptr_kappa_x = rcp(new PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits>(*p,dl));
      fm0.template registerEvaluator<EvalT>(ptr_kappa_x);
    }

    if (numDim > 1) {  //Shared parameter for sensitivity analysis: kappa_y
      const std::string param_name = "kappa_y Parameter";

      if(!PHAL::is_field_evaluated<EvalT>(fm0, param_name, dl->shared_param)) {
        RCP<ParameterList> p = rcp(new ParameterList("Thermal Conductivity: kappa_y"));
        p->set< RCP<ParamLib> >("Parameter Library", paramLib);
        p->set<std::string>("Parameter Name", param_name);
        p->set<Teuchos::RCP<ScalarParameterAccessors<EvalT>>>("Accessors", accessors);
        p->set<const Teuchos::ParameterList*>("Parameters List", &params->sublist("Parameters"));
        p->set<double>("Default Nominal Value", kappa[1]);
        RCP<PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits>> ptr_kappa_y;
        ptr_kappa_y = rcp(new PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits>(*p,dl));
        fm0.template registerEvaluator<EvalT>(ptr_kappa_y);
      }
    }
    if (numDim > 2) {  //Shared parameter for sensitivity analysis: kappa_z
      const std::string param_name = "kappa_z Parameter";

      if(!PHAL::is_field_evaluated<EvalT>(fm0, param_name, dl->shared_param)) {
        RCP<ParameterList> p = rcp(new ParameterList("Thermal Conductivity: kappa_z"));
        p->set< RCP<ParamLib> >("Parameter Library", paramLib);        
        p->set<std::string>("Parameter Name", param_name);
        p->set<Teuchos::RCP<ScalarParameterAccessors<EvalT>>>("Accessors", accessors);
        p->set<const Teuchos::ParameterList*>("Parameters List", &params->sublist("Parameters"));
        p->set<double>("Default Nominal Value", kappa[2]);
        RCP<PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits>> ptr_kappa_z;
        ptr_kappa_z = rcp(new PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits>(*p,dl));
        fm0.template registerEvaluator<EvalT>(ptr_kappa_z);
      }
    }
  }
  else //conductivityIsDistParam
  {
    RCP<ParameterList> p = rcp(new ParameterList);
    Albany::StateStruct::MeshFieldEntity entity = Albany::StateStruct::NodalDistParameter;
    std::string stateName = "thermal_conductivity";
    std::string fieldName = "ThermalConductivity";
    p = stateMgr.registerStateVariable(stateName, dl->node_scalar, meshSpecs.ebName, true, &entity, "");

    //Gather parameter (similarly to what done with the solution)
    ev = evalUtils.constructGatherScalarNodalParameter(stateName,fieldName);
    fm0.template registerEvaluator<EvalT>(ev);

    // Scalar Nodal parameter is stored as a ParamScalarT, while the residual evaluator expect a ScalarT.
    // Hence, if ScalarT!=ParamScalarT, we need to convert the field into a ScalarT 
    if(!std::is_same<typename EvalT::ScalarT,typename EvalT::ParamScalarT>::value) {
      p->set<Teuchos::RCP<PHX::DataLayout> >("Data Layout", dl->node_scalar);
      p->set<std::string>("Field Name", fieldName);
      ev = Teuchos::rcp(new PHAL::ConvertFieldTypePSTtoST<EvalT,PHAL::AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }
    fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFInterpolationEvaluator(fieldName));
    /*stateName = "thermal_conductivity_sensitivity";
    p = stateMgr.registerStateVariable(stateName, dl->node_scalar, meshSpecs.ebName, true, &entity, "");*/
  }

  {  // Temperature Resid
    RCP<ParameterList> p = rcp(new ParameterList("Temperature Resid"));

    // Input
    p->set<string>("Weighted BF Name", "wBF");
    p->set<string>("QP Time Derivative Variable Name", "Temperature_dot");
    p->set<string>("Gradient QP Variable Name", "Temperature Gradient");
    p->set<string>("Weighted Gradient BF Name", "wGrad BF");
    p->set<string>("Source Name", "Temperature Source");
    p->set<string>("QP Coordinate Vector Name", "Coord Vec");

    p->set<RCP<DataLayout>>("Node QP Scalar Data Layout", dl->node_qp_scalar);
    p->set<RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);
    p->set<RCP<DataLayout>>("QP Vector Data Layout", dl->qp_vector);
    p->set<RCP<DataLayout>>("Node QP Vector Data Layout", dl->node_qp_vector);
    p->set<RCP<DataLayout>>("Node Scalar Data Layout", dl->node_scalar);
    if (!conductivityIsDistParam) {  
      p->set<std::string>("Thermal Conductivity: kappa_x","kappa_x Parameter");
      if (numDim > 1) p->set<std::string>("Thermal Conductivity: kappa_y","kappa_y Parameter");
      if (numDim > 2) p->set<std::string>("Thermal Conductivity: kappa_z","kappa_z Parameter");
    }
    else {
      p->set<string>("ThermalConductivity Name", "ThermalConductivity");
      p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    }
    if (SolutionType != SolutionMethodType::Transient) {
      p->set<bool>("Disable Transient", true);
    }
    else {
      p->set<bool>("Disable Transient", false);
    }
    p->set<double>("Heat Capacity", C);
    p->set<double>("Density", rho);
    p->set<bool>("Distributed Thermal Conductivity", conductivityIsDistParam);
    p->set<std::string>("Thermal Source", thermal_source); 

    // Output
    p->set<string>("Residual Name", "Temperature Residual");

    ev = rcp(new PHAL::ThermalResid<EvalT, AlbanyTraits>(*p, dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (fieldManagerChoice == Albany::BUILD_RESID_FM) {
    PHX::Tag<typename EvalT::ScalarT> res_tag("Scatter", dl->dummy);
    fm0.requireField<EvalT>(res_tag);
    return res_tag.clone();
  }

  else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM) {
    Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
    return respUtils.constructResponses(
        fm0, *responseList, Teuchos::null, stateMgr);
  }

  return Teuchos::null;
}

#endif 
