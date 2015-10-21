//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_STOKESFOPROBLEM_HPP
#define FELIX_STOKESFOPROBLEM_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_AbstractProblem.hpp"

#ifdef ALBANY_EPETRA
  #include "FELIX_GatherVerticallyAveragedVelocity.hpp"
#endif

#include "Phalanx.hpp"
#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"
//#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_SaveStateField.hpp"
#include "PHAL_LoadStateField.hpp"

//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN

namespace FELIX {

template<typename EvalT,typename Traits>
class HomotopyParamValue
{
public:
    static typename EvalT::ScalarT* value;
};

template<typename EvalT,typename Traits>
typename EvalT::ScalarT* HomotopyParamValue<EvalT,Traits>::value = NULL;

  /*!
   * \brief Abstract interface for representing a 1-D finite element
   * problem.
   */
  class StokesFO : public Albany::AbstractProblem {
  public:

    //! Default constructor
    StokesFO(const Teuchos::RCP<Teuchos::ParameterList>& params,
     const Teuchos::RCP<ParamLib>& paramLib,
     const int numDim_);

    //! Destructor
    ~StokesFO();

    //! Return number of spatial dimensions
    virtual int spatialDimension() const { return numDim; }

    //! Build the PDE instantiations, boundary conditions, and initial solution
    virtual void buildProblem(
      Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
      Albany::StateManager& stateMgr);

    // Build evaluators
    virtual Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
    buildEvaluators(
      PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
      const Albany::MeshSpecsStruct& meshSpecs,
      Albany::StateManager& stateMgr,
      Albany::FieldManagerChoice fmchoice,
      const Teuchos::RCP<Teuchos::ParameterList>& responseList);

    //! Each problem must generate it's list of valide parameters
    Teuchos::RCP<const Teuchos::ParameterList> getValidProblemParameters() const;

  private:

    //! Private to prohibit copying
    StokesFO(const StokesFO&);

    //! Private to prohibit copying
    StokesFO& operator=(const StokesFO&);

  public:

    //! Main problem setup routine. Not directly called, but indirectly by following functions
    template <typename EvalT> Teuchos::RCP<const PHX::FieldTag>
    constructEvaluators(
      PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
      const Albany::MeshSpecsStruct& meshSpecs,
      Albany::StateManager& stateMgr,
      Albany::FieldManagerChoice fmchoice,
      const Teuchos::RCP<Teuchos::ParameterList>& responseList);

    void constructDirichletEvaluators(const Albany::MeshSpecsStruct& meshSpecs);
    void constructNeumannEvaluators(const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs);

  protected:

  // Used to build basal friction evaluator for all evaluation types
  struct ConstructBasalEvaluatorOp
  {
      StokesFO& prob_;
      std::vector<Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > >& evaluators_;

      ConstructBasalEvaluatorOp (StokesFO& prob,
                                 std::vector<Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > >& evaluators) :
          prob_(prob), evaluators_(evaluators) {}
      template<typename T>
      void operator() (T x) {
      evaluators_.push_back(prob_.template buildBasalFrictionCoefficientEvaluator<T>());
      evaluators_.push_back(prob_.template buildSlidingVelocityEvaluator<T>());
      evaluators_.push_back(prob_.template buildEffectivePressureEvaluator<T>());
      }
  };

  template<typename EvalT>
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> >
  buildBasalFrictionCoefficientEvaluator ();

  template<typename EvalT>
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> >
  buildSlidingVelocityEvaluator();

  template<typename EvalT>
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> >
  buildEffectivePressureEvaluator ();

    int numDim;
    double gravity;  //gravity
    double rho;  //ice density
    double rho_w;  //water density
    Teuchos::RCP<Albany::Layouts> dl;

  };

} // Namespace FELIX

#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"

#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_EvaluatorUtils.hpp"
#include "Albany_ResponseUtilities.hpp"

#include "FELIX_StokesFOResid.hpp"
#ifdef CISM_HAS_FELIX
#include "FELIX_CismSurfaceGradFO.hpp"
#endif
#include "FELIX_StokesFOBodyForce.hpp"
#include "FELIX_ViscosityFO.hpp"
#include "FELIX_FieldNorm.hpp"
#include "FELIX_BasalFrictionCoefficient.hpp"
#include "FELIX_EffectivePressure.hpp"
#include "PHAL_Neumann.hpp"
#include "PHAL_Source.hpp"
#include <type_traits>


template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
FELIX::StokesFO::constructEvaluators(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
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
  using std::string;
  using std::map;
  using PHAL::AlbanyTraits;

  RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > >
    intrepidBasis = Albany::getIntrepidBasis(meshSpecs.ctd);
  RCP<shards::CellTopology> cellType = rcp(new shards::CellTopology (&meshSpecs.ctd));

  const int numNodes = intrepidBasis->getCardinality();
  const int worksetSize = meshSpecs.worksetSize;

  Intrepid::DefaultCubatureFactory<RealType> cubFactory;
  RCP <Intrepid::Cubature<RealType> > cubature = cubFactory.create(*cellType, meshSpecs.cubatureDegree);

  const int numQPts = cubature->getNumPoints();
  const int numVertices = cellType->getNodeCount();
  int vecDim = neq;
  std::string elementBlockName = meshSpecs.ebName;

#ifdef OUTPUT_TO_SCREEN
  *out << "Field Dimensions: Workset=" << worksetSize
       << ", Vertices= " << numVertices
       << ", Nodes= " << numNodes
       << ", QuadPts= " << numQPts
       << ", Dim= " << numDim
       << ", vecDim= " << vecDim << std::endl;
#endif

   Albany::StateStruct::MeshFieldEntity entity;
   dl = rcp(new Albany::Layouts(worksetSize,numVertices,numNodes,numQPts,numDim, vecDim));
   Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
   int offset=0;

   entity= Albany::StateStruct::ElemData;

   // Temporary variable used numerous times below
      RCP<PHX::Evaluator<AlbanyTraits> > ev;
   {
     std::string stateName("temperature");
     RCP<ParameterList> p = stateMgr.registerStateVariable(stateName, dl->cell_scalar2, elementBlockName,true, &entity);
     ev = rcp(new PHAL::LoadStateField<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
   }

   {
     std::string stateName("flow_factor");
     RCP<ParameterList> p = stateMgr.registerStateVariable(stateName, dl->cell_scalar2, elementBlockName,true, &entity);
     ev = rcp(new PHAL::LoadStateField<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
   }

   entity= Albany::StateStruct::NodalDataToElemNode;

   {
     std::string stateName("surface_height");
     RCP<ParameterList> p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName,true, &entity);
     ev = rcp(new PHAL::LoadStateField<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
   }
#ifdef CISM_HAS_FELIX
   {
     std::string stateName("xgrad_surface_height"); //ds/dx which can be passed from CISM (defined at nodes)
     RCP<ParameterList> p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName,true, &entity);
     ev = rcp(new PHAL::LoadStateField<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
   }
   {
     std::string stateName("ygrad_surface_height"); //ds/dy which can be passed from CISM (defined at nodes)
     RCP<ParameterList> p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName,true, &entity);
     ev = rcp(new PHAL::LoadStateField<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
   }
#endif
   {
     std::string stateName("thickness");
     RCP<ParameterList> p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName,true, &entity);
     ev = rcp(new PHAL::LoadStateField<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
   }

   {
     std::string stateName("basal_friction");
     RCP<ParameterList> p;

     //Finding whether basal friction is a distributed parameter
     bool isStateAParameter(false);
     const std::string* meshPart;
     const std::string emptyString("");
     if(this->params->isSublist("Distributed Parameters")) {
       Teuchos::ParameterList& dist_params_list =  this->params->sublist("Distributed Parameters");
       Teuchos::ParameterList* param_list;
       int numParams = dist_params_list.get<int>("Number of Parameter Vectors",0);
       for(int p_index=0; p_index< numParams; ++p_index) {
         std::string parameter_sublist_name = Albany::strint("Distributed Parameter", p_index);
         if(dist_params_list.isSublist(parameter_sublist_name)) {
           param_list = &dist_params_list.sublist(parameter_sublist_name);
           if(param_list->get<std::string>("Name", emptyString) == stateName) {
             meshPart = &param_list->get<std::string>("Mesh Part",emptyString);
             isStateAParameter = true;
             break;
           }
         } else {
           if(stateName == dist_params_list.get(Albany::strint("Parameter", p_index), emptyString)) {
             isStateAParameter = true;
             meshPart = &emptyString;
             break;
           }
         }
       }
     }

     if(isStateAParameter) { //basal friction is a distributed parameter
       entity= Albany::StateStruct::NodalDistParameter;
       p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName,true, &entity, *meshPart);
       fm0.template registerEvaluator<EvalT>
           (evalUtils.constructGatherScalarNodalParameter(stateName));
       std::stringstream key; key << stateName <<  "Is Distributed Parameter";
       this->params->set<int>(key.str(), 1);
     } else {
       entity= Albany::StateStruct::NodalDataToElemNode;
       p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName,true, &entity);
       ev = rcp(new PHAL::LoadStateField<EvalT,AlbanyTraits>(*p));
       fm0.template registerEvaluator<EvalT>(ev);
     }
   }

   {
      std::string stateName("bed_topography");
      entity= Albany::StateStruct::NodalDataToElemNode;
      RCP<ParameterList> p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName,true, &entity);
      ev = rcp(new PHAL::LoadStateField<EvalT,AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }


#if defined(CISM_HAS_FELIX) || defined(MPAS_HAS_FELIX)
   {
    // Here is how to register the field for dirichlet condition.
    std::string stateName("dirichlet_field");
    entity= Albany::StateStruct::NodalDistParameter;
    stateMgr.registerStateVariable(stateName, dl->node_vector, elementBlockName, true, &entity, "");
   }
#endif


   // Define Field Names

  Teuchos::ArrayRCP<std::string> dof_names(1);
  Teuchos::ArrayRCP<std::string> dof_names_dot(1);
  Teuchos::ArrayRCP<std::string> resid_names(1);
  dof_names[0] = "Velocity";
  //dof_names_dot[0] = dof_names[0]+"_dot";
  resid_names[0] = "Stokes Residual";
  fm0.template registerEvaluator<EvalT>
  (evalUtils.constructGatherSolutionEvaluator_noTransient(true, dof_names, offset));

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructDOFVecInterpolationEvaluator(dof_names[0]));

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructDOFVecGradInterpolationEvaluator(dof_names[0]));

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructScatterResidualEvaluator(true, resid_names,offset, "Scatter Stokes"));
  offset += numDim;

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructGatherCoordinateVectorEvaluator());

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructMapToPhysicalFrameEvaluator(cellType, cubature));

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructComputeBasisFunctionsEvaluator(cellType, intrepidBasis, cubature));

  std::string sh = "surface_height";
  fm0.template registerEvaluator<EvalT>
      (evalUtils.constructDOFInterpolationEvaluator(sh));
  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructDOFGradInterpolationEvaluator_noDeriv(sh));


  { // FO Stokes Resid
    RCP<ParameterList> p = rcp(new ParameterList("Stokes Resid"));

    //Input
    p->set<std::string>("Weighted BF Name", "wBF");
    p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
    p->set<std::string>("QP Variable Name", "Velocity");
    p->set<std::string>("QP Time Derivative Variable Name", "Velocity_dot");
    p->set<std::string>("Gradient QP Variable Name", "Velocity Gradient");
    p->set<std::string>("Body Force Name", "Body Force");
    p->set<std::string>("FELIX Viscosity QP Variable Name", "FELIX Viscosity");

    p->set<std::string>("Coordinate Vector Name", "Coord Vec");

    Teuchos::ParameterList& mapParamList = params->sublist("Stereographic Map");
    p->set<Teuchos::ParameterList*>("Stereographic Map", &mapParamList);

    Teuchos::ParameterList& paramList = params->sublist("Equation Set");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    //Output
    p->set<std::string>("Residual Name", "Stokes Residual");

    ev = rcp(new FELIX::StokesFOResid<EvalT,AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }
  { // FELIX viscosity
    RCP<ParameterList> p = rcp(new ParameterList("FELIX Viscosity"));

    //Input
    p->set<std::string>("Coordinate Vector Name", "Coord Vec");
    p->set<std::string>("Gradient QP Variable Name", "Velocity Gradient");
    p->set<std::string>("QP Variable Name", "Velocity");
    p->set<std::string>("temperature Name", "temperature");
    p->set<std::string>("flow_factor Name", "flow_factor");

    Teuchos::ParameterList& mapParamList = params->sublist("Stereographic Map");
    p->set<Teuchos::ParameterList*>("Stereographic Map", &mapParamList);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("FELIX Viscosity");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    //Output
    p->set<std::string>("FELIX Viscosity QP Variable Name", "FELIX Viscosity");

    ev = rcp(new FELIX::ViscosityFO<EvalT,AlbanyTraits>(*p,dl));

    typename EvalT::ScalarT** value = &HomotopyParamValue<EvalT,PHAL::AlbanyTraits>::value;
    if (*value==NULL)
    {
        typedef typename Sacado::ParameterAccessor<EvalT, SPL_Traits> sacado_accessor_type;
        sacado_accessor_type* pa_ptr;
        pa_ptr = dynamic_cast<sacado_accessor_type*>(&(*ev));
        if (pa_ptr==0)
        {
            std::cout << "Error! Cannot cast the pointer...\n";
            std::abort();
        }
        *value = &pa_ptr->getValue("Glen's Law Homotopy Parameter");
    }
    fm0.template registerEvaluator<EvalT>(ev);

  }

  // Sliding velocity calculation
  {
    ev = buildSlidingVelocityEvaluator<EvalT>();
    fm0.template registerEvaluator<EvalT>(ev);
  }
  // Effective pressur
  {
    ev = buildEffectivePressureEvaluator<EvalT>();
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // FELIX basal friction coefficient
  {
    ev = buildBasalFrictionCoefficientEvaluator<EvalT>();
    fm0.template registerEvaluator<EvalT>(ev);
  }
#ifdef CISM_HAS_FELIX
  { // FELIX surface gradient from CISM
    RCP<ParameterList> p = rcp(new ParameterList("FELIX Surface Gradient"));

    //Input
    p->set<std::string>("xgrad_surface_height Name", "xgrad_surface_height");
    p->set<std::string>("ygrad_surface_height Name", "ygrad_surface_height");
    p->set<std::string>("BF Name", "BF");

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("FELIX Surface Gradient");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    //Output
    p->set<std::string>("FELIX Surface Gradient QP Name", "FELIX Surface Gradient");

    ev = rcp(new FELIX::CismSurfaceGradFO<EvalT,AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);

  }
#endif

  { // Body Force
    RCP<ParameterList> p = rcp(new ParameterList("Body Force"));

    //Input
    p->set<std::string>("FELIX Viscosity QP Variable Name", "FELIX Viscosity");
#ifdef CISM_HAS_FELIX
    p->set<std::string>("FELIX Surface Gradient QP Variable Name", "FELIX Surface Gradient");
#endif
    p->set<std::string>("Coordinate Vector Name", "Coord Vec");
    p->set<std::string>("surface_height Gradient Name", "surface_height Gradient");
    p->set<std::string>("surface_height Name", "surface_height");

    Teuchos::ParameterList& paramList = params->sublist("Body Force");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    Teuchos::ParameterList& mapParamList = params->sublist("Stereographic Map");
    p->set<Teuchos::ParameterList*>("Stereographic Map", &mapParamList);

    Teuchos::ParameterList& physParamList = params->sublist("FELIX Physical Parameters");
    p->set<Teuchos::ParameterList*>("Physical Parameter List", &physParamList);

    //Output
    p->set<std::string>("Body Force Name", "Body Force");

    ev = rcp(new FELIX::StokesFOBodyForce<EvalT,AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (fieldManagerChoice == Albany::BUILD_RESID_FM)
  {
    PHX::Tag<typename EvalT::ScalarT> res_tag("Scatter Stokes", dl->dummy);
    fm0.requireField<EvalT>(res_tag);
  }
  else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM)
  {
    RCP<const Albany::MeshSpecsStruct> meshSpecsPtr = Teuchos::rcpFromRef(meshSpecs);

    RCP<ParameterList> paramList = rcp(new ParameterList("Param List"));
    paramList->set<RCP<const Albany::MeshSpecsStruct> >("Mesh Specs Struct", meshSpecsPtr);
    paramList->set<RCP<ParamLib> >("Parameter Library", paramLib);

    entity= Albany::StateStruct::NodalDataToElemNode;
    {
      std::string stateName("surface_velocity");
      RCP<ParameterList> p = stateMgr.registerStateVariable(stateName, dl->node_vector, elementBlockName,true,&entity);
      ev = rcp(new PHAL::LoadStateField<EvalT,AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }

    {
      std::string stateName("surface_velocity_rms");
      RCP<ParameterList> p = stateMgr.registerStateVariable(stateName, dl->node_vector, elementBlockName,true,&entity);
      ev = rcp(new PHAL::LoadStateField<EvalT,AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }

    if(this->params->isSublist("Parameter Fields"))
    {
      Teuchos::ParameterList& params_list =  this->params->sublist("Parameter Fields");
      if(params_list.get<int>("Register Surface Mass Balance",0)){
        std::string stateName("surface_mass_balance");
        RCP<ParameterList> p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName,true, &entity);
        ev = rcp(new PHAL::LoadStateField<EvalT,AlbanyTraits>(*p));
        fm0.template registerEvaluator<EvalT>(ev);
      }
    }

#ifdef ALBANY_EPETRA
    {
       RCP<ParameterList> p = rcp(new ParameterList("Gather Averaged Velocity"));
       p->set<string>("Averaged Velocity Name", "Averaged Velocity");
       p->set<Teuchos::RCP<const CellTopologyData> >("Cell Topology",rcp(new CellTopologyData(meshSpecs.ctd)));
       ev = rcp(new GatherVerticallyAveragedVelocity<EvalT,AlbanyTraits>(*p,dl));
       fm0.template registerEvaluator<EvalT>(ev);
    }
#endif

    Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
    return respUtils.constructResponses(fm0, *responseList, paramList, stateMgr);
  }

  return Teuchos::null;
}

template<typename EvalT>
Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> >
FELIX::StokesFO::buildEffectivePressureEvaluator ()
{
  Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Effective Pressure"));

  Teuchos::ParameterList& physics = this->params->sublist("FELIX Physical Parameters");
  Teuchos::ParameterList& paramList = this->params->sublist("FELIX Basal Friction Coefficient");

  // Input
  p->set<Teuchos::ParameterList*>("Physical Parameters", &physics);
  p->set<bool>("Has Hydraulic Potential",paramList.get<bool>("Has Hydraulic Potential",false));
  p->set<double>("Hydraulic-Over-Hydrostatic Potential Ratio",paramList.get<double>("Hydraulic To Hydrostatic Potential Ratio",0.9));
  p->set<std::string> ("Hydraulic Potential Variable Name","Phi");
  p->set<std::string> ("Surface Height Variable Name","Phi");
  p->set<std::string> ("Ice Thickness Variable Name","Phi");

  // Output
  p->set<std::string> ("Effective Pressure Variable Name","N");

  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;
  ev = Teuchos::rcp(new FELIX::EffectivePressure<EvalT,PHAL::AlbanyTraits>(*p,dl));

  return ev;
}

template<typename EvalT>
Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> >
FELIX::StokesFO::buildSlidingVelocityEvaluator ()
{
  Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Velocity Norm"));
  p->set<std::string>("Field Name","Velocity");
  p->set<std::string>("Field Norm Name","Velocity Norm");

  // Need a more specific pointer to access the setHomotopyParamPtr method
  Teuchos::RCP<FELIX::FieldNorm<EvalT,PHAL::AlbanyTraits> > ev;
  ev = Teuchos::rcp(new FELIX::FieldNorm<EvalT,PHAL::AlbanyTraits>(*p,dl));
  ev->setHomotopyParamPtr(HomotopyParamValue<EvalT,PHAL::AlbanyTraits>::value);

  return ev;
}

template<typename EvalT>
Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> >
FELIX::StokesFO::buildBasalFrictionCoefficientEvaluator ()
{
  Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Basal Friction Coefficient"));

  //Input fields
  p->set<std::string>("Velocity Norm Variable Name", "Velocity Norm");
  p->set<std::string>("Given Beta Variable Name", "basal_friction");
  p->set<std::string>("Ice Thickness Variable Name", "thickness");
  p->set<std::string>("Effective Pressure Variable Name", "N");

  //Input physics parameters
  Teuchos::ParameterList& physics = this->params->sublist("FELIX Physical Parameters");

  Teuchos::ParameterList& paramList = this->params->sublist("FELIX Basal Friction Coefficient");
  p->set<Teuchos::ParameterList*>("Parameter List", &paramList);
  p->set<Teuchos::ParameterList*>("Physical Parameters", &physics);

  //Output
  p->set<std::string>("FELIX Basal Friction Coefficient Name", "beta_field");

  Teuchos::RCP<FELIX::BasalFrictionCoefficient<EvalT,PHAL::AlbanyTraits> > ev;
  ev = Teuchos::rcp(new FELIX::BasalFrictionCoefficient<EvalT,PHAL::AlbanyTraits>(*p,dl));
  ev->setHomotopyParamPtr(HomotopyParamValue<EvalT,PHAL::AlbanyTraits>::value);

  return ev;
}

#endif // FELIX_STOKESFOPROBLEM_HPP
