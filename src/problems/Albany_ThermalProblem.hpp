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
      Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
      StateManager& stateMgr);

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
    ThermalProblem(const ThermalProblem&);

    //! Private to prohibit copying
    ThermalProblem& operator=(const ThermalProblem&);

  public:

    //! Main problem setup routine. Not directly called, but indirectly by following functions
    template <typename EvalT>
    Teuchos::RCP<const PHX::FieldTag>
    constructEvaluators(
      PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
      const Albany::MeshSpecsStruct& meshSpecs,
      Albany::StateManager& stateMgr,
      Albany::FieldManagerChoice fmchoice,
      const Teuchos::RCP<Teuchos::ParameterList>& responseList);

    void constructDirichletEvaluators(const std::vector<std::string>& nodeSetIDs);
    void constructNeumannEvaluators(const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs);

  protected:

   int numDim;
   Teuchos::Array<double> kappa;  // thermal conductivity
   double                 C;      // heat capacity
   double                 rho;    // density
   std::string            thermal_source; //thermal source name 

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
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fieldManagerChoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  using PHAL::AlbanyTraits;
  using PHX::DataLayout;
  using PHX::MDALayout;
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

  Intrepid2::DefaultCubatureFactory     cubFactory;
  RCP<Intrepid2::Cubature<PHX::Device>> cellCubature =
      cubFactory.create<PHX::Device, RealType, RealType>(
          *cellType, meshSpecs.cubatureDegree);

  int const numQPtsCell = cellCubature->getNumPoints();
  int const numVertices = cellType->getNodeCount();

  // Get the solution method type
  SolutionMethodType SolutionType = getSolutionMethod();

  ALBANY_PANIC(
      SolutionType == SolutionMethodType::Unknown,
      "Solution Method must be Steady, Transient, "
      "Continuation, or Eigensolve");
  
  if (SolutionType == SolutionMethodType::Transient) { // Problem is transient
    ALBANY_PANIC(
        number_of_time_deriv != 1,
        "You are using a transient solution method in Albany::ThermalProblem but number of time derivatives != 1!"); 
  }
  else { //Problem is steady
    ALBANY_PANIC(
        number_of_time_deriv > 0,
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

  {  //Shared parameter for sensitivity analysis: kappa_x
    RCP<ParameterList> p = rcp(new ParameterList("Thermal Conductivity: kappa_x"));
    p->set< RCP<ParamLib> >("Parameter Library", paramLib);
    const std::string param_name = "kappa_x Parameter";
    p->set<std::string>("Parameter Name", param_name);
    RCP<PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits,Albany::ParamEnum,Albany::ParamEnum::Kappa_x>> ptr_kappa_x;
    ptr_kappa_x = rcp(new PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits,Albany::ParamEnum,Albany::ParamEnum::Kappa_x>(*p,dl));
    ptr_kappa_x->setNominalValue(params->sublist("Parameters"), kappa[0]);
    fm0.template registerEvaluator<EvalT>(ptr_kappa_x);
  }
  if (numDim > 1) {  //Shared parameter for sensitivity analysis: kappa_y
    RCP<ParameterList> p = rcp(new ParameterList("Thermal Conductivity: kappa_y"));
    p->set< RCP<ParamLib> >("Parameter Library", paramLib);
    const std::string param_name = "kappa_y Parameter";
    p->set<std::string>("Parameter Name", param_name);
    RCP<PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits,Albany::ParamEnum,Albany::ParamEnum::Kappa_y>> ptr_kappa_y;
    ptr_kappa_y = rcp(new PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits,Albany::ParamEnum,Albany::ParamEnum::Kappa_y>(*p,dl));
    ptr_kappa_y->setNominalValue(params->sublist("Parameters"), kappa[1]); 
    fm0.template registerEvaluator<EvalT>(ptr_kappa_y);
  }
  if (numDim > 2) {  //Shared parameter for sensitivity analysis: kappa_z
    RCP<ParameterList> p = rcp(new ParameterList("Thermal Conductivity: kappa_z"));
    p->set< RCP<ParamLib> >("Parameter Library", paramLib);
    const std::string param_name = "kappa_z Parameter";
    p->set<std::string>("Parameter Name", param_name);
    RCP<PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits,Albany::ParamEnum,Albany::ParamEnum::Kappa_z>> ptr_kappa_z;
    ptr_kappa_z = rcp(new PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits,Albany::ParamEnum,Albany::ParamEnum::Kappa_z>(*p,dl));
    ptr_kappa_z->setNominalValue(params->sublist("Parameters"), kappa[2]); 
    fm0.template registerEvaluator<EvalT>(ptr_kappa_z);
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
    p->set<std::string>("Thermal Conductivity: kappa_x","kappa_x Parameter");
    if (numDim > 1) p->set<std::string>("Thermal Conductivity: kappa_y","kappa_y Parameter");
    if (numDim > 2) p->set<std::string>("Thermal Conductivity: kappa_z","kappa_z Parameter");
    if (SolutionType != SolutionMethodType::Transient) {
      p->set<bool>("Disable Transient", true);
    }
    else {
      p->set<bool>("Disable Transient", false);
    }
    p->set<double>("Heat Capacity", C);
    p->set<double>("Density", rho);
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
