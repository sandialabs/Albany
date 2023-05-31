//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_ADVECTION_PROBLEM_HPP
#define ALBANY_ADVECTION_PROBLEM_HPP

#include "Albany_AbstractProblem.hpp"

#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"
#include "Albany_ProblemUtils.hpp"

#include "PHAL_ConvertFieldType.hpp"
#include "Albany_MaterialDatabase.hpp"

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_EvaluatorUtils.hpp"
#include "Albany_ResponseUtilities.hpp"
#include "PHAL_AdvectionResid.hpp"
#include "PHAL_SharedParameter.hpp"
#include "Albany_ParamEnum.hpp"

namespace Albany {

/*!
 * \brief Abstract interface for representing a 1-D finite element
 * problem.
 */
class AdvectionProblem : public AbstractProblem
{
public:
  //! Default constructor
  AdvectionProblem (const Teuchos::RCP<Teuchos::ParameterList>& params,
                    const Teuchos::RCP<ParamLib>& paramLib,
                    const int numDim_,
                    const Teuchos::RCP<const Teuchos_Comm >& comm_); 

  //! Destructor
  ~AdvectionProblem() = default;

  //! Return number of spatial dimensions
  int spatialDimension() const { return numDim; }
  
  //! Get boolean telling code if SDBCs are utilized  
  bool useSDBCs() const {return use_sdbcs_; }

  //! Build the PDE instantiations, boundary conditions, and initial solution
  void buildProblem (Teuchos::RCP<MeshSpecs>  meshSpecs,
                     StateManager& stateMgr);

  // Build evaluators
  Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
  buildEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                   const MeshSpecs& meshSpecs,
                   StateManager& stateMgr,
                   FieldManagerChoice fmchoice,
                   const Teuchos::RCP<Teuchos::ParameterList>& responseList);

  //! Each problem must generate it's list of valid parameters
  Teuchos::RCP<const Teuchos::ParameterList> getValidProblemParameters() const;

  //! Main problem setup routine. Not directly called, but indirectly by following functions
  template <typename EvalT>
  Teuchos::RCP<const PHX::FieldTag>
  constructEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                       const MeshSpecs& meshSpecs,
                       StateManager& stateMgr,
                       FieldManagerChoice fmchoice,
                       const Teuchos::RCP<Teuchos::ParameterList>& responseList);

  void constructDirichletEvaluators(const std::vector<std::string>& nodeSetIDs);
  void constructNeumannEvaluators(const Teuchos::RCP<MeshSpecs>& meshSpecs);

protected:

 int numDim;
 Teuchos::Array<double> a;  // advection_coefficient
 std::string            advection_source; //advection source name 
 bool                   advectionIsDistParam;

 //! Problem PL 
 const Teuchos::RCP<Teuchos::ParameterList> params; 

 Teuchos::RCP<const Teuchos_Comm> comm; 

 Teuchos::RCP<Layouts> dl;

 /// Boolean marking whether SDBCs are used 
 bool use_sdbcs_; 
};

// ------------------ IMPLEMENTATION ------------------- //

template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
AdvectionProblem::
constructEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                     const MeshSpecs& meshSpecs,
                     StateManager& stateMgr,
                     FieldManagerChoice fieldManagerChoice,
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

  auto intrepidBasis = getIntrepid2Basis(*elem_top);
  auto cellType = rcp(new shards::CellTopology(elem_top));

  int const numNodes    = intrepidBasis->getCardinality();
  int const worksetSize = meshSpecs.worksetSize;

  int cubDegree = params->get("Cubature Degree", 3);
  Intrepid2::DefaultCubatureFactory     cubFactory;
  auto cellCubature = cubFactory.create<PHX::Device, RealType, RealType>(*cellType, cubDegree);

  int const numQPtsCell = cellCubature->getNumPoints();
  int const numVertices = cellType->getNodeCount();

  // Get the solution method type
  SolutionMethodType SolutionType = getSolutionMethod();

  ALBANY_ASSERT(
      SolutionType == SolutionMethodType::Transient,
      "Solution Method must be Transient for Advection Problem!\n "); 
  
  ALBANY_ASSERT(
      number_of_time_deriv == 1,
      "Wrong number of time derivatives fo AdvectionProblem.\n"
      "  - expected number: 1\n"
      "  - actual number: " + std::to_string(number_of_time_deriv) + "\n");

  *out << "Field Dimensions: Workset=" << worksetSize
       << ", Vertices= " << numVertices << ", Nodes= " << numNodes
       << ", QuadPts= " << numQPtsCell << ", Dim= " << numDim << std::endl;

  dl = rcp(new Layouts(worksetSize, numVertices, numNodes, numQPtsCell, numDim));
  EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);

  // Temporary variable used numerous times below
  Teuchos::RCP<PHX::Evaluator<AlbanyTraits>> ev;

  Teuchos::ArrayRCP<string> dof_names(neq);
  dof_names[0] = "solution";
  Teuchos::ArrayRCP<string> dof_names_dot(neq);
  dof_names_dot[0] = "solution_dot";
  Teuchos::ArrayRCP<string> resid_names(neq);
  resid_names[0] = "Advection Residual";

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

  if (!advectionIsDistParam) {  
    //Shared parameter for sensitivity analysis: a_x
    RCP<ParameterList> p = rcp(new ParameterList("Advection Coefficient: a_x"));
    p->set< RCP<ParamLib> >("Parameter Library", paramLib);
    const std::string param_name = "a_x Parameter";
    p->set<std::string>("Parameter Name", param_name);
    p->set<Teuchos::RCP<ScalarParameterAccessors<EvalT>>>("Accessors", this->getAccessors()->template at<EvalT>());
    p->set<const Teuchos::ParameterList*>("Parameters List", &params->sublist("Parameters"));
    p->set<double>("Default Nominal Value", a[0]);
    RCP<PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits>> ptr_a_x;
    ptr_a_x = rcp(new PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ptr_a_x);

    if (numDim > 1) {  //Shared parameter for sensitivity analysis: a_y
      RCP<ParameterList> p = rcp(new ParameterList("Advection Coefficient: a_y"));
      p->set< RCP<ParamLib> >("Parameter Library", paramLib);
      const std::string param_name = "a_y Parameter";
      p->set<std::string>("Parameter Name", param_name);
      p->set<Teuchos::RCP<ScalarParameterAccessors<EvalT>>>("Accessors", this->getAccessors()->template at<EvalT>());
      p->set<const Teuchos::ParameterList*>("Parameters List", &params->sublist("Parameters"));
      p->set<double>("Default Nominal Value", a[1]);
      RCP<PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits>> ptr_a_y;
      ptr_a_y = rcp(new PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits>(*p,dl));
      fm0.template registerEvaluator<EvalT>(ptr_a_y);
    }
    if (numDim > 2) {  //Shared parameter for sensitivity analysis: a_z
      RCP<ParameterList> p = rcp(new ParameterList("Advection Coefficient: a_z"));
      p->set< RCP<ParamLib> >("Parameter Library", paramLib);
      const std::string param_name = "a_z Parameter";
      p->set<std::string>("Parameter Name", param_name);
      p->set<Teuchos::RCP<ScalarParameterAccessors<EvalT>>>("Accessors", this->getAccessors()->template at<EvalT>());
      p->set<const Teuchos::ParameterList*>("Parameters List", &params->sublist("Parameters"));
      p->set<double>("Default Nominal Value", a[2]);
      RCP<PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits>> ptr_a_z;
      ptr_a_z = rcp(new PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits>(*p,dl));
      fm0.template registerEvaluator<EvalT>(ptr_a_z);
    }
  }
  else //advectionIsDistParam
  {
    RCP<ParameterList> p = rcp(new ParameterList);
    StateStruct::MeshFieldEntity entity = StateStruct::NodalDistParameter;
    std::string stateName = "advection_coefficient";
    std::string fieldName = "AdvectionCoefficient";
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
  
    //Construct gradient of AdvectionCoefficient, for defining source term
    fm0.template registerEvaluator<EvalT>(
		  evalUtils.constructDOFGradInterpolationEvaluator(fieldName));
  }

  {  // Advection Resid
    RCP<ParameterList> p = rcp(new ParameterList("Advection Resid"));

    // Input
    p->set<string>("Weighted BF Name", "wBF");
    p->set<string>("QP Time Derivative Variable Name", "solution_dot");
    p->set<string>("Gradient QP Variable Name", "solution Gradient");
    p->set<string>("Source Name", "Advection Source");
    p->set<string>("QP Coordinate Vector Name", "Coord Vec");

    p->set<RCP<DataLayout>>("Node QP Scalar Data Layout", dl->node_qp_scalar);
    p->set<RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);
    p->set<RCP<DataLayout>>("QP Vector Data Layout", dl->qp_vector);
    p->set<RCP<DataLayout>>("Node QP Vector Data Layout", dl->node_qp_vector);
    p->set<RCP<DataLayout>>("Node Scalar Data Layout", dl->node_scalar);
    if (!advectionIsDistParam) {  
      p->set<std::string>("Advection Coefficient: a_x","a_x Parameter");
      if (numDim > 1) p->set<std::string>("Advection Coefficient: a_y","a_y Parameter");
      if (numDim > 2) p->set<std::string>("Advection Coefficient: a_z","a_z Parameter");
    } else {
      p->set<string>("AdvectionCoefficient Name", "AdvectionCoefficient");
      p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
      p->set<std::string>("AdvectionCoefficient Gradient Name", "AdvectionCoefficient Gradient");
    }
    p->set<bool>("Distributed Advection Coefficient", advectionIsDistParam);
    p->set<std::string>("Advection Source", advection_source); 

    // Output
    p->set<string>("Residual Name", "Advection Residual");

    ev = rcp(new PHAL::AdvectionResid<EvalT, AlbanyTraits>(*p, dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (fieldManagerChoice == BUILD_RESID_FM) {
    PHX::Tag<typename EvalT::ScalarT> res_tag("Scatter", dl->dummy);
    fm0.requireField<EvalT>(res_tag);
    return res_tag.clone();
  } else if (fieldManagerChoice == BUILD_RESPONSE_FM) {
    ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
    return respUtils.constructResponses(fm0, *responseList, Teuchos::null, stateMgr);
  }

  return Teuchos::null;
}

} // namespace Albany

#endif // ALBANY_ADVECTION_PROBLEM_HPP
