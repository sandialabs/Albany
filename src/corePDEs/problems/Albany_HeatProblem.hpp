//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_HEAT_PROBLEM_HPP
#define ALBANY_HEAT_PROBLEM_HPP

#include "Albany_AbstractProblem.hpp"

#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"
#include "Albany_ProblemUtils.hpp"

#include "PHAL_ConvertFieldType.hpp"
#include "Albany_MaterialDatabase.hpp"

#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_EvaluatorUtils.hpp"
#include "Albany_ResponseUtilities.hpp"

#include "PHAL_ThermalConductivity.hpp"
#include "PHAL_Absorption.hpp"
#include "PHAL_Source.hpp"
#include "PHAL_HeatEqResid.hpp"

#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

namespace Albany {

/*!
 * \brief Abstract interface for representing a 1-D finite element
 * problem.
 */
class HeatProblem : public AbstractProblem
{
public:

  //! Default constructor
  HeatProblem (const Teuchos::RCP<Teuchos::ParameterList>& params,
               const Teuchos::RCP<ParamLib>& paramLib,
               const int numDim_,
               const Teuchos::RCP<const Teuchos_Comm >& comm_);

  //! Destructor
  ~HeatProblem() = default;

  //! Return number of spatial dimensions
  int spatialDimension() const { return numDim; }
  
  //! Get boolean telling code if SDBCs are utilized  
  bool useSDBCs() const {return use_sdbcs_; }

  //! Build the PDE instantiations, boundary conditions, and initial solution
  void buildProblem (Teuchos::ArrayRCP<Teuchos::RCP<MeshSpecs>>  meshSpecs,
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

 //! Boundary conditions on source term
 bool haveSource     = false;
 bool haveAbsorption = false;
 bool conductivityIsDistParam = false;
 bool dirichletIsDistParam    = false;
 bool periodic;
 std::string meshPartDirichlet;
 int numDim;

 //! Problem PL 
 const Teuchos::RCP<Teuchos::ParameterList> params; 

 Teuchos::RCP<MaterialDatabase> materialDB;
 Teuchos::RCP<const Teuchos_Comm> comm; 

 Teuchos::RCP<Layouts> dl;

 /// Boolean marking whether SDBCs are used 
 bool use_sdbcs_ = false; 
};

// --------------- IMPLEMENTATION ------------------ //

template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
HeatProblem::
constructEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                     const MeshSpecs& meshSpecs,
                     StateManager& stateMgr,
                     FieldManagerChoice fieldManagerChoice,
                     const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::ParameterList;
  using PHX::DataLayout;
  using std::vector;
  using std::string;
  using PHAL::AlbanyTraits;

  const CellTopologyData * const elem_top = &meshSpecs.ctd;

  auto intrepidBasis = getIntrepid2Basis(*elem_top);
  auto cellType = rcp(new shards::CellTopology (elem_top));

  const int numNodes = intrepidBasis->getCardinality();
  const int worksetSize = meshSpecs.worksetSize;

  Intrepid2::DefaultCubatureFactory cubFactory;
  int cubDegree = params->get("Cubature Degree", 3);
  auto cellCubature = cubFactory.create<PHX::Device, RealType, RealType>(*cellType, cubDegree);

  const int numQPtsCell = cellCubature->getNumPoints();
  const int numVertices = cellType->getNodeCount();

  // Problem is steady or transient
  TEUCHOS_TEST_FOR_EXCEPTION(
      number_of_time_deriv < 0 || number_of_time_deriv > 1,
      std::logic_error,
      "Albany_HeatProblem must be defined as a steady or transient calculation.");

  *out << "Field Dimensions: Workset=" << worksetSize
       << ", Vertices= " << numVertices
       << ", Nodes= " << numNodes
       << ", QuadPts= " << numQPtsCell
       << ", Dim= " << numDim << std::endl;

  dl = rcp(new Layouts(worksetSize,numVertices,numNodes,numQPtsCell,numDim));
  EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);

  // Temporary variable used numerous times below
  Teuchos::RCP<PHX::Evaluator<AlbanyTraits> > ev;

  Teuchos::ArrayRCP<string> dof_names(neq);
  dof_names[0] = "Temperature";
  Teuchos::ArrayRCP<string> dof_names_dot(neq);
  if (number_of_time_deriv > 0)
    dof_names_dot[0] = "Temperature_dot";
  Teuchos::ArrayRCP<string> resid_names(neq);
  resid_names[0] = "Temperature Residual";

  if (number_of_time_deriv == 1)
    fm0.template registerEvaluator<EvalT>
       (evalUtils.constructGatherSolutionEvaluator(false, dof_names, dof_names_dot));
  else
    fm0.template registerEvaluator<EvalT>
       (evalUtils.constructGatherSolutionEvaluator_noTransient(false, dof_names));

  fm0.template registerEvaluator<EvalT>
     (evalUtils.constructScatterResidualEvaluator(false, resid_names));

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructGatherCoordinateVectorEvaluator());

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructMapToPhysicalFrameEvaluator( cellType, cellCubature));

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructComputeBasisFunctionsEvaluator(cellType, intrepidBasis, cellCubature));

  for (unsigned int i=0; i<neq; i++) {
    fm0.template registerEvaluator<EvalT>
      (evalUtils.constructDOFInterpolationEvaluator(dof_names[i]));

    if(number_of_time_deriv == 1)
      fm0.template registerEvaluator<EvalT>
        (evalUtils.constructDOFInterpolationEvaluator(dof_names_dot[i]));

    fm0.template registerEvaluator<EvalT>
      (evalUtils.constructDOFGradInterpolationEvaluator(dof_names[i]));
  }

  if(!conductivityIsDistParam) { // Thermal conductivity
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<string>("QP Variable Name", "ThermalConductivity");
    p->set<string>("QP Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Node Data Layout", dl->node_scalar);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    //p->set<RCP<DistributedParameterLibrary> >("Distributed Parameter Library", distParamLib);
    Teuchos::ParameterList& paramList = params->sublist("ThermalConductivity");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    // Here we assume that the instance of this problem applies on a single element block
    p->set<string>("Element Block Name", meshSpecs.ebName);

    if(materialDB != Teuchos::null)
      p->set< RCP<MaterialDatabase> >("MaterialDB", materialDB);

    ev = rcp(new PHAL::ThermalConductivity<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (haveAbsorption) { // Absorption
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<string>("QP Variable Name", "Absorption");
    p->set<string>("QP Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Absorption");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = rcp(new PHAL::Absorption<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (dirichletIsDistParam) {
    // Here is how to register the field for dirichlet condition.
    RCP<ParameterList> p = rcp(new ParameterList);
    StateStruct::MeshFieldEntity entity = StateStruct::NodalDistParameter;
    std::string stateName = "dirichlet_field";
    p = stateMgr.registerStateVariable(stateName, dl->node_scalar, meshSpecs.ebName, true, &entity, meshPartDirichlet);
  }

  if (conductivityIsDistParam) {
    RCP<ParameterList> p = rcp(new ParameterList);
    StateStruct::MeshFieldEntity entity = StateStruct::NodalDistParameter;
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

    stateName = "thermal_conductivity_sensitivity";
    p = stateMgr.registerStateVariable(stateName, dl->node_scalar, meshSpecs.ebName, true, &entity, "");
  }

  // Check and see if a source term is specified for this problem in the main input file.
  bool problemSpecifiesASource = params->isSublist("Source Functions");

  if(problemSpecifiesASource) {
    // Sources the same everywhere if they are present at all

    haveSource = true;
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<string>("Source Name", "Source");
    p->set<string>("Variable Name", "Temperature");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

    p->set<string>("QP Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Source Functions");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);
    Teuchos::ParameterList& scalarParamesList = params->sublist("Parameters");
    p->set<Teuchos::RCP<ScalarParameterAccessors<EvalT>>>("Accessors", this->getAccessors()->template at<EvalT>());
    p->set<Teuchos::ParameterList*>("Scalar Parameters List", &scalarParamesList);

    ev = rcp(new PHAL::Source<EvalT,AlbanyTraits>(*p, fm0, dl));
    fm0.template registerEvaluator<EvalT>(ev);
  } else if(materialDB != Teuchos::null) { // Sources can be specified in terms of materials or element blocks

    // Is the source function active for "this" element block?

    haveSource =  materialDB->isElementBlockSublist(meshSpecs.ebName, "Source Functions");

    if (haveSource) {
      RCP<ParameterList> p = rcp(new ParameterList);

      p->set<string>("Source Name", "Source");
      p->set<string>("Variable Name", "Temperature");
      p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

      p->set<RCP<ParamLib> >("Parameter Library", paramLib);
      Teuchos::ParameterList& paramList = materialDB->getElementBlockSublist(meshSpecs.ebName, "Source Functions");
      p->set<Teuchos::ParameterList*>("Parameter List", &paramList);
      Teuchos::ParameterList& scalarParamesList = params->sublist("Parameters");
      p->set<Teuchos::RCP<ScalarParameterAccessors<EvalT>>>("Accessors", this->getAccessors()->template at<EvalT>());
      p->set<Teuchos::ParameterList*>("Scalar Parameters List", &scalarParamesList);

      ev = rcp(new PHAL::Source<EvalT,AlbanyTraits>(*p, fm0, dl));
      fm0.template registerEvaluator<EvalT>(ev);
    }
  }

  { // Temperature Resid
    RCP<ParameterList> p = rcp(new ParameterList("Temperature Resid"));

    //Input
    p->set<string>("Weighted BF Name", "wBF");
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", dl->node_qp_scalar);
    p->set<string>("QP Variable Name", "Temperature");

    if(number_of_time_deriv == 0)
      p->set<bool>("Disable Transient", true);
    p->set<string>("QP Time Derivative Variable Name", "Temperature_dot");

    p->set<bool>("Have Source", haveSource);
    p->set<bool>("Have Absorption", haveAbsorption);
    p->set<string>("Source Name", "Source");

    p->set<string>("ThermalConductivity Name", "ThermalConductivity");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

    p->set<string>("Absorption Name", "ThermalConductivity");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

    p->set<string>("Gradient QP Variable Name", "Temperature Gradient");
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    p->set<string>("Weighted Gradient BF Name", "wGrad BF");
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);
    if (params->isType<string>("Convection Velocity"))
      p->set<string>("Convection Velocity",
                       params->get<string>("Convection Velocity"));
    if (params->isType<bool>("Have Rho Cp"))
      p->set<bool>("Have Rho Cp", params->get<bool>("Have Rho Cp"));

    //Output
    p->set<string>("Residual Name", "Temperature Residual");
    p->set< RCP<DataLayout> >("Node Scalar Data Layout", dl->node_scalar);

    ev = rcp(new PHAL::HeatEqResid<EvalT,AlbanyTraits>(*p));
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

#endif // ALBANY_HEAT_PROBLEM_HPP
