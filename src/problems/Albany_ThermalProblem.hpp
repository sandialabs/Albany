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
      Teuchos::RCP<const Teuchos::Comm<int> >& commT_); 

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

   //! Problem PL 
   const Teuchos::RCP<Teuchos::ParameterList> params; 

   Teuchos::RCP<const Teuchos::Comm<int> > commT; 

   Teuchos::RCP<Albany::Layouts> dl;

   /// Boolean marking whether SDBCs are used 
   bool use_sdbcs_; 

  };

}

#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_EvaluatorUtils.hpp"
#include "Albany_ResponseUtilities.hpp"

#include "PHAL_ThermalConductivity.hpp"
#include "PHAL_Absorption.hpp"
#include "PHAL_Source.hpp"
//#include "PHAL_Neumann.hpp"
#include "PHAL_ThermalResid.hpp"


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

  // Problem is steady or transient
  ALBANY_PANIC(
      number_of_time_deriv != 1,
      "Albany_ThermalProblem must be defined as transient "
      "calculation.");

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

  {  // Temperature Resid
    RCP<ParameterList> p = rcp(new ParameterList("Temperature Resid"));

    // Input
    p->set<string>("Weighted BF Name", "wBF");
    p->set<string>("QP Time Derivative Variable Name", "Temperature_dot");
    p->set<string>("Gradient QP Variable Name", "Temperature Gradient");
    p->set<string>("Weighted Gradient BF Name", "wGrad BF");

    p->set<RCP<DataLayout>>("Node QP Scalar Data Layout", dl->node_qp_scalar);
    p->set<RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);
    p->set<RCP<DataLayout>>("QP Vector Data Layout", dl->qp_vector);
    p->set<RCP<DataLayout>>("Node QP Vector Data Layout", dl->node_qp_vector);

    p->set<Teuchos::Array<double>>("Thermal Conductivity", kappa);
    p->set<double>("Heat Capacity", C);
    p->set<double>("Density", rho);

    // Output
    p->set<string>("Residual Name", "Temperature Residual");
    p->set<RCP<DataLayout>>("Node Scalar Data Layout", dl->node_scalar);

    ev = rcp(new PHAL::ThermalResid<EvalT, AlbanyTraits>(*p));
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
