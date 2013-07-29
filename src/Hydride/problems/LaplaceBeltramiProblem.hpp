//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LABELTRAMIPROBLEM_HPP
#define LABELTRAMIPROBLEM_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_AbstractProblem.hpp"

#include "Phalanx.hpp"
#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"
#include "Albany_ProblemUtils.hpp"

namespace Albany {

/*!
 * \brief Laplace Beltrami problem for r-adaptation
 * problem.
 */
class LaplaceBeltramiProblem : public AbstractProblem {
  public:

    //! Default constructor
    LaplaceBeltramiProblem(const Teuchos::RCP<Teuchos::ParameterList>& params,
                           const Teuchos::RCP<ParamLib>& paramLib,
                           const int numDim_,
                           const Teuchos::RCP<const Epetra_Comm>& comm_);

    //! Destructor
    ~LaplaceBeltramiProblem();

    //! Return number of spatial dimensions
    virtual int spatialDimension() const {
      return numDim;
    }

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

    //! Each problem must generate it's list of valid parameters
    Teuchos::RCP<const Teuchos::ParameterList> getValidProblemParameters() const;

  private:

    //! Private to prohibit copying
    LaplaceBeltramiProblem(const LaplaceBeltramiProblem&);

    //! Private to prohibit copying
    LaplaceBeltramiProblem& operator=(const LaplaceBeltramiProblem&);

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

  protected:

    int numDim;

    Teuchos::RCP<const Epetra_Comm> comm;

    Teuchos::RCP<Albany::Layouts> dl;

};

}

#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_EvaluatorUtils.hpp"
#include "Albany_ResponseUtilities.hpp"

#include "PHAL_SaveStateField.hpp"

#include "LaplaceBeltramiResid.hpp"
#include "PHAL_NSContravarientMetricTensor.hpp"
#include "PHAL_GatherCoordinateFromSolutionVector.hpp"


template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
Albany::LaplaceBeltramiProblem::constructEvaluators(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fieldManagerChoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList) {
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::ParameterList;
  using PHX::DataLayout;
  using PHX::MDALayout;
  using std::vector;
  using PHAL::AlbanyTraits;

  // get the name of the current element block
  string elementBlockName = meshSpecs.ebName;

  const CellTopologyData* const elem_top = &meshSpecs.ctd;

  RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > >
  intrepidBasis = Albany::getIntrepidBasis(*elem_top);
  RCP<shards::CellTopology> cellType = rcp(new shards::CellTopology(elem_top));


  const int numNodes = intrepidBasis->getCardinality();
  const int worksetSize = meshSpecs.worksetSize;

  Intrepid::DefaultCubatureFactory<RealType> cubFactory;
  RCP <Intrepid::Cubature<RealType> > cubature = cubFactory.create(*cellType, meshSpecs.cubatureDegree);

  const int numQPtsCell = cubature->getNumPoints();
  const int numVertices = cellType->getNodeCount();


  *out << "Field Dimensions: Workset=" << worksetSize
       << ", Vertices= " << numVertices
       << ", Nodes= " << numNodes
       << ", QuadPts= " << numQPtsCell
       << ", Dim= " << numDim << endl;

  dl = rcp(new Albany::Layouts(worksetSize, numVertices, numNodes, numQPtsCell, numDim));
  TEUCHOS_TEST_FOR_EXCEPTION(dl->vectorAndGradientLayoutsAreEquivalent == false, std::logic_error,
                             "Data Layout Usage in Laplace Beltrami problem assumes vecDim = numDim");
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);

  // Temporary variable used numerous times below
  Teuchos::RCP<PHX::Evaluator<AlbanyTraits> > ev;

  // The Laplace Beltrami Equations

  Teuchos::ArrayRCP<string> dof_names(1);
  Teuchos::ArrayRCP<string> resid_names(1);

  dof_names[0] = "Coordinates";
  resid_names[0] = "Coordinates Residual";

  fm0.template registerEvaluator<EvalT>
  (evalUtils.constructDOFVecInterpolationEvaluator(dof_names[0]));

  fm0.template registerEvaluator<EvalT>
  (evalUtils.constructDOFVecGradInterpolationEvaluator(dof_names[0]));

  fm0.template registerEvaluator<EvalT>
  (evalUtils.constructGatherSolutionEvaluator_noTransient(true, dof_names));

  fm0.template registerEvaluator<EvalT>
  (evalUtils.constructScatterResidualEvaluator(true, resid_names));

  fm0.template registerEvaluator<EvalT>
  (evalUtils.constructMapToPhysicalFrameEvaluator(cellType, cubature));

  fm0.template registerEvaluator<EvalT>
  (evalUtils.constructComputeBasisFunctionsEvaluator(cellType, intrepidBasis, cubature));

  fm0.template registerEvaluator<EvalT>
  (evalUtils.constructGatherCoordinateVectorEvaluator());

  // Gather the coordinates from the solution vector and place in "Coord Vec"
  // Solution here are the node coordinates
  /*
    {

     RCP<ParameterList> p = rcp(new ParameterList("Gather Coordinate From Solution Vector"));

      // Output:: Coordindate Vector at vertices
      p->set< string >("Coordinate Vector Name", "Coord Vec");
      p->set< string >("Solution Names", dof_names[0]);

      ev = rcp(new PHAL::GatherCoordinateFromSolutionVector<EvalT,AlbanyTraits>(*p,dl));
      fm0.template registerEvaluator<EvalT>(ev);

    }
  */


  std::string& method = params->get("Method", "TPSLaplace");

#if 0

  //  if (method == "TPSLaplace") { // Compute Contravarient Metric Tensor
  if(method == "LaplaceBeltrami") {  // Compute Contravarient Metric Tensor
    RCP<ParameterList> p =
      rcp(new ParameterList("Contravarient Metric Tensor"));

    // Inputs: X, Y at nodes, Cubature, and Basis
    p->set<string>("Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Coordinate Data Layout", dl->vertices_vector);
    p->set< RCP<Intrepid::Cubature<RealType> > >("Cubature", cubature);

    p->set<RCP<shards::CellTopology> >("Cell Type", cellType);

    // Outputs: BF, weightBF, Grad BF, weighted-Grad BF, all in physical space
    p->set<string>("Contravarient Metric Tensor Name", "Gc");
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);

    ev = rcp(new PHAL::NSContravarientMetricTensor<EvalT, AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

#endif

  {
    // Laplace Beltrami Resid
    RCP<ParameterList> p = rcp(new ParameterList("Laplace Beltrami Resid"));

    //Input
    p->set<string>("Smoothing Method", method);
    p->set<string>("Coordinate Vector Name", "Coord Vec");
    p->set< string >("Solution Vector Name", dof_names[0]);

    if(method == "LaplaceBeltrami")   // Compute Contravarient Metric Tensor
      p->set<std::string>("Contravarient Metric Tensor Name", "Gc");

    p->set< RCP<Intrepid::Cubature<RealType> > >("Cubature", cubature);
    p->set<RCP<shards::CellTopology> >("Cell Type", cellType);
    p->set< RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >
    ("Intrepid Basis", intrepidBasis);

    //Output
    p->set<string>("Residual Name", resid_names[0]);

    ev = rcp(new PHAL::LaplaceBeltramiResid<EvalT, AlbanyTraits>(*p, dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

#if 0
  {
    // Constraint BC - hold the nodes on the nodeset fixed
    RCP<ParameterList> p = rcp(new ParameterList("Constraint BC"));

    //Input
    p->set<int>("Equation Offset", 0);
    p->set<int>("Number of Equations", numDim);
    Teuchos::Array<std::string> defaultData;
    Teuchos::Array<std::string> nodesets = params->get<Teuchos::Array<std::string> >("Fixed Node Set BC", defaultData);
    p->set<Teuchos::Array<std::string> >("Fixed Node Set IDs", nodesets);

    p->set<string>("BC Metric Name", "Dummy Metric");


    ev = rcp(new PHAL::ConstrainedBC<EvalT, AlbanyTraits>(*p, dl));
    fm0.template registerEvaluator<EvalT>(ev);

    p = stateMgr.registerStateVariable("Dummy Metric", dl->workset_scalar, dl->dummy,
                                       elementBlockName, "scalar", 0.0, true);
    ev = rcp(new PHAL::SaveStateField<EvalT, AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

  }
#endif

  if(fieldManagerChoice == Albany::BUILD_RESID_FM)  {
    PHX::Tag<typename EvalT::ScalarT> res_tag("Scatter", dl->dummy);
    fm0.requireField<EvalT>(res_tag);
    return res_tag.clone();
  }

  else if(fieldManagerChoice == Albany::BUILD_RESPONSE_FM) {
    Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
    return respUtils.constructResponses(fm0, *responseList, Teuchos::null, stateMgr);
  }

  //TEUCHOS_TEST_FOR_EXCEPT(true);


  return Teuchos::null;
}


#endif // ALBANY_LAPLACEBELTRAMI_HPP
