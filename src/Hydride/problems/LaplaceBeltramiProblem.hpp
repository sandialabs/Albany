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

#include "Time.hpp"


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

#include "LaplaceResid.hpp"
#include "TPSLaplaceResid.hpp"
#include "TPSALaplaceResid.hpp"
#include "LaplaceBeltramiResid.hpp"
#include "ContravariantTargetMetricTensor.hpp"
#include "CalcInstantaneousCoords.hpp"

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
  std::string elementBlockName = meshSpecs.ebName;

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
       << ", Dim= " << numDim << std::endl;

  TEUCHOS_TEST_FOR_EXCEPTION(numNodes != numVertices,
                        std::logic_error,
                        "Error in LaplaceBeltrami problem: specification of coordinate vector vs. solution layout is incorrect."
                        << std::endl);

  dl = rcp(new Albany::Layouts(worksetSize, numVertices, numNodes, numQPtsCell, numDim));
  TEUCHOS_TEST_FOR_EXCEPTION(dl->vectorAndGradientLayoutsAreEquivalent == false, std::logic_error,
                             "Data Layout Usage in Laplace Beltrami problem assumes vecDim = numDim");

  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);

  // Temporary variable used numerous times below
  Teuchos::RCP<PHX::Evaluator<AlbanyTraits> > ev;

  std::string& method = params->get("Method", "Laplace");
  std::string& solution_strategy = params->get("Solution Method", "Steady");

  Teuchos::ArrayRCP<std::string> soln_name(1);
  Teuchos::ArrayRCP<std::string> soln_resid_name(1);
  Teuchos::ArrayRCP<std::string> tgt_name(1);
  Teuchos::ArrayRCP<std::string> tgt_resid_name(1);

  soln_name[0] = "Coordinates";
  tgt_name[0] = "Tgt Coords";
  soln_resid_name[0] = "Coordinates Residual";
  tgt_resid_name[0] = "Tgt Coords Residual";

  // vqp(cell,qp,i) += val_node(cell, node, i) * BF(cell, node, qp);
//  fm0.template registerEvaluator<EvalT>
//  (evalUtils.constructDOFVecInterpolationEvaluator(soln_name[0]));

 //grad_val_qp(cell,qp,i,dim) += val_node(cell, node, i) * GradBF(cell, node, qp, dim);
//  fm0.template registerEvaluator<EvalT>
//  (evalUtils.constructDOFVecGradInterpolationEvaluator(soln_name[0]));

  //gets solution vector
  fm0.template registerEvaluator<EvalT>
  (evalUtils.constructGatherSolutionEvaluator_noTransient(true, soln_name));

  // Puts residual vector
  fm0.template registerEvaluator<EvalT>
  (evalUtils.constructScatterResidualEvaluator(true, soln_resid_name));

  // Fills the coordVec field
  fm0.template registerEvaluator<EvalT>
  (evalUtils.constructGatherCoordinateVectorEvaluator());

  if(solution_strategy == "Continuation"){ // Time

    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<std::string>("Time Name", "Time");
    p->set<std::string>("Delta Time Name", "Delta Time");
    p->set< RCP<DataLayout> >("Workset Scalar Data Layout", dl->workset_scalar);
    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    p->set<bool>("Disable Transient", true);

    ev = rcp(new LCM::Time<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
    p = stateMgr.registerStateVariable("Time",dl->workset_scalar, dl->dummy, elementBlockName, "scalar", 0.0, true);
    ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }


  if(method == "Laplace"){

    // Laplace equation Resid
    RCP<ParameterList> p = rcp(new ParameterList("Laplace Resid"));

    //Input
    p->set<std::string>("Coordinate Vector Name", "Coord Vec");
    p->set< std::string >("Solution Vector Name", soln_name[0]);

    p->set< RCP<Intrepid::Cubature<RealType> > >("Cubature", cubature);
    p->set<RCP<shards::CellTopology> >("Cell Type", cellType);
    p->set< RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >
         ("Intrepid Basis", intrepidBasis);

    //Output
    p->set<std::string>("Residual Name", soln_resid_name[0]);

    ev = rcp(new PHAL::LaplaceResid<EvalT, AlbanyTraits>(*p, dl));
    fm0.template registerEvaluator<EvalT>(ev);

  }
  else if(method == "TPSLaplace"){

    // Laplace equation Resid
    RCP<ParameterList> p = rcp(new ParameterList("TPS Laplace Resid"));

    //Input
    p->set< std::string >("Solution Vector Name", soln_name[0]);

    p->set< RCP<Intrepid::Cubature<RealType> > >("Cubature", cubature);
    p->set<RCP<shards::CellTopology> >("Cell Type", cellType);
    p->set< RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >
         ("Intrepid Basis", intrepidBasis);

    //Output
    p->set<std::string>("Residual Name", soln_resid_name[0]);

    ev = rcp(new PHAL::TPSLaplaceResid<EvalT, AlbanyTraits>(*p, dl));
    fm0.template registerEvaluator<EvalT>(ev);

  }
  else if(method == "DispTPSLaplace"){

    {
      // Add coords and displacements to get instantaneous coordinates
      RCP<ParameterList> p = rcp(new ParameterList("Instantaneous Coordinates"));

      //Input
      p->set<std::string>("Coordinate Vector Name", "Coord Vec"); // Reference node coordinates
      p->set< std::string >("Solution Vector Name", soln_name[0]); // Displacements

      //Output
      p->set<std::string>("Instantaneous Coordinates Name", "Current Coords");

      ev = rcp(new PHAL::CalcInstantaneousCoords<EvalT, AlbanyTraits>(*p, dl));
      fm0.template registerEvaluator<EvalT>(ev);
    }

    {
      // Laplace equation Resid
      RCP<ParameterList> p = rcp(new ParameterList("Disp TPS Laplace Resid"));

      //Input
      p->set< std::string >("Solution Vector Name", "Current Coords");

      p->set< RCP<Intrepid::Cubature<RealType> > >("Cubature", cubature);
      p->set<RCP<shards::CellTopology> >("Cell Type", cellType);
      p->set< RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >
         ("Intrepid Basis", intrepidBasis);

      //Output
      p->set<std::string>("Residual Name", soln_resid_name[0]);

      ev = rcp(new PHAL::TPSLaplaceResid<EvalT, AlbanyTraits>(*p, dl));
      fm0.template registerEvaluator<EvalT>(ev);
    }

  }
  else if(method == "TPSALaplace"){

    {
      // Add coords and displacements to get instantaneous coordinates
      RCP<ParameterList> p = rcp(new ParameterList("Instantaneous Coordinates"));

      //Input
      p->set<std::string>("Coordinate Vector Name", "Coord Vec"); // Reference node coordinates
      p->set< std::string >("Solution Vector Name", soln_name[0]); // Displacements

      //Output
      p->set<std::string>("Instantaneous Coordinates Name", "Current Coords");

      ev = rcp(new PHAL::CalcInstantaneousCoords<EvalT, AlbanyTraits>(*p, dl));
      fm0.template registerEvaluator<EvalT>(ev);
    }

    fm0.template registerEvaluator<EvalT>
      (evalUtils.constructComputeBasisFunctionsEvaluator(cellType, intrepidBasis, cubature));

    {

      // TPS Laplace Resid
      RCP<ParameterList> p = rcp(new ParameterList("TPSA Laplace Resid"));

      //Input
      p->set< std::string >("Solution Vector Name", "Current Coords");
      p->set<std::string>("Gradient BF Name", "Grad BF");
      p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");

      //Output
      p->set<std::string>("Residual Name", soln_resid_name[0]);

      ev = rcp(new PHAL::TPSALaplaceResid<EvalT, AlbanyTraits>(*p, dl));
      fm0.template registerEvaluator<EvalT>(ev);
    }

  }
  else if(method == "LaplaceBeltrami"){

   // Add the target solution

    //gets solution vector
    fm0.template registerEvaluator<EvalT>
    (evalUtils.constructGatherSolutionEvaluator_noTransient(true, tgt_name, numDim));

    // Puts residual vector
    fm0.template registerEvaluator<EvalT>
    (evalUtils.constructScatterResidualEvaluator(true, tgt_resid_name, numDim));

    {
      // Laplace equation Resid - solve for the target space
      RCP<ParameterList> p = rcp(new ParameterList("Target Laplace Resid"));

      //Input
      // Target is calculated from the actual solution
      p->set< std::string >("Solution Vector Name", soln_name[0]);

      p->set< RCP<Intrepid::Cubature<RealType> > >("Cubature", cubature);
      p->set<RCP<shards::CellTopology> >("Cell Type", cellType);
      p->set< RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >
         ("Intrepid Basis", intrepidBasis);

      //Output
      p->set<std::string>("Residual Name", tgt_resid_name[0]);

      ev = rcp(new PHAL::TPSLaplaceResid<EvalT, AlbanyTraits>(*p, dl));
      fm0.template registerEvaluator<EvalT>(ev);
    }

    {
      // Calculate the target metric tensor

      RCP<ParameterList> p =
        rcp(new ParameterList("Contravariant Metric Tensor"));

      // Inputs: 
      // Note that the target solution is used to build Gc
      p->set< std::string >("Solution Vector Name", tgt_name[0]);
      p->set< RCP<Intrepid::Cubature<RealType> > >("Cubature", cubature);

      p->set<RCP<shards::CellTopology> >("Cell Type", cellType);

      // Outputs: BF, weightBF, Grad BF, weighted-Grad BF, all in physical space
      p->set<std::string>("Contravariant Metric Tensor Name", "Gc");

      ev = rcp(new PHAL::ContravariantTargetMetricTensor<EvalT, AlbanyTraits>(*p, dl));
      fm0.template registerEvaluator<EvalT>(ev);

    }

    {

      // Laplace Beltrami Resid - Solve for the coordinates
      RCP<ParameterList> p = rcp(new ParameterList("Laplace Beltrami Resid"));

      //Input
      p->set< std::string >("Solution Vector Name", soln_name[0]);
      p->set<std::string>("Contravariant Metric Tensor Name", "Gc");

      p->set< RCP<Intrepid::Cubature<RealType> > >("Cubature", cubature);
      p->set<RCP<shards::CellTopology> >("Cell Type", cellType);
      p->set< RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >
        ("Intrepid Basis", intrepidBasis);

      //Output
      p->set<std::string>("Residual Name", soln_resid_name[0]);

      ev = rcp(new PHAL::LaplaceBeltramiResid<EvalT, AlbanyTraits>(*p, dl));
      fm0.template registerEvaluator<EvalT>(ev);
    }
  }
  else if(method == "DispLaplaceBeltramiA"){

    {
      // Add coords and displacements to get instantaneous coordinates
      RCP<ParameterList> p = rcp(new ParameterList("Instantaneous Coordinates"));

      //Input
      p->set<std::string>("Coordinate Vector Name", "Coord Vec"); // Reference node coordinates
      p->set< std::string >("Solution Vector Name", soln_name[0]); // Displacements

      //Output
      p->set<std::string>("Instantaneous Coordinates Name", "Current Coords");

      ev = rcp(new PHAL::CalcInstantaneousCoords<EvalT, AlbanyTraits>(*p, dl));
      fm0.template registerEvaluator<EvalT>(ev);
    }

    {
      // Add coords and displacements to get instantaneous coordinates
      RCP<ParameterList> p = rcp(new ParameterList("Instantaneous Target Coordinates"));

      //Input
      p->set<std::string>("Coordinate Vector Name", "Coord Vec"); // Reference node coordinates
      p->set< std::string >("Solution Vector Name", tgt_name[0]); // Displacements

      //Output
      p->set<std::string>("Instantaneous Coordinates Name", "Current Tgt Coords");

      ev = rcp(new PHAL::CalcInstantaneousCoords<EvalT, AlbanyTraits>(*p, dl));
      fm0.template registerEvaluator<EvalT>(ev);
    }

   // Add the target solution

    //gets solution vector
    fm0.template registerEvaluator<EvalT>
    (evalUtils.constructGatherSolutionEvaluator_noTransient(true, tgt_name, numDim));

    // Puts residual vector
    fm0.template registerEvaluator<EvalT>
    (evalUtils.constructScatterResidualEvaluator(true, tgt_resid_name, numDim));

    {
      // Laplace equation Resid - solve for the target space
      RCP<ParameterList> p = rcp(new ParameterList("Disp Target Laplace Resid"));

      //Input
      // Target is calculated from the actual solution
      p->set< std::string >("Solution Vector Name", "Current Coords");

      p->set< RCP<Intrepid::Cubature<RealType> > >("Cubature", cubature);
      p->set<RCP<shards::CellTopology> >("Cell Type", cellType);
      p->set< RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >
         ("Intrepid Basis", intrepidBasis);

      //Output
      p->set<std::string>("Residual Name", tgt_resid_name[0]);

      ev = rcp(new PHAL::TPSLaplaceResid<EvalT, AlbanyTraits>(*p, dl));
      fm0.template registerEvaluator<EvalT>(ev);
    }

    {
      // Calculate the target metric tensor

      RCP<ParameterList> p =
        rcp(new ParameterList("Disp Contravariant Metric Tensor"));

      // Inputs: 
      // Note that the target solution is used to build Gc
      p->set< std::string >("Solution Vector Name", "Current Tgt Coords");
      p->set< RCP<Intrepid::Cubature<RealType> > >("Cubature", cubature);

      p->set<RCP<shards::CellTopology> >("Cell Type", cellType);

      // Outputs
      p->set<std::string>("Contravariant Metric Tensor Name", "Gc");

      ev = rcp(new PHAL::ContravariantTargetMetricTensor<EvalT, AlbanyTraits>(*p, dl));
      fm0.template registerEvaluator<EvalT>(ev);

    }

    {

      // Laplace Beltrami Resid - Solve for the coordinates
      RCP<ParameterList> p = rcp(new ParameterList("Disp Laplace Beltrami Resid"));

      //Input
      p->set< std::string >("Solution Vector Name", "Current Coords");
      p->set<std::string>("Contravariant Metric Tensor Name", "Gc");

      p->set< RCP<Intrepid::Cubature<RealType> > >("Cubature", cubature);
      p->set<RCP<shards::CellTopology> >("Cell Type", cellType);
      p->set< RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >
        ("Intrepid Basis", intrepidBasis);

      //Output
      p->set<std::string>("Residual Name", soln_resid_name[0]);

      ev = rcp(new PHAL::LaplaceBeltramiResid<EvalT, AlbanyTraits>(*p, dl));
      fm0.template registerEvaluator<EvalT>(ev);
    }
  }
  else if(method == "DispLaplaceBeltrami"){

    {
      // Add coords and displacements to get instantaneous coordinates
      RCP<ParameterList> p = rcp(new ParameterList("Instantaneous Coordinates"));

      //Input
      p->set<std::string>("Coordinate Vector Name", "Coord Vec"); // Reference node coordinates
      p->set< std::string >("Solution Vector Name", soln_name[0]); // Displacements

      //Output
      p->set<std::string>("Instantaneous Coordinates Name", "Current Coords");

      ev = rcp(new PHAL::CalcInstantaneousCoords<EvalT, AlbanyTraits>(*p, dl));
      fm0.template registerEvaluator<EvalT>(ev);
    }

    {
      // Add coords and displacements to get instantaneous coordinates
      RCP<ParameterList> p = rcp(new ParameterList("Instantaneous Target Coordinates"));

      //Input
      p->set<std::string>("Coordinate Vector Name", "Coord Vec"); // Reference node coordinates
      p->set< std::string >("Solution Vector Name", tgt_name[0]); // Displacements

      //Output
      p->set<std::string>("Instantaneous Coordinates Name", "Current Tgt Coords");

      ev = rcp(new PHAL::CalcInstantaneousCoords<EvalT, AlbanyTraits>(*p, dl));
      fm0.template registerEvaluator<EvalT>(ev);
    }

   // Add the target solution

    //gets solution vector
    fm0.template registerEvaluator<EvalT>
    (evalUtils.constructGatherSolutionEvaluator_noTransient(true, tgt_name, numDim));

    // Puts residual vector
    fm0.template registerEvaluator<EvalT>
    (evalUtils.constructScatterResidualEvaluator(true, tgt_resid_name, numDim));

    fm0.template registerEvaluator<EvalT>
      (evalUtils.constructComputeBasisFunctionsEvaluator(cellType, intrepidBasis, cubature));

    {

      // TPS Laplace Resid
      RCP<ParameterList> p = rcp(new ParameterList("Disp TPSA Laplace Resid"));

      //Input
      p->set< std::string >("Solution Vector Name", "Current Coords");
      p->set<std::string>("Gradient BF Name", "Grad BF");
      p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");

      //Output
      p->set<std::string>("Residual Name", tgt_resid_name[0]);

      ev = rcp(new PHAL::TPSALaplaceResid<EvalT, AlbanyTraits>(*p, dl));
      fm0.template registerEvaluator<EvalT>(ev);
    }

    {
      // Calculate the target metric tensor

      RCP<ParameterList> p =
        rcp(new ParameterList("Disp Contravariant Metric Tensor"));

      // Inputs: 
      // Note that the target solution is used to build Gc
      p->set< std::string >("Solution Vector Name", "Current Tgt Coords");
      p->set< RCP<Intrepid::Cubature<RealType> > >("Cubature", cubature);

      p->set<RCP<shards::CellTopology> >("Cell Type", cellType);

      // Outputs
      p->set<std::string>("Contravariant Metric Tensor Name", "Gc");

      ev = rcp(new PHAL::ContravariantTargetMetricTensor<EvalT, AlbanyTraits>(*p, dl));
      fm0.template registerEvaluator<EvalT>(ev);

    }

    {

      // Laplace Beltrami Resid - Solve for the coordinates
      RCP<ParameterList> p = rcp(new ParameterList("Disp Laplace Beltrami Resid"));

      //Input
      p->set< std::string >("Solution Vector Name", "Current Coords");
      p->set<std::string>("Contravariant Metric Tensor Name", "Gc");

      p->set< RCP<Intrepid::Cubature<RealType> > >("Cubature", cubature);
      p->set<RCP<shards::CellTopology> >("Cell Type", cellType);
      p->set< RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >
        ("Intrepid Basis", intrepidBasis);

      //Output
      p->set<std::string>("Residual Name", soln_resid_name[0]);

      ev = rcp(new PHAL::LaplaceBeltramiResid<EvalT, AlbanyTraits>(*p, dl));
      fm0.template registerEvaluator<EvalT>(ev);
    }
  }
  else {

     TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Smoothing method requested is not implemented.\n");

  }

  if(fieldManagerChoice == Albany::BUILD_RESID_FM)  {
    PHX::Tag<typename EvalT::ScalarT> res_tag("Scatter", dl->dummy);
    fm0.requireField<EvalT>(res_tag);
    return res_tag.clone();
  }

  else if(fieldManagerChoice == Albany::BUILD_RESPONSE_FM) {
    Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
    return respUtils.constructResponses(fm0, *responseList, Teuchos::null, stateMgr);
  }

  return Teuchos::null;

}


#endif // ALBANY_LAPLACEBELTRAMI_HPP
