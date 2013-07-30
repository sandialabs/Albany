//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_CAHNHILLPROBLEM_HPP
#define ALBANY_CAHNHILLPROBLEM_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_AbstractProblem.hpp"

#include "Phalanx.hpp"
#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"
#include "Albany_ProblemUtils.hpp"

namespace Albany {

  /*!
   * \brief Abstract interface for representing a 1-D finite element
   * problem.
   */
  class CahnHillProblem : public AbstractProblem {
  public:
  
    //! Default constructor
    CahnHillProblem(const Teuchos::RCP<Teuchos::ParameterList>& params,
		const Teuchos::RCP<ParamLib>& paramLib,
		const int numDim_,
    const Teuchos::RCP<const Epetra_Comm>& comm_);

    //! Destructor
    ~CahnHillProblem();

    //! Return number of spatial dimensions
    virtual int spatialDimension() const { return numDim; }

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
    CahnHillProblem(const CahnHillProblem&);
    
    //! Private to prohibit copying
    CahnHillProblem& operator=(const CahnHillProblem&);

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

    bool haveNoise; // Langevin noise present

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

#include "PHAL_CahnHillChemTerm.hpp"
#include "PHAL_LangevinNoiseTerm.hpp"
#include "PHAL_CahnHillRhoResid.hpp"
#include "PHAL_CahnHillWResid.hpp"


template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
Albany::CahnHillProblem::constructEvaluators(
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
   using PHAL::AlbanyTraits;

   const CellTopologyData * const elem_top = &meshSpecs.ctd;

   RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > >
     intrepidBasis = Albany::getIntrepidBasis(*elem_top);
   RCP<shards::CellTopology> cellType = rcp(new shards::CellTopology (elem_top));


   const int numNodes = intrepidBasis->getCardinality();
   const int worksetSize = meshSpecs.worksetSize;

   Intrepid::DefaultCubatureFactory<RealType> cubFactory;
   RCP <Intrepid::Cubature<RealType> > cellCubature = cubFactory.create(*cellType, meshSpecs.cubatureDegree);

   const int numQPtsCell = cellCubature->getNumPoints();
   const int numVertices = cellType->getNodeCount();


   *out << "Field Dimensions: Workset=" << worksetSize 
        << ", Vertices= " << numVertices
        << ", Nodes= " << numNodes
        << ", QuadPts= " << numQPtsCell
        << ", Dim= " << numDim << std::endl;

   dl = rcp(new Albany::Layouts(worksetSize,numVertices,numNodes,numQPtsCell,numDim));
   Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);

  // Temporary variable used numerous times below
  Teuchos::RCP<PHX::Evaluator<AlbanyTraits> > ev;

   Teuchos::ArrayRCP<string> dof_names(neq);
     dof_names[0] = "Rho"; // The concentration difference variable 0 \leq \rho \leq 1
     dof_names[1] = "W"; // The chemical potential difference variable
   Teuchos::ArrayRCP<string> dof_names_dot(neq);
     dof_names_dot[0] = "rhoDot";
     dof_names_dot[1] = "wDot"; // not currently used
   Teuchos::ArrayRCP<string> resid_names(neq);
     resid_names[0] = "Rho Residual";
     resid_names[1] = "W Residual";

  fm0.template registerEvaluator<EvalT>
     (evalUtils.constructGatherSolutionEvaluator(false, dof_names, dof_names_dot));

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

    fm0.template registerEvaluator<EvalT>
      (evalUtils.constructDOFInterpolationEvaluator(dof_names_dot[i]));

    fm0.template registerEvaluator<EvalT>
      (evalUtils.constructDOFGradInterpolationEvaluator(dof_names[i]));
  }


  { // Form the Chemical Energy term in Eq. 2.2

    RCP<ParameterList> p = rcp(new ParameterList("Chem Energy Term"));

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);

    // b value in Equation 1.1
    p->set<double>("b Value", params->get<double>("b"));

    //Input
    p->set<string>("Rho QP Variable Name", "Rho");
    p->set<string>("W QP Variable Name", "W");

    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    //Output
    p->set<string>("Chemical Energy Term", "Chemical Energy Term");

    ev = rcp(new PHAL::CahnHillChemTerm<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if(params->isParameter("Langevin Noise SD")){

   // Form the Langevin noise term

    haveNoise = true;

    RCP<ParameterList> p = rcp(new ParameterList("Langevin Noise Term"));

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);

    // Standard deviation of the noise
    p->set<double>("SD Value", params->get<double>("Langevin Noise SD"));
    // Time period over which to apply the noise (-1 means over the whole time)
    p->set<Teuchos::Array<int> >("Langevin Noise Time Period", 
        params->get<Teuchos::Array<int> >("Langevin Noise Time Period", Teuchos::tuple<int>(-1, -1)));

    //Input
    p->set<string>("Rho QP Variable Name", "Rho");

    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    //Output
    p->set<string>("Langevin Noise Term", "Langevin Noise Term");

    ev = rcp(new PHAL::LangevinNoiseTerm<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  { // Rho Resid
    RCP<ParameterList> p = rcp(new ParameterList("Rho Resid"));

    //Input
    p->set<string>("Weighted BF Name", "wBF");
    p->set<string>("Weighted Gradient BF Name", "wGrad BF");
    if(haveNoise)
      p->set<string>("Langevin Noise Term", "Langevin Noise Term");
    // Accumulate in the Langevin noise term?
    p->set<bool>("Have Noise", haveNoise);

    p->set<string>("Chemical Energy Term", "Chemical Energy Term");
    p->set<string>("Gradient QP Variable Name", "Rho Gradient");

    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", dl->node_qp_scalar);
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);

    // gamma value in Equation 2.2
    p->set<double>("gamma Value", params->get<double>("gamma"));

    //Output
    p->set<string>("Residual Name", "Rho Residual");
    p->set< RCP<DataLayout> >("Node Scalar Data Layout", dl->node_scalar);

    ev = rcp(new PHAL::CahnHillRhoResid<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  { // W Resid
    RCP<ParameterList> p = rcp(new ParameterList("W Resid"));

    //Input
    p->set<string>("Weighted BF Name", "wBF");
    p->set<string>("BF Name", "BF");
    p->set<string>("Rho QP Time Derivative Variable Name", "rhoDot");
    p->set<string>("Weighted Gradient BF Name", "wGrad BF");
    p->set<string>("Gradient QP Variable Name", "W Gradient");

    // Mass lump time term?
    p->set<bool>("Lump Mass", params->get<bool>("Lump Mass"));

    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", dl->node_qp_scalar);
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);

    //Output
    p->set<string>("Residual Name", "W Residual");
    p->set< RCP<DataLayout> >("Node Scalar Data Layout", dl->node_scalar);

    ev = rcp(new PHAL::CahnHillWResid<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (fieldManagerChoice == Albany::BUILD_RESID_FM)  {
    PHX::Tag<typename EvalT::ScalarT> res_tag("Scatter", dl->dummy);
    fm0.requireField<EvalT>(res_tag);
    return res_tag.clone();
  }

  else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM) {
    Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
    return respUtils.constructResponses(fm0, *responseList, Teuchos::null, stateMgr);
  }

  return Teuchos::null;
}


#endif // ALBANY_CAHNHILLPROBLEM_HPP
