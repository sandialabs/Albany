/********************************************************************\
*            Albany, Copyright (2012) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Glen Hansen, gahanse@sandia.gov                    *
\********************************************************************/


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

    //! Each problem must generate it's list of valide parameters
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
        << ", Dim= " << numDim << endl;

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

  { // W Resid
    RCP<ParameterList> p = rcp(new ParameterList("W Resid"));

    //Input
    p->set<string>("Weighted BF Name", "wBF");
    p->set<string>("Rho QP Time Derivative Variable Name", "rhoDot");
    p->set<string>("Weighted Gradient BF Name", "wGrad BF");
    p->set<string>("Gradient QP Variable Name", "W Gradient");

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

  { // Form the Chemical Energy term in Eq. 2.2

    RCP<ParameterList> p = rcp(new ParameterList("Chem Energy Term"));

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);

    // b value in Equation 1.1
    p->set<double>("b Value", params->get<double>("b"));

    //Input
    p->set<string>("Rho QP Variable Name", "Rho");
    p->set<string>("W QP Variable Name", "W");

    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);

    //Output
    p->set<string>("Chemical Energy Term", "Chemical Energy Term");

    ev = rcp(new PHAL::CahnHillChemTerm<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  { // Rho Resid
    RCP<ParameterList> p = rcp(new ParameterList("Rho Resid"));

    //Input
    p->set<string>("Weighted BF Name", "wBF");
    p->set<string>("Weighted Gradient BF Name", "wGrad BF");

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

  if (fieldManagerChoice == Albany::BUILD_RESID_FM)  {
    PHX::Tag<typename EvalT::ScalarT> res_tag("Scatter", dl->dummy);
    fm0.requireField<EvalT>(res_tag);
    return res_tag.clone();
  }

  else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM) {
    Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
    return respUtils.constructResponses(fm0, *responseList, stateMgr);
  }

  return Teuchos::null;
}


#endif // ALBANY_CAHNHILLPROBLEM_HPP
