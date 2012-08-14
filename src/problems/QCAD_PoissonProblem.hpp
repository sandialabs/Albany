/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
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
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


#ifndef QCAD_POISSONPROBLEM_HPP
#define QCAD_POISSONPROBLEM_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_AbstractProblem.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_ResponseUtilities.hpp"

#include "Phalanx.hpp"
#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"
#include "QCAD_MaterialDatabase.hpp"


//! Code Base for Quantum Device Simulation Tools LDRD
namespace QCAD {

  /*!
   * \brief Abstract interface for representing a 1-D finite element
   * problem.
   */
  class PoissonProblem : public Albany::AbstractProblem 
  {
  public:
  
    //! Default constructor
    PoissonProblem(const Teuchos::RCP<Teuchos::ParameterList>& params,
		   const Teuchos::RCP<ParamLib>& paramLib,
		   const int numDim_,
		   const Teuchos::RCP<const Epetra_Comm>& comm_);

    //! Destructor
    ~PoissonProblem();

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
    PoissonProblem(const PoissonProblem&);
    
    //! Private to prohibit copying
    PoissonProblem& operator=(const PoissonProblem&);

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

    void constructDirichletEvaluators(const Albany::MeshSpecsStruct& meshSpecs);
    void constructNeumannEvaluators(const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs);

  protected:

    //! Boundary conditions on source term
    bool periodic;

    //! Parameters to use when constructing evaluators
    Teuchos::RCP<const Epetra_Comm> comm;
    bool haveSource;
    int numDim;
    double length_unit_in_m;
    double temperature;
    Teuchos::RCP<QCAD::MaterialDatabase> materialDB;
    Teuchos::RCP<Albany::Layouts> dl;

    //! Parameters for coupling to Schrodinger
    int nEigenvectors;


    /* Now Poisson source Params
    bool bUseSchrodingerSource;
    bool bUsePredictorCorrector;
    bool bIncludeVxc; 
    */
         
  };

}

#include "QCAD_MaterialDatabase.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_EvaluatorUtils.hpp"

#include "PHAL_GatherEigenvectors.hpp"
#include "QCAD_Permittivity.hpp"
#include "PHAL_SharedParameter.hpp"
#include "QCAD_PoissonSource.hpp"
#include "PHAL_DOFInterpolation.hpp"
#include "QCAD_PoissonResid.hpp"
#include "QCAD_ResponseSaddleValue.hpp"

#include "PHAL_SaveStateField.hpp"

template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
QCAD::PoissonProblem::constructEvaluators(
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

   RCP<shards::CellTopology> cellType = rcp(new shards::CellTopology (&meshSpecs.ctd));
   RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > >
     intrepidBasis = Albany::getIntrepidBasis(meshSpecs.ctd);

   const int numNodes = intrepidBasis->getCardinality();
   const int worksetSize = meshSpecs.worksetSize;

   Intrepid::DefaultCubatureFactory<RealType> cubFactory;
   RCP <Intrepid::Cubature<RealType> > cubature = cubFactory.create(*cellType, meshSpecs.cubatureDegree);

   const int numQPts = cubature->getNumPoints();
   const int numVertices = cellType->getNodeCount();

   *out << "Field Dimensions: Workset=" << worksetSize 
        << ", Vertices= " << numVertices
        << ", Nodes= " << numNodes
        << ", QuadPts= " << numQPts
        << ", Dim= " << numDim << endl;

   RCP<DataLayout> shared_param = rcp(new MDALayout<Dim>(1));

   dl = rcp(new Albany::Layouts(worksetSize,numVertices,numNodes,numQPts,numDim));
   Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
   bool supportsTransient=false;

   // Temporary variable used numerous times below
   Teuchos::RCP<PHX::Evaluator<AlbanyTraits> > ev;

   // Define Field Names

   Teuchos::ArrayRCP<string> dof_names(neq);
     dof_names[0] = "Potential";

   Teuchos::ArrayRCP<string> dof_names_dot(neq);
   if (supportsTransient) {
     for (unsigned int i=0; i<neq; i++) dof_names_dot[i] = dof_names[i]+"_dot";
   }

   Teuchos::ArrayRCP<string> resid_names(neq);
     for (unsigned int i=0; i<neq; i++) resid_names[i] = dof_names[i]+" Residual";

   if (supportsTransient) fm0.template registerEvaluator<EvalT>
       (evalUtils.constructGatherSolutionEvaluator(false, dof_names, dof_names_dot));
   else fm0.template registerEvaluator<EvalT>
       (evalUtils.constructGatherSolutionEvaluator_noTransient(false, dof_names));

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructScatterResidualEvaluator(false, resid_names));

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructGatherCoordinateVectorEvaluator());

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructMapToPhysicalFrameEvaluator(cellType, cubature));

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructComputeBasisFunctionsEvaluator(cellType, intrepidBasis, cubature));

   for (unsigned int i=0; i<neq; i++) {
     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructDOFInterpolationEvaluator(dof_names[i]));

     if (supportsTransient)
     fm0.template registerEvaluator<EvalT>
         (evalUtils.constructDOFInterpolationEvaluator(dof_names_dot[i]));

     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructDOFGradInterpolationEvaluator(dof_names[i]));
  }

  { // Gather Eigenvectors
    RCP<ParameterList> p = rcp(new ParameterList);
    p->set<string>("Eigenvector field name root", "Evec");
    p->set<int>("Number of eigenvectors", nEigenvectors);
    p->set< RCP<DataLayout> >("Data Layout", dl->node_scalar);

    ev = rcp(new PHAL::GatherEigenvectors<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  { // Permittivity
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<string>("QP Variable Name", "Permittivity");
    p->set<string>("Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Node Data Layout", dl->node_scalar);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    p->set< RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Permittivity");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    p->set< RCP<QCAD::MaterialDatabase> >("MaterialDB", materialDB);

    ev = rcp(new QCAD::Permittivity<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  { // Temperature shared parameter (single scalar value, not spatially varying)
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<string>("Parameter Name", "Temperature");
    p->set<double>("Parameter Value", temperature);
    p->set< RCP<DataLayout> >("Data Layout", shared_param);
    p->set< RCP<ParamLib> >("Parameter Library", paramLib);

    ev = rcp(new PHAL::SharedParameter<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (haveSource) 
  { // Source
    RCP<ParameterList> p = rcp(new ParameterList);

    //Input
    p->set< string >("Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    p->set<string>("Variable Name", "Potential");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);

    // Poisson Source ParameterList
    Teuchos::ParameterList& paramList = params->sublist("Poisson Source");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);
    
    // Dirichlet BCs ParameterList 
    Teuchos::ParameterList& dbcPList = params->sublist("Dirichlet BCs");
    p->set<Teuchos::ParameterList*>("Dirichlet BCs ParameterList", &dbcPList);

    //Output
    p->set<string>("Source Name", "Poisson Source");

    //Global Problem Parameters
    p->set<double>("Length unit in m", length_unit_in_m);
    p->set<string>("Temperature Name", "Temperature");
    p->set< RCP<DataLayout> >("Shared Param Data Layout", shared_param);
    p->set< RCP<QCAD::MaterialDatabase> >("MaterialDB", materialDB);

    // Schrodinger coupling
    p->set<string>("Eigenvector field name root", "Evec");

    /* Now Poisson source params
    p->set<bool>("Use Schrodinger source", bUseSchrodingerSource);
    p->set<int>("Schrodinger eigenvectors", nEigenvectors);
    p->set<bool>("Use predictor-corrector method", bUsePredictorCorrector);
    p->set<bool>("Include exchange-correlation potential", bIncludeVxc); 
    */

    ev = rcp(new QCAD::PoissonSource<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // Interpolate Input Eigenvectors (if any) to quad points
  char buf[100];  
  for( int k = 0; k < nEigenvectors; k++)
  { 
    // DOF: Interpolate nodal Eigenvector values to quad points
    RCP<ParameterList> p;

    //REAL PART
    sprintf(buf, "Poisson Eigenvector Re %d interpolate to qps", k);
    p = rcp(new ParameterList(buf));

    // Input
    sprintf(buf, "Evec_Re%d", k);
    p->set<string>("Variable Name", buf);
    
    p->set<string>("BF Name", "BF");
    
    // Output (assumes same Name as input)
    
    sprintf(buf, "Eigenvector Re %d interpolate to qps", k);
    ev = rcp(new PHAL::DOFInterpolation<EvalT,AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
    
    
    //IMAGINARY PART
    sprintf(buf, "Eigenvector Im %d interpolate to qps", k);
    p = rcp(new ParameterList(buf));
    
    // Input
    sprintf(buf, "Evec_Im%d", k);
    p->set<string>("Variable Name", buf);
    
    p->set<string>("BF Name", "BF");
    
    // Output (assumes same Name as input)
    
    sprintf(buf, "Eigenvector Im %d interpolate to qps", k);
    ev = rcp(new PHAL::DOFInterpolation<EvalT,AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  { // Potential Resid
    RCP<ParameterList> p = rcp(new ParameterList("Potential Resid"));

    //Input
    p->set<string>("Weighted BF Name", "wBF");
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", dl->node_qp_scalar);
    p->set<string>("QP Variable Name", "Potential");

    p->set<string>("QP Time Derivative Variable Name", "Potential_dot");

    p->set<bool>("Have Source", haveSource);
    p->set<string>("Source Name", "Poisson Source");

    p->set<string>("Permittivity Name", "Permittivity");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

    p->set<string>("Gradient QP Variable Name", "Potential Gradient");
    p->set<string>("Flux QP Variable Name", "Potential Flux");
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    p->set<string>("Weighted Gradient BF Name", "wGrad BF");
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);

    //Output
    p->set<string>("Residual Name", "Potential Residual");
    p->set< RCP<DataLayout> >("Node Scalar Data Layout", dl->node_scalar);

    ev = rcp(new QCAD::PoissonResid<EvalT,AlbanyTraits>(*p));
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
#endif
