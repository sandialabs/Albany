//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_POISSONADVDIFFSYSTEM_HPP
#define ALBANY_POISSONADVDIFFSYSTEM_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_AbstractProblem.hpp"

#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"

namespace Albany {

  /*!
   * \brief Abstract interface for representing a 1-D finite element
   * problem.
   */
  class PoissonAdvDiffSystem : public AbstractProblem {
  public:
  
    //! Default constructor
    PoissonAdvDiffSystem(const Teuchos::RCP<Teuchos::ParameterList>& params,
		 const Teuchos::RCP<ParamLib>& paramLib,
		 const int numDim_);

    //! Destructor
    ~PoissonAdvDiffSystem();

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

    //! Each problem must generate it's list of valid parameters
    Teuchos::RCP<const Teuchos::ParameterList> getValidProblemParameters() const;

  private:

    //! Private to prohibit copying
    PoissonAdvDiffSystem(const PoissonAdvDiffSystem&);
    
    //! Private to prohibit copying
    PoissonAdvDiffSystem& operator=(const PoissonAdvDiffSystem&);

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

  protected:
    int numDim;
  
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

#include "PHAL_DOFVecGradInterpolation.hpp"

#include "PHAL_PoissonAdvDiffSystemResid.hpp"

#include "PHAL_Source.hpp"

template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
Albany::PoissonAdvDiffSystem::constructEvaluators(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& /* stateMgr */,
  Albany::FieldManagerChoice fieldManagerChoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::ParameterList;
  using PHX::DataLayout;
  using std::vector;
  using std::string;
  using std::map;
  using PHAL::AlbanyTraits;
  
  RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> >
    intrepidBasis = Albany::getIntrepid2Basis(meshSpecs.ctd);
  RCP<shards::CellTopology> cellType = rcp(new shards::CellTopology (&meshSpecs.ctd));
  
  const int numNodes = intrepidBasis->getCardinality();
  const int worksetSize = meshSpecs.worksetSize;
  const int cubDegree = this->params->get("Cubature Degree", 3);
  Intrepid2::DefaultCubatureFactory cubFactory;
  RCP <Intrepid2::Cubature<PHX::Device> > cubature = cubFactory.create<PHX::Device, RealType, RealType>(*cellType, cubDegree);
  
  const int numQPts = cubature->getNumPoints();
  const int numVertices = cellType->getNodeCount();
  
  *out << "Field Dimensions: Workset=" << worksetSize 
       << ", Vertices= " << numVertices
       << ", Nodes= " << numNodes
       << ", QuadPts= " << numQPts
       << ", Dim= " << numDim << std::endl;
  
   int vecDim = neq;

   RCP<Albany::Layouts> dl = rcp(new Albany::Layouts(worksetSize,numVertices,numNodes,numQPts,numDim, vecDim));
   Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
   bool supportsTransient = false;
   if(number_of_time_deriv > 0) 
      supportsTransient = true;
   int offset=0;

   // This problem appears to be only defined as a transient problem, throw exception if it is not
   TEUCHOS_TEST_FOR_EXCEPTION(
      number_of_time_deriv != 0,
      std::logic_error,
      "Albany_PoissonAdvDiffSystem must be defined as a steady calculation.");

   // Temporary variable used numerous times below
   Teuchos::RCP<PHX::Evaluator<AlbanyTraits> > ev;

   // Define Field Names

     Teuchos::ArrayRCP<string> dof_names(1);
     Teuchos::ArrayRCP<string> dof_names_dot(1);
     Teuchos::ArrayRCP<string> resid_names(1);
     dof_names[0] = "Phi";
     //IKT 12/23/2024: will need to change this.  Phi will never had a dot, but 
     //the other variable rhop will.  Keeping 'enable transient' logic for now.
     if (supportsTransient)
       dof_names_dot[0] = dof_names[0]+"_dot";
     resid_names[0] = "PoissonAdvDiff System Residual";

     if (supportsTransient) fm0.template registerEvaluator<EvalT>
           (evalUtils.constructGatherSolutionEvaluator(true, dof_names, dof_names_dot, offset));
     else fm0.template registerEvaluator<EvalT>
           (evalUtils.constructGatherSolutionEvaluator_noTransient(true, dof_names, offset));

     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructDOFVecInterpolationEvaluator(dof_names[0], offset));

     if (supportsTransient) fm0.template registerEvaluator<EvalT>
       (evalUtils.constructDOFVecInterpolationEvaluator(dof_names_dot[0], offset));

     //     fm0.template registerEvaluator<EvalT>
     //  (evalUtils.constructDOFVecGradInterpolationEvaluator(dof_names[0], offset));

     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructScatterResidualEvaluator(true, resid_names, offset, "Scatter PoissonAdvDiff System"));

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructGatherCoordinateVectorEvaluator());

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructMapToPhysicalFrameEvaluator(cellType, cubature));

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructComputeBasisFunctionsEvaluator(cellType, intrepidBasis, cubature));

   { // Specialized DofVecGrad Interpolation for this problem
    
     RCP<ParameterList> p = rcp(new ParameterList("DOFVecGrad Interpolation "+dof_names[0]));
     // Input
     p->set<string>("Variable Name", dof_names[0]);
     p->set<string>("Gradient BF Name", "Grad BF");
     p->set<int>("Offset of First DOF", offset);
     
     // Output (assumes same Name as input)
     p->set<string>("Gradient Variable Name", dof_names[0]+" Gradient");
     
     ev = rcp(new PHAL::DOFVecGradInterpolation<EvalT,AlbanyTraits>(*p,dl));
     fm0.template registerEvaluator<EvalT>(ev);
   }

  { // PoissonAdvDiff System Resid
    RCP<ParameterList> p = rcp(new ParameterList("PoissonAdvDiff System Resid"));

    //Input
    p->set<string>("Weighted BF Name", "wBF");
    p->set<string>("Weighted Gradient BF Name", "wGrad BF");
    p->set<string>("QP Variable Name", "Phi");
    p->set<string>("Gradient QP Variable Name", "Phi Gradient");
 
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_vecgradient);
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", dl->node_qp_scalar);
    p->set< RCP<DataLayout> >("Node QP Gradient Data Layout", dl->node_qp_gradient);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Options");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    //Output
    p->set<string>("Residual Name", "PoissonAdvDiff System Residual");
    p->set< RCP<DataLayout> >("Node Vector Data Layout", dl->node_vector);

    ev = rcp(new PHAL::PoissonAdvDiffSystemResid<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }


  if (fieldManagerChoice == Albany::BUILD_RESID_FM)  {
    PHX::Tag<typename EvalT::ScalarT> res_tag("Scatter PoissonAdvDiff System", dl->dummy);
    fm0.requireField<EvalT>(res_tag);
  }
  else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM) {
    Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
    return respUtils.constructResponses(fm0, *responseList, Teuchos::null);
  }

  return Teuchos::null;
}
#endif // ALBANY_POISSONADVDIFFSYSTEM_HPP
