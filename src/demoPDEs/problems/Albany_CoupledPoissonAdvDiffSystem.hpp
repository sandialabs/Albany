//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_COUPLEDPOISSONADVDIFFSYSTEM_HPP
#define ALBANY_COUPLEDPOISSONADVDIFFSYSTEM_HPP

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
  class CoupledPoissonAdvDiffSystem : public AbstractProblem {
  public:
  
    //! Default constructor
    CoupledPoissonAdvDiffSystem(const Teuchos::RCP<Teuchos::ParameterList>& params,
		 const Teuchos::RCP<ParamLib>& paramLib,
		 const int numDim_);

    //! Destructor
    ~CoupledPoissonAdvDiffSystem();

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
    CoupledPoissonAdvDiffSystem(const CoupledPoissonAdvDiffSystem&);
    
    //! Private to prohibit copying
    CoupledPoissonAdvDiffSystem& operator=(const CoupledPoissonAdvDiffSystem&);

  public:

    //! Main problem setup routine. Not directly called, but indirectly by following functions
    template <typename EvalT> Teuchos::RCP<const PHX::FieldTag>
    constructEvaluators(
      PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
      const Albany::MeshSpecsStruct& meshSpecs,
      Albany::StateManager& stateMgr,
      Albany::FieldManagerChoice fmchoice,
      const Teuchos::RCP<Teuchos::ParameterList>& responseList);
    
    void constructDirichletEvaluators(const std::vector<std::string>& nodeSetIDs);

  protected:

    
    bool periodic;     //! periodic BCs
    int numDim;        //! number of spatial dimensions

    const Teuchos::RCP<Teuchos::ParameterList> params; 
  
    Teuchos::RCP<Albany::Layouts> dl;
  
    /// Boolean marking whether SDBCs are used 
    bool use_sdbcs_; 
  };

}

#include <boost/type_traits/is_same.hpp>

#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"

#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_EvaluatorUtils.hpp"
#include "Albany_ResponseUtilities.hpp"

#include "PHAL_CoupledPoissonResid.hpp"
#include "PHAL_CoupledAdvDiffResid.hpp"

template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
Albany::CoupledPoissonAdvDiffSystem::constructEvaluators(
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
  const int cubDegree = params->get("Cubature Degree", 3);
  Intrepid2::DefaultCubatureFactory cubFactory;
  RCP <Intrepid2::Cubature<PHX::Device> > cubature = cubFactory.create<PHX::Device, RealType, RealType>(*cellType, cubDegree);
  
  const int numQPts = cubature->getNumPoints();
  const int numVertices = cellType->getNodeCount();

  // Print only for the residual specialization
  
  if(boost::is_same<EvalT, PHAL::AlbanyTraits::Residual>::value)

    *out << "Field Dimensions: Workset=" << worksetSize 
       << ", Vertices= " << numVertices
       << ", Nodes= " << numNodes
       << ", QuadPts= " << numQPts
       << ", Dim= " << numDim << std::endl;
  
   dl = rcp(new Albany::Layouts(worksetSize,numVertices,numNodes,numQPts,numDim));
   TEUCHOS_TEST_FOR_EXCEPTION(dl->vectorAndGradientLayoutsAreEquivalent==false, std::logic_error,
                              "Data Layout Usage in CoupledPoissonAdvDiffSystem problem assumes vecDim = numDim");

   Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
   int offset=0;

   // Problem is transient
   TEUCHOS_TEST_FOR_EXCEPTION(
      number_of_time_deriv < 0 || number_of_time_deriv > 1,
      std::logic_error,
      "Albany_CoupledPoissonAdvDiffSystemProblem must be defined as a steady or transient calculation.");

   // Temporary variable used numerous times below
   Teuchos::RCP<PHX::Evaluator<AlbanyTraits> > ev;

   // Define Field Names

   { // Gather Solution Phi (electric potential) 
     Teuchos::ArrayRCP<string> dof_names(1);
     Teuchos::ArrayRCP<string> dof_names_dot(1);
     Teuchos::ArrayRCP<string> resid_names(1);
     dof_names[0] = "Phi";
     if(number_of_time_deriv > 0)
       dof_names_dot[0] = dof_names[0]+" Dot";
     resid_names[0] = dof_names[0]+" Residual";
     
     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructGatherSolutionEvaluator_noTransient(false, dof_names, offset));

     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructDOFInterpolationEvaluator(dof_names[0], offset));

     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructDOFGradInterpolationEvaluator(dof_names[0], offset));

     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructScatterResidualEvaluator(false, resid_names, offset, "Scatter Phi"));
     offset ++;
   }

   { // Gather Solution rhop (positive ion density)
     Teuchos::ArrayRCP<string> dof_names(1);
     Teuchos::ArrayRCP<string> dof_names_dot(1);
     Teuchos::ArrayRCP<string> resid_names(1);
     dof_names[0] = "rhop";
     if(number_of_time_deriv > 0)
       dof_names_dot[0] = dof_names[0]+" Dot";
     resid_names[0] = dof_names[0]+" Residual";
     if(number_of_time_deriv > 0)
       fm0.template registerEvaluator<EvalT>
         (evalUtils.constructGatherSolutionEvaluator(false, dof_names, dof_names_dot, offset));
     else
       fm0.template registerEvaluator<EvalT>
         (evalUtils.constructGatherSolutionEvaluator_noTransient(false, dof_names, offset));

     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructDOFInterpolationEvaluator(dof_names[0], offset));

     if(number_of_time_deriv > 0)
       fm0.template registerEvaluator<EvalT>
         (evalUtils.constructDOFInterpolationEvaluator(dof_names_dot[0], offset));

     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructDOFGradInterpolationEvaluator(dof_names[0], offset));

     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructScatterResidualEvaluator(false, resid_names, offset, "Scatter rhop"));
     offset ++;
   }

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructGatherCoordinateVectorEvaluator());

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructMapToPhysicalFrameEvaluator(cellType, cubature));

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructComputeBasisFunctionsEvaluator(cellType, intrepidBasis, cubature));


   { // Phi Residual
    RCP<ParameterList> p = rcp(new ParameterList("Phi Resid"));

    //Input
    p->set<string>("Weighted BF Name", "wBF");
    p->set<string>("Weighted Gradient BF Name", "wGrad BF");
    p->set<string>("QP Variable Name", "Phi");
    p->set<string>("QP Variable Name", "rhop");
    p->set<string>("Gradient QP Variable Name", "Phi Gradient");
 
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", dl->node_qp_scalar);
    p->set< RCP<DataLayout> >("Node QP Gradient Data Layout", dl->node_qp_gradient);
    
    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Options");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    //Output
    p->set<string>("Residual Name", "Phi Residual");
    p->set< RCP<DataLayout> >("Node Scalar Data Layout", dl->node_scalar);

    ev = rcp(new PHAL::CoupledPoissonResid<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  { // rhop Residual
    RCP<ParameterList> p = rcp(new ParameterList("rhop Resid"));

    //Input
    p->set<string>("Weighted BF Name", "wBF");
    p->set<string>("Weighted Gradient BF Name", "wGrad BF");
    p->set<string>("QP Variable Name", "rhop");
    p->set<string>("Gradient QP Variable Name", "rhop Gradient");
    p->set<string>("Gradient QP Variable Name", "Phi Gradient");
    p->set<string>("QP Time Derivative Variable Name", "rhop Dot");
 
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", dl->node_qp_scalar);
    p->set< RCP<DataLayout> >("Node QP Gradient Data Layout", dl->node_qp_gradient);
    
    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Options");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    //Output
    p->set<string>("Residual Name", "rhop Residual");
    p->set< RCP<DataLayout> >("Node Scalar Data Layout", dl->node_scalar);

    ev = rcp(new PHAL::CoupledAdvDiffResid<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (fieldManagerChoice == Albany::BUILD_RESID_FM)  {
    Teuchos::RCP<const PHX::FieldTag> ret_tag;
    PHX::Tag<typename EvalT::ScalarT> phi_tag("Scatter Phi", dl->dummy);
    fm0.requireField<EvalT>(phi_tag);
    PHX::Tag<typename EvalT::ScalarT> rhop_tag("Scatter rhop", dl->dummy);
    fm0.requireField<EvalT>(rhop_tag);
    ret_tag = rhop_tag.clone();
    return ret_tag;
  }
  else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM) {
    Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
    return respUtils.constructResponses(fm0, *responseList, Teuchos::null);
  }

  return Teuchos::null;
}
#endif // ALBANY_COUPLEDPOISSONADVDIFFSYSTEM_HPP
