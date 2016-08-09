//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_COMPRNSPROBLEM_HPP
#define ALBANY_COMPRNSPROBLEM_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_AbstractProblem.hpp"

#include "Phalanx.hpp"
#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"

namespace Albany {

  /*!
   * \brief Abstract interface for representing a 1-D finite element
   * problem.
   */
  class ComprNSProblem : public AbstractProblem {
  public:
  
    //! Default constructor
    ComprNSProblem(const Teuchos::RCP<Teuchos::ParameterList>& params,
		 const Teuchos::RCP<ParamLib>& paramLib,
		 const int numDim_);

    //! Destructor
    ~ComprNSProblem();

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
    ComprNSProblem(const ComprNSProblem&);
    
    //! Private to prohibit copying
    ComprNSProblem& operator=(const ComprNSProblem&);

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
    void constructNeumannEvaluators(const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs);

  protected:
    int numDim;
    Teuchos::RCP<Albany::Layouts> dl;


  };

}

#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"

#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_EvaluatorUtils.hpp"
#include "Albany_ResponseUtilities.hpp"

#include "PHAL_DOFVecGradInterpolation.hpp"

#include "PHAL_ComprNSResid.hpp"

#include "PHAL_Neumann.hpp"
#include "PHAL_Source.hpp"
#include "PHAL_ComprNSBodyForce.hpp"
#include "PHAL_ComprNSViscosity.hpp"

template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
Albany::ComprNSProblem::constructEvaluators(
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
  using std::map;
  using PHAL::AlbanyTraits;
  
  RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> >
    intrepidBasis = Albany::getIntrepid2Basis(meshSpecs.ctd);
  RCP<shards::CellTopology> cellType = rcp(new shards::CellTopology (&meshSpecs.ctd));

  // Problem is transient
  TEUCHOS_TEST_FOR_EXCEPTION(
      number_of_time_deriv != 1,
      std::logic_error,
      "Albany_ComprNSProblem must be defined as a transient calculation.");
  
  const int numNodes = intrepidBasis->getCardinality();
  const int worksetSize = meshSpecs.worksetSize;
  
  Intrepid2::DefaultCubatureFactory cubFactory;
  RCP <Intrepid2::Cubature<PHX::Device> > cubature = cubFactory.create<PHX::Device, RealType, RealType>(*cellType, meshSpecs.cubatureDegree);
  
  const int numQPts = cubature->getNumPoints();
  const int numVertices = cellType->getNodeCount();
  
  *out << "Field Dimensions: Workset=" << worksetSize 
       << ", Vertices= " << numVertices
       << ", Nodes= " << numNodes
       << ", QuadPts= " << numQPts
       << ", Dim= " << numDim << std::endl;
  
   int vecDim = neq;

   dl = rcp(new Albany::Layouts(worksetSize,numVertices,numNodes,numQPts,numDim, vecDim));
   Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
   bool supportsTransient=true;
   int offset=0;

   // Temporary variable used numerous times below
   Teuchos::RCP<PHX::Evaluator<AlbanyTraits> > ev;

   // Define Field Names

     Teuchos::ArrayRCP<string> dof_names(1);
     Teuchos::ArrayRCP<string> dof_names_dot(1);
     Teuchos::ArrayRCP<string> resid_names(1);
     dof_names[0] = "qFluct";
     dof_names_dot[0] = dof_names[0]+"_dot";
     resid_names[0] = "ComprNS Residual";
     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructGatherSolutionEvaluator(true, dof_names, dof_names_dot, offset));

     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructDOFVecInterpolationEvaluator(dof_names[0], offset));

     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructDOFVecInterpolationEvaluator(dof_names_dot[0], offset));

     //     fm0.template registerEvaluator<EvalT>
     //  (evalUtils.constructDOFVecGradInterpolationEvaluator(dof_names[0], offset));

     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructScatterResidualEvaluator(true, resid_names,offset, "Scatter ComprNS"));

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
     
     ev = rcp(new PHAL::DOFVecGradInterpolation<EvalT,AlbanyTraits, typename EvalT::ScalarT>(*p,dl));
     fm0.template registerEvaluator<EvalT>(ev);
   }

  { // ComprNS Resid
    RCP<ParameterList> p = rcp(new ParameterList("ComprNS Resid"));

    //Input
    p->set<string>("Weighted BF Name", "wBF");
    p->set<string>("Weighted Gradient BF Name", "wGrad BF");
    p->set<string>("QP Variable Name", "qFluct");
    p->set<string>("QP Time Derivative Variable Name", "qFluct_dot");
    p->set<string>("Gradient QP Variable Name", "qFluct Gradient");
    p->set<string>("Body Force Name", "Body Force");   
 
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_vecgradient);
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", dl->node_qp_scalar);
    p->set< RCP<DataLayout> >("Node QP Gradient Data Layout", dl->node_qp_gradient);

    p->set<string>("Viscosity Mu QP Variable Name", "Viscosity Mu");
    p->set<string>("Viscosity Lambda QP Variable Name", "Viscosity Lambda");
    p->set<string>("Viscosity Kappa QP Variable Name", "Viscosity Kappa");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    
    p->set<string>("Stress Tau11 QP Variable Name", "Stress Tau11");
    p->set<string>("Stress Tau12 QP Variable Name", "Stress Tau12");
    p->set<string>("Stress Tau13 QP Variable Name", "Stress Tau13");
    p->set<string>("Stress Tau22 QP Variable Name", "Stress Tau22");
    p->set<string>("Stress Tau23 QP Variable Name", "Stress Tau23");
    p->set<string>("Stress Tau33 QP Variable Name", "Stress Tau33");
    
    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Equation Set");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    //Output
    p->set<string>("Residual Name", "ComprNS Residual");
    p->set< RCP<DataLayout> >("Node Vector Data Layout", dl->node_vector);

    ev = rcp(new PHAL::ComprNSResid<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }
  { // Viscosity
    RCP<ParameterList> p = rcp(new ParameterList("Viscosity"));

    //Input
    p->set<string>("Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);
    p->set< RCP<DataLayout> >("QP Gradient Data Layout", dl->qp_gradient); 
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_vecgradient);
    p->set<string>("QP Variable Name", "qFluct");
    p->set<string>("Gradient QP Variable Name", "qFluct Gradient");
    
    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Viscosity");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);
  
    //Output
    p->set<string>("Viscosity Mu QP Variable Name", "Viscosity Mu");
    p->set<string>("Viscosity Lambda QP Variable Name", "Viscosity Lambda");
    p->set<string>("Viscosity Kappa QP Variable Name", "Viscosity Kappa");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set<string>("Stress Tau11 QP Variable Name", "Stress Tau11");
    p->set<string>("Stress Tau12 QP Variable Name", "Stress Tau12");
    p->set<string>("Stress Tau13 QP Variable Name", "Stress Tau13");
    p->set<string>("Stress Tau22 QP Variable Name", "Stress Tau22");
    p->set<string>("Stress Tau23 QP Variable Name", "Stress Tau23");
    p->set<string>("Stress Tau33 QP Variable Name", "Stress Tau33");

    ev = rcp(new PHAL::ComprNSViscosity<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
    
  }

   { // ComprNS Body Force
    RCP<ParameterList> p = rcp(new ParameterList("Body Force"));

   //Input
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);
    p->set< RCP<DataLayout> >("QP Gradient Data Layout", dl->qp_gradient);
    p->set<string>("Coordinate Vector Name", "Coord Vec");

    Teuchos::ParameterList& paramList = params->sublist("Body Force");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

   //Output
     p->set<string>("Body Force Name", "Body Force");

    ev = rcp(new PHAL::ComprNSBodyForce<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }


  if (fieldManagerChoice == Albany::BUILD_RESID_FM)  {
    PHX::Tag<typename EvalT::ScalarT> res_tag("Scatter ComprNS", dl->dummy);
    fm0.requireField<EvalT>(res_tag);
  }
  else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM) {
    Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
    return respUtils.constructResponses(fm0, *responseList, Teuchos::null, stateMgr);
  }

  return Teuchos::null;
}
#endif // ALBANY_COMPRNSPROBLEM_HPP
