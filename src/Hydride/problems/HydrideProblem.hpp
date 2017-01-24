//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef HYDRIDEPROBLEM_HPP
#define HYDRIDEPROBLEM_HPP

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
  class HydrideProblem : public AbstractProblem {
  public:
  
    //! Default constructor
    HydrideProblem(const Teuchos::RCP<Teuchos::ParameterList>& params,
		const Teuchos::RCP<ParamLib>& paramLib,
		const int numDim_,
                Teuchos::RCP<const Teuchos::Comm<int> >& commT_);

    //! Destructor
    ~HydrideProblem();

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
    HydrideProblem(const HydrideProblem&);
    
    //! Private to prohibit copying
    HydrideProblem& operator=(const HydrideProblem&);

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

    bool haveNoise; // Langevin noise present

    Teuchos::RCP<const Teuchos::Comm<int> > commT;

    Teuchos::RCP<Albany::Layouts> dl;

  };

}

#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_EvaluatorUtils.hpp"
#include "Albany_ResponseUtilities.hpp"

#include "HydrideChemTerm.hpp"
#include "HydrideStressTerm.hpp"
#include "HydrideStress.hpp"
#include "PHAL_LangevinNoiseTerm.hpp"
#include "HydrideCResid.hpp"
#include "HydrideWResid.hpp"

#include "PHAL_SaveStateField.hpp"
#include "Strain.hpp"
#include "ElasticityResid.hpp"

#include "ElasticModulus.hpp"
#include "PoissonsRatio.hpp"
#include "Stress.hpp"


template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
Albany::HydrideProblem::constructEvaluators(
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

   // get the name of the current element block
   std::string elementBlockName = meshSpecs.ebName;

   const CellTopologyData * const elem_top = &meshSpecs.ctd;

   RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> >
     intrepidBasis = Albany::getIntrepid2Basis(*elem_top);
   RCP<shards::CellTopology> cellType = rcp(new shards::CellTopology (elem_top));


   const int numNodes = intrepidBasis->getCardinality();
   const int worksetSize = meshSpecs.worksetSize;

   Intrepid2::DefaultCubatureFactory cubFactory;
   RCP <Intrepid2::Cubature<PHX::Device> > cellCubature = cubFactory.create<PHX::Device, RealType, RealType>(*cellType, meshSpecs.cubatureDegree);

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

  int offset=0;

// The displacement equations

  {
     Teuchos::ArrayRCP<std::string> dof_names(1);
     Teuchos::ArrayRCP<std::string> dof_names_dot(1);
     Teuchos::ArrayRCP<std::string> resid_names(1);
     dof_names[0] = "Displacement";
     dof_names_dot[0] = dof_names[0]+"DotDot";
     resid_names[0] = "Displacement Residual";

     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructGatherSolutionEvaluator(true, dof_names, dof_names_dot, offset));

     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructDOFVecInterpolationEvaluator(dof_names[0], offset));

     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructDOFVecInterpolationEvaluator(dof_names_dot[0], offset));

     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructDOFVecGradInterpolationEvaluator(dof_names[0], offset));

     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructScatterResidualEvaluator(true, resid_names, offset, "Scatter Displacement"));

     offset += numDim;

   }

// Equations for c and w

   {
     int nscalars = 2;
     Teuchos::ArrayRCP<std::string> dof_names(nscalars);
       dof_names[0] = "C"; // The concentration difference variable 0 \leq C \leq 1
       dof_names[1] = "W"; // The chemical potential difference variable
     Teuchos::ArrayRCP<std::string> dof_names_dot(nscalars);
       dof_names_dot[0] = dof_names[0]+"Dot";
       dof_names_dot[1] = dof_names[1]+"Dot"; // not currently used
     Teuchos::ArrayRCP<std::string> resid_names(nscalars);
       resid_names[0] = "C Residual";
       resid_names[1] = "W Residual";
  
    fm0.template registerEvaluator<EvalT>
       (evalUtils.constructGatherSolutionEvaluator(false, dof_names, dof_names_dot, offset));
  
    fm0.template registerEvaluator<EvalT>
       (evalUtils.constructScatterResidualEvaluator(false, resid_names, offset, "Scatter c and w"));
  
    for (unsigned int i=0; i<nscalars; i++) {
      fm0.template registerEvaluator<EvalT>
        (evalUtils.constructDOFInterpolationEvaluator(dof_names[i], offset));
  
      fm0.template registerEvaluator<EvalT>
        (evalUtils.constructDOFInterpolationEvaluator(dof_names_dot[i], offset));
  
      fm0.template registerEvaluator<EvalT>
        (evalUtils.constructDOFGradInterpolationEvaluator(dof_names[i], offset));
    }

    offset += nscalars;

  }

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructGatherCoordinateVectorEvaluator());

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructMapToPhysicalFrameEvaluator( cellType, cellCubature));

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructComputeBasisFunctionsEvaluator(cellType, intrepidBasis, cellCubature));


  { // Form the Chemical Energy term in Eq. 2.2

    RCP<ParameterList> p = rcp(new ParameterList("Chem Energy Term"));

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);

    // b value in Equation 1.1
    p->set<double>("b Value", params->get<double>("b"));

    //Input
    p->set<std::string>("C QP Variable Name", "C");
    p->set<std::string>("W QP Variable Name", "W");

    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    //Output
    p->set<std::string>("Chemical Energy Term", "Chemical Energy Term");

    ev = rcp(new HYD::HydrideChemTerm<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  { // Form the Stress term in Eq. 2.2

    RCP<ParameterList> p = rcp(new ParameterList("Stress Term"));

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);

    // b value in Equation 1.1
    p->set<double>("e Value", params->get<double>("e"));

    //Input
    p->set<std::string>("Stress Name", "Stress"); 
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);

    //Output
    p->set<std::string>("Stress Term", "Stress Term");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

    ev = rcp(new HYD::HydrideStressTerm<EvalT,AlbanyTraits>(*p));
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
    p->set<std::string>("C QP Variable Name", "C");

    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    //Output
    p->set<std::string>("Langevin Noise Term", "Langevin Noise Term");

    ev = rcp(new PHAL::LangevinNoiseTerm<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  { // Strain
    RCP<ParameterList> p = rcp(new ParameterList("Strain"));

    //Input
    p->set<std::string>("Gradient QP Variable Name", "Displacement Gradient");
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);

    //Output
    p->set<std::string>("Strain Name", "Strain"); //dl->qp_tensor also

    ev = rcp(new LCM::Strain<EvalT,AlbanyTraits>(*p, dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }


#if 0
	{ // Hydride stress
      RCP<ParameterList> p = rcp(new ParameterList("HydrideStress"));

      //Input
      p->set<std::string>("Strain Name", "Strain");
      p->set<std::string>("C QP Variable Name", "C");
      p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
      p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);

      // e value in Equation between 1.2 and 1.3
      p->set<double>("e Value", params->get<double>("e"));

      //Output
      p->set<std::string>("Stress Name", "Stress"); //dl->qp_tensor also

      ev = rcp(new HYD::HydrideStress<EvalT,AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
      p = stateMgr.registerStateVariable("Stress",dl->qp_tensor, dl->dummy, elementBlockName, "scalar", 0.0);
      ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);

	}
#else
  { // Elastic Modulus
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<std::string>("QP Variable Name", "Elastic Modulus");
    p->set<std::string>("QP Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Node Data Layout", dl->node_scalar);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Elastic Modulus");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = rcp(new LCM::ElasticModulus<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  { // Poissons Ratio 
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<std::string>("QP Variable Name", "Poissons Ratio");
    p->set<std::string>("QP Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Node Data Layout", dl->node_scalar);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Poissons Ratio");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = rcp(new LCM::PoissonsRatio<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }
	{ // Linear elasticity stress
      RCP<ParameterList> p = rcp(new ParameterList("Stress"));

      //Input
      p->set<std::string>("Strain Name", "Strain");
      p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);

      p->set<std::string>("Elastic Modulus Name", "Elastic Modulus");
      p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

      p->set<std::string>("Poissons Ratio Name", "Poissons Ratio");  // dl->qp_scalar also

      //Output
      p->set<std::string>("Stress Name", "Stress"); //dl->qp_tensor also

      ev = rcp(new LCM::Stress<EvalT,AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
      p = stateMgr.registerStateVariable("Stress",dl->qp_tensor, dl->dummy, elementBlockName, "scalar", 0.0);
      ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
	}
#endif


  { // C Resid
    RCP<ParameterList> p = rcp(new ParameterList("C Resid"));

    //Input
    p->set<std::string>("Weighted BF Name", "wBF");
    p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
    if(haveNoise)
      p->set<std::string>("Langevin Noise Term", "Langevin Noise Term");
    // Accumulate in the Langevin noise term?
    p->set<bool>("Have Noise", haveNoise);

    p->set<std::string>("Chemical Energy Term", "Chemical Energy Term");
    p->set<std::string>("Gradient QP Variable Name", "C Gradient");
    p->set<std::string>("Stress Term", "Stress Term");

    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", dl->node_qp_scalar);
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);

    // gamma value in Equation 2.2
    p->set<double>("gamma Value", params->get<double>("gamma"));

    //Output
    p->set<std::string>("Residual Name", "C Residual");
    p->set< RCP<DataLayout> >("Node Scalar Data Layout", dl->node_scalar);

    ev = rcp(new HYD::HydrideCResid<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  { // W Resid
    RCP<ParameterList> p = rcp(new ParameterList("W Resid"));

    //Input
    p->set<std::string>("Weighted BF Name", "wBF");
    p->set<std::string>("BF Name", "BF");
    p->set<std::string>("C QP Time Derivative Variable Name", "CDot");
    p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
    p->set<std::string>("Gradient QP Variable Name", "W Gradient");

    // Mass lump time term?
    p->set<bool>("Lump Mass", params->get<bool>("Lump Mass"));

    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", dl->node_qp_scalar);
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);

    //Output
    p->set<std::string>("Residual Name", "W Residual");
    p->set< RCP<DataLayout> >("Node Scalar Data Layout", dl->node_scalar);

    ev = rcp(new HYD::HydrideWResid<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  { // Displacement Resid
    RCP<ParameterList> p = rcp(new ParameterList("Displacement Resid"));

    //Input
    p->set<std::string>("Stress Name", "Stress");
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);

    p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);
    p->set<bool>("Disable Transient", true);

    //Output
    p->set<std::string>("Residual Name", "Displacement Residual");
    p->set< RCP<DataLayout> >("Node Vector Data Layout", dl->node_vector);

    ev = rcp(new LCM::ElasticityResid<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (fieldManagerChoice == Albany::BUILD_RESID_FM)  {
    PHX::Tag<typename EvalT::ScalarT> disp_tag("Scatter Displacement", dl->dummy);
    fm0.requireField<EvalT>(disp_tag);
    PHX::Tag<typename EvalT::ScalarT> c_w_tag("Scatter c and w", dl->dummy);
    fm0.requireField<EvalT>(c_w_tag);
    return disp_tag.clone();
  }

  else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM) {
    Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
    return respUtils.constructResponses(fm0, *responseList, Teuchos::null, stateMgr);
  }

  return Teuchos::null;
}


#endif // ALBANY_CAHNHILLPROBLEM_HPP
