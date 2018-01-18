//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LAMEPROBLEM_HPP
#define LAMEPROBLEM_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_AbstractProblem.hpp"

#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"
#include "PHAL_AlbanyTraits.hpp"


namespace Albany {

  /*!
   * \brief Abstract interface for representing a 2-D finite element
   * problem.
   */
  class LameProblem : public Albany::AbstractProblem {
  public:

    //! Default constructor
    LameProblem(const Teuchos::RCP<Teuchos::ParameterList>& params,
                const Teuchos::RCP<ParamLib>& paramLib,
                const int numEqm,
                Teuchos::RCP<const Teuchos::Comm<int>>& commT);

    //! Destructor
    virtual ~LameProblem();

    //! Return number of spatial dimensions
    virtual int spatialDimension() const { return numDim; }

    //! Get boolean telling code if SDBCs are utilized
    virtual bool useSDBCs() const {return use_sdbcs_; }

    //! Build the PDE instantiations, boundary conditions, and initial solution
    virtual void buildProblem(
      Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct>>  meshSpecs,
      StateManager& stateMgr);

    // Build evaluators
    virtual Teuchos::Array< Teuchos::RCP<const PHX::FieldTag>>
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
    LameProblem(const LameProblem&);

    //! Private to prohibit copying
    LameProblem& operator=(const LameProblem&);

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

  protected:

    //! Boundary conditions on source term
    bool haveSource;
    int numDim;
    bool haveMatDB;
    std::string mtrlDbFilename;
    Teuchos::RCP<Albany::MaterialDatabase> materialDB;

    /// Boolean marking whether SDBCs are used
    bool use_sdbcs_;

  };

}

#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_ResponseUtilities.hpp"
#include "Albany_EvaluatorUtils.hpp"

#include "PHAL_Source.hpp"
#include "Strain.hpp"
#include "DefGrad.hpp"
#ifdef ALBANY_LAME
#include "lame/LameStress.hpp"
#endif
#ifdef ALBANY_LAMENT
#include "lame/LamentStress.hpp"
#endif
#include "lame/LameUtils.hpp"
#include "PHAL_SaveStateField.hpp"
#include "ElasticityResid.hpp"
#include "TLElasResid.hpp"

template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
Albany::LameProblem::constructEvaluators(
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
  string elementBlockName = meshSpecs.ebName;

   RCP<shards::CellTopology> cellType = rcp(new shards::CellTopology (&meshSpecs.ctd));
   RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>>
     intrepidBasis = Albany::getIntrepid2Basis(meshSpecs.ctd);

   const int numNodes = intrepidBasis->getCardinality();
   const int worksetSize = meshSpecs.worksetSize;

   Intrepid2::DefaultCubatureFactory cubFactory;
   RCP <Intrepid2::Cubature<PHX::Device>>> cubature = cubFactory.create<PHX::Device, RealType, RealType>(*cellType, meshSpecs.cubatureDegree);

   numDim = cubature->getDimension();
   const int numQPts = cubature->getNumPoints();
   const int numVertices = cellType->getNodeCount();
   //   const int numVertices = cellType->getVertexCount();

   *out << "Field Dimensions: Workset=" << worksetSize
        << ", Vertices= " << numVertices
        << ", Nodes= " << numNodes
        << ", QuadPts= " << numQPts
        << ", Dim= " << numDim << std::endl;

   // Construct standard FEM evaluators with standard field names
   RCP<Albany::Layouts> dl = rcp(new Albany::Layouts(worksetSize,numVertices,numNodes,numQPts,numDim));
   TEUCHOS_TEST_FOR_EXCEPTION(dl->vectorAndGradientLayoutsAreEquivalent==false, std::logic_error,
                              "Data Layout Usage in Mechanics problems assume vecDim = numDim");
   Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
   bool supportsTransient=true;

   // Define Field Names
   Teuchos::ArrayRCP<string> dof_names(1);
     dof_names[0] = "Displacement";
   Teuchos::ArrayRCP<string> dof_names_dotdot(1);
   if (supportsTransient)
     dof_names_dotdot[0] = dof_names[0]+"_dotdot";
   Teuchos::ArrayRCP<string> resid_names(1);
     resid_names[0] = dof_names[0]+" Residual";

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructDOFVecInterpolationEvaluator(dof_names[0]));

   if (supportsTransient) fm0.template registerEvaluator<EvalT>
       (evalUtils.constructDOFVecInterpolationEvaluator(dof_names_dotdot[0]));

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructDOFVecGradInterpolationEvaluator(dof_names[0]));

   if (supportsTransient) fm0.template registerEvaluator<EvalT>
       (evalUtils.constructGatherSolutionEvaluator(true, dof_names, dof_names_dotdot));
   else fm0.template registerEvaluator<EvalT>
       (evalUtils.constructGatherSolutionEvaluator_noTransient(true, dof_names));

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructScatterResidualEvaluator(true, resid_names));

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructGatherCoordinateVectorEvaluator());

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructMapToPhysicalFrameEvaluator(cellType, cubature));

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructComputeBasisFunctionsEvaluator(cellType, intrepidBasis, cubature));

  // Temporary variable used numerous times below
  Teuchos::RCP<PHX::Evaluator<AlbanyTraits>> ev;

  if (haveSource) { // Source
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                       "Error!  Sources not implemented in Elasticity yet!");

    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<string>("Source Name", "Source");
    p->set<string>("Variable Name", "Displacement");
    p->set< RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);

    p->set<RCP<ParamLib>>("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Source Functions");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = rcp(new PHAL::Source<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  { // Strain
    RCP<ParameterList> p = rcp(new ParameterList("Strain"));

    //Input
    p->set<string>("Gradient QP Variable Name", "Displacement Gradient");

    //Output
    p->set<string>("Strain Name", "Strain"); //dl->qp_tensor also

    ev = rcp(new LCM::Strain<EvalT,AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  { // Deformation Gradient
    RCP<ParameterList> p = rcp(new ParameterList("DefGrad"));

    //Input
    // If true, compute determinate of deformation gradient at all integration points, then replace all of them with the simple average for the element.  This give a constant volumetric response.
    const bool avgJ = params->get("avgJ", false);
    p->set<bool>("avgJ Name", avgJ);
    // If true, compute determinate of deformation gradient at all integration points, then replace all of them with the volume average for the element (integrate J over volume of element, divide by total volume).  This give a constant volumetric response.
    const bool volavgJ = params->get("volavgJ", false);
    p->set<bool>("volavgJ Name", volavgJ);
    const bool weighted_Volume_Averaged_J = params->get("weighted_Volume_Averaged_J", false);
    p->set<bool>("weighted_Volume_Averaged_J Name", weighted_Volume_Averaged_J);
    // Integration weights for each quadrature point
    p->set<string>("Weights Name","Weights");
    p->set< RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);
    p->set<string>("Gradient QP Variable Name", "Displacement Gradient");
    p->set< RCP<DataLayout>>("QP Tensor Data Layout", dl->qp_tensor);

    //Output
    p->set<string>("DefGrad Name", "Deformation Gradient"); //dl->qp_tensor also
    p->set<string>("DetDefGrad Name", "Determinant of Deformation Gradient");
    p->set< RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);

    ev = rcp(new LCM::DefGrad<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  { // LameStress
    RCP<ParameterList> p = rcp(new ParameterList("Stress"));

    // Material properties that will be passed to LAME material model
    string lameMaterialModel = params->get("Lame Material Model","Elastic");
    p->set<string>("Lame Material Model", lameMaterialModel);

    // Info to get material data from materials xml database file
    p->set<bool>("Have MatDB", haveMatDB);
        p->set<string>("Element Block Name", meshSpecs.ebName);

    if(haveMatDB)
      p->set< RCP<Albany::MaterialDatabase>>("MaterialDB", materialDB);

    // Materials specification
    Teuchos::ParameterList& lameMaterialParametersList = p->sublist("Lame Material Parameters");
    lameMaterialParametersList = params->sublist("Lame Material Parameters");

    // Input
    p->set<string>("Strain Name", "Strain");
    p->set< RCP<DataLayout>>("QP Tensor Data Layout", dl->qp_tensor);

    p->set<string>("DefGrad Name", "Deformation Gradient"); // dl->qp_tensor also

    // Output
    p->set<string>("Stress Name", "Stress"); // dl->qp_tensor also

    // A LAME material model may register additional state variables (type is always double)
    p->set< RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);

#ifdef ALBANY_LAME
    ev = rcp(new LCM::LameStress<EvalT,AlbanyTraits>(*p));
#endif

#ifdef ALBANY_LAMENT
    ev = rcp(new LCM::LamentStress<EvalT,AlbanyTraits>(*p));
#endif

    fm0.template registerEvaluator<EvalT>(ev);

    // Declare state data that need to be saved
    // (register with state manager and create corresponding evaluator)
    RCP<ParameterList> p2;
    p2 = stateMgr.registerStateVariable("Stress",dl->qp_tensor, dl->dummy, elementBlockName, "scalar", 0.0, true);
    ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p2));
    fm0.template registerEvaluator<EvalT>(ev);

    p2 = stateMgr.registerStateVariable("Deformation Gradient",dl->qp_tensor, dl->dummy, elementBlockName, "identity", 1.0, true);
    ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p2));
    fm0.template registerEvaluator<EvalT>(ev);

    std::vector<std::string> lameMaterialModelStateVariableNames
       = LameUtils::getStateVariableNames(lameMaterialModel, lameMaterialParametersList);
    std::vector<double> lameMaterialModelStateVariableInitialValues
       = LameUtils::getStateVariableInitialValues(lameMaterialModel, lameMaterialParametersList);
    for(unsigned int i=0 ; i<lameMaterialModelStateVariableNames.size() ; ++i){
      p2 = stateMgr.registerStateVariable(lameMaterialModelStateVariableNames[i],
                                          dl->qp_scalar,
                                          dl->dummy,
                                          elementBlockName,
                                          doubleToInitString(lameMaterialModelStateVariableInitialValues[i]),true);
      ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p2));
      fm0.template registerEvaluator<EvalT>(ev);
    }
  }

  { // Displacement Resid
    RCP<ParameterList> p = rcp(new ParameterList("Displacement Resid"));

    //Input
    p->set<string>("Stress Name", "Stress");
    p->set< RCP<DataLayout>>("QP Tensor Data Layout", dl->qp_tensor);

    // \todo Is the required?
    p->set<string>("DefGrad Name", "Deformation Gradient"); //dl->qp_tensor also
    p->set<string>("DetDefGrad Name", "Determinant of Deformation Gradient");
    p->set< RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);

    p->set<string>("Weighted Gradient BF Name", "wGrad BF");
    p->set<RCP<DataLayout>>("Node QP Vector Data Layout", dl->node_qp_vector);
    p->set<RCP<ParamLib>>("Parameter Library", paramLib);

    // extra input for time dependent term
    p->set<string>("Weighted BF Name", "wBF");
    p->set< RCP<DataLayout>>("Node QP Scalar Data Layout", dl->node_qp_scalar);
    p->set<string>("Time Dependent Variable Name", "Displacement_dotdot");
    p->set< RCP<DataLayout>>("QP Vector Data Layout", dl->qp_vector);

    //Output
    p->set<string>("Residual Name", "Displacement Residual");
    p->set< RCP<DataLayout>>("Node Vector Data Layout", dl->node_vector);

    //ev = rcp(new LCM::ElasticityResid<EvalT,AlbanyTraits>(*p));
    ev = rcp(new LCM::TLElasResid<EvalT,AlbanyTraits>(*p));
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

#endif // LAMEPROBLEM_HPP
