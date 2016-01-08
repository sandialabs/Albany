//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef GRADIENTDAMAGEPROBLEM_HPP
#define GRADIENTDAMAGEPROBLEM_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_AbstractProblem.hpp"

#include "Phalanx.hpp"
#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"
#include "PHAL_AlbanyTraits.hpp"

namespace Albany {

  class GradientDamageProblem : public Albany::AbstractProblem {
  public:
  
    //! Default constructor
    GradientDamageProblem( const Teuchos::RCP<Teuchos::ParameterList>& params,
                           const Teuchos::RCP<ParamLib>& paramLib,
                           const int numEq);

    //! Destructor
    virtual ~GradientDamageProblem();

    //! Return number of spatial dimensions
    virtual int spatialDimension() const { return numDim; }

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

    void getAllocatedStates(
         Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid2::FieldContainer<RealType>>>> oldState_,
         Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid2::FieldContainer<RealType>>>> newState_
         ) const;

  private:

    //! Private to prohibit copying
    GradientDamageProblem(const GradientDamageProblem&);
    
    //! Private to prohibit copying
    GradientDamageProblem& operator=(const GradientDamageProblem&);

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

    // counting helpers
    int numQPts;
    int numNodes;
    int numVertices;

    int D_offset;  //Position of T unknown in nodal DOFs
    int X_offset;  //Position of X unknown in nodal DOFs, followed by Y,Z
    int numDim;    //Number of spatial dimensions and displacement variable 

    // string to store material model name
    std::string matModel;

    // state containers
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid2::FieldContainer<RealType>>>> oldState;
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid2::FieldContainer<RealType>>>> newState;
  };

}

#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_ResponseUtilities.hpp"
#include "Albany_EvaluatorUtils.hpp"

#include "BulkModulus.hpp"
#include "ShearModulus.hpp"
#include "PHAL_Source.hpp"
#include "DefGrad.hpp"
#include "HardeningModulus.hpp"
#include "YieldStrength.hpp"
#include "SaturationModulus.hpp"
#include "SaturationExponent.hpp"
#include "PHAL_SaveStateField.hpp"
#include "TLElasResid.hpp"
#include "J2Damage.hpp"
#include "DamageLS.hpp"
#include "DamageSource.hpp"
#include "DamageResid.hpp"

template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
Albany::GradientDamageProblem::constructEvaluators(
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
   using std::map;
   using PHAL::AlbanyTraits;

   // get the name of the current element block
   std::string elementBlockName = meshSpecs.ebName;

   RCP<shards::CellTopology> cellType = rcp(new shards::CellTopology (&meshSpecs.ctd));
   RCP<Intrepid2::Basis<RealType, Intrepid2::FieldContainer<RealType>>>
     intrepidBasis = Albany::getIntrepid2Basis(meshSpecs.ctd);

   numNodes = intrepidBasis->getCardinality();
   const int worksetSize = meshSpecs.worksetSize;

   Intrepid2::DefaultCubatureFactory<RealType> cubFactory;
   RCP <Intrepid2::Cubature<RealType>> cubature = cubFactory.create(*cellType, meshSpecs.cubatureDegree);

   const int numQPts = cubature->getNumPoints();
   const int numVertices = cellType->getNodeCount();

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
   std::string scatterName="Scatter Damage";

   // Displacement Variable
   Teuchos::ArrayRCP<std::string> dof_names(1);
   dof_names[0] = "Displacement";
   Teuchos::ArrayRCP<std::string> resid_names(1);
   resid_names[0] = "Mechanical Residual";

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructDOFVecInterpolationEvaluator(dof_names[0], X_offset));

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructDOFVecGradInterpolationEvaluator(dof_names[0], X_offset));

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructGatherSolutionEvaluator_noTransient(true, dof_names, X_offset)); 
   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructScatterResidualEvaluator(true, resid_names, X_offset));

   // Damage Variable
   Teuchos::ArrayRCP<std::string> ddof_names(1);
   ddof_names[0] = "Damage";
   Teuchos::ArrayRCP<std::string> ddof_names_dot(1);
   ddof_names_dot[0] = ddof_names[0]+"_dot";
   Teuchos::ArrayRCP<std::string> dresid_names(1);
   dresid_names[0] = ddof_names[0]+" Residual";

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructDOFInterpolationEvaluator(ddof_names[0], D_offset));

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructDOFInterpolationEvaluator(ddof_names_dot[0], D_offset));

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructDOFGradInterpolationEvaluator(ddof_names[0], D_offset));

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructGatherSolutionEvaluator(false, ddof_names, ddof_names_dot, D_offset));

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructScatterResidualEvaluator(false, dresid_names, D_offset, scatterName));

   // General FEM Stuff
   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructGatherCoordinateVectorEvaluator());

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructMapToPhysicalFrameEvaluator(cellType, cubature));

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructComputeBasisFunctionsEvaluator(cellType, intrepidBasis, cubature));

   // Temporary variable used numerous times below
   Teuchos::RCP<PHX::Evaluator<AlbanyTraits>> ev;

   { // Bulk Modulus
     RCP<ParameterList> p = rcp(new ParameterList);

     p->set<std::string>("QP Variable Name", "Bulk Modulus");
     p->set<std::string>("QP Coordinate Vector Name", "Coord Vec");
     p->set< RCP<DataLayout>>("Node Data Layout", dl->node_scalar);
     p->set< RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);
     p->set< RCP<DataLayout>>("QP Vector Data Layout", dl->qp_vector);

     p->set<RCP<ParamLib>>("Parameter Library", paramLib);
     Teuchos::ParameterList& paramList = params->sublist("Bulk Modulus");
     p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

     ev = rcp(new LCM::BulkModulus<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
   }

   { // Shear Modulus 
     RCP<ParameterList> p = rcp(new ParameterList);

     p->set<std::string>("QP Variable Name", "Shear Modulus");
     p->set<std::string>("QP Coordinate Vector Name", "Coord Vec");
     p->set< RCP<DataLayout>>("Node Data Layout", dl->node_scalar);
     p->set< RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);
     p->set< RCP<DataLayout>>("QP Vector Data Layout", dl->qp_vector);

     p->set<RCP<ParamLib>>("Parameter Library", paramLib);
     Teuchos::ParameterList& paramList = params->sublist("Shear Modulus");
     p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

     ev = rcp(new LCM::ShearModulus<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
   }

   if (haveSource) { // Source
     TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                                "Error!  Sources not implemented in Elasticity yet!");

     RCP<ParameterList> p = rcp(new ParameterList);

     p->set<std::string>("Source Name", "Source");
     p->set<std::string>("Variable Name", "Displacement");
     p->set< RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);

     p->set<RCP<ParamLib>>("Parameter Library", paramLib);
     Teuchos::ParameterList& paramList = params->sublist("Source Functions");
     p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

     ev = rcp(new PHAL::Source<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
   }

   { // Deformation Gradient
     RCP<ParameterList> p = rcp(new ParameterList("DefGrad"));

     //Inputs: flags, weights, GradU
     const bool avgJ = params->get("avgJ", false);
     p->set<bool>("avgJ Name", avgJ);
     const bool volavgJ = params->get("volavgJ", false);
     p->set<bool>("volavgJ Name", volavgJ);
     const bool weighted_Volume_Averaged_J = params->get("weighted_Volume_Averaged_J", false);
     p->set<bool>("weighted_Volume_Averaged_J Name", weighted_Volume_Averaged_J);
     p->set<std::string>("Weights Name","Weights");
     p->set<std::string>("Gradient QP Variable Name", "Displacement Gradient");
     p->set< RCP<DataLayout>>("QP Tensor Data Layout", dl->qp_tensor);

     //Outputs: F, J
     p->set<std::string>("DefGrad Name", "Deformation Gradient"); //dl->qp_tensor also
     p->set<std::string>("DetDefGrad Name", "Determinant of Deformation Gradient"); 
     p->set< RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);

     ev = rcp(new LCM::DefGrad<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
   }
   { // Hardening Modulus
     RCP<ParameterList> p = rcp(new ParameterList);

     p->set<std::string>("QP Variable Name", "Hardening Modulus");
     p->set<std::string>("QP Coordinate Vector Name", "Coord Vec");
     p->set< RCP<DataLayout>>("Node Data Layout", dl->node_scalar);
     p->set< RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);
     p->set< RCP<DataLayout>>("QP Vector Data Layout", dl->qp_vector);

     p->set<RCP<ParamLib>>("Parameter Library", paramLib);
     Teuchos::ParameterList& paramList = params->sublist("Hardening Modulus");
     p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

     ev = rcp(new LCM::HardeningModulus<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
   }

   { // Yield Strength
     RCP<ParameterList> p = rcp(new ParameterList);

     p->set<std::string>("QP Variable Name", "Yield Strength");
     p->set<std::string>("QP Coordinate Vector Name", "Coord Vec");
     p->set< RCP<DataLayout>>("Node Data Layout", dl->node_scalar);
     p->set< RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);
     p->set< RCP<DataLayout>>("QP Vector Data Layout", dl->qp_vector);

     p->set<RCP<ParamLib>>("Parameter Library", paramLib);
     Teuchos::ParameterList& paramList = params->sublist("Yield Strength");
     p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

     ev = rcp(new LCM::YieldStrength<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
   }

   { // Saturation Modulus
     RCP<ParameterList> p = rcp(new ParameterList);

     p->set<std::string>("Saturation Modulus Name", "Saturation Modulus");
     p->set<std::string>("QP Coordinate Vector Name", "Coord Vec");
     p->set< RCP<DataLayout>>("Node Data Layout", dl->node_scalar);
     p->set< RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);
     p->set< RCP<DataLayout>>("QP Vector Data Layout", dl->qp_vector);

     p->set<RCP<ParamLib>>("Parameter Library", paramLib);
     Teuchos::ParameterList& paramList = params->sublist("Saturation Modulus");
     p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

     ev = rcp(new LCM::SaturationModulus<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
   }

   { // Saturation Exponent
     RCP<ParameterList> p = rcp(new ParameterList);

     p->set<std::string>("Saturation Exponent Name", "Saturation Exponent");
     p->set<std::string>("QP Coordinate Vector Name", "Coord Vec");
     p->set< RCP<DataLayout>>("Node Data Layout", dl->node_scalar);
     p->set< RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);
     p->set< RCP<DataLayout>>("QP Vector Data Layout", dl->qp_vector);

     p->set<RCP<ParamLib>>("Parameter Library", paramLib);
     Teuchos::ParameterList& paramList = params->sublist("Saturation Exponent");
     p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

     ev = rcp(new LCM::SaturationExponent<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
   }

   {// Stress
     RCP<ParameterList> p = rcp(new ParameterList("Stress"));

     //Input
     p->set<std::string>("DefGrad Name", "Deformation Gradient");
     p->set< RCP<DataLayout>>("QP Tensor Data Layout", dl->qp_tensor);

     p->set<std::string>("Bulk Modulus Name", "Bulk Modulus");
     p->set< RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);

     p->set<std::string>("Shear Modulus Name", "Shear Modulus");  // dl->qp_scalar also
     p->set<std::string>("Hardening Modulus Name", "Hardening Modulus"); // dl->qp_scalar also
     p->set<std::string>("Yield Strength Name", "Yield Strength"); // dl->qp_scalar also
     p->set<std::string>("Saturation Modulus Name", "Saturation Modulus"); // dl->qp_scalar also
     p->set<std::string>("Saturation Exponent Name", "Saturation Exponent"); // dl->qp_scalar also
     p->set<std::string>("DetDefGrad Name", "Determinant of Deformation Gradient");  // dl->qp_scalar also
     p->set<std::string>("Damage Name", "Damage");

     //Output
     p->set<std::string>("Stress Name", "Stress"); //dl->qp_tensor also
     p->set<std::string>("DP Name", "DP"); // dl->qp_scalar also
     p->set<std::string>("Effective Stress Name", "Effective Stress"); // dl->qp_scalar also
     p->set<std::string>("Energy Name", "Energy"); // dl->qp_scalar also

     p->set<std::string>("Fp Name", "Fp");  // dl->qp_tensor also
     p->set<std::string>("Eqps Name", "eqps");  // dl->qp_scalar also

 
     //Declare what state data will need to be saved (name, layout, init_type)
     // A :true: as 5th argument declares that the previous state needs to be saved

     ev = rcp(new LCM::J2Damage<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
     p = stateMgr.registerStateVariable("Stress",dl->qp_tensor, dl->dummy, elementBlockName, "scalar", 0.0);
     ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
     p = stateMgr.registerStateVariable("Fp",dl->qp_tensor, dl->dummy, elementBlockName, "identity",1.0,true);
     ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
     p = stateMgr.registerStateVariable("eqps",dl->qp_scalar, dl->dummy, elementBlockName, "scalar", 0.0,true);
     ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
   }

   { // Displacement Resid
     RCP<ParameterList> p = rcp(new ParameterList("Mechanical Residual"));

     //Input
     p->set<std::string>("Stress Name", "Stress");
     p->set< RCP<DataLayout>>("QP Tensor Data Layout", dl->qp_tensor);

     p->set<std::string>("DefGrad Name", "Deformation Gradient"); //dl->qp_tensor also

     p->set<std::string>("DetDefGrad Name", "Determinant of Deformation Gradient");
     p->set< RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);

     p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
     p->set< RCP<DataLayout>>("Node QP Vector Data Layout", dl->node_qp_vector);

     p->set<std::string>("Weighted BF Name", "wBF");
     p->set< RCP<DataLayout>>("Node QP Scalar Data Layout", dl->node_qp_scalar);
     p->set<RCP<ParamLib>>("Parameter Library", paramLib);

     //Output
     p->set<std::string>("Residual Name", "Mechanical Residual");
     p->set< RCP<DataLayout>>("Node Vector Data Layout", dl->node_vector);

     ev = rcp(new LCM::TLElasResid<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
   }

   { // Damage length scale
     RCP<ParameterList> p = rcp(new ParameterList);

     p->set<std::string>("QP Variable Name", "Damage Length Scale");
     p->set<std::string>("QP Coordinate Vector Name", "Coord Vec");
     p->set< RCP<DataLayout>>("Node Data Layout", dl->node_scalar);
     p->set< RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);
     p->set< RCP<DataLayout>>("QP Vector Data Layout", dl->qp_vector);

     p->set<RCP<ParamLib>>("Parameter Library", paramLib);
     Teuchos::ParameterList& paramList = params->sublist("Damage Length Scale");
     p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

     ev = rcp(new LCM::DamageLS<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
   }

   { // Damage Source
     RCP<ParameterList> p = rcp(new ParameterList("Damage Source"));

     //Input
     RealType gc = params->get("gc", 1.0);
     p->set<RealType>("gc Name", gc);
     p->set<std::string>("Bulk Modulus Name", "Bulk Modulus");
     p->set< RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);
     p->set<std::string>("Damage Name", "Damage");
     p->set<std::string>("DP Name", "DP");
     p->set<std::string>("Effective Stress Name", "Effective Stress");
     p->set<std::string>("Energy Name", "Energy");
     p->set<std::string>("DetDefGrad Name", "Determinant of Deformation Gradient");
     p->set<std::string>("Damage Length Scale Name", "Damage Length Scale");

     //Output
     p->set<std::string>("Damage Source Name", "Damage Source");

     ev = rcp(new LCM::DamageSource<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
     p = stateMgr.registerStateVariable("Damage Source",dl->qp_scalar, dl->dummy, elementBlockName, "scalar", 0.0,true);
     ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
     p = stateMgr.registerStateVariable("Damage",dl->qp_scalar, dl->dummy, elementBlockName, "scalar", 0.0,true);
     ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
   }

   { // Damage Resid
     RCP<ParameterList> p = rcp(new ParameterList("Damage Resid"));

     //Input
     RealType gc = params->get("gc", 0.0);
     p->set<RealType>("gc Name", gc);
     p->set<std::string>("Weighted BF Name", "wBF");
     p->set< RCP<DataLayout>>("Node QP Scalar Data Layout", dl->node_qp_scalar);
     p->set<std::string>("QP Variable Name", "Damage");

     p->set<std::string>("QP Time Derivative Variable Name", "Damage_dot");

     p->set<std::string>("Damage Source Name", "Damage Source");  //dl->qp_scalar

     p->set<std::string>("Damage Length Scale Name", "Damage Length Scale");
     p->set< RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);

     p->set<std::string>("Gradient QP Variable Name", "Damage Gradient");
     p->set< RCP<DataLayout>>("QP Vector Data Layout", dl->qp_vector);

     p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
     p->set< RCP<DataLayout>>("Node QP Vector Data Layout", dl->node_qp_vector);

     //Output
     p->set<std::string>("Residual Name", "Damage Residual");
     p->set< RCP<DataLayout>>("Node Scalar Data Layout", dl->node_scalar);

     ev = rcp(new LCM::DamageResid<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
   }

   if (fieldManagerChoice == Albany::BUILD_RESID_FM)  {
     PHX::Tag<typename EvalT::ScalarT> res_tag("Scatter", dl->dummy);
     fm0.requireField<EvalT>(res_tag);

     PHX::Tag<typename EvalT::ScalarT> res_tag2(scatterName, dl->dummy);
     fm0.requireField<EvalT>(res_tag2);

     return res_tag.clone();
   }
   else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM) {
     Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
     return respUtils.constructResponses(fm0, *responseList, stateMgr);
   }

   return Teuchos::null;
}

#endif // ALBANY_GRADIENTDAMAGEPROBLEM_HPP
