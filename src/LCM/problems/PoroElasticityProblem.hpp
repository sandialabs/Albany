//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef POROELASTICITYPROBLEM_HPP
#define POROELASTICITYPROBLEM_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_AbstractProblem.hpp"

#include "Phalanx.hpp"
#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_ResponseUtilities.hpp"
#include "Albany_EvaluatorUtils.hpp"
#include "PHAL_AlbanyTraits.hpp"

namespace Albany {

  /*!
   * \brief Problem definition for Poro-Elasticity
   */
  class PoroElasticityProblem : public Albany::AbstractProblem {
  public:
  
    //! Default constructor
    PoroElasticityProblem(const Teuchos::RCP<Teuchos::ParameterList>& params,
			  const Teuchos::RCP<ParamLib>& paramLib,
			  const int numEq);

    //! Destructor
    virtual ~PoroElasticityProblem();

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
         Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Kokkos::DynRankView<RealType, PHX::Device>>>> oldState_,
	 Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Kokkos::DynRankView<RealType, PHX::Device>>>> newState_
	 ) const;

  private:

    //! Private to prohibit copying
    PoroElasticityProblem(const PoroElasticityProblem&);
    
    //! Private to prohibit copying
    PoroElasticityProblem& operator=(const PoroElasticityProblem&);

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
    int T_offset;  //Position of T unknown in nodal DOFs
    int X_offset;  //Position of X unknown in nodal DOFs, followed by Y,Z
    int numDim;    //Number of spatial dimensions and displacement variable 

    std::string matModel;

    Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Kokkos::DynRankView<RealType, PHX::Device>>>> oldState;
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Kokkos::DynRankView<RealType, PHX::Device>>>> newState;
  };
}

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_AbstractProblem.hpp"

#include "Phalanx.hpp"
#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_ResponseUtilities.hpp"
#include "Albany_EvaluatorUtils.hpp"

#include "Time.hpp"
#include "Strain.hpp"
#include "AssumedStrain.hpp"
#include "StabParameter.hpp"
#include "PHAL_SaveStateField.hpp"
#include "Porosity.hpp"
#include "BiotCoefficient.hpp"
#include "BiotModulus.hpp"
#include "PHAL_ThermalConductivity.hpp"
#include "KCPermeability.hpp"
#include "ElasticModulus.hpp"
#include "ShearModulus.hpp"
#include "PoissonsRatio.hpp"
#include "TotalStress.hpp"
#include "Stress.hpp"
#include "PoroElasticityResidMomentum.hpp"
#include "PHAL_Source.hpp"
#include "PoroElasticityResidMass.hpp"

#include "PHAL_NSMaterialProperty.hpp"

#include "CapExplicit.hpp"
#include "CapImplicit.hpp"


template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
Albany::PoroElasticityProblem::constructEvaluators(
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

   // get the name of the current element block
   std::string elementBlockName = meshSpecs.ebName;

   RCP<shards::CellTopology> cellType = rcp(new shards::CellTopology (&meshSpecs.ctd));
   RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> >
     intrepidBasis = Albany::getIntrepid2Basis(meshSpecs.ctd);

   const int numNodes = intrepidBasis->getCardinality();
   const int worksetSize = meshSpecs.worksetSize;

   Intrepid2::DefaultCubatureFactory cubFactory;
   RCP <Intrepid2::Cubature<PHX::Device> >> cubature = cubFactory.create<PHX::Device, RealType, RealType>(*cellType, meshSpecs.cubatureDegree);

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
   std::string scatterName="Scatter PoreFluid";


   // ----------------------setup the solution field ---------------//

   // Displacement Variable
   Teuchos::ArrayRCP<std::string> dof_names(1);
   dof_names[0] = "Displacement";
   Teuchos::ArrayRCP<std::string> resid_names(1);
   resid_names[0] = dof_names[0]+" Residual";

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructDOFVecInterpolationEvaluator(dof_names[0], X_offset));

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructDOFVecGradInterpolationEvaluator(dof_names[0], X_offset));

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructGatherSolutionEvaluator_noTransient(true, dof_names, X_offset));

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructScatterResidualEvaluator(true, resid_names, X_offset));

   // Pore Pressure Variable
   Teuchos::ArrayRCP<std::string> tdof_names(1);
   tdof_names[0] = "Pore Pressure";
   Teuchos::ArrayRCP<std::string> tdof_names_dot(1);
   tdof_names_dot[0] = tdof_names[0]+"_dot";
   Teuchos::ArrayRCP<std::string> tresid_names(1);
   tresid_names[0] = tdof_names[0]+" Residual";

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructDOFInterpolationEvaluator(tdof_names[0], T_offset));

   (evalUtils.constructDOFInterpolationEvaluator(tdof_names_dot[0], T_offset));

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructDOFGradInterpolationEvaluator(tdof_names[0], T_offset));

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructGatherSolutionEvaluator(false, tdof_names, tdof_names_dot, T_offset));

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructScatterResidualEvaluator(false, tresid_names, T_offset, scatterName));

   // ----------------------setup the solution field ---------------//

   // General FEM stuff
   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructGatherCoordinateVectorEvaluator());

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructMapToPhysicalFrameEvaluator(cellType, cubature));

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructComputeBasisFunctionsEvaluator(cellType, intrepidBasis, cubature));

   // Temporary variable used numerous times below
   Teuchos::RCP<PHX::Evaluator<AlbanyTraits>> ev;

   { // Time
     RCP<ParameterList> p = rcp(new ParameterList);

     p->set<std::string>("Time Name", "Time");
     p->set<std::string>("Delta Time Name", " Delta Time");
     p->set< RCP<DataLayout>>("Workset Scalar Data Layout", dl->workset_scalar);
     p->set<RCP<ParamLib>>("Parameter Library", paramLib);
     p->set<bool>("Disable Transient", true);

     ev = rcp(new LCM::Time<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
     p = stateMgr.registerStateVariable("Time",dl->workset_scalar, dl->dummy, elementBlockName, "scalar", 0.0, true);
     ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
   }

   { // Spatial Stabilization Parameter Field
     RCP<ParameterList> p = rcp(new ParameterList);

     p->set<std::string>("Stabilization Parameter Name", "Stabilization Parameter");
     p->set< RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);

     p->set<RCP<ParamLib>>("Parameter Library", paramLib);
     Teuchos::ParameterList& paramList = params->sublist("Stabilization Parameter");
     p->set<Teuchos::ParameterList*>("Parameter List", &paramList);


     // Additional information to construct stabilization parameter field
     p->set<std::string>("Gradient QP Variable Name", "Pore Pressure Gradient");
     p->set< RCP<DataLayout>>("QP Vector Data Layout", dl->qp_vector);

     p->set<std::string>("Gradient BF Name", "Grad BF");
     p->set< RCP<DataLayout>>("Node QP Vector Data Layout", dl->node_qp_vector);

     p->set<std::string>("Diffusive Parameter Name", "Kozeny-Carman Permeability");
     p->set< RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);

     ev = rcp(new LCM::StabParameter<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);

     p = stateMgr.registerStateVariable("Stabilization Parameter",dl->qp_scalar, dl->dummy, elementBlockName, "scalar", 5.0e-7, true);
     ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
   }


   { // Strain
     RCP<ParameterList> p = rcp(new ParameterList("Strain"));

     //Input
     p->set<std::string>("Gradient QP Variable Name", "Displacement Gradient");

     //Output
     p->set<std::string>("Strain Name", "Strain"); //dl->qp_tensor also

     ev = rcp(new LCM::Strain<EvalT,AlbanyTraits>(*p,dl));
     fm0.template registerEvaluator<EvalT>(ev);
     p = stateMgr.registerStateVariable("Strain",dl->qp_tensor, dl->dummy, elementBlockName, "scalar", 0.0,true);
     ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
   }

   { // Assumed Strain
     RCP<ParameterList> p = rcp(new ParameterList("Assumed Strain"));

     //Inputs: flags, weights, GradU
     const bool avgJ = params->get("avgJ", false);
     p->set<bool>("avgJ Name", avgJ);
     const bool volavgJ = params->get("volavgJ", false);
     p->set<bool>("volavgJ Name", volavgJ);
     const bool weighted_Volume_Averaged_J = params->get("weighted_Volume_Averaged_J", false);
     p->set<bool>("weighted_Volume_Averaged_J Name", weighted_Volume_Averaged_J);
     p->set<std::string>("Weights Name","Weights");
     p->set< RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);
     p->set<std::string>("Gradient QP Variable Name", "Displacement Gradient");
     p->set< RCP<DataLayout>>("QP Tensor Data Layout", dl->qp_tensor);

     //Outputs: F, J
     p->set<std::string>("DefGrad Name", "Deformation Gradient"); //dl->qp_tensor also
     p->set<std::string>("Assumed Strain Name", "Assumed Strain"); //dl->qp_tensor also
     p->set<std::string>("DetDefGrad Name", "Jacobian");
     p->set< RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);

     ev = rcp(new LCM::AssumedStrain<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
     p = stateMgr.registerStateVariable("Assumed Strain",dl->qp_tensor, dl->dummy, elementBlockName, "scalar", 0.0,true);
     ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
   }

   {  // Porosity
     RCP<ParameterList> p = rcp(new ParameterList);

     p->set<std::string>("Porosity Name", "Porosity");
     p->set<std::string>("QP Coordinate Vector Name", "Coord Vec");
     p->set<RCP<ParamLib>>("Parameter Library", paramLib);
     Teuchos::ParameterList& paramList = params->sublist("Porosity");
     p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

     // Setting this turns on dependence of strain and pore pressure)
     p->set<std::string>("Strain Name", "Strain");
     p->set<std::string>("QP Pore Pressure Name", "Pore Pressure");
     p->set< RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);
     p->set<std::string>("Biot Coefficient Name", "Biot Coefficient");

     ev = rcp(new LCM::Porosity<EvalT,AlbanyTraits>(*p,dl));
     fm0.template registerEvaluator<EvalT>(ev);
     p = stateMgr.registerStateVariable("Porosity",dl->qp_scalar, dl->dummy, elementBlockName, "scalar", 0.4, true);
     ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
   }



   { // Biot Coefficient
     RCP<ParameterList> p = rcp(new ParameterList);

     p->set<std::string>("Biot Coefficient Name", "Biot Coefficient");
     p->set<std::string>("QP Coordinate Vector Name", "Coord Vec");
     p->set< RCP<DataLayout>>("Node Data Layout", dl->node_scalar);
     p->set< RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);
     p->set< RCP<DataLayout>>("QP Vector Data Layout", dl->qp_vector);

     p->set<RCP<ParamLib>>("Parameter Library", paramLib);
     Teuchos::ParameterList& paramList = params->sublist("Biot Coefficient");
     p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

     // Setting this turns on linear dependence on porosity
     p->set<std::string>("Porosity Name", "Porosity");

     ev = rcp(new LCM::BiotCoefficient<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
     p = stateMgr.registerStateVariable("Biot Coefficient",dl->qp_scalar, dl->dummy, elementBlockName, "scalar", 1.0);
     ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
   }

   { // Biot Modulus
     RCP<ParameterList> p = rcp(new ParameterList);

     p->set<std::string>("Biot Modulus Name", "Biot Modulus");
     p->set<std::string>("QP Coordinate Vector Name", "Coord Vec");
     p->set< RCP<DataLayout>>("Node Data Layout", dl->node_scalar);
     p->set< RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);
     p->set< RCP<DataLayout>>("QP Vector Data Layout", dl->qp_vector);

     p->set<RCP<ParamLib>>("Parameter Library", paramLib);
     Teuchos::ParameterList& paramList = params->sublist("Biot Modulus");
     p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

     // Setting this turns on linear dependence on porosity and Biot's coeffcient
     p->set<std::string>("Porosity Name", "Porosity");
     p->set<std::string>("Biot Coefficient Name", "Biot Coefficient");

     ev = rcp(new LCM::BiotModulus<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
     p = stateMgr.registerStateVariable("Biot Modulus",dl->qp_scalar, dl->dummy, elementBlockName, "scalar", 1.0e10);
     ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
   }

   { // Thermal conductivity
     RCP<ParameterList> p = rcp(new ParameterList);

     p->set<std::string>("QP Variable Name", "Thermal Conductivity");
     p->set<std::string>("QP Coordinate Vector Name", "Coord Vec");
     p->set< RCP<DataLayout>>("Node Data Layout", dl->node_scalar);
     p->set< RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);
     p->set< RCP<DataLayout>>("QP Vector Data Layout", dl->qp_vector);

     p->set<RCP<ParamLib>>("Parameter Library", paramLib);
     Teuchos::ParameterList& paramList = params->sublist("Thermal Conductivity");
     p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

     ev = rcp(new PHAL::ThermalConductivity<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
   }

   { // Kozeny-Carman Permeaiblity
     RCP<ParameterList> p = rcp(new ParameterList);

     p->set<std::string>("Kozeny-Carman Permeability Name", "Kozeny-Carman Permeability");
     p->set<std::string>("QP Coordinate Vector Name", "Coord Vec");
     p->set< RCP<DataLayout>>("Node Data Layout", dl->node_scalar);
     p->set< RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);
     p->set< RCP<DataLayout>>("QP Vector Data Layout", dl->qp_vector);

     p->set<RCP<ParamLib>>("Parameter Library", paramLib);
     Teuchos::ParameterList& paramList = params->sublist("Kozeny-Carman Permeability");
     p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

     // Setting this turns on Kozeny-Carman relation
     p->set<std::string>("Porosity Name", "Porosity");
     p->set< RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);

     ev = rcp(new LCM::KCPermeability<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
     p = stateMgr.registerStateVariable("Kozeny-Carman Permeability",dl->qp_scalar, dl->dummy, elementBlockName, "scalar", 0.0);
     ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
   }

   // Skeleton parameter

   { // Elastic Modulus
     RCP<ParameterList> p = rcp(new ParameterList);

     p->set<std::string>("QP Variable Name", "Elastic Modulus");
     p->set<std::string>("QP Coordinate Vector Name", "Coord Vec");
     p->set< RCP<DataLayout>>("Node Data Layout", dl->node_scalar);
     p->set< RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);
     p->set< RCP<DataLayout>>("QP Vector Data Layout", dl->qp_vector);

     p->set<RCP<ParamLib>>("Parameter Library", paramLib);
     Teuchos::ParameterList& paramList = params->sublist("Elastic Modulus");
     p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

     p->set<std::string>("Porosity Name", "Porosity"); // porosity is defined at Cubature points
     p->set< RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);

     ev = rcp(new LCM::ElasticModulus<EvalT,AlbanyTraits>(*p));
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

   { // Poissons Ratio 
     RCP<ParameterList> p = rcp(new ParameterList);

     p->set<std::string>("QP Variable Name", "Poissons Ratio");
     p->set<std::string>("QP Coordinate Vector Name", "Coord Vec");
     p->set< RCP<DataLayout>>("Node Data Layout", dl->node_scalar);
     p->set< RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);
     p->set< RCP<DataLayout>>("QP Vector Data Layout", dl->qp_vector);

     p->set<RCP<ParamLib>>("Parameter Library", paramLib);
     Teuchos::ParameterList& paramList = params->sublist("Poissons Ratio");
     p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

     // Setting this turns on linear dependence of nu on T, nu = nu_ + dnudT*T)
     //p->set<std::string>("QP Pore Pressure Name", "Pore Pressure");

     ev = rcp(new LCM::PoissonsRatio<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
   }

   if (matModel == "CapExplicit" || matModel == "CapImplicit")
   {
     { // Cap model stress
       RCP<ParameterList> p = rcp(new ParameterList("Stress"));

       //Input
       p->set<std::string>("Strain Name", "Assumed Strain");
       p->set< RCP<DataLayout>>("QP Tensor Data Layout", dl->qp_tensor);

       p->set<std::string>("Elastic Modulus Name", "Elastic Modulus");
       p->set< RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);

       p->set<std::string>("Poissons Ratio Name", "Poissons Ratio");  // dl->qp_scalar also

       //Output
       p->set<std::string>("Stress Name", "Stress"); //dl->qp_tensor also
       p->set<std::string>("Back Stress Name", "backStress"); //dl->qp_tensor also
       p->set<std::string>("Cap Parameter Name", "capParameter"); //dl->qp_tensor also

       //if (matModel == "CapImplicit"){
       //p->set<std::string>("Friction Name", "friction"); //dl->qp_scalar also
       //p->set<std::string>("Dilatancy Name", "dilatancy"); //dl->qp_scalar also
       //p->set<std::string>("Hardening Modulus Name", "hardeningModulus"); //dl->qp_scalar also
       //}
       p->set<std::string>("Eqps Name", "eqps"); //dl->qp_scalar also
       p->set<std::string>("Vol Plastic Strain Name", "volPlasticStrain"); //dl->qp_scalar also

       RealType A = params->get("A", 1.0);
       RealType B = params->get("B", 1.0);
       RealType C = params->get("C", 1.0);
       RealType theta = params->get("theta", 1.0);
       RealType R = params->get("R", 1.0);
       RealType kappa0 = params->get("kappa0", 1.0);
       RealType W = params->get("W", 1.0);
       RealType D1 = params->get("D1", 1.0);
       RealType D2 = params->get("D2", 1.0);
       RealType calpha = params->get("calpha", 1.0);
       RealType psi = params->get("psi", 1.0);
       RealType N = params->get("N", 1.0);
       RealType L = params->get("L", 1.0);
       RealType phi = params->get("phi", 1.0);
       RealType Q = params->get("Q", 1.0);

       p->set<RealType>("A Name", A);
       p->set<RealType>("B Name", B);
       p->set<RealType>("C Name", C);
       p->set<RealType>("Theta Name", theta);
       p->set<RealType>("R Name", R);
       p->set<RealType>("Kappa0 Name", kappa0);
       p->set<RealType>("W Name", W);
       p->set<RealType>("D1 Name", D1);
       p->set<RealType>("D2 Name", D2);
       p->set<RealType>("Calpha Name", calpha);
       p->set<RealType>("Psi Name", psi);
       p->set<RealType>("N Name", N);
       p->set<RealType>("L Name", L);
       p->set<RealType>("Phi Name", phi);
       p->set<RealType>("Q Name", Q);

       //Declare what state data will need to be saved (name, layout, init_type)
       if(matModel == "CapExplicit"){
    	  ev = rcp(new LCM::CapExplicit<EvalT,AlbanyTraits>(*p,dl));
       }

       if(matModel == "CapImplicit"){
    	  ev = rcp(new LCM::CapImplicit<EvalT,AlbanyTraits>(*p,dl));
       }

       fm0.template registerEvaluator<EvalT>(ev);
       p = stateMgr.registerStateVariable("Stress",dl->qp_tensor, dl->dummy, elementBlockName, "scalar", 0.0, true);
       ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
       fm0.template registerEvaluator<EvalT>(ev);
       p = stateMgr.registerStateVariable("backStress",dl->qp_tensor, dl->dummy, elementBlockName, "scalar", 0.0, true);
       ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
       fm0.template registerEvaluator<EvalT>(ev);
       p = stateMgr.registerStateVariable("capParameter",dl->qp_scalar, dl->dummy, elementBlockName, "scalar", kappa0, true);
       ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
       fm0.template registerEvaluator<EvalT>(ev);

       //if(matModel == "CapImplicit"){
       //p = stateMgr.registerStateVariable("friction",dl->qp_scalar, dl->dummy, elementBlockName, "scalar", 0.0);
       //ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
       //fm0.template registerEvaluator<EvalT>(ev);
       //p = stateMgr.registerStateVariable("dilatancy",dl->qp_scalar, dl->dummy, elementBlockName, "scalar", 0.0);
       //ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
       //fm0.template registerEvaluator<EvalT>(ev);
       //p = stateMgr.registerStateVariable("hardeningModulus",dl->qp_scalar, dl->dummy, elementBlockName, "scalar", 0.0);
       //ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
       //fm0.template registerEvaluator<EvalT>(ev);
       //}
       p = stateMgr.registerStateVariable("eqps",dl->qp_scalar, dl->dummy, elementBlockName, "scalar", 0.0, true);
       ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
       fm0.template registerEvaluator<EvalT>(ev);
       p = stateMgr.registerStateVariable("volPlasticStrain",dl->qp_scalar, dl->dummy, elementBlockName, "scalar", 0.0,true);
       ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
       fm0.template registerEvaluator<EvalT>(ev);
     }
   }
   else
   {
     { // Linear elasticity stress
       RCP<ParameterList> p = rcp(new ParameterList("Stress"));

       //Input
       p->set<std::string>("Strain Name", "Assumed Strain");
       p->set< RCP<DataLayout>>("QP Tensor Data Layout", dl->qp_tensor);

       p->set<std::string>("Elastic Modulus Name", "Elastic Modulus");
       p->set< RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);

       p->set<std::string>("Poissons Ratio Name", "Poissons Ratio");  // dl->qp_scalar also

       //Output
       p->set<std::string>("Stress Name", "Stress"); //dl->qp_tensor also

       ev = rcp(new LCM::Stress<EvalT,AlbanyTraits>(*p));
       fm0.template registerEvaluator<EvalT>(ev);
       p = stateMgr.registerStateVariable("Stress",dl->qp_tensor, dl->dummy, elementBlockName, "scalar", 0.0);
       ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
       fm0.template registerEvaluator<EvalT>(ev);
     }
   }

   { // Total Stress
     RCP<ParameterList> p = rcp(new ParameterList("Total Stress"));

     //Input
     p->set<std::string>("Effective Stress Name", "Stress");
     p->set< RCP<DataLayout>>("QP Tensor Data Layout", dl->qp_tensor);

     p->set<std::string>("Biot Coefficient Name", "Biot Coefficient");  // dl->qp_scalar also
     p->set< RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);

     p->set<std::string>("QP Variable Name", "Pore Pressure");
     p->set< RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);

     //Output
     p->set<std::string>("Total Stress Name", "Total Stress"); //dl->qp_tensor also

     ev = rcp(new LCM::TotalStress<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
     p = stateMgr.registerStateVariable("Total Stress",dl->qp_tensor, dl->dummy, elementBlockName, "scalar", 0.0);
     ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
     p = stateMgr.registerStateVariable("Pore Pressure",dl->qp_scalar, dl->dummy, elementBlockName, "scalar", 0.0, true);
     ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);

   }


   /*  { // Total Stress
       RCP<ParameterList> p = rcp(new ParameterList("Total Stress"));

       //Input
       p->set<std::string>("Strain Name", "Strain");
       p->set< RCP<DataLayout>>("QP Tensor Data Layout", dl->qp_tensor);

       p->set<std::string>("Elastic Modulus Name", "Elastic Modulus");
       p->set< RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);

       p->set<std::string>("Poissons Ratio Name", "Poissons Ratio");  // dl->qp_scalar also

       p->set<std::string>("Biot Coefficient Name", "Biot Coefficient");  // dl->qp_scalar also
       p->set< RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);

       p->set<std::string>("QP Variable Name", "Pore Pressure");
       p->set< RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);

       //Output
       p->set<std::string>("Total Stress Name", "Total Stress"); //dl->qp_tensor also

       ev = rcp(new LCM::TotalStress<EvalT,AlbanyTraits>(*p));
       fm0.template registerEvaluator<EvalT>(ev);
       p = stateMgr.registerStateVariable("Total Stress",dl->qp_tensor, dl->dummy, elementBlockName, "scalar", 0.0);
       ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
       fm0.template registerEvaluator<EvalT>(ev);
       p = stateMgr.registerStateVariable("Pore Pressure",dl->qp_scalar, dl->dummy, elementBlockName, "scalar", 0.0, true);
       ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
       fm0.template registerEvaluator<EvalT>(ev);

       } */

   if (haveSource) { // Source
     RCP<ParameterList> p = rcp(new ParameterList);

     p->set<std::string>("Source Name", "Source");
     p->set<std::string>("QP Variable Name", "Pore Pressure");
     p->set< RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);

     p->set<RCP<ParamLib>>("Parameter Library", paramLib);
     Teuchos::ParameterList& paramList = params->sublist("Source Functions");
     p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

     ev = rcp(new PHAL::Source<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
   }

   { // Displacement Resid
     RCP<ParameterList> p = rcp(new ParameterList("Displacement Resid"));

     //Input
     p->set<std::string>("Total Stress Name", "Total Stress");
     p->set< RCP<DataLayout>>("QP Tensor Data Layout", dl->qp_tensor);

     p->set<std::string>("Weighted BF Name", "wBF");
     p->set< RCP<DataLayout>>("Node QP Scalar Data Layout", dl->node_qp_scalar);

     p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
     p->set< RCP<DataLayout>>("Node QP Vector Data Layout", dl->node_qp_vector);

     p->set<bool>("Disable Transient", true);

     //Output
     p->set<std::string>("Residual Name", "Displacement Residual");
     p->set< RCP<DataLayout>>("Node Vector Data Layout", dl->node_vector);

     ev = rcp(new LCM::PoroElasticityResidMomentum<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
   }



   { // Pore Pressure Resid
     RCP<ParameterList> p = rcp(new ParameterList("Pore Pressure Resid"));

     //Input

     // Input from nodal points
     p->set<std::string>("Weighted BF Name", "wBF");
     p->set< RCP<DataLayout>>("Node QP Scalar Data Layout", dl->node_qp_scalar);

     p->set<bool>("Have Source", false);
     p->set<std::string>("Source Name", "Source");

     p->set<bool>("Have Absorption", false);

     // Input from cubature points
     p->set<std::string>("QP Pore Pressure Name", "Pore Pressure");
     p->set< RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);

     p->set<std::string>("QP Time Derivative Variable Name", "Pore Pressure");

     p->set<std::string>("Material Property Name", "Stabilization Parameter");
     p->set<std::string>("Thermal Conductivity Name", "Thermal Conductivity");
     p->set<std::string>("Porosity Name", "Porosity");
     p->set<std::string>("Kozeny-Carman Permeability Name", "Kozeny-Carman Permeability");
     p->set<std::string>("Biot Coefficient Name", "Biot Coefficient");
     p->set<std::string>("Biot Modulus Name", "Biot Modulus");
     p->set<std::string>("Elastic Modulus Name", "Elastic Modulus");
     p->set<std::string>("Poissons Ratio Name", "Poissons Ratio");

     p->set<std::string>("Gradient QP Variable Name", "Pore Pressure Gradient");
     p->set< RCP<DataLayout>>("QP Vector Data Layout", dl->qp_vector);

     p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
     p->set< RCP<DataLayout>>("Node QP Vector Data Layout", dl->node_qp_vector);

     p->set<std::string>("Strain Name", "Strain");
     p->set< RCP<DataLayout>>("QP Tensor Data Layout", dl->qp_tensor);

     // Inputs: X, Y at nodes, Cubature, and Basis
     p->set<std::string>("Coordinate Vector Name","Coord Vec");
     p->set< RCP<DataLayout>>("Coordinate Data Layout", dl->vertices_vector);
     p->set< RCP<Intrepid2::Cubature<PHX::Device> >>>("Cubature", cubature);
     p->set<RCP<shards::CellTopology>>("Cell Type", cellType);

     p->set<std::string>("Weights Name","Weights");

     p->set<std::string>("Delta Time Name", " Delta Time");
     p->set< RCP<DataLayout>>("Workset Scalar Data Layout", dl->workset_scalar);

     //Output
     p->set<std::string>("Residual Name", "Pore Pressure Residual");
     p->set< RCP<DataLayout>>("Node Scalar Data Layout", dl->node_scalar);

     ev = rcp(new LCM::PoroElasticityResidMass<EvalT,AlbanyTraits>(*p));
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

#endif // POROELASTICITYPROBLEM_HPP
