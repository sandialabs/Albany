//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef THERMO_MECHANICAL_PROBLEM_HPP
#define THERMO_MECHANICAL_PROBLEM_HPP

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
   * \brief ThermoMechanical Coupling Problem
   */
  class ThermoMechanicalProblem : public Albany::AbstractProblem {
  public:
  
    //! Default constructor
    ThermoMechanicalProblem( const Teuchos::RCP<Teuchos::ParameterList>& params,
			     const Teuchos::RCP<ParamLib>& paramLib,
			     const int numEq );

    //! Destructor
    virtual ~ThermoMechanicalProblem();

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

    void getAllocatedStates( Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid::FieldContainer<RealType>>>> oldState_,
			     Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid::FieldContainer<RealType>>>> newState_ ) const;

  private:

    //! Private to prohibit copying
    ThermoMechanicalProblem( const ThermoMechanicalProblem& );
    
    //! Private to prohibit copying
    ThermoMechanicalProblem& operator = ( const ThermoMechanicalProblem& );

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

    void constructDirichletEvaluators( const Albany::MeshSpecsStruct& meshSpecs );

  protected:

    //! Boundary conditions on source term
    bool haveSource;
    std::string model;
    int T_offset;  //Position of T unknown in nodal DOFs
    int X_offset;  //Position of X unknown in nodal DOFs, followed by Y,Z
    int numDim;    //Number of spatial dimensions and displacement variable 

    Teuchos::ArrayRCP< Teuchos::ArrayRCP< Teuchos::RCP< Intrepid::FieldContainer< RealType >>>> oldState;
    Teuchos::ArrayRCP< Teuchos::ArrayRCP< Teuchos::RCP< Intrepid::FieldContainer< RealType >>>> newState;
  };

}

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

#include "ShearModulus.hpp"
#include "BulkModulus.hpp"
#include "YieldStrength.hpp"
#include "HardeningModulus.hpp"
#include "SaturationModulus.hpp"
#include "SaturationExponent.hpp"
#include "PHAL_Source.hpp"
#include "DefGrad.hpp"
#include "ThermoMechanicalStress.hpp"
#include "PHAL_SaveStateField.hpp"
#include "ThermoMechanicalEnergyResidual.hpp"
#include "ThermoMechanicalMomentumResidual.hpp"
#include "PHAL_ThermalConductivity.hpp"
#include "PHAL_Source.hpp"
#include "PHAL_HeatEqResid.hpp"
#include "Time.hpp"

template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
Albany::ThermoMechanicalProblem::constructEvaluators(
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
  RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType>>>
    intrepidBasis = Albany::getIntrepidBasis(meshSpecs.ctd);

  const int numNodes = intrepidBasis->getCardinality();
  const int worksetSize = meshSpecs.worksetSize;

  Intrepid::DefaultCubatureFactory<RealType> cubFactory;
  RCP <Intrepid::Cubature<RealType>> cubature = cubFactory.create(*cellType, meshSpecs.cubatureDegree);

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
  std::string scatterName="Scatter Heat";


  // Displacement Variable
  Teuchos::ArrayRCP<std::string> dof_names(1);
  dof_names[0] = "Displacement";
  Teuchos::ArrayRCP<std::string> resid_names(1);
  resid_names[0] = "Thermo Mechanical Momentum Residual";

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructDOFVecInterpolationEvaluator(dof_names[0], X_offset));

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructDOFVecGradInterpolationEvaluator(dof_names[0], X_offset));

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructGatherSolutionEvaluator_noTransient(true, dof_names, X_offset));

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructScatterResidualEvaluator(true, resid_names, X_offset));

  // Temperature Variable
  Teuchos::ArrayRCP<std::string> tdof_names(1);
  tdof_names[0] = "Temperature";
  Teuchos::ArrayRCP<std::string> tdof_names_dot(1);
  tdof_names_dot[0] = tdof_names[0]+"_dot";
  Teuchos::ArrayRCP<std::string> tresid_names(1);
  tresid_names[0] = "Thermo Mechanical Energy Residual";

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructDOFInterpolationEvaluator(tdof_names[0], T_offset));

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructDOFInterpolationEvaluator(tdof_names_dot[0], T_offset));

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructDOFGradInterpolationEvaluator(tdof_names[0], T_offset));

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructGatherSolutionEvaluator(false, tdof_names, tdof_names_dot, T_offset));

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructScatterResidualEvaluator(false, tresid_names, T_offset, scatterName));

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
    p->set<std::string>("Delta Time Name", "Delta Time");
    p->set< RCP<DataLayout>>("Workset Scalar Data Layout", dl->workset_scalar);
    p->set<RCP<ParamLib>>("Parameter Library", paramLib);
    p->set<bool>("Disable Transient", true);

    ev = rcp(new LCM::Time<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
    p = stateMgr.registerStateVariable("Time",dl->workset_scalar, dl->dummy, elementBlockName, "scalar", 0.0, true);
    ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (haveSource) { // Source
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                               "Error!  Sources not implemented in Mechanical yet!");

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
    RCP<ParameterList> p = rcp(new ParameterList("Deformation Gradient"));

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
    p->set<std::string>("DetDefGrad Name", "Determinant of the Deformation Gradient"); 
    p->set< RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);

    ev = rcp(new LCM::DefGrad<EvalT,AlbanyTraits>(*p));
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

    // Setting this turns on linear dependence of mu on T, mu = mu + dmudT*(T - Tref)
    p->set<std::string>("QP Temperature Name", "Temperature");
    RealType refTemp = params->get("Reference Temperature", 0.0);
    p->set<RealType>("Reference Temperature", refTemp);
 
    ev = rcp(new LCM::ShearModulus<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

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

    // Setting this turns on linear dependence of K on T, K = K + dKdT*(T - Tref)
    p->set<std::string>("QP Temperature Name", "Temperature");
    RealType refTemp = params->get("Reference Temperature", 0.0);
    p->set<RealType>("Reference Temperature", refTemp);
 
    ev = rcp(new LCM::BulkModulus<EvalT,AlbanyTraits>(*p));
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

    // Setting this turns on linear dependence of Y on T, Y = Y + dYdT*(T - Tref)
    p->set<std::string>("QP Temperature Name", "Temperature");
    RealType refTemp = params->get("Reference Temperature", 0.0);
    p->set<RealType>("Reference Temperature", refTemp);

    ev = rcp(new LCM::YieldStrength<EvalT,AlbanyTraits>(*p));
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

    // Setting this turns on linear dependence of H on T, H = H + dHdT*(T - Tref)
    p->set<std::string>("QP Temperature Name", "Temperature");
    RealType refTemp = params->get("Reference Temperature", 0.0);
    p->set<RealType>("Reference Temperature", refTemp);

    ev = rcp(new LCM::HardeningModulus<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if ( model == "ThermoMechanical" )
  {
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

      // Setting this turns on linear dependence of S on T, S = S + dSdT*(T - Tref)
      p->set<std::string>("QP Temperature Name", "Temperature");
      RealType refTemp = params->get("Reference Temperature", 0.0);
      p->set<RealType>("Reference Temperature", refTemp);

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

    { // Stress
      RCP<ParameterList> p = rcp(new ParameterList("Stress"));

      //Input
      p->set<std::string>("DefGrad Name", "Deformation Gradient");
      p->set< RCP<DataLayout>>("QP Tensor Data Layout", dl->qp_tensor);

      p->set<std::string>("Shear Modulus Name", "Shear Modulus");

      p->set< RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);

      p->set<std::string>("Bulk Modulus Name", "Bulk Modulus");  // dl->qp_scalar also
      p->set<std::string>("DetDefGrad Name", "Determinant of the Deformation Gradient");  // dl->qp_scalar also
      p->set<std::string>("Yield Strength Name", "Yield Strength");
      p->set<std::string>("Hardening Modulus Name", "Hardening Modulus");
      p->set<std::string>("Saturation Modulus Name", "Saturation Modulus"); // dl->qp_scalar also
      p->set<std::string>("Saturation Exponent Name", "Saturation Exponent"); // dl->qp_scalar also
      p->set<std::string>("Temperature Name", "Temperature");
      RealType refTemp = params->get("Reference Temperature", 0.0);
      p->set<RealType>("Reference Temperature", refTemp);
      RealType coeff = params->get("Thermal Expansion Coefficient", 0.0);
      p->set<RealType>("Thermal Expansion Coefficient", coeff);

      p->set<std::string>("Delta Time Name", "Delta Time");
      p->set< RCP<DataLayout>>("Workset Scalar Data Layout", dl->workset_scalar);

      //Output
      p->set<std::string>("Stress Name", "Stress"); //dl->qp_tensor also
      p->set<std::string>("Fp Name", "Fp");
      p->set<std::string>("eqps Name", "eqps");
      p->set<std::string>("Mechanical Source Name", "Mechanical Source");

      ev = rcp(new LCM::ThermoMechanicalStress<EvalT,AlbanyTraits>(*p));
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
  }

  else if ( model == "BCJ" )
  {
    // put BCJ specific things here

    { // Stress
      // RCP<ParameterList> p = rcp(new ParameterList("Stress"));

      // //Input
      // p->set<std::string>("DefGrad Name", "Deformation Gradient");
      // p->set< RCP<DataLayout>>("QP Tensor Data Layout", dl->qp_tensor);

      // p->set<std::string>("Shear Modulus Name", "Shear Modulus");

      // p->set< RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);

      // p->set<std::string>("Bulk Modulus Name", "Bulk Modulus");  // dl->qp_scalar also
      // p->set<std::string>("DetDefGrad Name", "Determinant of the Deformation Gradient");  // dl->qp_scalar also
      // p->set<std::string>("Yield Strength Name", "Yield Strength");
      // p->set<std::string>("Hardening Modulus Name", "Hardening Modulus");
      // p->set<std::string>("Temperature Name", "Temperature");

      // RealType refTemp = params->get("Reference Temperature", 0.0);
      // p->set<RealType>("Reference Temperature", refTemp);
      // RealType coeff = params->get("Thermal Expansion Coefficient", 0.0);
      // p->set<RealType>("Thermal Expansion Coefficient", coeff);

      // p->set<std::string>("Delta Time Name", "Delta Time");
      // p->set< RCP<DataLayout>>("Workset Scalar Data Layout", dl->workset_scalar);
      
      // // Output
      // p->set<std::string>("Stress Name", "Stress"); //dl->qp_tensor also
      // p->set<std::string>("Fp Name", "Fp");
      // p->set<std::string>("eqps Name", "eqps");
      // p->set<std::string>("Mechanical Source Name", "Mechanical Source");

      // ev = rcp(new LCM::BCJ<EvalT,AlbanyTraits>(*p));
      // fm0.template registerEvaluator<EvalT>(ev);
      // p = stateMgr.registerStateVariable("Stress",dl->qp_tensor, dl->dummy, elementBlockName, "scalar", 0.0);
      // ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
      // fm0.template registerEvaluator<EvalT>(ev);
      // p = stateMgr.registerStateVariable("Fp",dl->qp_tensor, dl->dummy, elementBlockName, "identity",1.0,true);
      // ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
      // fm0.template registerEvaluator<EvalT>(ev);
      // p = stateMgr.registerStateVariable("eqps",dl->qp_scalar, dl->dummy, elementBlockName, "scalar", 0.0,true);
      // ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
      // fm0.template registerEvaluator<EvalT>(ev);
    }
  }

  { // ThermoMechanical Momentum Residual
    RCP<ParameterList> p = rcp(new ParameterList("Thermo Mechanical Momentum Residual"));

    //Input
    p->set<std::string>("Stress Name", "Stress");
    p->set< RCP<DataLayout>>("QP Tensor Data Layout", dl->qp_tensor);

    p->set<std::string>("DetDefGrad Name", "Determinant of the Deformation Gradient");
    p->set< RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);

    p->set<std::string>("DefGrad Name", "Deformation Gradient");

    p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
    p->set< RCP<DataLayout>>("Node QP Vector Data Layout", dl->node_qp_vector);

    p->set<RCP<ParamLib>>("Parameter Library", paramLib);

    p->set<bool>("Disable Transient", true);

    //Output
    p->set<std::string>("Residual Name", "Thermo Mechanical Momentum Residual");
    p->set< RCP<DataLayout>>("Node Vector Data Layout", dl->node_vector);

    ev = rcp(new LCM::ThermoMechanicalMomentumResidual<EvalT,AlbanyTraits>(*p));
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

  if (haveSource) { // Source
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<std::string>("Source Name", "Source");
    p->set<std::string>("Variable Name", "Temperature");
    p->set< RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);

    p->set<RCP<ParamLib>>("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Source Functions");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = rcp(new PHAL::Source<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  { // ThermoMechanical Energy Residual
    RCP<ParameterList> p = rcp(new ParameterList("Thermo Mechanical Energy Residual"));

    //Input
    p->set<std::string>("Weighted BF Name", "wBF");
    p->set< RCP<DataLayout>>("Node QP Scalar Data Layout", dl->node_qp_scalar);
    p->set<std::string>("QP Variable Name", "Temperature");

    p->set<bool>("Have Source", haveSource);
    p->set<std::string>("Source Name", "Source");

    p->set<std::string>("Deformation Gradient Name", "Deformation Gradient");
    p->set< RCP<DataLayout>>("QP Tensor Data Layout", dl->qp_tensor);

    p->set<std::string>("Thermal Conductivity Name", "Thermal Conductivity");
    p->set< RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);

    p->set<std::string>("Gradient QP Variable Name", "Temperature Gradient");
    p->set< RCP<DataLayout>>("QP Vector Data Layout", dl->qp_vector);

    p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
    p->set< RCP<DataLayout>>("Node QP Vector Data Layout", dl->node_qp_vector);

    p->set<std::string>("Mechanical Source Name", "Mechanical Source");
    p->set<std::string>("Delta Time Name", "Delta Time");
    p->set< RCP<DataLayout>>("Workset Scalar Data Layout", dl->workset_scalar);
    RealType density = params->get("Density", 1.0);
    p->set<RealType>("Density", density);
    RealType Cv = params->get("Heat Capacity", 1.0);
    p->set<RealType>("Heat Capacity", Cv);

    //Output
    p->set<std::string>("Residual Name", "Thermo Mechanical Energy Residual");
    p->set< RCP<DataLayout>>("Node Scalar Data Layout", dl->node_scalar);

    ev = rcp(new LCM::ThermoMechanicalEnergyResidual<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
    p = stateMgr.registerStateVariable("Temperature",dl->qp_scalar, dl->dummy, elementBlockName, "scalar", 0.0,true);
    ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
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
#endif // ALBANY_ELASTICITYPROBLEM_HPP
