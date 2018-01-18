//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef THERMOELASTICITYPROBLEM_HPP
#define THERMOELASTICITYPROBLEM_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_AbstractProblem.hpp"

#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_ResponseUtilities.hpp"
#include "Albany_EvaluatorUtils.hpp"
#include "PHAL_AlbanyTraits.hpp"

namespace Albany {

  /*!
   * \brief Abstract interface for representing a 2-D finite element
   * problem.
   */
  class ThermoElasticityProblem : public Albany::AbstractProblem {
  public:

    //! Default constructor
    ThermoElasticityProblem(
			    const Teuchos::RCP<Teuchos::ParameterList>& params,
			    const Teuchos::RCP<ParamLib>& paramLib,
			    const int numEq);

    //! Destructor
    virtual ~ThermoElasticityProblem();

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

    void getAllocatedStates(Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Kokkos::DynRankView<RealType, PHX::Device>>>> oldState_,
			    Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Kokkos::DynRankView<RealType, PHX::Device>>>> newState_
			    ) const;

  private:

    //! Private to prohibit copying
    ThermoElasticityProblem(const ThermoElasticityProblem&);

    //! Private to prohibit copying
    ThermoElasticityProblem& operator=(const ThermoElasticityProblem&);

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

    ///
    ///Boolean marking whether SDBCs are used
    bool use_sdbcs_;

    //! Boundary conditions on source term
    bool haveSource;
    int T_offset;  //Position of T unknown in nodal DOFs
    int X_offset;  //Position of X unknown in nodal DOFs, followed by Y,Z
    int numDim;    //Number of spatial dimensions and displacement variable

    Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Kokkos::DynRankView<RealType, PHX::Device>>>> oldState;
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Kokkos::DynRankView<RealType, PHX::Device>>>> newState;

  };
}

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_AbstractProblem.hpp"

#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_ResponseUtilities.hpp"
#include "Albany_EvaluatorUtils.hpp"
#include "PHAL_AlbanyTraits.hpp"

// Explicity add evaluators defined in the model below
#include "ElasticModulus.hpp"
#include "PoissonsRatio.hpp"
#include "PHAL_Source.hpp"
#include "Strain.hpp"
#include "Stress.hpp"
#include "PHAL_SaveStateField.hpp"
#include "ElasticityResid.hpp"
#include "PHAL_ThermalConductivity.hpp"
#include "PHAL_Source.hpp"
#include "PHAL_HeatEqResid.hpp"


template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
Albany::ThermoElasticityProblem::constructEvaluators(
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
   RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>>
     intrepidBasis = Albany::getIntrepid2Basis(meshSpecs.ctd);

   const int numNodes = intrepidBasis->getCardinality();
   const int worksetSize = meshSpecs.worksetSize;

   Intrepid2::DefaultCubatureFactory cubFactory;
   RCP <Intrepid2::Cubature<PHX::Device>  > cubature = cubFactory.create<PHX::Device, RealType, RealType>(*cellType, meshSpecs.cubatureDegree);

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
     resid_names[0] = dof_names[0]+" Residual";

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
     tresid_names[0] = tdof_names[0]+" Residual";

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

    // Setting this turns on linear dependence of E on T, E = E_ + dEdT*T)
    p->set<std::string>("QP Temperature Name", "Temperature");

    ev = rcp(new LCM::ElasticModulus<EvalT,AlbanyTraits>(*p));
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
    p->set<std::string>("QP Temperature Name", "Temperature");

    ev = rcp(new LCM::PoissonsRatio<EvalT,AlbanyTraits>(*p));
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

  { // Strain
    RCP<ParameterList> p = rcp(new ParameterList("Strain"));

    //Input
    p->set<std::string>("Gradient QP Variable Name", "Displacement Gradient");

    //Output
    p->set<std::string>("Strain Name", "Strain");

    ev = rcp(new LCM::Strain<EvalT,AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  { // Stress
    RCP<ParameterList> p = rcp(new ParameterList("Stress"));

    //Input
    p->set<std::string>("Strain Name", "Strain");
    p->set< RCP<DataLayout>>("QP Tensor Data Layout", dl->qp_tensor);

    p->set<std::string>("Elastic Modulus Name", "Elastic Modulus");
    p->set< RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);

    p->set<std::string>("Poissons Ratio Name", "Poissons Ratio");  // dl->qp_scalar also

    //Output
    p->set<std::string>("Stress Name", "Stress"); //dl->qp_tensor also

    ev = rcp(new LCM::Stress<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
    p = stateMgr.registerStateVariable("Stress",dl->qp_tensor, dl->dummy, elementBlockName, "zero");
    ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  { // Displacement Resid
    RCP<ParameterList> p = rcp(new ParameterList("Displacement Resid"));

    //Input
    p->set<std::string>("Stress Name", "Stress");
    p->set< RCP<DataLayout>>("QP Tensor Data Layout", dl->qp_tensor);

    p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
    p->set< RCP<DataLayout>>("Node QP Vector Data Layout", dl->node_qp_vector);

    p->set<bool>("Disable Transient", true);

    //Output
    p->set<std::string>("Residual Name", "Displacement Residual");
    p->set< RCP<DataLayout>>("Node Vector Data Layout", dl->node_vector);

    ev = rcp(new LCM::ElasticityResid<EvalT,AlbanyTraits>(*p));
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
  { // Temperature Resid
    RCP<ParameterList> p = rcp(new ParameterList("Temperature Resid"));

    //Input
    p->set<std::string>("Weighted BF Name", "wBF");
    p->set< RCP<DataLayout>>("Node QP Scalar Data Layout", dl->node_qp_scalar);
    p->set<std::string>("QP Variable Name", "Temperature");

    p->set<std::string>("QP Time Derivative Variable Name", "Temperature_dot");

    p->set<bool>("Have Source", haveSource);
    p->set<std::string>("Source Name", "Source");

    p->set<bool>("Have Absorption", false);

    p->set<std::string>("Thermal Conductivity Name", "Thermal Conductivity");
    p->set< RCP<DataLayout>>("QP Scalar Data Layout", dl->qp_scalar);

    p->set<std::string>("Gradient QP Variable Name", "Temperature Gradient");
    p->set< RCP<DataLayout>>("QP Vector Data Layout", dl->qp_vector);

    p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
    p->set< RCP<DataLayout>>("Node QP Vector Data Layout", dl->node_qp_vector);

    //Output
    p->set<std::string>("Residual Name", "Temperature Residual");
    p->set< RCP<DataLayout>>("Node Scalar Data Layout", dl->node_scalar);

    ev = rcp(new PHAL::HeatEqResid<EvalT,AlbanyTraits>(*p));
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
