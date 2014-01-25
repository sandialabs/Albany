//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_NAVIERSTOKES_HPP
#define ALBANY_NAVIERSTOKES_HPP

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
  class NavierStokes : public AbstractProblem {
  public:
  
    //! Default constructor
    NavierStokes(const Teuchos::RCP<Teuchos::ParameterList>& params,
		 const Teuchos::RCP<ParamLib>& paramLib,
		 const int numDim_);

    //! Destructor
    ~NavierStokes();

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
    NavierStokes(const NavierStokes&);
    
    //! Private to prohibit copying
    NavierStokes& operator=(const NavierStokes&);

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
    void constructNeumannEvaluators(const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs); 

  protected:

    //! Enumerated type describing how a variable appears
    enum NS_VAR_TYPE {
      NS_VAR_TYPE_NONE,      //! Variable does not appear
      NS_VAR_TYPE_CONSTANT,  //! Variable is a constant
      NS_VAR_TYPE_DOF        //! Variable is a degree-of-freedom
    };

    void getVariableType(Teuchos::ParameterList& paramList,
			 const std::string& defaultType,
			 NS_VAR_TYPE& variableType,
			 bool& haveVariable,
			 bool& haveEquation);
    std::string variableTypeToString(const NS_VAR_TYPE variableType);

  protected:
    
    bool periodic;     //! periodic BCs
    int numDim;        //! number of spatial dimensions

    NS_VAR_TYPE flowType; //! type of flow variables
    NS_VAR_TYPE heatType; //! type of heat variables
    NS_VAR_TYPE neutType; //! type of neutron variables

    bool haveFlow;     //! have flow variables (momentum+continuity)
    bool haveHeat;     //! have heat variables (temperature)
    bool haveNeut;     //! have neutron flux variables

    bool haveFlowEq;     //! have flow equations (momentum+continuity)
    bool haveHeatEq;     //! have heat equation (temperature)
    bool haveNeutEq;     //! have neutron flux equation

    bool haveSource;   //! have source term in heat equation
    bool haveNeutSource;   //! have source term in neutron flux equation
    bool havePSPG;     //! have pressure stabilization
    bool haveSUPG;     //! have SUPG stabilization
    bool porousMedia;  //! flow through porous media problem
    
    Teuchos::RCP<Albany::Layouts> dl;
  };

}

#include <boost/type_traits/is_same.hpp>

#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"

#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_EvaluatorUtils.hpp"
#include "Albany_ResponseUtilities.hpp"

#include "PHAL_NSContravarientMetricTensor.hpp"
#include "PHAL_NSMaterialProperty.hpp"
#include "PHAL_Source.hpp"
#include "PHAL_NSBodyForce.hpp"
#include "PHAL_NSPermeabilityTerm.hpp"
#include "PHAL_NSForchheimerTerm.hpp"
#include "PHAL_NSRm.hpp"
#include "PHAL_NSTauM.hpp"
#include "PHAL_NSTauT.hpp"
#include "PHAL_NSMomentumResid.hpp"
#include "PHAL_NSContinuityResid.hpp"
#include "PHAL_NSThermalEqResid.hpp"
#include "PHAL_NSNeutronEqResid.hpp"

template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
Albany::NavierStokes::constructEvaluators(
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
 
  const CellTopologyData * const elem_top = &meshSpecs.ctd;
 
  RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > >
    intrepidBasis = Albany::getIntrepidBasis(meshSpecs.ctd);
  RCP<shards::CellTopology> cellType = rcp(new shards::CellTopology (&meshSpecs.ctd));
  
  const int numNodes = intrepidBasis->getCardinality();
  const int worksetSize = meshSpecs.worksetSize;
  
  Intrepid::DefaultCubatureFactory<RealType> cubFactory;
  RCP <Intrepid::Cubature<RealType> > cubature = cubFactory.create(*cellType, meshSpecs.cubatureDegree);
  
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
                              "Data Layout Usage in NavierStokes problem assumes vecDim = numDim");

   Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
   bool supportsTransient=true;
   int offset=0;

   // Temporary variable used numerous times below
   Teuchos::RCP<PHX::Evaluator<AlbanyTraits> > ev;

   // Define Field Names

   if (haveFlowEq) {
     Teuchos::ArrayRCP<string> dof_names(1);
     Teuchos::ArrayRCP<string> dof_names_dot(1);
     Teuchos::ArrayRCP<string> resid_names(1);
     dof_names[0] = "Velocity";
     dof_names_dot[0] = dof_names[0]+"_dot";
     resid_names[0] = "Momentum Residual";
     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructGatherSolutionEvaluator(true, dof_names, dof_names_dot, offset));

     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructDOFVecInterpolationEvaluator(dof_names[0], offset));

     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructDOFVecInterpolationEvaluator(dof_names_dot[0], offset));

     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructDOFVecGradInterpolationEvaluator(dof_names[0], offset));

     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructScatterResidualEvaluator(true, resid_names,offset, "Scatter Momentum"));
     offset += numDim;
   }
   else if (haveFlow) { // Constant velocity
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<string>("Material Property Name", "Velocity");
    p->set< RCP<DataLayout> >("Data Layout", dl->qp_vector);
    p->set<string>("Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Coordinate Vector Data Layout", dl->qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Flow");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = rcp(new PHAL::NSMaterialProperty<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

   if (haveFlowEq) {
     Teuchos::ArrayRCP<string> dof_names(1);
     Teuchos::ArrayRCP<string> dof_names_dot(1);
     Teuchos::ArrayRCP<string> resid_names(1);
     dof_names[0] = "Pressure";
     dof_names_dot[0] = dof_names[0]+"_dot";
     resid_names[0] = "Continuity Residual";
     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructGatherSolutionEvaluator(false, dof_names, dof_names_dot, offset));

     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructDOFInterpolationEvaluator(dof_names[0], offset));

     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructDOFInterpolationEvaluator(dof_names_dot[0], offset));

     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructDOFGradInterpolationEvaluator(dof_names[0], offset));

     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructScatterResidualEvaluator(false, resid_names,offset, "Scatter Continuity"));
     offset ++;
   }

   if (haveHeatEq) { // Gather Solution Temperature
     Teuchos::ArrayRCP<string> dof_names(1);
     Teuchos::ArrayRCP<string> dof_names_dot(1);
     Teuchos::ArrayRCP<string> resid_names(1);
     dof_names[0] = "Temperature";
     dof_names_dot[0] = dof_names[0]+"_dot";
     resid_names[0] = dof_names[0]+" Residual";
     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructGatherSolutionEvaluator(false, dof_names, dof_names_dot, offset));

     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructDOFInterpolationEvaluator(dof_names[0], offset));

     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructDOFInterpolationEvaluator(dof_names_dot[0], offset));

     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructDOFGradInterpolationEvaluator(dof_names[0], offset));

     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructScatterResidualEvaluator(false, resid_names, offset, "Scatter Temperature"));
     offset ++;
   }
   else if (haveHeat) { // Constant temperature
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<string>("Material Property Name", "Temperature");
    p->set< RCP<DataLayout> >("Data Layout", dl->qp_scalar);
    p->set<string>("Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Coordinate Vector Data Layout", dl->qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Heat");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = rcp(new PHAL::NSMaterialProperty<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

   if (haveNeutEq) { // Gather Solution Neutron
     Teuchos::ArrayRCP<string> dof_names(1);
     Teuchos::ArrayRCP<string> dof_names_dot(1);
     Teuchos::ArrayRCP<string> resid_names(1);
     dof_names[0] = "Neutron Flux";
     dof_names_dot[0] = dof_names[0]+"_dot";
     resid_names[0] = dof_names[0]+" Residual";
     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructGatherSolutionEvaluator(false, dof_names, dof_names_dot, offset));

     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructDOFInterpolationEvaluator(dof_names[0], offset));

     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructDOFInterpolationEvaluator(dof_names_dot[0], offset));

     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructDOFGradInterpolationEvaluator(dof_names[0], offset));

     fm0.template registerEvaluator<EvalT>
       (evalUtils.constructScatterResidualEvaluator(false, resid_names, offset, "Scatter Neutron"));
     offset ++;
   }
   else if (haveNeut) { // Constant neutron flux
     RCP<ParameterList> p = rcp(new ParameterList);

     p->set<string>("Material Property Name", "Neutron Flux");
     p->set< RCP<DataLayout> >("Data Layout", dl->qp_scalar);
     p->set<string>("Coordinate Vector Name", "Coord Vec");
     p->set< RCP<DataLayout> >("Coordinate Vector Data Layout", dl->qp_vector);
     
     p->set<RCP<ParamLib> >("Parameter Library", paramLib);
     Teuchos::ParameterList& paramList = params->sublist("Neutronics");
     p->set<Teuchos::ParameterList*>("Parameter List", &paramList);
     
     ev = rcp(new PHAL::NSMaterialProperty<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
   }

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructGatherCoordinateVectorEvaluator());

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructMapToPhysicalFrameEvaluator(cellType, cubature));

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructComputeBasisFunctionsEvaluator(cellType, intrepidBasis, cubature));

  if (havePSPG || haveSUPG) { // Compute Contravarient Metric Tensor
    RCP<ParameterList> p = 
      rcp(new ParameterList("Contravarient Metric Tensor"));

    // Inputs: X, Y at nodes, Cubature, and Basis
    p->set<string>("Coordinate Vector Name","Coord Vec");
    p->set< RCP<DataLayout> >("Coordinate Data Layout", dl->vertices_vector);
    p->set< RCP<Intrepid::Cubature<RealType> > >("Cubature", cubature);

    p->set<RCP<shards::CellTopology> >("Cell Type", cellType);

    // Outputs: BF, weightBF, Grad BF, weighted-Grad BF, all in physical space
    p->set<string>("Contravarient Metric Tensor Name", "Gc");
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);

    ev = rcp(new PHAL::NSContravarientMetricTensor<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (haveFlowEq || haveHeatEq) { // Density
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<string>("Material Property Name", "Density");
    p->set< RCP<DataLayout> >("Data Layout", dl->qp_scalar);
    p->set<string>("Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Coordinate Vector Data Layout", dl->qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Density");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = rcp(new PHAL::NSMaterialProperty<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (haveFlowEq) { // Viscosity
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<string>("Material Property Name", "Viscosity");
    p->set< RCP<DataLayout> >("Data Layout", dl->qp_scalar);
    p->set<string>("Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Coordinate Vector Data Layout", dl->qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Viscosity");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = rcp(new PHAL::NSMaterialProperty<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (haveHeatEq) { // Specific Heat
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<string>("Material Property Name", "Specific Heat");
     p->set< RCP<DataLayout> >("Data Layout", dl->qp_scalar);
    p->set<string>("Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Coordinate Vector Data Layout", dl->qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Specific Heat");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = rcp(new PHAL::NSMaterialProperty<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (haveHeatEq) { // Thermal conductivity
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<string>("Material Property Name", "Thermal Conductivity");
    p->set< RCP<DataLayout> >("Data Layout", dl->qp_scalar);
    p->set<string>("Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Coordinate Vector Data Layout", dl->qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Thermal Conductivity");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = rcp(new PHAL::NSMaterialProperty<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (haveNeutEq) { // Neutron diffusion
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<string>("Material Property Name", "Neutron Diffusion Coefficient");
    p->set< RCP<DataLayout> >("Data Layout", dl->qp_scalar);
    p->set<string>("Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Coordinate Vector Data Layout", dl->qp_vector);
    p->set<string>("Temperature Variable Name", "Temperature");
    p->set<string>("Absorption Cross Section Name", 
		   "Absorption Cross Section");
    p->set<string>("Scattering Cross Section Name", 
		   "Scattering Cross Section");
    p->set<string>("Average Scattering Angle Name", 
		   "Average Scattering Angle");

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = 
      params->sublist("Neutron Diffusion Coefficient");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = rcp(new PHAL::NSMaterialProperty<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (haveNeutEq) { // Reference temperature
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<string>("Material Property Name", "Reference Temperature");
    p->set< RCP<DataLayout> >("Data Layout", dl->qp_scalar);
    p->set<string>("Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Coordinate Vector Data Layout", dl->qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = 
      params->sublist("Reference Temperature");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = rcp(new PHAL::NSMaterialProperty<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (haveNeutEq) { // Neutron absorption cross section
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<string>("Material Property Name", "Absorption Cross Section");
    p->set< RCP<DataLayout> >("Data Layout", dl->qp_scalar);
    p->set<string>("Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Coordinate Vector Data Layout", dl->qp_vector);
    p->set<string>("Temperature Variable Name", "Temperature");

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = 
      params->sublist("Absorption Cross Section");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = rcp(new PHAL::NSMaterialProperty<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (haveNeutEq) { // Neutron scattering cross section
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<string>("Material Property Name", 
		   "Scattering Cross Section");
    p->set< RCP<DataLayout> >("Data Layout", dl->qp_scalar);
    p->set<string>("Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Coordinate Vector Data Layout", dl->qp_vector);
    p->set<string>("Temperature Variable Name", "Temperature");

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = 
      params->sublist("Scattering Cross Section");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = rcp(new PHAL::NSMaterialProperty<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (haveNeutEq || (haveNeut && haveHeatEq)) { // Neutron fission cross section
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<string>("Material Property Name", "Fission Cross Section");
    p->set< RCP<DataLayout> >("Data Layout", dl->qp_scalar);
    p->set<string>("Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Coordinate Vector Data Layout", dl->qp_vector);
    p->set<string>("Temperature Variable Name", "Temperature");

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = 
      params->sublist("Fission Cross Section");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = rcp(new PHAL::NSMaterialProperty<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (haveNeutEq) { // Neutron released per fission
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<string>("Material Property Name", "Neutrons per Fission");
    p->set< RCP<DataLayout> >("Data Layout", dl->qp_scalar);
    p->set<string>("Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Coordinate Vector Data Layout", dl->qp_vector);
    p->set<string>("Temperature Variable Name", "Temperature");

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = 
      params->sublist("Neutrons per Fission");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = rcp(new PHAL::NSMaterialProperty<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (haveNeutEq) { // Average scattering angle
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<string>("Material Property Name", "Average Scattering Angle");
    p->set< RCP<DataLayout> >("Data Layout", dl->qp_scalar);
    p->set<string>("Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Coordinate Vector Data Layout", dl->qp_vector);
    p->set<string>("Temperature Variable Name", "Temperature");
    p->set("Default Value", 0.0);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = 
      params->sublist("Average Scattering Angle");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = rcp(new PHAL::NSMaterialProperty<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (haveNeut && haveHeatEq) { // Energy released per fission
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<string>("Material Property Name", "Energy Released per Fission");
    p->set< RCP<DataLayout> >("Data Layout", dl->qp_scalar);
    p->set<string>("Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Coordinate Vector Data Layout", dl->qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = 
      params->sublist("Energy Released per Fission");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = rcp(new PHAL::NSMaterialProperty<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (haveFlowEq && haveHeat) { // Volumetric Expansion Coefficient
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<string>("Material Property Name", 
		   "Volumetric Expansion Coefficient");
    p->set< RCP<DataLayout> >("Data Layout", dl->qp_scalar);
    p->set<string>("Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Coordinate Vector Data Layout", dl->qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = 
      params->sublist("Volumetric Expansion Coefficient");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = rcp(new PHAL::NSMaterialProperty<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (porousMedia) { // Porosity, \phi
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<string>("Material Property Name", 
		   "Porosity");
    p->set< RCP<DataLayout> >("Data Layout", dl->qp_scalar);
    p->set<string>("Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Coordinate Vector Data Layout", dl->qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = 
      params->sublist("Porosity");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = rcp(new PHAL::NSMaterialProperty<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (porousMedia) { // Permeability Tensor, K
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<string>("Material Property Name", 
		   "Permeability");
    p->set< RCP<DataLayout> >("Data Layout", dl->qp_scalar);
    p->set<string>("Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Coordinate Vector Data Layout", dl->qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = 
      params->sublist("Permeability");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = rcp(new PHAL::NSMaterialProperty<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (porousMedia) { // Forchheimer Tensor, F
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<string>("Material Property Name", 
		   "Forchheimer");
    p->set< RCP<DataLayout> >("Data Layout", dl->qp_scalar);
    p->set<string>("Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Coordinate Vector Data Layout", dl->qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = 
      params->sublist("Forchheimer");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = rcp(new PHAL::NSMaterialProperty<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  } 

  if (haveHeatEq && haveSource) { // Source
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<string>("Source Name", "Heat Source");
    p->set<string>("Variable Name", "Temperature");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);
    p->set<string>("QP Coordinate Vector Name", "Coord Vec");
    p->set<string>("Neutron Flux Name", "Neutron Flux");
    p->set<string>("Fission Cross Section Name", "Fission Cross Section");
    p->set<string>("Energy Released per Fission Name", 
		   "Energy Released per Fission");

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Source Functions");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = rcp(new PHAL::Source<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (haveNeutEq && haveNeutSource) { // Source
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<string>("Source Name", "Neutron Source");
    p->set<string>("Variable Name", "Neutron Flux");
   p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);
    p->set<string>("QP Coordinate Vector Name", "Coord Vec");

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Neutron Source");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = rcp(new PHAL::Source<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (haveFlowEq) { // Body Force
    RCP<ParameterList> p = rcp(new ParameterList("Body Force"));

    //Input
    p->set<bool>("Have Heat", haveHeat);
    p->set<string>("Temperature QP Variable Name", "Temperature");
    p->set<string>("Density QP Variable Name", "Density");
    p->set<string>("Volumetric Expansion Coefficient QP Variable Name", 
		   "Volumetric Expansion Coefficient");

    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector); 

    Teuchos::ParameterList& paramList = params->sublist("Body Force");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);
  
    //Output
    p->set<string>("Body Force Name", "Body Force");

    ev = rcp(new PHAL::NSBodyForce<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

/*  if (haveFlowEq && haveNeumannx) { // Neumann BC's listed in input file and sidesets are present

    RCP<ParameterList> p = rcp(new ParameterList);
    // cell side
      
    const CellTopologyData * const side_top = elem_top->side[0].topology;

    p->set<string>("Side Set ID", meshSpecs.ssNames[0]);
    
    RCP<shards::CellTopology> sideType = rcp(new shards::CellTopology(side_top));
    RCP <Intrepid::Cubature<RealType> > sideCubature = cubFactory.create(*sideType, meshSpecs.cubatureDegree);
  
    // Inputs: X, Y at nodes, Cubature, and Basis
    p->set<string>("Node Variable Name", "Neumannx");
    p->set<string>("Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Coordinate Data Layout", dl->vertices_vector);
    p->set< RCP<Intrepid::Cubature<RealType> > >("Cubature", cubature);
    
    p->set< RCP<Intrepid::Cubature<RealType> > >("Side Cubature", sideCubature);
    
    p->set< RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >
        ("Intrepid Basis", intrepidBasis);

    p->set<RCP<shards::CellTopology> >("Cell Type", cellType);
    p->set<RCP<shards::CellTopology> >("Side Type", sideType);

    // Output
    p->set< RCP<DataLayout> >("Node Scalar Data Layout", dl->node_scalar);
  
//    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Neumann BCs");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);
    
    ev = rcp(new PHAL::Neumann<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

 if (haveFlowEq && haveNeumanny) { // Neumann BC's listed in input file and sidesets are present

    RCP<ParameterList> p = rcp(new ParameterList);
    // cell side

    const CellTopologyData * const side_top = elem_top->side[0].topology;

    p->set<string>("Side Set ID", meshSpecs.ssNames[0]);

    RCP<shards::CellTopology> sideType = rcp(new shards::CellTopology(side_top));
    RCP <Intrepid::Cubature<RealType> > sideCubature = cubFactory.create(*sideType, meshSpecs.cubatureDegree);

    // Inputs: X, Y at nodes, Cubature, and Basis
    p->set<string>("Node Variable Name", "Neumanny");
    p->set<string>("Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Coordinate Data Layout", dl->vertices_vector);
    p->set< RCP<Intrepid::Cubature<RealType> > >("Cubature", cubature);

    p->set< RCP<Intrepid::Cubature<RealType> > >("Side Cubature", sideCubature);

    p->set< RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >
        ("Intrepid Basis", intrepidBasis);

    p->set<RCP<shards::CellTopology> >("Cell Type", cellType);
    p->set<RCP<shards::CellTopology> >("Side Type", sideType);

    // Output
    p->set< RCP<DataLayout> >("Node Scalar Data Layout", dl->node_scalar);

//    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Neumann BCs");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = rcp(new PHAL::Neumann<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (haveFlowEq && haveNeumannz) { // Neumann BC's listed in input file and sidesets are present

    RCP<ParameterList> p = rcp(new ParameterList);
    // cell side

    const CellTopologyData * const side_top = elem_top->side[0].topology;

    p->set<string>("Side Set ID", meshSpecs.ssNames[0]);

    RCP<shards::CellTopology> sideType = rcp(new shards::CellTopology(side_top));
    RCP <Intrepid::Cubature<RealType> > sideCubature = cubFactory.create(*sideType, meshSpecs.cubatureDegree);

    // Inputs: X, Y at nodes, Cubature, and Basis
    p->set<string>("Node Variable Name", "Neumannz");
    p->set<string>("Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Coordinate Data Layout", dl->vertices_vector);
    p->set< RCP<Intrepid::Cubature<RealType> > >("Cubature", cubature);

    p->set< RCP<Intrepid::Cubature<RealType> > >("Side Cubature", sideCubature);

    p->set< RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >
        ("Intrepid Basis", intrepidBasis);

    p->set<RCP<shards::CellTopology> >("Cell Type", cellType);
    p->set<RCP<shards::CellTopology> >("Side Type", sideType);

    // Output
    p->set< RCP<DataLayout> >("Node Scalar Data Layout", dl->node_scalar);

//    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Neumann BCs");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = rcp(new PHAL::Neumann<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
   }
*/

   if (porousMedia) { // Permeability term in momentum equation
    std::cout <<"This is flow through porous media problem" << std::endl;
    RCP<ParameterList> p = rcp(new ParameterList("Permeability Term"));

    //Input
    p->set<string>("Velocity QP Variable Name", "Velocity");
    p->set<string>("Density QP Variable Name", "Density");
    p->set<string>("Viscosity QP Variable Name", "Viscosity");
    p->set<string>("Porosity QP Variable Name", "Porosity");
    p->set<string>("Permeability QP Variable Name", "Permeability");

    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
  
    //Output
    p->set<string>("Permeability Term", "Permeability Term");

    ev = rcp(new PHAL::NSPermeabilityTerm<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  } 

  if (porousMedia) { // Forchheimer term in momentum equation
    RCP<ParameterList> p = rcp(new ParameterList("Forchheimer Term"));

    //Input
    p->set<string>("Velocity QP Variable Name", "Velocity");
    p->set<string>("Density QP Variable Name", "Density");
    p->set<string>("Viscosity QP Variable Name", "Viscosity");
    p->set<string>("Porosity QP Variable Name", "Porosity");
    p->set<string>("Permeability QP Variable Name", "Permeability");
    p->set<string>("Forchheimer QP Variable Name", "Forchheimer");

    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
  
    //Output
    p->set<string>("Forchheimer Term", "Forchheimer Term");

    ev = rcp(new PHAL::NSForchheimerTerm<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  } 

  if (haveFlowEq) { // Rm
    RCP<ParameterList> p = rcp(new ParameterList("Rm"));

    //Input
    p->set<string>("Velocity QP Variable Name", "Velocity");
    p->set<string>("Velocity Dot QP Variable Name", "Velocity_dot");
    p->set<string>("Velocity Gradient QP Variable Name", "Velocity Gradient");
    p->set<string>("Pressure Gradient QP Variable Name", "Pressure Gradient");
    p->set<string>("Density QP Variable Name", "Density");
    p->set<string>("Body Force QP Variable Name", "Body Force");
    p->set<string>("Porosity QP Variable Name", "Porosity");
    p->set<string>("Permeability Term", "Permeability Term");
    p->set<string>("Forchheimer Term", "Forchheimer Term");
 
    p->set<bool>("Porous Media", porousMedia);

    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
  
    //Output
    p->set<string>("Rm Name", "Rm");

    ev = rcp(new PHAL::NSRm<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (haveFlowEq && (haveSUPG || havePSPG)) { // Tau M
    RCP<ParameterList> p = rcp(new ParameterList("Tau M"));

    //Input
    p->set<string>("Velocity QP Variable Name", "Velocity");
    p->set<std::string>("Contravarient Metric Tensor Name", "Gc"); 
    p->set<string>("Density QP Variable Name", "Density");
    p->set<string>("Viscosity QP Variable Name", "Viscosity");

    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);

    //Output
    p->set<string>("Tau M Name", "Tau M");

    ev = rcp(new PHAL::NSTauM<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (haveHeatEq && haveFlow && haveSUPG) { // Tau T
    RCP<ParameterList> p = rcp(new ParameterList("Tau T"));

    //Input
    p->set<string>("Velocity QP Variable Name", "Velocity");
    p->set<std::string>("Contravarient Metric Tensor Name", "Gc"); 
    p->set<string>("Thermal Conductivity Name", "Thermal Conductivity");
    p->set<string>("Density QP Variable Name", "Density");
    p->set<string>("Specific Heat QP Variable Name", "Specific Heat");

    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);

    //Output
    p->set<string>("Tau T Name", "Tau T");

    ev = rcp(new PHAL::NSTauT<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (haveFlowEq) { // Momentum Resid
    RCP<ParameterList> p = rcp(new ParameterList("Momentum Resid"));

    //Input
    p->set<string>("Weighted BF Name", "wBF");
    p->set<string>("Weighted Gradient BF Name", "wGrad BF");
    p->set<string>("Velocity Gradient QP Variable Name", "Velocity Gradient");
    p->set<string>("Pressure QP Variable Name", "Pressure");
    p->set<string>("Pressure Gradient QP Variable Name", "Pressure Gradient");
    p->set<string>("Viscosity QP Variable Name", "Viscosity");
    p->set<string>("Rm Name", "Rm");

    p->set<bool>("Have SUPG", haveSUPG);
    p->set<string>("Velocity QP Variable Name", "Velocity");
    p->set<string>("Density QP Variable Name", "Density");
    p->set<string> ("Tau M Name", "Tau M");

    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);    
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", dl->node_qp_scalar);
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);
    
    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
  
    //Output
    p->set<string>("Residual Name", "Momentum Residual");
    p->set< RCP<DataLayout> >("Node Vector Data Layout", dl->node_vector);

    ev = rcp(new PHAL::NSMomentumResid<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (haveFlowEq) { // Continuity Resid
    RCP<ParameterList> p = rcp(new ParameterList("Continuity Resid"));

    //Input
    p->set<string>("Weighted BF Name", "wBF");
    p->set<string>("Gradient QP Variable Name", "Velocity Gradient");
    p->set<string>("Density QP Variable Name", "Density");

    p->set<bool>("Have PSPG", havePSPG);
    p->set<string>("Weighted Gradient BF Name", "wGrad BF");
    p->set<std::string> ("Tau M Name", "Tau M");
    p->set<std::string> ("Rm Name", "Rm");

    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", dl->node_qp_scalar);
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);
    p->set< RCP<PHX::DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<PHX::DataLayout> >("QP Vector Data Layout", dl->qp_vector);
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);

    //Output
    p->set<string>("Residual Name", "Continuity Residual");
    p->set< RCP<DataLayout> >("Node Scalar Data Layout", dl->node_scalar);

    ev = rcp(new PHAL::NSContinuityResid<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

   if (haveHeatEq) { // Temperature Resid
    RCP<ParameterList> p = rcp(new ParameterList("Temperature Resid"));

    //Input
    p->set<string>("Weighted BF Name", "wBF");
    p->set<string>("Weighted Gradient BF Name", "wGrad BF");
    p->set<string>("QP Variable Name", "Temperature");
    p->set<string>("QP Time Derivative Variable Name", "Temperature_dot");
    p->set<string>("Gradient QP Variable Name", "Temperature Gradient");
    p->set<string>("Density QP Variable Name", "Density");
    p->set<string>("Specific Heat QP Variable Name", "Specific Heat");
    p->set<string>("Thermal Conductivity Name", "Thermal Conductivity");
    
    p->set<bool>("Have Source", haveSource);
    p->set<string>("Source Name", "Heat Source");

    p->set<bool>("Have Flow", haveFlow);
    p->set<string>("Velocity QP Variable Name", "Velocity");
    
    p->set<bool>("Have SUPG", haveSUPG);
    p->set<string> ("Tau T Name", "Tau T");
 
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", dl->node_qp_scalar);
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);

    //Output
    p->set<string>("Residual Name", "Temperature Residual");
    p->set< RCP<DataLayout> >("Node Scalar Data Layout", dl->node_scalar);

    ev = rcp(new PHAL::NSThermalEqResid<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

   if (haveNeutEq) { // Neutron Resid
    RCP<ParameterList> p = rcp(new ParameterList("Neutron Resid"));

    //Input
    p->set<string>("Weighted BF Name", "wBF");
    p->set<string>("Weighted Gradient BF Name", "wGrad BF");
    p->set<string>("QP Variable Name", "Neutron Flux");
    p->set<string>("Gradient QP Variable Name", "Neutron Flux Gradient");
    p->set<string>("Neutron Diffusion Name", "Neutron Diffusion Coefficient");
    p->set<string>("Neutron Absorption Name", "Absorption Cross Section");
    p->set<string>("Neutron Fission Name", "Fission Cross Section");
    p->set<string>("Neutrons per Fission Name", "Neutrons per Fission");
    
    p->set<bool>("Have Neutron Source", haveNeutSource);
    p->set<string>("Source Name", "Neutron Source");
 
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", dl->node_qp_scalar);
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);

    //Output
    p->set<string>("Residual Name", "Neutron Flux Residual");
    p->set< RCP<DataLayout> >("Node Scalar Data Layout", dl->node_scalar);

    ev = rcp(new PHAL::NSNeutronEqResid<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (fieldManagerChoice == Albany::BUILD_RESID_FM)  {
    Teuchos::RCP<const PHX::FieldTag> ret_tag;
    if (haveFlowEq) {
      PHX::Tag<typename EvalT::ScalarT> mom_tag("Scatter Momentum", dl->dummy);
      fm0.requireField<EvalT>(mom_tag);
      PHX::Tag<typename EvalT::ScalarT> con_tag("Scatter Continuity", dl->dummy);
      fm0.requireField<EvalT>(con_tag);
      ret_tag = mom_tag.clone();
    }
    if (haveHeatEq) {
      PHX::Tag<typename EvalT::ScalarT> heat_tag("Scatter Temperature", dl->dummy);
      fm0.requireField<EvalT>(heat_tag);
      ret_tag = heat_tag.clone();
    }
    if (haveNeutEq) {
      PHX::Tag<typename EvalT::ScalarT> neut_tag("Scatter Neutron", dl->dummy);
      fm0.requireField<EvalT>(neut_tag);
      ret_tag = neut_tag.clone();
    }
    return ret_tag;
  }
  else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM) {
    Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
    return respUtils.constructResponses(fm0, *responseList, Teuchos::null, stateMgr);
  }

  return Teuchos::null;
}
#endif // ALBANY_NAVIERSTOKES_HPP
