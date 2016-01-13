//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef HYDRIDEMORPHPROBLEM_HPP
#define HYDRIDEMORPHPROBLEM_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_AbstractProblem.hpp"

#include "Phalanx.hpp"
#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"
#include "Albany_ProblemUtils.hpp"

#include "QCAD_MaterialDatabase.hpp"

namespace Albany {

  /*!
   * \brief Abstract interface for representing a 1-D finite element
   * problem.
   */
  class HydMorphProblem : public AbstractProblem {
  public:

    //! Default constructor
    HydMorphProblem(const Teuchos::RCP<Teuchos::ParameterList>& params,
		const Teuchos::RCP<ParamLib>& paramLib,
		const int numDim_,
                Teuchos::RCP<const Teuchos::Comm<int> >& commT_); 

    //! Destructor
    ~HydMorphProblem();

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
    HydMorphProblem(const HydMorphProblem&);

    //! Private to prohibit copying
    HydMorphProblem& operator=(const HydMorphProblem&);

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

    bool haveHeatSource;
    int numDim;

    Teuchos::RCP<QCAD::MaterialDatabase> materialDB;
    Teuchos::RCP<const Teuchos::Comm<int> >commT; 

    Teuchos::RCP<Albany::Layouts> dl;

  };

}

#include "Intrepid2_FieldContainer.hpp"
#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_EvaluatorUtils.hpp"
#include "Albany_ResponseUtilities.hpp"

#include "PHAL_ThermalConductivity.hpp"
#include "JThermConductivity.hpp"
#include "PHAL_Source.hpp"
//#include "PHAL_Neumann.hpp"
#include "PHAL_HeatEqResid.hpp"
#include "HydFractionResid.hpp"


template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
Albany::HydMorphProblem::constructEvaluators(
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

   const CellTopologyData * const elem_top = &meshSpecs.ctd;

   RCP<Intrepid2::Basis<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > >
     intrepidBasis = Albany::getIntrepid2Basis(*elem_top);
   RCP<shards::CellTopology> cellType = rcp(new shards::CellTopology (elem_top));


   const int numNodes = intrepidBasis->getCardinality();
   const int worksetSize = meshSpecs.worksetSize;

   Intrepid2::DefaultCubatureFactory<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > cubFactory;
   RCP <Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout,PHX::Device> > > cellCubature = cubFactory.create(*cellType, meshSpecs.cubatureDegree);

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

// The coupled heat and hydrogen diffusion equations

  Teuchos::ArrayRCP<std::string> dof_names(neq);
  Teuchos::ArrayRCP<std::string> dof_names_dot(neq);
  Teuchos::ArrayRCP<std::string> resid_names(neq);

  dof_names[0] = "Temperature";
  dof_names_dot[0] = "Temperature_dot";
  resid_names[0] = "Temperature Residual";

  dof_names[1] = "HydFraction";
  dof_names_dot[1] = "HydFraction_dot";
  resid_names[1] = "HydFraction Residual";

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructGatherSolutionEvaluator(false, dof_names, dof_names_dot));

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructScatterResidualEvaluator(false, resid_names));

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructGatherCoordinateVectorEvaluator());

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructMapToPhysicalFrameEvaluator( cellType, cellCubature));

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructComputeBasisFunctionsEvaluator(cellType, intrepidBasis, cellCubature));

  for (unsigned int i=0; i<neq; i++) {
    fm0.template registerEvaluator<EvalT>
      (evalUtils.constructDOFInterpolationEvaluator(dof_names[i]));

    fm0.template registerEvaluator<EvalT>
      (evalUtils.constructDOFInterpolationEvaluator(dof_names_dot[i]));

    fm0.template registerEvaluator<EvalT>
      (evalUtils.constructDOFGradInterpolationEvaluator(dof_names[i]));
  }

  { // Thermal conductivity
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<std::string>("QP Variable Name", "Thermal Conductivity");
    p->set<std::string>("QP Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Node Data Layout", dl->node_scalar);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Thermal Conductivity");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    // Here we assume that the instance of this problem applies on a single element block
    p->set<std::string>("Element Block Name", meshSpecs.ebName);

    if(materialDB != Teuchos::null)
      p->set< RCP<QCAD::MaterialDatabase> >("MaterialDB", materialDB);

    ev = rcp(new PHAL::ThermalConductivity<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

// Check and see if a source term is specified for this problem in the main input file.
  bool problemSpecifiesASource = params->isSublist("Source Functions");

  if(problemSpecifiesASource){

      // Sources the same everywhere if they are present at all

      haveHeatSource = true;
      RCP<ParameterList> p = rcp(new ParameterList);

      p->set<std::string>("Source Name", "Source");
      p->set<std::string>("Variable Name", "Temperature");
      p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

      p->set<RCP<ParamLib> >("Parameter Library", paramLib);
      Teuchos::ParameterList& paramList = params->sublist("Source Functions");
      p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

      ev = rcp(new PHAL::Source<EvalT,AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);

  }
  else if(materialDB != Teuchos::null){ // Sources can be specified in terms of materials or element blocks

      // Is the source function active for "this" element block?

      haveHeatSource =  materialDB->isElementBlockSublist(meshSpecs.ebName, "Source Functions");

      if(haveHeatSource){

        RCP<ParameterList> p = rcp(new ParameterList);

        p->set<std::string>("Source Name", "Source");
        p->set<std::string>("Variable Name", "Temperature");
        p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

        p->set<RCP<ParamLib> >("Parameter Library", paramLib);
        Teuchos::ParameterList& paramList = materialDB->getElementBlockSublist(meshSpecs.ebName, "Source Functions");
        p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

        ev = rcp(new PHAL::Source<EvalT,AlbanyTraits>(*p));
        fm0.template registerEvaluator<EvalT>(ev);
    }
  }

  { // Temperature Resid
    RCP<ParameterList> p = rcp(new ParameterList("Temperature Resid"));

    //Input
    p->set<std::string>("Weighted BF Name", "wBF");
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", dl->node_qp_scalar);
    p->set<std::string>("QP Variable Name", "Temperature");

    p->set<std::string>("QP Time Derivative Variable Name", "Temperature_dot");

    p->set<bool>("Have Source", haveHeatSource);
    p->set<bool>("Have Absorption", false);
    p->set<std::string>("Source Name", "Source");

    p->set<std::string>("Thermal Conductivity Name", "Thermal Conductivity");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

    p->set<std::string>("Gradient QP Variable Name", "Temperature Gradient");
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);
    if (params->isType<std::string>("Convection Velocity"))
    	p->set<std::string>("Convection Velocity",
                       params->get<std::string>("Convection Velocity"));
    if (params->isType<bool>("Have Rho Cp"))
    	p->set<bool>("Have Rho Cp", params->get<bool>("Have Rho Cp"));

    //Output
    p->set<std::string>("Residual Name", "Temperature Residual");
    p->set< RCP<DataLayout> >("Node Scalar Data Layout", dl->node_scalar);

    ev = rcp(new PHAL::HeatEqResid<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  { // The coefficient in front of Grad T
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<std::string>("Temperature Name", "Temperature");
    p->set<std::string>("QP Variable Name", "J Conductivity");

    Teuchos::ParameterList& paramList = params->sublist("Material Parameters");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    // Here we assume that the instance of this problem applies on a single element block
    p->set<std::string>("Element Block Name", meshSpecs.ebName);

    if(materialDB != Teuchos::null)
      p->set< RCP<QCAD::MaterialDatabase> >("MaterialDB", materialDB);

    ev = rcp(new PHAL::JThermConductivity<EvalT,AlbanyTraits>(*p, dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  { // Hydrogen Concentration Resid
    RCP<ParameterList> p = rcp(new ParameterList("Hydrogen Concentration Resid"));

    //Input
    p->set<std::string>("Weighted BF Name", "wBF");
    p->set<std::string>("Temperature Name", "Temperature");
    p->set<std::string>("Temp Time Derivative Name", "Temperature_dot");
    p->set<std::string>("QP Time Derivative Variable Name", "HydFraction_dot");
    p->set<std::string>("J Conductivity Name", "J Conductivity");
    p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
    p->set<std::string>("QP Variable Name", "HydFraction");

    Teuchos::ParameterList& paramList = params->sublist("Material Parameters");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    // Here we assume that the instance of this problem applies on a single element block
    p->set<std::string>("Element Block Name", meshSpecs.ebName);

    if(materialDB != Teuchos::null)
      p->set< RCP<QCAD::MaterialDatabase> >("MaterialDB", materialDB);

    //Output
    p->set<std::string>("Residual Name", "HydFraction Residual");

    ev = rcp(new PHAL::HydFractionResid<EvalT,AlbanyTraits>(*p, dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (fieldManagerChoice == Albany::BUILD_RESID_FM)  {
    PHX::Tag<typename EvalT::ScalarT> res_tag("Scatter", dl->dummy);
    fm0.requireField<EvalT>(res_tag);
    return res_tag.clone();
  }

  else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM) {
    Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
    return respUtils.constructResponses(fm0, *responseList, Teuchos::null, stateMgr);
  }

  return Teuchos::null;
}


#endif // ALBANY_HYDMORPHPROBLEM_HPP
