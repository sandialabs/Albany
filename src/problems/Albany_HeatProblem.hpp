//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_HEATPROBLEM_HPP
#define ALBANY_HEATPROBLEM_HPP

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
  class HeatProblem : public AbstractProblem {
  public:

    //! Default constructor
    HeatProblem(
      const Teuchos::RCP<Teuchos::ParameterList>& params,
      const Teuchos::RCP<ParamLib>& paramLib,
      //const Teuchos::RCP<DistParamLib>& distParamLib,
      const int numDim_,
      const Teuchos::RCP<const Epetra_Comm>& comm_);

    //! Destructor
    ~HeatProblem();

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
    HeatProblem(const HeatProblem&);

    //! Private to prohibit copying
    HeatProblem& operator=(const HeatProblem&);

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

    //! Boundary conditions on source term
    bool periodic;
    bool haveSource;
    bool haveAbsorption;
    int numDim;

   Teuchos::RCP<QCAD::MaterialDatabase> materialDB;
   Teuchos::RCP<const Epetra_Comm> comm;

   Teuchos::RCP<Albany::Layouts> dl;

  };

}

#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_EvaluatorUtils.hpp"
#include "Albany_ResponseUtilities.hpp"

#include "PHAL_ThermalConductivity.hpp"
#include "PHAL_Absorption.hpp"
#include "PHAL_Source.hpp"
//#include "PHAL_Neumann.hpp"
#include "PHAL_HeatEqResid.hpp"


template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
Albany::HeatProblem::constructEvaluators(
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

   RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > >
     intrepidBasis = Albany::getIntrepidBasis(*elem_top);
   RCP<shards::CellTopology> cellType = rcp(new shards::CellTopology (elem_top));


   const int numNodes = intrepidBasis->getCardinality();
   const int worksetSize = meshSpecs.worksetSize;

   Intrepid::DefaultCubatureFactory<RealType> cubFactory;
   RCP <Intrepid::Cubature<RealType> > cellCubature = cubFactory.create(*cellType, meshSpecs.cubatureDegree);

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

   Teuchos::ArrayRCP<string> dof_names(neq);
     dof_names[0] = "Temperature";
   Teuchos::ArrayRCP<string> dof_names_dot(neq);
     dof_names_dot[0] = "Temperature_dot";
   Teuchos::ArrayRCP<string> resid_names(neq);
     resid_names[0] = "Temperature Residual";

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

    p->set<string>("QP Variable Name", "Thermal Conductivity");
    p->set<string>("QP Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Node Data Layout", dl->node_scalar);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    //p->set<RCP<DistParamLib> >("Distributed Parameter Library", distParamLib);
    Teuchos::ParameterList& paramList = params->sublist("Thermal Conductivity");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    // Here we assume that the instance of this problem applies on a single element block
    p->set<string>("Element Block Name", meshSpecs.ebName);

    if(materialDB != Teuchos::null)
      p->set< RCP<QCAD::MaterialDatabase> >("MaterialDB", materialDB);

    ev = rcp(new PHAL::ThermalConductivity<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (haveAbsorption) { // Absorption
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<string>("QP Variable Name", "Absorption");
    p->set<string>("QP Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Absorption");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = rcp(new PHAL::Absorption<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

// Check and see if a source term is specified for this problem in the main input file.
  bool problemSpecifiesASource = params->isSublist("Source Functions");

  if(problemSpecifiesASource){

      // Sources the same everywhere if they are present at all

      haveSource = true;
      RCP<ParameterList> p = rcp(new ParameterList);

      p->set<string>("Source Name", "Source");
      p->set<string>("Variable Name", "Temperature");
      p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

      p->set<RCP<ParamLib> >("Parameter Library", paramLib);
      Teuchos::ParameterList& paramList = params->sublist("Source Functions");
      p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

      ev = rcp(new PHAL::Source<EvalT,AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);

  }
  else if(materialDB != Teuchos::null){ // Sources can be specified in terms of materials or element blocks

      // Is the source function active for "this" element block?

      haveSource =  materialDB->isElementBlockSublist(meshSpecs.ebName, "Source Functions");

      if(haveSource){

        RCP<ParameterList> p = rcp(new ParameterList);

        p->set<string>("Source Name", "Source");
        p->set<string>("Variable Name", "Temperature");
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
    p->set<string>("Weighted BF Name", "wBF");
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", dl->node_qp_scalar);
    p->set<string>("QP Variable Name", "Temperature");

    p->set<string>("QP Time Derivative Variable Name", "Temperature_dot");

    p->set<bool>("Have Source", haveSource);
    p->set<bool>("Have Absorption", haveAbsorption);
    p->set<string>("Source Name", "Source");

    p->set<string>("Thermal Conductivity Name", "Thermal Conductivity");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

    p->set<string>("Absorption Name", "Thermal Conductivity");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

    p->set<string>("Gradient QP Variable Name", "Temperature Gradient");
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    p->set<string>("Weighted Gradient BF Name", "wGrad BF");
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);
    if (params->isType<string>("Convection Velocity"))
        p->set<string>("Convection Velocity",
                       params->get<string>("Convection Velocity"));
    if (params->isType<bool>("Have Rho Cp"))
        p->set<bool>("Have Rho Cp", params->get<bool>("Have Rho Cp"));

    //Output
    p->set<string>("Residual Name", "Temperature Residual");
    p->set< RCP<DataLayout> >("Node Scalar Data Layout", dl->node_scalar);

    ev = rcp(new PHAL::HeatEqResid<EvalT,AlbanyTraits>(*p));
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


#endif // ALBANY_HEATNONLINEARSOURCEPROBLEM_HPP
