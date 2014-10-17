//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_STOKESFOPROBLEM_HPP
#define FELIX_STOKESFOPROBLEM_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_AbstractProblem.hpp"

#include "Phalanx.hpp"
#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"
//#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_SaveStateField.hpp"
#include "PHAL_LoadStateField.hpp"

//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN

namespace FELIX {

  /*!
   * \brief Abstract interface for representing a 1-D finite element
   * problem.
   */
  class StokesFO : public Albany::AbstractProblem {
  public:
  
    //! Default constructor
    StokesFO(const Teuchos::RCP<Teuchos::ParameterList>& params,
		 const Teuchos::RCP<ParamLib>& paramLib,
		 const int numDim_);

    //! Destructor
    ~StokesFO();

    //! Return number of spatial dimensions
    virtual int spatialDimension() const { return numDim; }

    //! Build the PDE instantiations, boundary conditions, and initial solution
    virtual void buildProblem(
      Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
      Albany::StateManager& stateMgr);

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
    StokesFO(const StokesFO&);
    
    //! Private to prohibit copying
    StokesFO& operator=(const StokesFO&);

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

#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"

#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_EvaluatorUtils.hpp"
#include "Albany_ResponseUtilities.hpp"

#include "FELIX_StokesFOResid.hpp"
#include "FELIX_ViscosityFO.hpp"
#include "FELIX_StokesFOBodyForce.hpp"
#include "PHAL_Neumann.hpp"
#include "PHAL_Source.hpp"


template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
FELIX::StokesFO::constructEvaluators(
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
  
  RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > >
    intrepidBasis = Albany::getIntrepidBasis(meshSpecs.ctd);
  RCP<shards::CellTopology> cellType = rcp(new shards::CellTopology (&meshSpecs.ctd));
  
  const int numNodes = intrepidBasis->getCardinality();
  const int worksetSize = meshSpecs.worksetSize;
  
  Intrepid::DefaultCubatureFactory<RealType> cubFactory;
  RCP <Intrepid::Cubature<RealType> > cubature = cubFactory.create(*cellType, meshSpecs.cubatureDegree);
  
  const int numQPts = cubature->getNumPoints();
  const int numVertices = cellType->getNodeCount();
  int vecDim = neq;
  std::string elementBlockName = meshSpecs.ebName;
  
#ifdef OUTPUT_TO_SCREEN
  *out << "Field Dimensions: Workset=" << worksetSize 
       << ", Vertices= " << numVertices
       << ", Nodes= " << numNodes
       << ", QuadPts= " << numQPts
       << ", Dim= " << numDim 
       << ", vecDim= " << vecDim << std::endl;
#endif
  
   Albany::StateStruct::MeshFieldEntity entity;
   dl = rcp(new Albany::Layouts(worksetSize,numVertices,numNodes,numQPts,numDim, vecDim));
   Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
   int offset=0;

   entity= Albany::StateStruct::ElemData;

   // Temporary variable used numerous times below
      RCP<PHX::Evaluator<AlbanyTraits> > ev;
   {
     std::string stateName("temperature");
     RCP<ParameterList> p = stateMgr.registerStateVariable(stateName, dl->cell_scalar2, elementBlockName,true, &entity);
     ev = rcp(new PHAL::LoadStateField<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
   }

   {
     std::string stateName("flow_factor");
     RCP<ParameterList> p = stateMgr.registerStateVariable(stateName, dl->cell_scalar2, elementBlockName,true, &entity);
     ev = rcp(new PHAL::LoadStateField<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
   }

   entity= Albany::StateStruct::NodalDataToElemNode;

   {
     std::string stateName("surface_height");
     RCP<ParameterList> p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName,true, &entity);
     ev = rcp(new PHAL::LoadStateField<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
   }

   {
     std::string stateName("thickness");
     RCP<ParameterList> p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName,true, &entity);
     ev = rcp(new PHAL::LoadStateField<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
   }

   entity= Albany::StateStruct::NodalDistParameter;

   {
     std::string stateName("basal_friction");
     const std::string& meshPart = this->params->sublist("Distributed Parameters").get("Mesh Part","");
     RCP<ParameterList> p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName,true, &entity, meshPart);
    }



   // Define Field Names

  Teuchos::ArrayRCP<std::string> dof_names(1);
  Teuchos::ArrayRCP<std::string> dof_names_dot(1);
  Teuchos::ArrayRCP<std::string> resid_names(1);
  dof_names[0] = "Velocity";
  //dof_names_dot[0] = dof_names[0]+"_dot";
  resid_names[0] = "Stokes Residual";
  fm0.template registerEvaluator<EvalT>
  (evalUtils.constructGatherSolutionEvaluator_noTransient(true, dof_names, offset));

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructDOFVecInterpolationEvaluator(dof_names[0]));

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructDOFVecGradInterpolationEvaluator(dof_names[0]));

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructScatterResidualEvaluator(true, resid_names,offset, "Scatter Stokes"));
  offset += numDim;

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructGatherCoordinateVectorEvaluator());

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructMapToPhysicalFrameEvaluator(cellType, cubature));

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructComputeBasisFunctionsEvaluator(cellType, intrepidBasis, cubature));

  std::string sh = "surface_height";
  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructDOFGradInterpolationEvaluator_noDeriv(sh));

  { // FO Stokes Resid
    RCP<ParameterList> p = rcp(new ParameterList("Stokes Resid"));
   
    //Input
    p->set<std::string>("Weighted BF Name", "wBF");
    p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
    p->set<std::string>("QP Variable Name", "Velocity");
    p->set<std::string>("QP Time Derivative Variable Name", "Velocity_dot");
    p->set<std::string>("Gradient QP Variable Name", "Velocity Gradient");
    p->set<std::string>("Body Force Name", "Body Force");
    p->set<std::string>("FELIX Viscosity QP Variable Name", "FELIX Viscosity");
    
    Teuchos::ParameterList& paramList = params->sublist("Equation Set");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    //Output
    p->set<std::string>("Residual Name", "Stokes Residual");

    ev = rcp(new FELIX::StokesFOResid<EvalT,AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }
  { // FELIX viscosity
    RCP<ParameterList> p = rcp(new ParameterList("FELIX Viscosity"));

    //Input
    p->set<std::string>("Coordinate Vector Name", "Coord Vec");
    p->set<std::string>("Gradient QP Variable Name", "Velocity Gradient");
    p->set<std::string>("temperature Name", "temperature");
    p->set<std::string>("flow_factor Name", "flow_factor");
    
    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("FELIX Viscosity");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);
  
    //Output
    p->set<std::string>("FELIX Viscosity QP Variable Name", "FELIX Viscosity");

    ev = rcp(new FELIX::ViscosityFO<EvalT,AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
    
  }

  { // Body Force
    RCP<ParameterList> p = rcp(new ParameterList("Body Force"));

    //Input
    p->set<std::string>("FELIX Viscosity QP Variable Name", "FELIX Viscosity");
    p->set<std::string>("Coordinate Vector Name", "Coord Vec");
    p->set<std::string>("surface_height Gradient Name", "surface_height Gradient");
    
    Teuchos::ParameterList& paramList = params->sublist("Body Force");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);
      

    //Output
    p->set<std::string>("Body Force Name", "Body Force");

    ev = rcp(new FELIX::StokesFOBodyForce<EvalT,AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  RCP<ParameterList> paramList = rcp(new ParameterList("Param List"));
  { // response
    RCP<const Albany::MeshSpecsStruct> meshSpecsPtr = Teuchos::rcpFromRef(meshSpecs);
    paramList->set<RCP<const Albany::MeshSpecsStruct> >("Mesh Specs Struct", meshSpecsPtr);
    paramList->set<RCP<ParamLib> >("Parameter Library", paramLib);
  }

  if (fieldManagerChoice == Albany::BUILD_RESID_FM)  {
    PHX::Tag<typename EvalT::ScalarT> res_tag("Scatter Stokes", dl->dummy);
    fm0.requireField<EvalT>(res_tag);
  }
  else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM) {
  
    {
      std::string stateName("surface_velocity");
      RCP<ParameterList> p = stateMgr.registerStateVariable(stateName, dl->node_vector, elementBlockName,true,&entity);
      ev = rcp(new PHAL::LoadStateField<EvalT,AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }

    {
      std::string stateName("surface_velocity_rms");
      RCP<ParameterList> p = stateMgr.registerStateVariable(stateName, dl->node_vector, elementBlockName,true,&entity);
      ev = rcp(new PHAL::LoadStateField<EvalT,AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
     }

    Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
    return respUtils.constructResponses(fm0, *responseList, paramList, stateMgr);
  }

  return Teuchos::null;
}
#endif // FELIX_STOKESFOPROBLEM_HPP
