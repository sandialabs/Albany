/*
 * LandIce_ThicknessEvol.hpp
 *
 *  Created on: Sep 10, 2026
 *      Author: mperego
 */

#ifndef LANDICE_THICKNESSEVOLUTION_PROBLEM_HPP
#define LANDICE_THICKNESSEVOLUTION_PROBLEM_HPP

#include "Albany_Utils.hpp"
#include "Albany_EvaluatorUtils.hpp"
#include "Albany_GeneralPurposeFieldsNames.hpp"

#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"
#include "PHAL_AlbanyTraits.hpp"

#include "LandIce_Gather2DField.hpp"
#include "LandIce_ScatterResidual2D.hpp"
#include "LandIce_SimpleOperationEvaluator.hpp"
#include "LandIce_UpdateZCoordinate.hpp"
#include "LandIce_ThicknessResidCell.hpp"
#include "PHAL_GatherCoordinateVector.hpp"  

#include "LandIce_ParamEnum.hpp"
#include "LandIce_ResponseUtilities.hpp"

#include "Albany_AbstractProblem.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_ScalarOrdinalTypes.hpp"
#include "Albany_FieldUtils.hpp"
#include "Albany_StringUtils.hpp"

#include "PHAL_LoadStateField.hpp"
#include "PHAL_SaveStateField.hpp"
#include "PHAL_LoadSideSetStateField.hpp"
#include "PHAL_ScatterScalarNodalParameter.hpp"
#include "PHAL_SharedParameter.hpp"

#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include <set>

namespace LandIce
{

  class ThicknessEvolution : public Albany::AbstractProblem
  {
  public:
    //! Default constructor
    ThicknessEvolution(const Teuchos::RCP<Teuchos::ParameterList>& params,
             const Teuchos::RCP<Teuchos::ParameterList>& discParams,
             const Teuchos::RCP<ParamLib>& paramLib,
             const int numDim_);

    //! Destructor
    ~ThicknessEvolution();

    //! Return number of spatial dimensions
    virtual int spatialDimension() const { return numDim; }

    //! Get boolean telling code if SDBCs are utilized
    virtual bool useSDBCs() const {return use_sdbcs_; }

    //! Build the PDE instantiations, boundary conditions, and initial solution
    virtual void buildProblem(Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
                              Albany::StateManager& stateMgr);

    //! Build evaluators
    virtual Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> > buildEvaluators(PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                                                                const Albany::MeshSpecsStruct& meshSpecs,
                                                                                Albany::StateManager& stateMgr,
                                                                                Albany::FieldManagerChoice fmchoice,
                                                                                const Teuchos::RCP<Teuchos::ParameterList>& responseList);
    //! Build unmanaged fields
    void buildFields(PHX::FieldManager<PHAL::AlbanyTraits>& fm0);

    //! Each problem must generate its list of valid parameters
    Teuchos::RCP<const Teuchos::ParameterList> getValidProblemParameters() const;

    //! Main problem setup routine. Not directly called, but indirectly by following functions
    template <typename EvalT> Teuchos::RCP<const PHX::FieldTag>
    constructEvaluators(PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                        const Albany::MeshSpecsStruct& meshSpecs,
                        Albany::StateManager& stateMgr,
                        Albany::FieldManagerChoice fmchoice,
                        const Teuchos::RCP<Teuchos::ParameterList>& responseList);

    template <typename EvalT>
    void constructFields(PHX::FieldManager<PHAL::AlbanyTraits>& fm0);

    void constructDirichletEvaluators(const Albany::MeshSpecsStruct& meshSpecs);
    void constructNeumannEvaluators(const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs);

  protected:
    Teuchos::RCP<shards::CellTopology> cellType;
    Teuchos::RCP<shards::CellTopology> sideType;

    Teuchos::RCP<Intrepid2::Cubature<PHX::Device> >  cellCubature;
    Teuchos::RCP<Intrepid2::Cubature<PHX::Device> >  sideCubature;

    Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> > cellBasis;
    Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> > sideBasis;

    unsigned int numDim;
    Teuchos::RCP<Albany::Layouts> dl, dl_side;
    std::string elementBlockName;
    std::string lateralSideName;

    std::string ice_thickness_name, surface_height_name, initial_ice_thickness_name;
    int vecDim;

    bool unsteady;
    Teuchos::ArrayRCP<std::string> dof_names, dof_names_dot;
    Teuchos::ArrayRCP<int>         dof_offsets;
    Teuchos::ArrayRCP<std::string> resid_names;
    Teuchos::ArrayRCP<std::string> scatter_names;

    //! Discretization parameters
    Teuchos::RCP<Teuchos::ParameterList> discParams;

    std::set<std::string> volumeFields;

    // Storage for unmanaged fields
    Teuchos::RCP<Albany::FieldUtils> fieldUtils;

    /// Boolean marking whether SDBCs are used
    bool use_sdbcs_;
  };

} // end of the namespace LandIce

// ================================ IMPLEMENTATION ============================ //
template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
LandIce::ThicknessEvolution::constructEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                      const Albany::MeshSpecsStruct& meshSpecs ,
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

  using FRT = Albany::FieldRankType;

  Albany::StateStruct::MeshFieldEntity entity;

  Teuchos::RCP<ParameterList> p;

  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);

  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;

/*
  {

      Teuchos::ParameterList& info = discParams->sublist("Required Fields Info");
      int num_fields = info.get<int>("Number Of Fields",0);

      for (int ifield=0; ifield<num_fields; ++ifield) {
        Teuchos::ParameterList& thisFieldList =  info.sublist(util::strint("Field", ifield));

        // Get current state specs
        stateName = thisFieldList.get<std::string>("Field Name");
        entity = Albany::StateStruct::NodalDataToElemNode;
        if(stateName=="velocity")       
          p = stateMgr.registerStateVariable(stateName, fieldName, dl->node_vector, elementBlockName, true, &entity, "");
        else
          p = stateMgr.registerStateVariable(stateName, fieldName, dl->node_scalar, elementBlockName, true, &entity, "");
        ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
        fm0.template registerEvaluator<EvalT>(ev);
      }
  }
*/
  // Velocity
  {
    entity = Albany::StateStruct::NodalDataToElemNode;
    std::string stateName = "velocity";
    p = stateMgr.registerStateVariable(stateName, dl->node_vector, elementBlockName, true, &entity, "");
    ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // Surface Height
  {
    entity = Albany::StateStruct::NodalDataToElemNode;
    std::string stateName = "forcing";
    p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName,true, &entity);
    ev = Teuchos::rcp(new PHAL::LoadStateFieldMST<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // Initial Ice Thickness
  {
    entity = Albany::StateStruct::NodalDataToElemNode;
    std::string stateName = initial_ice_thickness_name;
    p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName,true, &entity);
    ev = Teuchos::rcp(new PHAL::LoadStateFieldMST<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }



  // --- Thickness Resid --- //
  p = Teuchos::rcp(new Teuchos::ParameterList("Thickness Resid"));

  //Input
  p->set<bool>("Unsteady", unsteady);
  if(unsteady) {
    p->set<std::string>("Thickness Dot Variable Name", dof_names_dot[0]);
  } else {
    if(this->params->isParameter("Time Step Ptr")) {
      p->set<Teuchos::RCP<double> >("Time Step Ptr", this->params->get<Teuchos::RCP<double> >("Time Step Ptr"));
    } else {
      Teuchos::RCP<double> dt = Teuchos::rcp(new double(this->params->get<double>("Time Step")));
      p->set<Teuchos::RCP<double> >("Time Step Ptr", dt);
    }
  }

  p->set<std::string>("Thickness Change Variable Name", dof_names[0]);
  p->set<std::string>("Initial Thickness Name", initial_ice_thickness_name);
  p->set<std::string>("Coordinate Vector Name", Albany::coord_vec_name);
  p->set<int>("Cubature Degree",4);
  p->set<Teuchos::RCP<const Albany::MeshSpecsStruct> >("Mesh Specs Struct", Teuchos::rcpFromRef(meshSpecs));
  p->set<std::string>("Velocity Name", "velocity");
  p->set<std::string>("Forcing Name", "forcing");
  p->set<std::string>("Stabilization", this->params->get<std::string>("Thickness Stabilization", "None"));
  p->set<bool>("Lump Mass Matrix", this->params->get<bool>("Lump Time Derivative Mass Matrix", false));
  p->set<std::string>("Lateral Side Set Name", lateralSideName);

  //Output
  p->set<std::string>("Residual Name", resid_names[0]);

  ev = Teuchos::rcp(new LandIce::ThicknessResidCell<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);


  int offset = 0;

  // --- Interpolation and utilities ---
  // ThicknessEvolution
  {


    if(unsteady)
      fm0.template registerEvaluator<EvalT> (evalUtils.constructGatherSolutionEvaluator(false, dof_names[0], dof_names_dot[0], offset));
    else
      fm0.template registerEvaluator<EvalT> (evalUtils.constructGatherSolutionEvaluator_noTransient(false, dof_names[0], offset));

    fm0.template registerEvaluator<EvalT> (evalUtils.constructScatterResidualEvaluator(false, resid_names, offset, "Scatter ThicknessEvolution"));

    fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFInterpolationEvaluator(dof_names[0], offset));

    fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFGradInterpolationEvaluator(dof_names[0], offset));
  }

  fm0.template registerEvaluator<EvalT> (evalUtils.constructGatherCoordinateVectorEvaluator());

  fm0.template registerEvaluator<EvalT> (evalUtils.constructMapToPhysicalFrameEvaluator(cellType, cellCubature));

  fm0.template registerEvaluator<EvalT> (evalUtils.constructComputeBasisFunctionsEvaluator(cellType, cellBasis, cellCubature));

  fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructDOFVecInterpolationEvaluator("velocity"));

  fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructDOFVecGradInterpolationEvaluator("velocity"));

  fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructDOFVecInterpolationEvaluator("ice_thickness"));

  fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructDOFVecGradInterpolationEvaluator("ice_thickness"));



  //--- LandIce Stokes FO Residual Thickness ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("Scatter ResidualH"));

  //Input
  p->set<std::string>("Residual Name", resid_names[0]);
  p->set<int>("Tensor Rank", 0);
  p->set<int>("Offset of First DOF", dof_offsets[0]);
  p->set<Teuchos::RCP<const shards::CellTopology> >("Cell Topology",cellType);

  //Output
  p->set<std::string>("Scatter Field Name", scatter_names[0]);

  ev = Teuchos::rcp(new PHAL::ScatterResidual<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  if (fieldManagerChoice == Albany::BUILD_RESID_FM) {
    // Require scattering of residual
    PHX::Tag<typename EvalT::ScalarT> scatterTag(scatter_names[0], dl->dummy);
    fm0.requireField<EvalT>(scatterTag);
  }

  return Teuchos::null;
}

template <typename EvalT>
void
LandIce::ThicknessEvolution::constructFields(PHX::FieldManager<PHAL::AlbanyTraits> &/* fm0 */)
{
  fieldUtils->setComputeBasisFunctionsFields<EvalT>();
}

#endif /* LANDICE_THICKNESSEVOLUTION_PROBLEM_HPP */
