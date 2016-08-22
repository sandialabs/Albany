//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_POPULATE_MESH_HPP
#define FELIX_POPULATE_MESH_HPP 1

#include <type_traits>

#include "Phalanx.hpp"
#include "Intrepid2_FieldContainer.hpp"
#include "Shards_CellTopology.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_AbstractProblem.hpp"
#include "Albany_EvaluatorUtils.hpp"

#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_Utilities.hpp"

#include "FELIX_DummyResidual.hpp"

namespace FELIX
{

/*!
 * \brief A dummy problem to load ascii fields into the mesh.

 *  This is a dummy problem, whose only goal is to make sure input fields are read
 *  from file and stuffed and saved in the mesh for future use.
 *  Note: the mesh is populated while Albany::Application is created, so the
 *        execution that follows could be anything (Solve, Analysis,...).
 *        In any case, the 'residual' of the problem is DummyResidual, whic
 *        sets the residual equal to the solution.
 */
class PopulateMesh : public Albany::AbstractProblem
{
public:

  //! Default constructor
  PopulateMesh (const Teuchos::RCP<Teuchos::ParameterList>& params_,
                const Teuchos::RCP<Teuchos::ParameterList>& discParams_,
                const Teuchos::RCP<ParamLib>& paramLib_);

  //! Destructor
  ~PopulateMesh();

  //! Return number of spatial dimensions
  virtual int spatialDimension() const { return 0; }

  //! Build the PDE instantiations, boundary conditions, and initial solution
  virtual void buildProblem (Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
                             Albany::StateManager& stateMgr);

  // Build evaluators
  virtual Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
  buildEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                   const Albany::MeshSpecsStruct& meshSpecs,
                   Albany::StateManager& stateMgr,
                   Albany::FieldManagerChoice fmchoice,
                   const Teuchos::RCP<Teuchos::ParameterList>& responseList);

  //! Each problem must generate it's list of valide parameters
  Teuchos::RCP<const Teuchos::ParameterList> getValidProblemParameters() const;

private:

  //! Private to prohibit copying
  PopulateMesh(const PopulateMesh&);

  //! Private to prohibit copying
  PopulateMesh& operator=(const PopulateMesh&);

public:

  //! Main problem setup routine. Not directly called, but indirectly by following functions
  template <typename EvalT> Teuchos::RCP<const PHX::FieldTag>
  constructEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                       const Albany::MeshSpecsStruct& meshSpecs,
                       Albany::StateManager& stateMgr,
                       Albany::FieldManagerChoice fmchoice,
                       const Teuchos::RCP<Teuchos::ParameterList>& responseList);

  void constructDirichletEvaluators(const Albany::MeshSpecsStruct& meshSpecs);
  void constructNeumannEvaluators(const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs);

protected:

  typedef Teuchos::RCP<shards::CellTopology>                                                                                  topologyType;
  typedef Teuchos::RCP<Intrepid2::Basis<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > >    basisType;

  basisType     cellBasis;
  topologyType  cellTopology;

  std::map<std::string,basisType>     sideBasis;
  std::map<std::string,topologyType>  sideTopology;

  Teuchos::RCP<Albany::Layouts> dl;

  std::string                       cellEBName;
  std::map<std::string,std::string> sideEBName;

  //! Discretization parameters
  Teuchos::RCP<Teuchos::ParameterList> discParams;
};

} // Namespace FELIX

// ================================ IMPLEMENTATION ============================ //

template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
FELIX::PopulateMesh::constructEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                          const Albany::MeshSpecsStruct& meshSpecs,
                                          Albany::StateManager& stateMgr,
                                          Albany::FieldManagerChoice fieldManagerChoice,
                                          const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);

  Albany::StateStruct::MeshFieldEntity entity;
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;
  Teuchos::RCP<Teuchos::ParameterList> p;

  // ---------------------------- Registering state variables ------------------------- //

  // Map string to StateStruct::MeshFieldEntity
  std::map<std::string,Albany::StateStruct::MeshFieldEntity> str2mfe;
  str2mfe["Node Scalar"] = Albany::StateStruct::NodalDataToElemNode;
  str2mfe["Node Vector"] = Albany::StateStruct::NodalDataToElemNode;
  str2mfe["Elem Scalar"] = Albany::StateStruct::ElemData;
  str2mfe["Elem Vector"] = Albany::StateStruct::ElemData;
  str2mfe["Node Layered Scalar"] = Albany::StateStruct::NodalDataToElemNode;
  str2mfe["Node Layered Vector"] = Albany::StateStruct::NodalDataToElemNode;
  str2mfe["Elem Layered Scalar"] = Albany::StateStruct::ElemData;
  str2mfe["Elem Layered Vector"] = Albany::StateStruct::ElemData;

  std::string fname, flayout;
  if (discParams->isSublist("Required Fields Info"))
  {
    // Map string to PHX layout
    std::map<std::string,Teuchos::RCP<PHX::DataLayout>> str2dl;
    str2dl["Node Scalar"] = dl->node_scalar;
    str2dl["Node Vector"] = dl->node_vector;
    str2dl["Elem Scalar"] = dl->cell_scalar2;
    str2dl["Elem Vector"] = dl->cell_vector;


    Teuchos::ParameterList& req_fields_info = discParams->sublist("Required Fields Info");
    int num_fields = req_fields_info.get<int>("Number Of Fields",0);
    for (int ifield=0; ifield<num_fields; ++ifield)
    {
      const Teuchos::ParameterList& thisFieldList =  req_fields_info.sublist(Albany::strint("Field", ifield));

      fname   = thisFieldList.get<std::string>("Field Name");
      flayout = thisFieldList.get<std::string>("Field Layout");

      if (flayout.find("Layered")!=std::string::npos)
      {
        Teuchos::RCP<PHX::DataLayout> ldl;
        int numLayers = thisFieldList.get<int>("Number Of Layers");
        if (flayout=="Node Layered Scalar")
          ldl = PHAL::ExtendLayout<LayerDim,Cell,Node>::apply(dl->node_scalar,numLayers);
        else if (flayout=="Node Layered Vector")
          ldl = PHAL::ExtendLayout<LayerDim,Cell,Node,Dim>::apply(dl->node_vector,numLayers);
        else if (flayout=="Elem Layered Scalar")
          ldl = PHAL::ExtendLayout<LayerDim,Cell>::apply(dl->cell_scalar2,numLayers);
        else if (flayout=="Elem Layered Vector")
          ldl = PHAL::ExtendLayout<LayerDim,Cell,Dim>::apply(dl->cell_vector,numLayers);
        else
          TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, "Error! Invalid layout for field '" << fname << "'.\n");

        p = stateMgr.registerStateVariable(fname, ldl, cellEBName, true, &str2mfe[flayout]);
      }
      else
        p = stateMgr.registerStateVariable(fname, str2dl[flayout], cellEBName, true, &str2mfe[flayout]);
    }
  }

  if (discParams->isSublist("Side Set Discretizations"))
  {
    Teuchos::ParameterList& ss_disc_pl = discParams->sublist("Side Set Discretizations");
    const Teuchos::Array<std::string>& ss_names = ss_disc_pl.get<Teuchos::Array<std::string>>("Side Sets");

    for (int is=0; is<ss_names.size(); ++is)
    {
      const std::string& ss_name = ss_names[is];
      Teuchos::ParameterList& this_ss_pl = ss_disc_pl.sublist(ss_name);

      if (this_ss_pl.isSublist("Required Fields Info"))
      {
        Teuchos::RCP<Albany::Layouts> sdl = dl->side_layouts[ss_name];

        // Map string to PHX layout
        std::map<std::string,Teuchos::RCP<PHX::DataLayout>> str2dl;
        str2dl["Node Scalar"] = sdl->node_scalar;
        str2dl["Node Vector"] = sdl->node_vector;
        str2dl["Elem Scalar"] = sdl->cell_scalar2;
        str2dl["Elem Vector"] = sdl->cell_vector;

        Teuchos::ParameterList& req_fields_info = this_ss_pl.sublist("Required Fields Info");
        int num_fields = req_fields_info.get<int>("Number Of Fields",0);
        for (int ifield=0; ifield<num_fields; ++ifield)
        {
          const Teuchos::ParameterList& thisFieldList =  req_fields_info.sublist(Albany::strint("Field", ifield));

          fname   = thisFieldList.get<std::string>("Field Name");
          flayout = thisFieldList.get<std::string>("Field Layout");

          if (flayout.find("Layered")!=std::string::npos)
          {
            Teuchos::RCP<PHX::DataLayout> ldl;
            int numLayers = thisFieldList.get<int>("Number Of Layers");
            if (flayout=="Node Layered Scalar")
              ldl = PHAL::ExtendLayout<LayerDim,Cell,Side,Node>::apply(sdl->node_scalar,numLayers);
            else if (flayout=="Node Layered Vector")
              ldl = PHAL::ExtendLayout<LayerDim,Cell,Side,Node,Dim>::apply(sdl->node_vector,numLayers);
            else if (flayout=="Elem Layered Scalar")
              ldl = PHAL::ExtendLayout<LayerDim,Cell,Side>::apply(sdl->cell_scalar2,numLayers);
            else if (flayout=="Elem Layered Vector")
              ldl = PHAL::ExtendLayout<LayerDim,Cell,Side,Dim>::apply(sdl->cell_vector,numLayers);
            else
              TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, "Error! Invalid layout for field '" << fname << "'.\n");

            p = stateMgr.registerSideSetStateVariable(ss_name, fname, fname, ldl, sideEBName[ss_name], true, &str2mfe[flayout]);
          }
          else
          {
            p = stateMgr.registerSideSetStateVariable(ss_name, fname, fname, str2dl[flayout], sideEBName[ss_name], true, &str2mfe[flayout]);
          }
        }
      }
    }
  }

  // ------------------- Computing and Scattering a Dummy Residual ------------------ //

  // Residual and solution names
  Teuchos::ArrayRCP<std::string> dof_names(1), resid_names(1);
  dof_names[0] = "Solution";
  resid_names[0] = "Residual";
  int offset = 0;

  // --- Gather solution field --- //
  ev = evalUtils.constructGatherSolutionEvaluator_noTransient(false, dof_names);
  fm0.template registerEvaluator<EvalT> (ev);

  // --- Scatter residual --- //
  ev = evalUtils.constructScatterResidualEvaluator(false, resid_names, offset, "Scatter Dummy Residual");
  fm0.template registerEvaluator<EvalT> (ev);

  // --- Dummy Residual --- //
  p = Teuchos::rcp(new Teuchos::ParameterList("Dummy Residual"));
  p->set<std::string>("Solution Variable Name", dof_names[0]);
  p->set<std::string>("Residual Variable Name", resid_names[0]);
  ev = Teuchos::rcp(new FELIX::DummyResidual<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  if (fieldManagerChoice == Albany::BUILD_RESID_FM)
  {
    // Require scattering of residual
    PHX::Tag<typename EvalT::ScalarT> res_tag("Scatter Dummy Residual", dl->dummy);
    fm0.requireField<EvalT>(res_tag);
  }

  return Teuchos::null;
}

#endif // FELIX_POPULATE_MESH_HPP
