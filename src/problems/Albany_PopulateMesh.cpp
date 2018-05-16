//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <string>

#include "Shards_CellTopology.hpp"

#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_PopulateMesh.hpp"

namespace Albany
{

PopulateMesh::PopulateMesh (const Teuchos::RCP<Teuchos::ParameterList>& params_,
                            const Teuchos::RCP<Teuchos::ParameterList>& discParams_,
                            const Teuchos::RCP<ParamLib>& paramLib_) :
  AbstractProblem(params_, paramLib_),
  discParams(discParams_),
  use_sdbcs_(false)
{
  neq = 1;

  // Set the num PDEs for the null space object to pass to ML
  this->rigidBodyModes->setNumPDEs(neq);
}

PopulateMesh::~PopulateMesh()
{
  // Nothing to be done here
}

void PopulateMesh::buildProblem (Teuchos::ArrayRCP<Teuchos::RCP<MeshSpecsStruct>> meshSpecs,
                                 StateManager& stateMgr)
{
  Intrepid2::DefaultCubatureFactory   cubFactory;

  // Building cell type, basis and cubature
  const CellTopologyData * const cell_top_data = &meshSpecs[0]->ctd;

  cellEBName    = meshSpecs[0]->ebName;
  cellTopology  = Teuchos::rcp(new shards::CellTopology (cell_top_data));
  cellBasis     = getIntrepid2Basis(*cell_top_data);
  cellCubature  = cubFactory.create<PHX::Device, RealType, RealType>(*cellTopology, meshSpecs[0]->cubatureDegree);

  const int worksetSize     = meshSpecs[0]->worksetSize;
  const int numCellSides    = cellTopology->getFaceCount();
  const int numCellVertices = cellTopology->getNodeCount();
  const int numCellNodes    = cellBasis->getCardinality();
  const int numCellQPs      = cellCubature->getNumPoints();
  const int numCellDim      = meshSpecs[0]->numDim;
  const int numCellVecDim   = -1;

  dl = Teuchos::rcp(new Layouts(worksetSize,numCellVertices,numCellNodes,numCellQPs,numCellDim,numCellVecDim));

  if (discParams->isSublist("Side Set Discretizations"))
  {
    Teuchos::ParameterList& ss_disc_pl = discParams->sublist("Side Set Discretizations");
    const Teuchos::Array<std::string>& ss_names = ss_disc_pl.get<Teuchos::Array<std::string>>("Side Sets");
    for (auto ss_name : ss_names)
    {
      Teuchos::ParameterList& this_ss_pl = ss_disc_pl.sublist(ss_name);

      const MeshSpecsStruct& ssMeshSpecs = *meshSpecs[0]->sideSetMeshSpecs.at(ss_name)[0];

      // Building also side structures
      const CellTopologyData * const side_top_data = &ssMeshSpecs.ctd;

      sideEBName[ss_name]   = meshSpecs[0]->ebName;
      sideTopology[ss_name] = Teuchos::rcp(new shards::CellTopology (side_top_data));
      sideBasis[ss_name]    = getIntrepid2Basis(*side_top_data);
      sideCubature[ss_name] = cubFactory.create<PHX::Device, RealType, RealType>(*sideTopology[ss_name], ssMeshSpecs.cubatureDegree);

      const int numSideVertices = sideTopology[ss_name]->getNodeCount();
      const int numSideNodes    = sideBasis[ss_name]->getCardinality();
      const int numSideDim      = ssMeshSpecs.numDim;
      const int numSideQPs      = sideCubature[ss_name]->getNumPoints();
      const int numSideVecDim   = -1;

      dl->side_layouts[ss_name] = Teuchos::rcp(new Layouts(worksetSize,numSideVertices,numSideNodes,numSideQPs,
                                                           numSideDim,numCellDim,numCellSides,numSideVecDim));
    }
  }

  // ---------------------------- Registering state variables ------------------------- //

  StateStruct::MeshFieldEntity entity;
  Teuchos::RCP<Teuchos::ParameterList> p;

  std::string fname, flayout;
  Teuchos::ParameterList& req_fields_info = discParams->sublist("Required Fields Info");
  int num_fields = req_fields_info.get<int>("Number Of Fields",0);
  for (int ifield=0; ifield<num_fields; ++ifield)
  {
    const Teuchos::ParameterList& thisFieldList =  req_fields_info.sublist(strint("Field", ifield));

    fname   = thisFieldList.get<std::string>("Field Name");
    flayout = thisFieldList.get<std::string>("Field Type");

    bool is_nodal   = flayout.find("Node")!=std::string::npos;
    bool is_vector  = flayout.find("Vector")!=std::string::npos;
    bool is_layered = flayout.find("Layered")!=std::string::npos;

    entity = is_nodal ? StateStruct::NodalDataToElemNode : StateStruct::ElemData;

    // Incrementally build the layout
    Teuchos::RCP<PHX::DataLayout> layout;

    // Node vs cell
    if (is_nodal)
      layout = dl->node_scalar;
    else
      layout = dl->cell_scalar2;

    // Vector fields
    if (is_vector)
    {
      int vec_dim = thisFieldList.get<int>("Vector Dim");
      layout = is_nodal ? PHAL::ExtendLayout<Dim,Cell,Node>::apply(layout,vec_dim)
                        : PHAL::ExtendLayout<Dim,Cell>::apply(layout,vec_dim);
    }

    // Layered fields
    if (is_layered)
    {
      int num_layers = thisFieldList.get<int>("Number Of Layers");
      layout = is_vector
                  ? (is_nodal ? PHAL::ExtendLayout<LayerDim,Cell,Node,Dim>::apply(layout,num_layers)
                              : PHAL::ExtendLayout<LayerDim,Cell,Dim>::apply(layout,num_layers))
                  : (is_nodal ? PHAL::ExtendLayout<LayerDim,Cell,Node>::apply(layout,num_layers)
                              : PHAL::ExtendLayout<LayerDim,Cell>::apply(layout,num_layers));
    }

    // Finally, register the state
    p = stateMgr.registerStateVariable(fname, layout, cellEBName, true, &entity);
  }

  if (discParams->isSublist("Side Set Discretizations"))
  {
    Teuchos::ParameterList& ss_disc_pl = discParams->sublist("Side Set Discretizations");
    const Teuchos::Array<std::string>& ss_names = ss_disc_pl.get<Teuchos::Array<std::string>>("Side Sets");
    for (auto ss_name : ss_names)
    {
      Teuchos::ParameterList& this_ss_pl = ss_disc_pl.sublist(ss_name);
      Teuchos::ParameterList& req_fields_info = this_ss_pl.sublist("Required Fields Info");
      Teuchos::RCP<Layouts> sdl = dl->side_layouts[ss_name];

      int num_fields = req_fields_info.get<int>("Number Of Fields",0);

      for (int ifield=0; ifield<num_fields; ++ifield)
      {
        const Teuchos::ParameterList& thisFieldList =  req_fields_info.sublist(strint("Field", ifield));

        fname   = thisFieldList.get<std::string>("Field Name");
        flayout = thisFieldList.get<std::string>("Field Type");

        bool is_nodal   = flayout.find("Node")!=std::string::npos;
        bool is_vector  = flayout.find("Vector")!=std::string::npos;
        bool is_layered = flayout.find("Layered")!=std::string::npos;

        entity = is_nodal ? StateStruct::NodalDataToElemNode : StateStruct::ElemData;

        // Incrementally build the layout
        Teuchos::RCP<PHX::DataLayout> layout;

        // Node vs cell
        if (is_nodal)
          layout = sdl->node_scalar;
        else
          layout = sdl->cell_scalar2;

        // Vector fields
        if (is_vector)
        {
          int vec_dim = thisFieldList.get<int>("Vector Dim");
          layout = is_nodal ? PHAL::ExtendLayout<Dim,Cell,Side,Node>::apply(layout,vec_dim)
                            : PHAL::ExtendLayout<Dim,Cell,Side>::apply(layout,vec_dim);
        }

        // Layered fields
        if (is_layered)
        {
          int num_layers = thisFieldList.get<int>("Number Of Layers");
          layout = is_vector
                      ? (is_nodal ? PHAL::ExtendLayout<LayerDim,Cell,Side,Node,Dim>::apply(layout,num_layers)
                                  : PHAL::ExtendLayout<LayerDim,Cell,Side,Dim>::apply(layout,num_layers))
                      : (is_nodal ? PHAL::ExtendLayout<LayerDim,Cell,Side,Node>::apply(layout,num_layers)
                                  : PHAL::ExtendLayout<LayerDim,Cell,Side>::apply(layout,num_layers));
        }

        // Finally, register the state
        p = stateMgr.registerSideSetStateVariable(ss_name, fname, fname, layout, sideEBName[ss_name], true, &entity);
      }
    }
  }


  /* Construct All Phalanx Evaluators */
  TEUCHOS_TEST_FOR_EXCEPTION(meshSpecs.size()!=1,std::logic_error,"Problem supports one Material Block");
  fm.resize(1);
  fm[0]  = Teuchos::rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
  buildEvaluators(*fm[0], *meshSpecs[0], stateMgr, BUILD_RESID_FM,Teuchos::null);
}

Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
PopulateMesh::buildEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                               const MeshSpecsStruct& meshSpecs,
                               StateManager& stateMgr,
                               FieldManagerChoice fmchoice,
                               const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructeEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  ConstructEvaluatorsOp<PopulateMesh> op(
    *this, fm0, meshSpecs, stateMgr, fmchoice, responseList);
  Sacado::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes> fe(op);
  return *op.tags;
}

Teuchos::RCP<const Teuchos::ParameterList>
PopulateMesh::getValidProblemParameters () const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = this->getGenericProblemParams("ValidPopulateMeshProblemParams");
  return validPL;
}

} // Namespace Albany
