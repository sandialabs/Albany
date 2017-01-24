//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <string>

#include "Shards_CellTopology.hpp"

#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "FELIX_PopulateMesh.hpp"

namespace FELIX
{

PopulateMesh::PopulateMesh (const Teuchos::RCP<Teuchos::ParameterList>& params_,
                            const Teuchos::RCP<Teuchos::ParameterList>& discParams_,
                            const Teuchos::RCP<ParamLib>& paramLib_) :
  Albany::AbstractProblem(params_, paramLib_),
  discParams(discParams_)
{
  neq = 1;

  // Set the num PDEs for the null space object to pass to ML
  this->rigidBodyModes->setNumPDEs(neq);

  Teuchos::Array<std::string> empty_str_ar;
  Teuchos::Array<int> empty_int_ar;

  // Need to allocate a fields in mesh database
  Teuchos::Array<std::string> req = params->get<Teuchos::Array<std::string> > ("Required Fields",empty_str_ar);
  for (int i(0); i<req.size(); ++i)
    this->requirements.push_back(req[i]);

  Teuchos::ParameterList& p = params->sublist("Side Sets Requirements");

  Teuchos::Array<std::string> ss_names = p.get<Teuchos::Array<std::string>>("Side Sets Names",empty_str_ar);
  Teuchos::Array<int> ss_vec_dims_ar = p.get<Teuchos::Array<int>>("Side Sets Vec Dims",empty_int_ar);

  TEUCHOS_TEST_FOR_EXCEPTION (ss_names.size()!=ss_vec_dims_ar.size(), Teuchos::Exceptions::InvalidParameter,
                              "Error! You must specify a vector dimension for each side set.\n");

  for (int i=0; i<ss_names.size(); ++i)
  {
    ss_vec_dims[ss_names[i]] = ss_vec_dims_ar[i];

    Teuchos::Array<std::string> reqs = p.get<Teuchos::Array<std::string>>(ss_names[i]);

    for (int j=0; j<reqs.size(); ++j)
      this->ss_requirements[ss_names[i]].push_back(reqs[j]);
  }
}

PopulateMesh::~PopulateMesh()
{
  // Nothing to be done here
}

void PopulateMesh::buildProblem (Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct>> meshSpecs,
                                 Albany::StateManager& stateMgr)
{
  Intrepid2::DefaultCubatureFactory   cubFactory;

  // Building cell type, basis and cubature
  const CellTopologyData * const cell_top_data = &meshSpecs[0]->ctd;

  cellEBName    = meshSpecs[0]->ebName;
  cellTopology  = Teuchos::rcp(new shards::CellTopology (cell_top_data));
  cellBasis     = Albany::getIntrepid2Basis(*cell_top_data);
  cellCubature  = cubFactory.create<PHX::Device, RealType, RealType>(*cellTopology, meshSpecs[0]->cubatureDegree);

  const int worksetSize     = meshSpecs[0]->worksetSize;
  const int numCellSides    = cellTopology->getFaceCount();
  const int numCellVertices = cellTopology->getNodeCount();
  const int numCellNodes    = cellBasis->getCardinality();
  const int numCellQPs      = cellCubature->getNumPoints();
  const int numCellDim      = meshSpecs[0]->numDim;
  const int numCellVecDim   = params->get<int>("Cell Vec Dim",numCellDim);

  dl = Teuchos::rcp(new Albany::Layouts(worksetSize,numCellVertices,numCellNodes,numCellQPs,numCellDim,numCellVecDim));

  if (this->ss_requirements.size()>0)
  {
    TEUCHOS_TEST_FOR_EXCEPTION (!discParams->isSublist("Side Set Discretizations"), std::logic_error,
                                "Error! There are side set requirements in the problem section, but no side discretizations.\n");

    Teuchos::ParameterList& ss_disc_pl = discParams->sublist("Side Set Discretizations");
    const Teuchos::Array<std::string>& ss_names = ss_disc_pl.get<Teuchos::Array<std::string>>("Side Sets");

    for (auto it : this->ss_requirements)
    {
      const std::string& ss_name = it.first;
      TEUCHOS_TEST_FOR_EXCEPTION (!ss_disc_pl.isSublist(ss_name),std::logic_error,
                                  "Error! Side set '" << ss_name << "' is listed in the problem section but is missing in the discretization section.\n");

      Teuchos::ParameterList& this_ss_pl = ss_disc_pl.sublist(ss_name);

      const Albany::MeshSpecsStruct& ssMeshSpecs = *meshSpecs[0]->sideSetMeshSpecs.at(ss_name)[0];

      // Building also side structures
      const CellTopologyData * const side_top_data = &ssMeshSpecs.ctd;

      sideEBName[ss_name]   = meshSpecs[0]->ebName;
      sideTopology[ss_name] = Teuchos::rcp(new shards::CellTopology (side_top_data));
      sideBasis[ss_name]    = Albany::getIntrepid2Basis(*side_top_data);
      sideCubature[ss_name] = cubFactory.create<PHX::Device, RealType, RealType>(*sideTopology[ss_name], ssMeshSpecs.cubatureDegree);

      const int numSideVertices = sideTopology[ss_name]->getNodeCount();
      const int numSideNodes    = sideBasis[ss_name]->getCardinality();
      const int numSideDim      = ssMeshSpecs.numDim;
      const int numSideQPs      = sideCubature[ss_name]->getNumPoints();
      const int numSideVecDim   = ss_vec_dims[ss_name];

      dl->side_layouts[ss_name] = Teuchos::rcp(new Albany::Layouts(worksetSize,numSideVertices,numSideNodes,numSideQPs,
                                                                   numSideDim,numCellDim,numCellSides,numSideVecDim));
    }
  }

  // ---------------------------- Registering state variables ------------------------- //

  Albany::StateStruct::MeshFieldEntity entity;
  Teuchos::RCP<Teuchos::ParameterList> p;

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
  if (this->requirements.size()>0)
  {
    // Map string to PHX layout
    std::map<std::string,Teuchos::RCP<PHX::DataLayout>> str2dl;
    str2dl["Node Scalar"] = dl->node_scalar;
    str2dl["Node Vector"] = dl->node_vector;
    str2dl["Elem Scalar"] = dl->cell_scalar2;
    str2dl["Elem Vector"] = dl->cell_vector;


    Teuchos::ParameterList& req_fields_info = discParams->sublist("Required Fields Info");
    int num_fields = req_fields_info.get<int>("Number Of Fields",0);

    TEUCHOS_TEST_FOR_EXCEPTION (num_fields!=this->requirements.size(), std::logic_error,
                                "Error! The input 'Number Of Fields' in the mesh in the discretization section " <<
                                "does not match the number of requirements specified in the problem section.\n");

    std::map<std::string, bool> found;
    for (int ifield=0; ifield<num_fields; ++ifield)
      found[this->requirements[ifield]] = false;

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

    for (auto found_it : found)
      TEUCHOS_TEST_FOR_EXCEPTION(found_it.second==false, std::runtime_error,
                                 "Error! The requirement '" << found_it.first << "' was not found in the discretization section.\n");
  }

  if (this->ss_requirements.size()>0)
  {
    Teuchos::ParameterList& ss_disc_pl = discParams->sublist("Side Set Discretizations");
    const Teuchos::Array<std::string>& ss_names = ss_disc_pl.get<Teuchos::Array<std::string>>("Side Sets");

    for (auto it : this->ss_requirements)
    {
      const std::string& ss_name = it.first;
      Teuchos::ParameterList& this_ss_pl = ss_disc_pl.sublist(ss_name);
      Teuchos::ParameterList& req_fields_info = this_ss_pl.sublist("Required Fields Info");
      Teuchos::RCP<Albany::Layouts> sdl = dl->side_layouts[ss_name];

      // Map string to PHX layout
      std::map<std::string,Teuchos::RCP<PHX::DataLayout>> str2dl;
      str2dl["Node Scalar"] = sdl->node_scalar;
      str2dl["Node Vector"] = sdl->node_vector;
      str2dl["Elem Scalar"] = sdl->cell_scalar2;
      str2dl["Elem Vector"] = sdl->cell_vector;

      int num_fields = req_fields_info.get<int>("Number Of Fields",0);
      TEUCHOS_TEST_FOR_EXCEPTION (num_fields!=it.second.size(), std::logic_error,
                                  "Error! The input 'Number Of Fields' on side set '" << ss_name << "' in the discretization section " <<
                                  "does not match the number of side set requirements specified in the problem section.\n");

      std::map<std::string, bool> found;
      for (int ifield=0; ifield<num_fields; ++ifield)
        found[it.second[ifield]] = false;

      for (int ifield=0; ifield<num_fields; ++ifield)
      {
        const Teuchos::ParameterList& thisFieldList =  req_fields_info.sublist(Albany::strint("Field", ifield));

        fname   = thisFieldList.get<std::string>("Field Name");
        flayout = thisFieldList.get<std::string>("Field Layout");

        TEUCHOS_TEST_FOR_EXCEPTION (found.find(fname)==found.end(), std::logic_error,
                                    "Error! Field '" << fname << "' was not specified as a requirement in the problem section.\n");

        found[fname] = true;

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

      for (auto found_it : found)
        TEUCHOS_TEST_FOR_EXCEPTION(found_it.second==false, std::runtime_error,
                                   "Error! The requirement '" << found_it.first << "' on side set '" << ss_name << "' was not found in the discretization section.\n");
    }
  }

  /* Construct All Phalanx Evaluators */
  TEUCHOS_TEST_FOR_EXCEPTION(meshSpecs.size()!=1,std::logic_error,"Problem supports one Material Block");
  fm.resize(1);
  fm[0]  = Teuchos::rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
  buildEvaluators(*fm[0], *meshSpecs[0], stateMgr, Albany::BUILD_RESID_FM,Teuchos::null);
}

Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
PopulateMesh::buildEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                               const Albany::MeshSpecsStruct& meshSpecs,
                               Albany::StateManager& stateMgr,
                               Albany::FieldManagerChoice fmchoice,
                               const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructeEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  Albany::ConstructEvaluatorsOp<PopulateMesh> op(
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

} // Namespace FELIX
