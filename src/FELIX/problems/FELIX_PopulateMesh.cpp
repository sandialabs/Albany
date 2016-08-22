//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <string>

#include "Intrepid2_FieldContainer.hpp"
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

  // Need to allocate a fields in mesh database
  if (discParams->isSublist("Required Fields Info"))
  {
    std::string fname;
    Teuchos::ParameterList& req_fields_info = discParams->sublist("Required Fields Info");
    int num_fields = req_fields_info.get<int>("Number Of Fields",0);
    for (int ifield=0; ifield<num_fields; ++ifield)
    {
      const Teuchos::ParameterList& thisFieldList =  req_fields_info.sublist(Albany::strint("Field", ifield));

      fname = thisFieldList.get<std::string>("Field Name");

      this->requirements.push_back(fname);
    }
  }
  if (params->isParameter("Required Fields"))
  {
    // Need to allocate a fields in mesh database
    Teuchos::Array<std::string> req = params->get<Teuchos::Array<std::string> > ("Required Fields");
    for (int i(0); i<req.size(); ++i)
      this->requirements.push_back(req[i]);
  }
  if (discParams->isSublist("Side Set Discretizations"))
  {
    Teuchos::ParameterList& ss_disc_pl = discParams->sublist("Side Set Discretizations");
    const Teuchos::Array<std::string>& ss_names = ss_disc_pl.get<Teuchos::Array<std::string>>("Side Sets");

    std::string fname;
    for (int is=0; is<ss_names.size(); ++is)
    {
      const std::string& ss_name = ss_names[is];
      Teuchos::ParameterList& this_ss_pl = ss_disc_pl.sublist(ss_name);

      if (this_ss_pl.isSublist("Required Fields Info"))
      {
        Teuchos::ParameterList& req_fields_info = ss_disc_pl.sublist("Required Fields Info");
        int num_fields = req_fields_info.get<int>("Number Of Fields",0);
        for (int ifield=0; ifield<num_fields; ++ifield)
        {
          const Teuchos::ParameterList& thisFieldList =  req_fields_info.sublist(Albany::strint("Field", ifield));

          fname = thisFieldList.get<std::string>("Field Name");

          this->ss_requirements[ss_name].push_back(fname);
        }
      }
    }
  }
}

PopulateMesh::~PopulateMesh()
{
  // Nothing to be done here
}

void PopulateMesh::buildProblem (Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct>> meshSpecs,
                                 Albany::StateManager& stateMgr)
{
  using Teuchos::rcp;

  // Building cell basis
  const CellTopologyData * const cell_top = &meshSpecs[0]->ctd;
  cellBasis    = Albany::getIntrepid2Basis(*cell_top);
  cellTopology = rcp(new shards::CellTopology (cell_top));
  cellEBName = meshSpecs[0]->ebName;

  const int worksetSize     = meshSpecs[0]->worksetSize;
  const int numCellSides    = cellTopology->getFaceCount();
  const int numCellVertices = cellTopology->getNodeCount();
  const int numCellNodes    = cellBasis->getCardinality();
  const int numCellQPs      = 0;  // Not needed
  const int numCellDim      = meshSpecs[0]->numDim;
  const int numCellVecDim   = 0;  // Not needed

  dl = rcp(new Albany::Layouts(worksetSize,numCellVertices,numCellNodes,numCellQPs,numCellDim,numCellVecDim));

  if (discParams->isSublist("Side Set Discretizations"))
  {
    Teuchos::ParameterList& ss_disc_pl = discParams->sublist("Side Set Discretizations");
    const Teuchos::Array<std::string>& ss_names = ss_disc_pl.get<Teuchos::Array<std::string>>("Side Sets");

    for (int is=0; is<ss_names.size(); ++is)
    {
      const std::string& ss_name = ss_names[is];
      Teuchos::ParameterList& this_ss_pl = ss_disc_pl.sublist(ss_name);

      const Albany::MeshSpecsStruct& ssMeshSpecs = *meshSpecs[0]->sideSetMeshSpecs.at(ss_name)[0];

      // Building also side structures
      const CellTopologyData * const side_top = &ssMeshSpecs.ctd;
      sideTopology[ss_name] = rcp(new shards::CellTopology (side_top));
      sideBasis[ss_name]    = Albany::getIntrepid2Basis(*side_top);
      sideEBName[ss_name]   = ssMeshSpecs.ebName;

      int numSideVertices = sideTopology[ss_name]->getNodeCount();
      int numSideNodes    = sideBasis[ss_name]->getCardinality();
      int numSideDim      = ssMeshSpecs.numDim;
      int numSideQPs      = 0;    // Not needed
      int numSideVecDim   = 0;    // Not needed

      dl->side_layouts[ss_name] = rcp(new Albany::Layouts(worksetSize,numSideVertices,numSideNodes,numSideQPs,
                                                          numSideDim,numCellDim,numCellSides,numSideVecDim));
    }
  }
  /* Construct All Phalanx Evaluators */
  TEUCHOS_TEST_FOR_EXCEPTION(meshSpecs.size()!=1,std::logic_error,"Problem supports one Material Block");
  fm.resize(1);
  fm[0]  = rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
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
