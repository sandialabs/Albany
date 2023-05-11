//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "LandIce_ColumnCouplingTest.hpp"

#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "PHAL_FactoryTraits.hpp"
#include "Albany_BCUtils.hpp"
#include "Albany_ProblemUtils.hpp"
#include <string>

namespace LandIce
{

ColumnCouplingTest::
ColumnCouplingTest (const Teuchos::RCP<Teuchos::ParameterList>& params_,
                    const Teuchos::RCP<Teuchos::ParameterList>& discParams_,
                    const Teuchos::RCP<ParamLib>& paramLib)
 : Albany::AbstractProblem (params_, paramLib, 1)
 , discParams(discParams_)
 , use_sdbcs_(false)
{
  numDim = 3;

  dof_name = "U";
  resid_name = dof_name + "resid";
  scatter_name = "scatter_" + resid_name;

  sideSetName = params->get<std::string>("Side Set Name");
}

void ColumnCouplingTest::
buildProblem (Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
              Albany::StateManager& stateMgr)
{
  TEUCHOS_TEST_FOR_EXCEPTION (meshSpecs.size()!=1,std::logic_error,"Problem supports one Material Block");

  Intrepid2::DefaultCubatureFactory cubFactory;

  // Building cell basis and cubature
  const CellTopologyData * const cell_top = &meshSpecs[0]->ctd;
  cellType = Teuchos::rcp(new shards::CellTopology (cell_top));

  cellEBName = meshSpecs[0]->ebName;

  const int worksetSize     = meshSpecs[0]->worksetSize;
  const int numCellSides    = cellType->getSideCount();
  const int numCellVertices = cellType->getNodeCount();

  dl = Teuchos::rcp(new Albany::Layouts(worksetSize,numCellVertices,numCellVertices,-1,numDim));

  // Building also side structures
  const auto& ss_discs_params = discParams->sublist("Side Set Discretizations");
  const auto& ss_names = ss_discs_params.get<Teuchos::Array<std::string>>("Side Sets");
  for (const auto& ss_name : ss_names) {
    const Albany::MeshSpecsStruct& sideMeshSpecs = *meshSpecs[0]->sideSetMeshSpecs.at(ss_name)[0];

    const CellTopologyData * const side_top = &sideMeshSpecs.ctd;
    auto sideType = Teuchos::rcp(new shards::CellTopology (side_top));

    sideEBName[ss_name] = sideMeshSpecs.ebName;
    sideBasis[ss_name] = Albany::getIntrepid2Basis(*side_top);

    int cubDegree = this->params->get("Cubature Degree", 3);
    sideCubature[ss_name] = cubFactory.create<PHX::Device, RealType, RealType>(*sideType, cubDegree);

    int numSideVertices = sideType->getNodeCount();
    int numSideQPs      = sideCubature[ss_name]->getNumPoints();

    dl->side_layouts[ss_name] = Teuchos::rcp(new Albany::Layouts(numSideVertices,numSideVertices,
                                             numSideQPs,numDim-1,numDim,numCellSides,2,sideSetName));
  }
  dl_side = dl->side_layouts.at(sideSetName);

  *out << " Column Coupling Test problem:\n"
       << "   - dimension         : " << numDim          << "\n"
       << "   - workset size      : " << worksetSize     << "\n"
       << "   - num cell vertices : " << numCellVertices << "\n"
       << "   - num cell sides    : " << numCellSides    << "\n"
       << "   - num cell dofs     : " << numCellVertices << "\n"
       << "   - num cell qps      : N/A\n"
       << "   - residual side set : " << sideSetName << "\n"
       << "   - num side vertices : " << dl->side_layouts.at(sideSetName)->node_scalar->dimension(1) << "\n"
       << "   - num side dofs     : " << dl->side_layouts.at(sideSetName)->node_scalar->dimension(1) << "\n"
       << "   - num side qps      : N/A\n";

  /* Construct All Phalanx Evaluators */
  fm.resize(1);
  fm[0]  = Teuchos::rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);

  // Build evaluators
  buildEvaluators(*fm[0], *meshSpecs[0], stateMgr, Albany::BUILD_RESID_FM, Teuchos::null);

  constructDirichletEvaluators (*meshSpecs[0]);
}

Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
ColumnCouplingTest::buildEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                             const Albany::MeshSpecsStruct& meshSpecs,
                             Albany::StateManager& stateMgr,
                             Albany::FieldManagerChoice fmchoice,
                             const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructeEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  Albany::ConstructEvaluatorsOp<ColumnCouplingTest> op(*this, fm0, meshSpecs, stateMgr, fmchoice, responseList);
  Sacado::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes> fe(op);

  return *op.tags;
}

void ColumnCouplingTest::constructDirichletEvaluators(const Albany::MeshSpecsStruct& meshSpecs)
{
  // Construct Dirichlet evaluators for all nodesets and names
  std::vector<std::string> dirichletNames = {dof_name};

  Albany::BCUtils<Albany::DirichletTraits> dirUtils;
  dfm = dirUtils.constructBCEvaluators(meshSpecs.nsNames, dirichletNames, this->params, this->paramLib);
  use_sdbcs_ = dirUtils.useSDBCs(); 
}

Teuchos::RCP<const Teuchos::ParameterList>
ColumnCouplingTest::getValidProblemParameters () const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = this->getGenericProblemParams("ValidColumnCouplingTestProblemParams");

  validPL->set<std::string>("Side Set Name","","The name of the sideset where the side laplacian has to be solved (only for Dimension=3).");
  validPL->set<bool>("Extruded Column Coupled in 2D Residual",true,"Whether extruded columns dofs are coupled in the 2d residual");

  return validPL;
}

} // namespace Albany
