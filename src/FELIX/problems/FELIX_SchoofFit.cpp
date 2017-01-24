//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <string>

#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "Teuchos_FancyOStream.hpp"

#include "PHAL_FactoryTraits.hpp"
#include "Albany_Utils.hpp"
#include "Albany_BCUtils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "FELIX_SchoofFit.hpp"

//uncomment the following line if you want debug output to be printed to screen
#define OUTPUT_TO_SCREEN

FELIX::SchoofFit::
SchoofFit (const Teuchos::RCP<Teuchos::ParameterList>& params_,
           const Teuchos::RCP<ParamLib>& paramLib_,
           const int numDim_) :
  Albany::AbstractProblem(params_, paramLib_, numDim_),
  numDim(numDim_)
{
  neq = 1;

  // Set the num PDEs for the null space object to pass to ML
  this->rigidBodyModes->setNumPDEs(neq);

  // Need to allocate a fields in mesh database
  Teuchos::Array<std::string> s_req = params->get<Teuchos::Array<std::string> > ("Required Scalar Fields");
  for (int i(0); i<s_req.size(); ++i)
    this->requirements.push_back(s_req[i]);

  Teuchos::Array<std::string> v_req = params->get<Teuchos::Array<std::string> > ("Required Vector Fields");
  for (int i(0); i<v_req.size(); ++i)
    this->requirements.push_back(v_req[i]);
}

FELIX::SchoofFit::
~SchoofFit()
{
  // Nothing to be done here
}

void FELIX::SchoofFit::buildProblem (Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
                                    Albany::StateManager& stateMgr)
{
  using Teuchos::rcp;

  // Building cell basis and cubature
  const CellTopologyData * const cell_top = &meshSpecs[0]->ctd;
  cellBasis = Albany::getIntrepid2Basis(*cell_top);
  cellType = rcp(new shards::CellTopology (cell_top));

  Intrepid2::DefaultCubatureFactory   cubFactory;
  cellCubature = cubFactory.create<PHX::Device, RealType, RealType>(*cellType, meshSpecs[0]->cubatureDegree);

  elementBlockName = meshSpecs[0]->ebName;

  const int worksetSize     = meshSpecs[0]->worksetSize;
  const int vecDim          = 2;
  const int numCellSides    = cellType->getFaceCount();
  const int numCellVertices = cellType->getNodeCount();
  const int numCellNodes    = cellBasis->getCardinality();
  const int numCellQPs      = cellCubature->getNumPoints();

  dl = rcp(new Albany::Layouts(worksetSize,numCellVertices,numCellNodes,numCellQPs,numDim,vecDim));

#ifdef OUTPUT_TO_SCREEN
  Teuchos::RCP<Teuchos::FancyOStream> out = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));
  int commRank = Teuchos::GlobalMPISession::getRank();
  int commSize = Teuchos::GlobalMPISession::getNProc();
  out->setProcRankAndSize(commRank, commSize);
  out->setOutputToRootOnly(0);

  *out << "Field Dimensions: \n"
       << "  Workset             = " << worksetSize << "\n"
       << "  Vertices            = " << numCellVertices << "\n"
       << "  CellNodes           = " << numCellNodes << "\n"
       << "  CellQuadPts         = " << numCellQPs << "\n"
       << "  Dim                 = " << numDim << "\n"
       << "  VecDim              = " << vecDim << std::endl;
#endif

  /* Construct All Phalanx Evaluators */
  TEUCHOS_TEST_FOR_EXCEPTION(meshSpecs.size()!=1,std::logic_error,"Problem supports one Material Block");
  fm.resize(1);
  fm[0]  = rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
  buildEvaluators(*fm[0], *meshSpecs[0], stateMgr, Albany::BUILD_RESID_FM,Teuchos::null);
}

Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
FELIX::SchoofFit::
buildEvaluators(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fmchoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructeEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  Albany::ConstructEvaluatorsOp<SchoofFit> op(
    *this, fm0, meshSpecs, stateMgr, fmchoice, responseList);
  Sacado::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes> fe(op);
  return *op.tags;
}

Teuchos::RCP<const Teuchos::ParameterList>
FELIX::SchoofFit::getValidProblemParameters () const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidSchoofFitProblemParams");

  validPL->set<Teuchos::Array<std::string> > ("Required Scalar Fields", Teuchos::Array<std::string>(), "");
  validPL->set<Teuchos::Array<std::string> > ("Required Vector Fields", Teuchos::Array<std::string>(), "");
  validPL->set<Teuchos::Array<std::string> > ("Save Fields", Teuchos::Array<std::string>(), "");
  validPL->sublist("FELIX Basal Friction Coefficient", false, "Parameters needed to compute the basal friction coefficient");
  validPL->sublist("FELIX Effective Pressure Surrogate", false, "");
  validPL->sublist("FELIX Field Norm", false, "");
  validPL->sublist("FELIX Physical Parameters", false, "");

  return validPL;
}
