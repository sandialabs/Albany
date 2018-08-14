//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_config.h"

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#ifdef ALBANY_EPETRA
#include <Epetra_MpiComm.h>
#endif
#include <MiniTensor.h>
#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Phalanx_DataLayout_MDALayout.hpp"

#include "Albany_Layouts.hpp"
#include "Albany_STKDiscretization.hpp"
#include "Albany_StateManager.hpp"
#include "Albany_TmplSTKMeshStruct.hpp"
#include "Albany_Utils.hpp"
#include "HeliumODEs.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_SaveStateField.hpp"
#include "SetField.hpp"
//#include "ConstitutiveModelInterface.hpp"
//#include "ConstitutiveModelParameters.hpp"
//#include "FieldNameMap.hpp"

namespace {

typedef PHX::MDField<PHAL::AlbanyTraits::Residual::ScalarT>::size_type
                                                   size_type;
typedef PHAL::AlbanyTraits::Residual               Residual;
typedef PHAL::AlbanyTraits::Residual::ScalarT      ScalarT;
typedef PHAL::AlbanyTraits                         Traits;
typedef Kokkos::DynRankView<RealType, PHX::Device> FC;
typedef shards::CellTopology                       CT;
using minitensor::bun;
using minitensor::eye;
using minitensor::norm;
using minitensor::Tensor;
using minitensor::Vector;
using Teuchos::ArrayRCP;
using Teuchos::RCP;
using Teuchos::rcp;

TEUCHOS_UNIT_TEST(HeliumODEs, test1)
{
  // A mpi object must be instantiated
  Teuchos::GlobalMPISession        mpi_session(void);
  Teuchos::RCP<const Teuchos_Comm> commT =
      Albany::createTeuchosCommFromMpiComm(Albany_MPI_COMM_WORLD);

  // Get the name of the material model to be used (and make sure there is one)
  std::string element_block_name = "Block0";

  // set tolerance once and for all
  double tolerance = 1.0e-15;

  const int                  workset_size = 1;
  const int                  num_pts      = 1;
  const int                  num_dims     = 3;
  const int                  num_vertices = 8;
  const int                  num_nodes    = 8;
  const RCP<Albany::Layouts> dl           = rcp(new Albany::Layouts(
      workset_size, num_vertices, num_nodes, num_pts, num_dims));

  //--------------------------------------------------------------------------
  // total concentration
  ArrayRCP<ScalarT> total_concentration(1);
  total_concentration[0] = 5.6e-4;

  Teuchos::ParameterList tcPL;
  tcPL.set<std::string>("Evaluated Field Name", "Total Concentration");
  tcPL.set<ArrayRCP<ScalarT>>("Field Values", total_concentration);
  tcPL.set<RCP<PHX::DataLayout>>("Evaluated Field Data Layout", dl->qp_scalar);
  RCP<LCM::SetField<Residual, Traits>> setFieldTotalConcentration =
      rcp(new LCM::SetField<Residual, Traits>(tcPL));

  //--------------------------------------------------------------------------
  // delta time
  ArrayRCP<ScalarT> delta_time(1);
  delta_time[0] = 0.001;

  Teuchos::ParameterList dtPL;
  dtPL.set<std::string>("Evaluated Field Name", "Delta Time");
  dtPL.set<ArrayRCP<ScalarT>>("Field Values", delta_time);
  dtPL.set<RCP<PHX::DataLayout>>(
      "Evaluated Field Data Layout", dl->workset_scalar);
  RCP<LCM::SetField<Residual, Traits>> setFieldDeltaTime =
      rcp(new LCM::SetField<Residual, Traits>(dtPL));

  //--------------------------------------------------------------------------
  // diffusion coeffecient
  ArrayRCP<ScalarT> diff_coeff(1);
  diff_coeff[0] = 1.0;

  Teuchos::ParameterList dcPL;
  dcPL.set<std::string>("Evaluated Field Name", "Diffusion Coefficient");
  dcPL.set<ArrayRCP<ScalarT>>("Field Values", diff_coeff);
  dcPL.set<RCP<PHX::DataLayout>>("Evaluated Field Data Layout", dl->qp_scalar);
  RCP<LCM::SetField<Residual, Traits>> setFieldDiffCoeff =
      rcp(new LCM::SetField<Residual, Traits>(dcPL));

  //--------------------------------------------------------------------------
  // helium ODEs evaluator
  Teuchos::ParameterList hoPL;
  hoPL.set<std::string>("Total Concentration Name", "Total Concentration");
  hoPL.set<std::string>("Delta Time Name", "Delta Time");
  hoPL.set<std::string>("Diffusion Coefficient Name", "Diffusion Coefficient");
  hoPL.set<std::string>("He Concentration Name", "He Concentration");
  hoPL.set<std::string>("Total Bubble Density Name", "Total Bubble Density");
  hoPL.set<std::string>(
      "Bubble Volume Fraction Name", "Bubble Volume Fraction");
  // Transport Parameters
  Teuchos::ParameterList trans_params;
  trans_params.set<double>("Avogadro's Number", 6.0221413e11);
  hoPL.set<Teuchos::ParameterList*>("Transport Parameters", &trans_params);

  // Tritium Parameters
  Teuchos::ParameterList tri_params;
  tri_params.set<double>("Tritium Decay Constant", 1.79e-9);
  tri_params.set<double>("Helium Radius", 2.5e-4);
  tri_params.set<double>("Atoms Per Cluster", 10);
  hoPL.set<Teuchos::ParameterList*>("Tritium Parameters", &tri_params);

  // Molar Volume
  Teuchos::ParameterList mol_vol;
  mol_vol.set<double>("Value", 7.116);
  hoPL.set<Teuchos::ParameterList*>("Molar Volume", &mol_vol);

  RCP<LCM::HeliumODEs<Residual, Traits>> HeODEs =
      rcp(new LCM::HeliumODEs<Residual, Traits>(hoPL, dl));

  //--------------------------------------------------------------------------
  // Instantiate a field manager.
  PHX::FieldManager<Traits> field_manager;

  // Instantiate a field manager for States
  PHX::FieldManager<Traits> state_field_manager;

  // Register the evaluators with the field manager
  field_manager.registerEvaluator<Residual>(setFieldTotalConcentration);
  field_manager.registerEvaluator<Residual>(setFieldDeltaTime);
  field_manager.registerEvaluator<Residual>(setFieldDiffCoeff);
  field_manager.registerEvaluator<Residual>(HeODEs);

  // Register the evaluators with the state field manager
  state_field_manager.registerEvaluator<Residual>(setFieldTotalConcentration);
  state_field_manager.registerEvaluator<Residual>(setFieldDeltaTime);
  state_field_manager.registerEvaluator<Residual>(setFieldDiffCoeff);
  state_field_manager.registerEvaluator<Residual>(HeODEs);

  // Set the evaluated fields as required fields
  for (std::vector<RCP<PHX::FieldTag>>::const_iterator it =
           HeODEs->evaluatedFields().begin();
       it != HeODEs->evaluatedFields().end();
       it++)
    field_manager.requireField<Residual>(**it);

  //--------------------------------------------------------------------------
  // Instantiate a state manager
  Albany::StateManager stateMgr;

  // register the states
  //
  Teuchos::RCP<Teuchos::ParameterList> p;
  Teuchos::RCP<PHX::Evaluator<Traits>> ev;
  p = stateMgr.registerStateVariable(
      "Total Concentration",
      dl->qp_scalar,
      dl->dummy,
      element_block_name,
      "scalar",
      total_concentration[0],
      true,   // state
      true);  // output
  ev = Teuchos::rcp(new PHAL::SaveStateField<Residual, Traits>(*p));
  field_manager.registerEvaluator<Residual>(ev);
  state_field_manager.registerEvaluator<Residual>(ev);

  p = stateMgr.registerStateVariable(
      "He Concentration",
      dl->qp_scalar,
      dl->dummy,
      element_block_name,
      "scalar",
      0.0,
      true,   // state
      true);  // output
  ev = Teuchos::rcp(new PHAL::SaveStateField<Residual, Traits>(*p));
  field_manager.registerEvaluator<Residual>(ev);
  state_field_manager.registerEvaluator<Residual>(ev);

  p = stateMgr.registerStateVariable(
      "Total Bubble Density",
      dl->qp_scalar,
      dl->dummy,
      element_block_name,
      "scalar",
      0.0,
      true,   // state
      true);  // output
  ev = Teuchos::rcp(new PHAL::SaveStateField<Residual, Traits>(*p));
  field_manager.registerEvaluator<Residual>(ev);
  state_field_manager.registerEvaluator<Residual>(ev);

  p = stateMgr.registerStateVariable(
      "Bubble Volume Fraction",
      dl->qp_scalar,
      dl->dummy,
      element_block_name,
      "scalar",
      0.0,
      true,   // state
      true);  // output
  ev = Teuchos::rcp(new PHAL::SaveStateField<Residual, Traits>(*p));
  field_manager.registerEvaluator<Residual>(ev);
  state_field_manager.registerEvaluator<Residual>(ev);

  //--------------------------------------------------------------------------
  // Call postRegistrationSetup on the evaluators
  // JTO - I don't know what "Test String" is meant for...
  PHAL::AlbanyTraits::SetupData setupData = "Test String";
  field_manager.postRegistrationSetup(setupData);

  Teuchos::RCP<PHX::DataLayout> dummy =
      Teuchos::rcp(new PHX::MDALayout<Dummy>(0));
  std::vector<std::string> responseIDs =
      stateMgr.getResidResponseIDsToRequire(element_block_name);
  std::vector<std::string>::const_iterator it;
  for (it = responseIDs.begin(); it != responseIDs.end(); it++) {
    const std::string&                              responseID = *it;
    PHX::Tag<PHAL::AlbanyTraits::Residual::ScalarT> res_response_tag(
        responseID, dummy);
    state_field_manager.requireField<PHAL::AlbanyTraits::Residual>(
        res_response_tag);
  }
  state_field_manager.postRegistrationSetup("");

  // std::cout << "Process using 'dot -Tpng -O <name>'\n";
  field_manager.writeGraphvizFile<Residual>("FM", true, true);
  state_field_manager.writeGraphvizFile<Residual>("SFM", true, true);

  //---------------------------------------------------------------------------
  // grab the output file name
  //
  std::string output_file = "output.exo";

  //---------------------------------------------------------------------------
  // Create discretization, as required by the StateManager
  //
  Teuchos::RCP<Teuchos::ParameterList> discretizationParameterList =
      Teuchos::rcp(new Teuchos::ParameterList("Discretization"));
  discretizationParameterList->set<int>("1D Elements", workset_size);
  discretizationParameterList->set<int>("2D Elements", 1);
  discretizationParameterList->set<int>("3D Elements", 1);
  discretizationParameterList->set<std::string>("Method", "STK3D");
  discretizationParameterList->set<int>("Number Of Time Derivatives", 0);
  discretizationParameterList->set<std::string>(
      "Exodus Output File Name", output_file);
  Teuchos::RCP<Tpetra_Map>    mapT = Teuchos::rcp(new Tpetra_Map(
      workset_size * num_dims * num_nodes,
      0,
      commT,
      Tpetra::LocallyReplicated));
  Teuchos::RCP<Tpetra_Vector> solution_vectorT =
      Teuchos::rcp(new Tpetra_Vector(mapT));

  int numberOfEquations = 3;
  Albany::AbstractFieldContainer::FieldContainerRequirements req;

  Teuchos::RCP<Albany::AbstractSTKMeshStruct> stkMeshStruct =
      Teuchos::rcp(new Albany::TmplSTKMeshStruct<3>(
          discretizationParameterList, Teuchos::null, commT));
  stkMeshStruct->setFieldAndBulkData(
      commT,
      discretizationParameterList,
      numberOfEquations,
      req,
      stateMgr.getStateInfoStruct(),
      stkMeshStruct->getMeshSpecs()[0]->worksetSize);

  Teuchos::RCP<Albany::AbstractDiscretization> discretization =
      Teuchos::rcp(new Albany::STKDiscretization(
          discretizationParameterList, stkMeshStruct, commT));

  //---------------------------------------------------------------------------
  // Associate the discretization with the StateManager
  //
  stateMgr.setupStateArrays(discretization);

  //--------------------------------------------------------------------------
  // Create a workset
  PHAL::Workset workset;
  workset.numCells = workset_size;
  workset.stateArrayPtr =
      &stateMgr.getStateArray(Albany::StateManager::ELEM, 0);

  //--------------------------------------------------------------------------
  // loop over time and call evaluators
  double end_time = 100.0;
  for (double time(0.0); time < end_time; time += delta_time[0]) {
    total_concentration[0] = 0.005;

    //--------------------------------------------------------------------------
    // Call the evaluators, evaluateFields() computes things
    field_manager.preEvaluate<Residual>(workset);
    field_manager.evaluateFields<Residual>(workset);
    field_manager.postEvaluate<Residual>(workset);

    //--------------------------------------------------------------------------
    // Call the evaluators, evaluateFields() computes things
    state_field_manager.preEvaluate<Residual>(workset);
    state_field_manager.evaluateFields<Residual>(workset);
    state_field_manager.postEvaluate<Residual>(workset);

    stateMgr.updateStates();

    // output to the exodus file
    discretization->writeSolutionT(*solution_vectorT, time);
  }

  //--------------------------------------------------------------------------
  // Pull the He concentration
  PHX::MDField<ScalarT, Cell, QuadPoint> he_conc(
      "He Concentration", dl->qp_scalar);
  field_manager.getFieldData<Residual>(he_conc);

  // Record the expected concentration
  double expected_conc(0.0);
  for (size_type cell = 0; cell < workset_size; ++cell)
    for (size_type pt = 0; pt < num_pts; ++pt)
      TEST_COMPARE(fabs(he_conc(cell, pt) - expected_conc), <=, tolerance);

  //--------------------------------------------------------------------------
  // Pull the total bubble density
  PHX::MDField<ScalarT, Cell, QuadPoint> tot_bub_density(
      "Total Bubble Density", dl->qp_scalar);
  field_manager.getFieldData<Residual>(tot_bub_density);

  // Record the bubble density
  double expected_density(0.0);
  for (size_type cell = 0; cell < workset_size; ++cell)
    for (size_type pt = 0; pt < num_pts; ++pt)
      TEST_COMPARE(
          fabs(tot_bub_density(cell, pt) - expected_density), <=, tolerance);

  //--------------------------------------------------------------------------
  // Pull the bubble volume fraction
  PHX::MDField<ScalarT, Cell, QuadPoint> bub_vol_frac(
      "Bubble Volume Fraction", dl->qp_scalar);
  field_manager.getFieldData<Residual>(bub_vol_frac);

  // Record the bubble volume fraction
  double expected_vol_frac(0.0);
  for (size_type cell = 0; cell < workset_size; ++cell)
    for (size_type pt = 0; pt < num_pts; ++pt)
      TEST_COMPARE(
          fabs(bub_vol_frac(cell, pt) - expected_vol_frac), <=, tolerance);
}

}  // namespace
