
//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//---------------------------------------------------------------------------//
/*!
 * \file   dtk_interp_and_error.cpp
 * \author Irina Tezaur (ikalash@sandia.gov)
 * \brief  Projection of solution from source to target mesh, followed by
 *         discrete l2 error calculation using DTK.
 */
//---------------------------------------------------------------------------//

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <sstream>
#include <vector>

#include "DTK_MapOperatorFactory.hpp"
#include "DTK_STKMeshHelpers.hpp"
#include "DTK_STKMeshManager.hpp"

#include <Teuchos_Array.hpp>
#include <Teuchos_ArrayRCP.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_DefaultMpiComm.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_OpaqueWrapper.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_StandardCatchMacros.hpp>
#include <Teuchos_TimeMonitor.hpp>
#include <Teuchos_TypeTraits.hpp>
#include <Teuchos_VerboseObject.hpp>
#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_XMLParameterListCoreHelpers.hpp"
#include "Teuchos_YamlParameterListCoreHelpers.hpp"

#include <Tpetra_MultiVector.hpp>

#include <Intrepid_FieldContainer.hpp>

#include <stk_util/parallel/Parallel.hpp>

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/CoordinateSystems.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_mesh/base/Types.hpp>
#include <stk_topology/topology.hpp>

#include <stk_io/IossBridge.hpp>
#include <stk_io/StkMeshIoBroker.hpp>

#include <Ionit_Initializer.h>
#include <Ioss_SubSystem.h>

// #define DEBUG_OUTPUT

template <typename FieldType>
void
interp_and_calc_error(
    Teuchos::RCP<const Teuchos::Comm<int>> comm,
    Teuchos::RCP<Teuchos::ParameterList>   plist)
{
  Teuchos::RCP<Teuchos::FancyOStream> out =
      Teuchos::fancyOStream(Teuchos::VerboseObjectBase::getDefaultOStream());
  // Read command-line options

  std::string source_mesh_input_file =
      plist->get<std::string>("Source Mesh Input File");

  int src_snap_no = plist->get<int>(
      "Source Mesh Snapshot Number", 1);  // this value is 1-based

  std::string target_mesh_input_file =
      plist->get<std::string>("Target Mesh Input File");

  std::string target_mesh_output_file =
      plist->get<std::string>("Target Mesh Output File");

  int tgt_snap_no = plist->get<int>(
      "Target Mesh Snapshot Number", 1);  // this value is 1-based

  std::string source_field_name =
      plist->get<std::string>("Source Field Name", "solution");

  std::string target_field_name =
      plist->get<std::string>("Target Field Name", "solution");

  // IKT, 10/20/17 - the following tells the code whether to divide by the norm
  // of the reference solution vector when computing the relative error written
  // to the output file.  If false (default), each component will be scaled by
  // the norm of that component only in the reference solution.
  bool scale_by_norm_soln_vec = false;
  if (plist->isParameter("Scale by Norm of Solution Vector")) {
    scale_by_norm_soln_vec =
        plist->get<bool>("Scale by Norm of Solution Vector", false);
  }

  std::string src_field_name = source_field_name + "_src";

  std::string tgt_interp_field_name = target_field_name + "Ref";

  std::string rel_err_field_name = target_field_name + "RelErr";

  std::string abs_err_field_name = target_field_name + "AbsErr";

  std::vector<int> src_time_step_indices, tgt_time_step_indices;

  // Get the raw mpi communicator (basic typedef in STK).
  Teuchos::RCP<const Teuchos::MpiComm<int>> mpi_comm =
      Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(comm);

  Teuchos::RCP<const Teuchos::OpaqueWrapper<MPI_Comm>> opaque_comm =
      mpi_comm->getRawMpiComm();

  stk::ParallelMachine parallel_machine = (*opaque_comm)();

  // SOURCE MESH READ
  // ----------------
  stk::io::StkMeshIoBroker src_broker(parallel_machine);

  std::size_t src_input_index = src_broker.add_mesh_database(
      source_mesh_input_file, "exodus", stk::io::READ_MESH);

  src_broker.set_active_mesh(src_input_index);
  src_broker.create_input_mesh();

  // number of intervals to divide each input time step into
  int interpolation_intervals = 1;

  stk::io::MeshField::TimeMatchOption tmo = stk::io::MeshField::CLOSEST;

  if (interpolation_intervals > 1) {
    tmo = stk::io::MeshField::LINEAR_INTERPOLATION;
  }

  src_broker.add_all_mesh_fields_as_input_fields(tmo);
  src_broker.populate_bulk_data();

  Teuchos::RCP<stk::mesh::BulkData> src_bulk_data =
      Teuchos::rcpFromRef(src_broker.bulk_data());

  Teuchos::RCP<Ioss::Region> src_io_region = src_broker.get_input_io_region();

  STKIORequire(!Teuchos::is_null(src_io_region));

  // Get number of time steps in source mesh
  int src_timestep_count = src_io_region->get_property("state_count").get_int();

#ifdef DEBUG_OUTPUT
  *out << "   timestep_count in source mesh: " << src_timestep_count
       << std::endl;
#endif

  // Get source_field from source mesh
  FieldType* source_field = src_broker.meta_data().get_field<FieldType>(
      stk::topology::NODE_RANK, source_field_name);

  if (source_field != 0) {
    *out << "   Field with name " << source_field_name
         << " found in source mesh file!" << std::endl;
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        Teuchos::Exceptions::InvalidParameter,
        std::endl
            << "   Field with name " << source_field_name
            << " NOT found in source mesh file!" << std::endl);
  }

  int neq = source_field->max_size(stk::topology::NODE_RANK);
#ifdef DEBUG_OUTPUT
  *out << "   Source field has " << neq << " dofs/node." << std::endl;
#endif

  // TARGET MESH READ
  // ----------------

  // Load the target mesh.
  stk::io::StkMeshIoBroker tgt_broker(parallel_machine);
  std::size_t              tgt_input_index = tgt_broker.add_mesh_database(
      target_mesh_input_file, "exodus", stk::io::READ_MESH);
  tgt_broker.set_active_mesh(tgt_input_index);
  tgt_broker.create_input_mesh();
  tgt_broker.add_all_mesh_fields_as_input_fields(tmo);

  // Put fields on target mesh
  // Add a nodal field to the interpolated target part.
  FieldType& target_interp_field =
      tgt_broker.meta_data().declare_field<FieldType>(
          stk::topology::NODE_RANK, tgt_interp_field_name);

  stk::mesh::put_field_on_mesh(
      target_interp_field, tgt_broker.meta_data().universal_part(), neq, nullptr);

  // Add a absolute error nodal field to the target part.
  FieldType& target_abs_error_field =
      tgt_broker.meta_data().declare_field<FieldType>(
          stk::topology::NODE_RANK, abs_err_field_name);

  stk::mesh::put_field_on_mesh(
      target_abs_error_field, tgt_broker.meta_data().universal_part(), neq, nullptr);

  // Add a relative error nodal field to the target part.
  FieldType& target_rel_error_field =
      tgt_broker.meta_data().declare_field<FieldType>(
          stk::topology::NODE_RANK, rel_err_field_name);

  stk::mesh::put_field_on_mesh(
      target_rel_error_field, tgt_broker.meta_data().universal_part(), neq, nullptr);

  // Create the target bulk data.
  tgt_broker.populate_bulk_data();

  Teuchos::RCP<stk::mesh::BulkData> tgt_bulk_data =
      Teuchos::rcpFromRef(tgt_broker.bulk_data());

  // Add a nodal field to the interpolated target part.
  // Populate target_field
  FieldType* target_field = tgt_broker.meta_data().get_field<FieldType>(
      stk::topology::NODE_RANK, target_field_name);

  if (target_field != 0) {
    *out << "   Field with name " << target_field_name;
    *out << " found in target mesh file!" << std::endl;
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        Teuchos::Exceptions::InvalidParameter,
        std::endl
            << "   Field with name " << target_field_name
            << " NOT found in target mesh file!" << std::endl);
  }

  Teuchos::RCP<Ioss::Region> tgt_io_region = tgt_broker.get_input_io_region();
  STKIORequire(!Teuchos::is_null(tgt_io_region));

  // Get number of time steps in source mesh
  int tgt_timestep_count = tgt_io_region->get_property("state_count").get_int();

#ifdef DEBUG_OUTPUT
  *out << "   tgt_timestep_count in target mesh: " << tgt_timestep_count
       << std::endl;
#endif

  if (src_timestep_count < 1) {
    TEUCHOS_TEST_FOR_EXCEPTION(true,
                               Teuchos::Exceptions::InvalidParameter,
                               std::endl
                                   << "Source file has 0 snapshots!"
                                   << std::endl;)
  }
  if (tgt_timestep_count < 1) {
    TEUCHOS_TEST_FOR_EXCEPTION(true,
                               Teuchos::Exceptions::InvalidParameter,
                               std::endl
                                   << "Taret file has 0 snapshots!"
                                   << std::endl;)
  }
  if (((src_snap_no == -1) && (tgt_snap_no != -1)) ||
      ((tgt_snap_no == -1) && (src_snap_no != -1))) {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        Teuchos::Exceptions::InvalidParameter,
        std::endl
            << "Invalid value of Source Mesh Snapshot Number = " << src_snap_no
            << ".  If Target Mesh Snapshot Number = -1, Source Mesh Snapshot "
               "Number "
            << "must be -1." << std::endl;);
  }
  if ((src_snap_no == -1) && (tgt_snap_no == -1)) {
    if (tgt_timestep_count < src_timestep_count) {
      *out << "\n Number of snapshots in target mesh file = "
           << tgt_timestep_count
           << " < number of snapshots in source mesh file (= "
           << src_timestep_count
           << ").\n  Errors will be computed only up to snapshot #"
           << tgt_timestep_count << ".\n \n";
      src_timestep_count = tgt_timestep_count;
    }
    if (src_timestep_count < tgt_timestep_count) {
      *out << "\n Number of snapshots in source mesh file = "
           << src_timestep_count
           << " < number of snapshots in target mesh file (= "
           << tgt_timestep_count
           << ").\n  Errors will be computed only up to snapshot #"
           << src_timestep_count << ".\n \n";
      tgt_timestep_count = src_timestep_count;
    }
    if (tgt_timestep_count != src_timestep_count) {
      TEUCHOS_TEST_FOR_EXCEPTION(
          true,
          Teuchos::Exceptions::InvalidParameter,
          std::endl
              << "Number of snapshots in source mesh file must equal number of "
              << "snapshots in target mesh file when Target Mesh Snapshot "
                 "Number = Source Mesh "
              << "Snapshot Number = -1." << std::endl;)
    } else {
      tgt_time_step_indices.resize(tgt_timestep_count);
      src_time_step_indices.resize(tgt_timestep_count);
      for (int i = 0; i < tgt_timestep_count; i++)
        tgt_time_step_indices[i] = i + 1;
      for (int i = 0; i < tgt_timestep_count; i++)
        src_time_step_indices[i] = i + 1;
    }
  } else {
    if (src_snap_no > src_timestep_count) {
      TEUCHOS_TEST_FOR_EXCEPTION(
          true,
          Teuchos::Exceptions::InvalidParameter,
          std::endl
              << "Invalid value of Source Mesh Snapshot Number = "
              << src_snap_no << " > total number of snapshots in "
              << source_mesh_input_file << " = " << src_timestep_count << "."
              << std::endl;);
    }
    if (tgt_snap_no > tgt_timestep_count) {
      TEUCHOS_TEST_FOR_EXCEPTION(
          true,
          Teuchos::Exceptions::InvalidParameter,
          std::endl
              << "Invalid value of Target Mesh Snapshot Number = "
              << tgt_snap_no << " > total number of snapshots in "
              << target_mesh_input_file << " = " << tgt_timestep_count << "."
              << std::endl;);
    }
    if ((src_snap_no == 0) || (src_snap_no < -1)) {
      TEUCHOS_TEST_FOR_EXCEPTION(
          true,
          Teuchos::Exceptions::InvalidParameter,
          std::endl
              << "Invalid value of Source Mesh Snapshot Number = "
              << src_snap_no << "; valid values are -1 and >0." << std::endl;);
    }
    if ((tgt_snap_no == 0) || (tgt_snap_no < -1)) {
      TEUCHOS_TEST_FOR_EXCEPTION(
          true,
          Teuchos::Exceptions::InvalidParameter,
          std::endl
              << "Invalid value of Target Mesh Snapshot Number = "
              << tgt_snap_no << "; valid values are -1 and >0 ." << std::endl;);
    }
    tgt_time_step_indices.resize(1);
    src_time_step_indices.resize(1);
    tgt_time_step_indices[0] = tgt_snap_no;
    src_time_step_indices[0] = src_snap_no;
  }

  // DEFINE PARTS/SELECTOR
  // ----------------

  stk::mesh::Selector src_stk_selector =
      stk::mesh::Selector(src_broker.meta_data().universal_part());

  stk::mesh::BucketVector src_part_buckets =
      src_stk_selector.get_buckets(stk::topology::NODE_RANK);

  std::vector<stk::mesh::Entity> src_part_nodes;

  stk::mesh::get_selected_entities(
      src_stk_selector, src_part_buckets, src_part_nodes);

  Intrepid::FieldContainer<double> src_node_coords =
      DataTransferKit::STKMeshHelpers::getEntityNodeCoordinates(
          Teuchos::Array<stk::mesh::Entity>(src_part_nodes), *src_bulk_data);

  for (int index = 0; index < tgt_time_step_indices.size(); index++) {
    double time = src_io_region->get_state_time(src_time_step_indices[index]);
    if (src_time_step_indices[index] == src_timestep_count)
      interpolation_intervals = 1;

    int step_end = src_time_step_indices[index] < src_timestep_count ?
                       src_time_step_indices[index] + 1 :
                       src_time_step_indices[index];
    double tend  = src_io_region->get_state_time(step_end);
    double tbeg  = time;
    double delta = (tend - tbeg) / static_cast<double>(interpolation_intervals);

    for (int interval = 0; interval < interpolation_intervals; interval++) {
      time = tbeg + delta * static_cast<double>(interval);
      src_broker.read_defined_input_fields(time);
    }

    time = tgt_io_region->get_state_time(tgt_time_step_indices[index]);
    if (tgt_time_step_indices[index] == tgt_timestep_count) {
      interpolation_intervals = 1;
    }

    step_end = tgt_time_step_indices[index] < tgt_timestep_count ?
                   tgt_time_step_indices[index] + 1 :
                   tgt_time_step_indices[index];
    tend  = tgt_io_region->get_state_time(step_end);
    tbeg  = time;
    delta = (tend - tbeg) / static_cast<double>(interpolation_intervals);

    for (int interval = 0; interval < interpolation_intervals; interval++) {
      time = tbeg + delta * static_cast<double>(interval);
      tgt_broker.read_defined_input_fields(time);
    }

    // SOLUTION TRANSFER SETUP
    // -----------------------

    // Create a manager for the source part elements.
    DataTransferKit::STKMeshManager src_manager(
        src_bulk_data, src_stk_selector);

    // Create a manager for the target part nodes.
    stk::mesh::Selector tgt_stk_selector =
        stk::mesh::Selector(tgt_broker.meta_data().universal_part());

    DataTransferKit::STKMeshManager tgt_manager(
        tgt_bulk_data, tgt_stk_selector);

    // Create a solution vector for the source.
    Teuchos::RCP<Tpetra::MultiVector<double, int, DataTransferKit::SupportId>>
        src_vector = src_manager.createFieldMultiVector<FieldType>(
            Teuchos::ptr(source_field), neq);

    // Create a solution vector for the target.
    Teuchos::RCP<Tpetra::MultiVector<double, int, DataTransferKit::SupportId>>
        tgt_vector = tgt_manager.createFieldMultiVector<FieldType>(
            Teuchos::ptr(&target_interp_field), neq);

#ifdef DEBUG_OUTPUT
    // Print out source mesh info.
    Teuchos::RCP<Teuchos::Describable> src_describe =
        src_manager.functionSpace()->entitySet();

    *out << "   Source Mesh: " << std::endl;
    src_describe->describe(*out, Teuchos::VERB_HIGH);
    *out << std::endl;

    // Print out target mesh info.
    Teuchos::RCP<Teuchos::Describable> tgt_describe =
        tgt_manager.functionSpace()->entitySet();

    *out << "   Target Mesh: " << std::endl;
    tgt_describe->describe(*out, Teuchos::VERB_HIGH);
    *out << std::endl;
#endif

    // SOLUTION TRANSFER
    // -----------------

    // Create a map operator. The operator settings are in the
    // "DataTransferKit" parameter list.
    Teuchos::ParameterList& dtk_list = plist->sublist("DataTransferKit");
    DataTransferKit::MapOperatorFactory op_factory;

    Teuchos::RCP<DataTransferKit::MapOperator> map_op =
        op_factory.create(src_vector->getMap(), tgt_vector->getMap(), dtk_list);

    // Setup the map operator. This creates the underlying linear operators.
    map_op->setup(src_manager.functionSpace(), tgt_manager.functionSpace());

    // Apply the map operator. This interpolates the data from one STK field
    // to the other.
    map_op->apply(*src_vector, *tgt_vector);

#ifdef DEBUG_OUTPUT
    *out << "   src_vector: \n ";
    src_vector->describe(*out, Teuchos::VERB_EXTREME);
    *out << "   tgt_vector: \n ";
    tgt_vector->describe(*out, Teuchos::VERB_EXTREME);
#endif

    // COMPUTE THE SOLUTION ERROR
    // --------------------------

    double* tgt_field_data;

    double* rel_err_field_data;

    double* abs_err_field_data;

    std::vector<stk::mesh::Entity> tgt_ownednodes;

    stk::mesh::Selector select_owned_in_part =
        stk::mesh::Selector(tgt_broker.meta_data().universal_part()) &
        stk::mesh::Selector(tgt_broker.meta_data().locally_owned_part());

    stk::mesh::get_selected_entities(
        select_owned_in_part,
        tgt_broker.bulk_data().buckets(stk::topology::NODE_RANK),
        tgt_ownednodes);

    int tgt_num_owned_nodes = tgt_ownednodes.size();  // number owned nodes

    stk::mesh::BucketVector tgt_part_buckets =
        tgt_stk_selector.get_buckets(stk::topology::NODE_RANK);

    std::vector<stk::mesh::Entity> tgt_part_nodes;

    stk::mesh::get_selected_entities(
        tgt_stk_selector, tgt_part_buckets, tgt_part_nodes);

    Intrepid::FieldContainer<double> tgt_node_coords =
        DataTransferKit::STKMeshHelpers::getEntityNodeCoordinates(
            Teuchos::Array<stk::mesh::Entity>(tgt_part_nodes), *tgt_bulk_data);

    int num_tgt_part_nodes =
        tgt_part_nodes.size();  // number nodes (owned + overlap)

#ifdef DEBUG_OUTPUT
    std::cout << "   proc #: " << comm->getRank() << ", tgt_num_owned_nodes = ";
    std::cout << tgt_num_owned_nodes << std::endl;
#endif

    double error_l2_norm_global_vec{0.0};
    double rel_error_l2_norm_global_vec{0.0};
    double field_l2_norm_global_vec{0.0};

    for (int component = 0; component < neq; component++) {
      double error_l2_norm_sq{0.0};
      double field_l2_norm_sq{0.0};

      for (int n = 0; n < num_tgt_part_nodes; ++n) {
        // reference solution (i.e., target_interp_field)
        double* gold_value =
            stk::mesh::field_data(target_interp_field, tgt_part_nodes[n]);

        tgt_field_data =
            stk::mesh::field_data(*target_field, tgt_part_nodes[n]);

        rel_err_field_data =
            stk::mesh::field_data(target_rel_error_field, tgt_part_nodes[n]);

        abs_err_field_data =
            stk::mesh::field_data(target_abs_error_field, tgt_part_nodes[n]);

        rel_err_field_data[component] =
            std::abs(tgt_field_data[component] - gold_value[component]);

        abs_err_field_data[component] =
            std::abs(tgt_field_data[component] - gold_value[component]);

        // IKT, 10/20/17: originally, the relative error was computed in the
        // next line. This can cause problems and erroneous-looking figures in
        // the case the reference solution is close to 0.  It makes more sense to
        // divife the absolute error by the norm of the reference solution rather
        // than a single-point value (done below).
        /*if (std::abs(gold_value[component]) > 1.0e-14) {
          rel_err_field_data[component] /= std::abs(gold_value[component]);
        }*/

#ifdef DEBUG_OUTPUT
        *out << "      tgt_field_data, gold_value, abs_err, rel_err: "
             << tgt_field_data[component] << ", " << gold_value[component]
             << ", " << abs_err_field_data[component] << ", "
             << rel_err_field_data[component] << std::endl;
#endif

        error_l2_norm_sq +=
            abs_err_field_data[component] * abs_err_field_data[component];
        field_l2_norm_sq +=
            tgt_field_data[component] * tgt_field_data[component];
      }

      double error_l2_norm_global;

      double field_l2_norm_global;

      double rel_error_l2_norm_global;

      Teuchos::reduceAll(
          *comm,
          Teuchos::REDUCE_SUM,
          1,
          &error_l2_norm_sq,
          &error_l2_norm_global);

      Teuchos::reduceAll(
          *comm,
          Teuchos::REDUCE_SUM,
          1,
          &field_l2_norm_sq,
          &field_l2_norm_global);

      error_l2_norm_global_vec += error_l2_norm_global;
      field_l2_norm_global_vec += field_l2_norm_global;

      error_l2_norm_global = std::sqrt(error_l2_norm_global);
      field_l2_norm_global = std::sqrt(field_l2_norm_global);
      if (std::abs(field_l2_norm_global) > 1.0e-14) {
        rel_error_l2_norm_global = error_l2_norm_global / field_l2_norm_global;
      } else {
        rel_error_l2_norm_global = 0.0;
      }

      if (scale_by_norm_soln_vec == false) {
        for (int n = 0; n < num_tgt_part_nodes; ++n) {
          rel_err_field_data =
              stk::mesh::field_data(target_rel_error_field, tgt_part_nodes[n]);
          if (field_l2_norm_global > 1.0e-14) {
            rel_err_field_data[component] /= field_l2_norm_global;
          }
        }
      }

      *out << "  Target Snapshot = " << tgt_time_step_indices[index]
           << ", Source Snapshot = " << src_time_step_indices[index]
           << std::endl;
      *out << "      Dof = " << component
           << ", |e|_2 (abs error): " << error_l2_norm_global << std::endl;
      *out << "      Dof = " << component
           << ", |f|_2 (norm ref soln): " << field_l2_norm_global << std::endl;
      *out << "      Dof = " << component
           << ", |e|_2 / |f|_2 (rel error): " << rel_error_l2_norm_global
           << std::endl;
      *out << "     "
              "-------------------------------------------------------------"
           << "--------------------------" << std::endl;
    }

    error_l2_norm_global_vec = std::sqrt(error_l2_norm_global_vec);
    field_l2_norm_global_vec = std::sqrt(field_l2_norm_global_vec);

    if (std::abs(field_l2_norm_global_vec) > 1.0e-14) {
      rel_error_l2_norm_global_vec =
          error_l2_norm_global_vec / field_l2_norm_global_vec;
    } else {
      rel_error_l2_norm_global_vec = 0.0;
    }

    *out << "  Target Snapshot = " << tgt_time_step_indices[index]
         << ", Source Snapshot = " << src_time_step_indices[index] << std::endl;
    *out << "      All dofs, |e|_2 (abs error): " << error_l2_norm_global_vec
         << std::endl;
    *out << "      All dofs, |f|_2 (norm ref soln): "
         << field_l2_norm_global_vec << std::endl;
    *out << "      All dofs, |e|_2 / |f|_2 (rel error): "
         << rel_error_l2_norm_global_vec << std::endl;
    *out << "     -------------------------------------------------------------"
         << "--------------------------" << std::endl;

    if (scale_by_norm_soln_vec == true) {
      for (int component = 0; component < neq; component++) {
        for (int n = 0; n < num_tgt_part_nodes; ++n) {
          rel_err_field_data =
              stk::mesh::field_data(target_rel_error_field, tgt_part_nodes[n]);
          if (field_l2_norm_global_vec > 1.0e-14) {
            rel_err_field_data[component] /= field_l2_norm_global_vec;
          }
        }
      }
    }

    // TARGET MESH WRITE
    // -----------------
    std::size_t tgt_output_index;
    if (index == 0) {
      tgt_output_index = tgt_broker.create_output_mesh(
          target_mesh_output_file, stk::io::WRITE_RESULTS);
      // Add output fields
      tgt_broker.add_field(tgt_output_index, target_interp_field);
      tgt_broker.add_field(tgt_output_index, target_rel_error_field);
      tgt_broker.add_field(tgt_output_index, target_abs_error_field);
      tgt_broker.add_field(tgt_output_index, *target_field);
    }
    double state_time =
        tgt_io_region->get_state_time(tgt_time_step_indices[index]);
#ifdef DEBUG_OUTPUT
    *out << "tgt_output_index, time = " << tgt_output_index << ", "
         << state_time << std::endl;
#endif
    // Write step
    tgt_broker.begin_output_step(tgt_output_index, state_time);
    tgt_broker.write_defined_output_fields(tgt_output_index);
    tgt_broker.end_output_step(tgt_output_index);
  }
}

namespace {

std::string
getFileExtension(std::string const& filename)
{
  auto const pos = filename.find_last_of(".");
  return filename.substr(pos + 1);
}

}  // anonymous namespace

int
main(int argc, char* argv[])
{
  // INITIALIZATION
  // --------------

  std::cout << "" << std::endl;

  // Setup communication.
  Teuchos::GlobalMPISession mpiSession(&argc, &argv);

  Teuchos::RCP<const Teuchos::Comm<int>> comm =
      Teuchos::DefaultComm<int>::getComm();

  // Read in command line options.
  std::string                   yaml_input_filename;
  Teuchos::CommandLineProcessor clp(false);

  clp.setOption(
      "yaml-in-file",
      &yaml_input_filename,
      "The XML file to read into a parameter list");

  clp.parse(argc, argv);

  Teuchos::RCP<Teuchos::FancyOStream> out =
      Teuchos::fancyOStream(Teuchos::VerboseObjectBase::getDefaultOStream());

  // Build the parameter list from the yaml input.
  Teuchos::RCP<Teuchos::ParameterList> plist =
      Teuchos::rcp(new Teuchos::ParameterList());

  std::string const input_extension = getFileExtension(yaml_input_filename);
  if (input_extension == "yaml" || input_extension == "yml") {
    Teuchos::updateParametersFromYamlFile(
        yaml_input_filename, Teuchos::inoutArg(*plist));
  } else {
    Teuchos::updateParametersFromXmlFile(
        yaml_input_filename, Teuchos::inoutArg(*plist));
  }

  std::string field_type = plist->get<std::string>("Field Type", "Node Vector");

  int field_type_num;

  if (field_type == "Node Vector") {
    field_type_num = 0;
  } else if (field_type == "Node Scalar") {
    field_type_num = 1;
  } else if (field_type == "Node Tensor") {
    field_type_num = 2;
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        Teuchos::Exceptions::InvalidParameter,
        std::endl
            << "Error in dtk_interp_and_error.cpp: invalid field_type = "
            << field_type
            << "!  Valid field_types are 'Node Vector', 'Node "
               "Scalar' and 'Node Tensor'."
            << std::endl);
  }

  switch (field_type_num) {
    case 0:  // VectorFieldType
    {
      *out << " Interpolating and calculating error in fields of type Node "
              "Vector..."
           << std::endl;
      interp_and_calc_error<stk::mesh::Field<double, stk::mesh::Cartesian>>(
          comm, plist);
      break;
    }
    case 1:  // ScalarFieldType
    {
      *out << " Interpolating and calculating error in fields of type Node "
              "Scalar..."
           << std::endl;
      interp_and_calc_error<stk::mesh::Field<double>>(comm, plist);
      break;
    }
    case 2:  // TensorFieldType
    {
      *out << " Interpolating and calculating error in fields of type Node "
              "Tensor..."
           << std::endl;
      interp_and_calc_error<stk::mesh::Field<double, shards::ArrayDimension>>(
          comm, plist);
      break;
    }
  }

  *out << " ...done!" << std::endl;

}  // end file dtk_interp_and_error.cpp
