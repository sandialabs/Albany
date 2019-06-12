//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_ThyraUtils.hpp"
#include "PHAL_ReadStateField.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_DataLayout_MDALayout.hpp"
#include "Teuchos_TestForException.hpp"

#ifdef ALBANY_STK
#include "Albany_AbstractSTKFieldContainer.hpp"
#include "Albany_AbstractSTKMeshStruct.hpp"
#endif
#include "Albany_AbstractDiscretization.hpp"

namespace PHAL {

template <typename EvalT, typename Traits>
ReadStateField<EvalT, Traits>::ReadStateField(
    const Teuchos::ParameterList& /* p */)
{
  // States Not REad for Generic Type, only Specializations
  this->setName("Read State Field");
}

//
//
//
template <typename EvalT, typename Traits>
void
ReadStateField<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData /* d */,
    PHX::FieldManager<Traits>& /* fm */)
{
  // States Not Read for Generic Type, only Specializations
}

//
//
//
template <typename EvalT, typename Traits>
void ReadStateField<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData /* workset */)
{
  // States Not Saved for Generic Type, only Specializations
}

//
//
//
template <typename Traits>
ReadStateField<PHAL::AlbanyTraits::Residual, Traits>::ReadStateField(
    const Teuchos::ParameterList& p)
{
  field_name = p.get<std::string>("Field Name");
  state_name = p.get<std::string>("State Name");

  Teuchos::RCP<PHX::DataLayout> layout =
      p.get<Teuchos::RCP<PHX::DataLayout>>("State Field Layout");
  field      = decltype(field)(field_name, layout);
  field_type = layout->name(0);

  Teuchos::RCP<PHX::DataLayout> dummy =
      Teuchos::rcp(new PHX::MDALayout<Dummy>(0));
  read_state_operation = Teuchos::rcp(new PHX::Tag<ScalarT>(field_name, dummy));

  this->addDependentField(field.fieldTag());
  this->addEvaluatedField(*read_state_operation);

  this->setName(
      "Read Field " + field_name + " to State " + state_name + "Residual");
}

//
//
//
template <typename Traits>
void
ReadStateField<PHAL::AlbanyTraits::Residual, Traits>::postRegistrationSetup(
    typename Traits::SetupData /* d */,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(field, fm);
}

//
//
//
template <typename Traits>
void
ReadStateField<PHAL::AlbanyTraits::Residual, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  if (field_type == "Cell") {
    readElemState(workset);
  } else if (field_type == "Node") {
    readNodalState(workset);
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(
        field_type == "Cell" || field_type == "Node",
        std::runtime_error,
        "Error! Only read cell or node states for now.\n");
  }
}

template <typename Traits>
void
ReadStateField<PHAL::AlbanyTraits::Residual, Traits>::readElemState(
    typename Traits::EvalData workset)
{
  // Note: to read cell fields, we need to open up the mesh, and work directly
  // on it.
  //       For this reason, we assume the mesh is stk. Moreover, since the cell
  //       buckets and node buckets do not coincide (elem-bucket 12 does not
  //       necessarily contain all nodes in node-bucket 12), we need to work
  //       with stk fields. To do this we must extract entities from the bulk
  //       data and use them to access the values of the stk field.

#ifdef ALBANY_STK
  Teuchos::RCP<Albany::AbstractDiscretization> disc = workset.disc;
  TEUCHOS_TEST_FOR_EXCEPTION(
      disc == Teuchos::null,
      std::runtime_error,
      "Error! Discretization is needed to read a cell state.\n");

  Teuchos::RCP<Albany::AbstractSTKMeshStruct> mesh =
      Teuchos::rcp_dynamic_cast<Albany::AbstractSTKMeshStruct>(
          disc->getMeshStruct());
  TEUCHOS_TEST_FOR_EXCEPTION(
      mesh == Teuchos::null,
      std::runtime_error,
      "Error! Read cell state available only for stk meshes.\n");

  stk::mesh::MetaData& metaData = *mesh->metaData;
  stk::mesh::BulkData& bulkData = *mesh->bulkData;

  typedef Albany::AbstractSTKFieldContainer::ScalarFieldType SFT;
  typedef Albany::AbstractSTKFieldContainer::VectorFieldType VFT;

  SFT* scalar_field;
  VFT* vector_field;

  auto cell_vs = disc->getVectorSpace(state_name);

  std::vector<PHX::DataLayout::size_type> dims;
  field.dimensions(dims);

  GO                cellId;
  double*           values;
  auto const        num_local_cells = Albany::getNumLocalElements(cell_vs);
  stk::mesh::Entity e;

  switch (dims.size()) {
    case 2:  // cell_scalar
      scalar_field =
          metaData.get_field<SFT>(stk::topology::ELEM_RANK, state_name);
      TEUCHOS_TEST_FOR_EXCEPTION(
          scalar_field == 0, std::runtime_error, "Error! Field not found.\n");

      for (int cell = 0; cell < num_local_cells; ++cell) {
        cellId      = Albany::getGlobalElement(cell_vs, cell);
        e           = bulkData.get_entity(stk::topology::ELEM_RANK, cellId + 1);
        values      = stk::mesh::field_data(*scalar_field, e);
        field(cell) = values[0];
      }
      break;
    case 3:  // cell_vector
      vector_field =
          metaData.get_field<VFT>(stk::topology::NODE_RANK, state_name);
      TEUCHOS_TEST_FOR_EXCEPTION(
          vector_field == 0, std::runtime_error, "Error! Field not found.\n");
      for (int cell = 0; cell < num_local_cells; ++cell) {
        cellId = Albany::getGlobalElement(cell_vs, cell);
        e      = bulkData.get_entity(stk::topology::NODE_RANK, cellId + 1);
        values = stk::mesh::field_data(*vector_field, e);
        for (int i = 0; i < dims[2]; ++i) field(cell, i) = values[i];
      }
      break;
    default:  // error!
      TEUCHOS_TEST_FOR_EXCEPTION(
          true,
          std::runtime_error,
          "Error! Unexpected field dimension (only cell_scalar/cell_vector for "
          "now).\n");
  }
#else
  (void)workset;
  TEUCHOS_TEST_FOR_EXCEPTION(
      true,
      std::runtime_error,
      "Error! Cell states only available for stk meshes, but Trilinos was "
      "compiled without STK!\n");
#endif
}

template <typename Traits>
void
ReadStateField<PHAL::AlbanyTraits::Residual, Traits>::readNodalState(
    typename Traits::EvalData workset)
{
  // Note: to read nodal fields, we need to open up the mesh, and work directly
  // on it.
  //       For this reason, we assume the mesh is stk. Moreover, since the cell
  //       buckets and node buckets do not coincide (elem-bucket 12 does not
  //       necessarily contain all nodes in node-bucket 12), we need to work
  //       with stk fields. To do this we must extract entities from the bulk
  //       data and use them to access the values of the stk field.

#ifdef ALBANY_STK
  Teuchos::RCP<Albany::AbstractDiscretization> disc = workset.disc;
  TEUCHOS_TEST_FOR_EXCEPTION(
      disc == Teuchos::null,
      std::runtime_error,
      "Error! Discretization is needed to read a nodal state.\n");

  Teuchos::RCP<Albany::AbstractSTKMeshStruct> mesh =
      Teuchos::rcp_dynamic_cast<Albany::AbstractSTKMeshStruct>(
          disc->getMeshStruct());
  TEUCHOS_TEST_FOR_EXCEPTION(
      mesh == Teuchos::null,
      std::runtime_error,
      "Error! Read nodal state available only for stk meshes.\n");

  stk::mesh::MetaData& metaData = *mesh->metaData;
  stk::mesh::BulkData& bulkData = *mesh->bulkData;

  const auto& wsElNodeID = disc->getWsElNodeID();

  typedef Albany::AbstractSTKFieldContainer::ScalarFieldType SFT;
  typedef Albany::AbstractSTKFieldContainer::VectorFieldType VFT;

  SFT* scalar_field;
  VFT* vector_field;

  std::vector<PHX::DataLayout::size_type> dims;
  field.dimensions(dims);

  GO                nodeId;
  double*           values;
  stk::mesh::Entity e;
  switch (dims.size()) {
    case 2:  // node_scalar
      scalar_field =
          metaData.get_field<SFT>(stk::topology::NODE_RANK, state_name);
      TEUCHOS_TEST_FOR_EXCEPTION(
          scalar_field == 0, std::runtime_error, "Error! Field not found.\n");
      for (int cell = 0; cell < workset.numCells; ++cell)
        for (int node = 0; node < dims[1]; ++node) {
          nodeId    = wsElNodeID[workset.wsIndex][cell][node];
          e         = bulkData.get_entity(stk::topology::NODE_RANK, nodeId + 1);
          values    = stk::mesh::field_data(*scalar_field, e);
          values[0] = field(cell, node);
        }
      break;
    case 3:  // node_vector
      vector_field =
          metaData.get_field<VFT>(stk::topology::NODE_RANK, state_name);
      TEUCHOS_TEST_FOR_EXCEPTION(
          vector_field == 0, std::runtime_error, "Error! Field not found.\n");
      for (int cell = 0; cell < workset.numCells; ++cell)
        for (int node = 0; node < dims[1]; ++node) {
          nodeId = wsElNodeID[workset.wsIndex][cell][node];
          e      = bulkData.get_entity(stk::topology::NODE_RANK, nodeId + 1);
          values = stk::mesh::field_data(*vector_field, e);
          for (int i = 0; i < dims[2]; ++i) values[i] = field(cell, node, i);
        }
      break;
    default:  // error!
      TEUCHOS_TEST_FOR_EXCEPTION(
          true,
          std::runtime_error,
          "Error! Unexpected field dimension (only node_scalar/node_vector for "
          "now).\n");
  }
#else
  (void)workset;
  TEUCHOS_TEST_FOR_EXCEPTION(
      true,
      std::runtime_error,
      "Error! Nodal states only available for stk meshes, but Trilinos was "
      "compiled without STK!\n");
#endif
}
}  // namespace PHAL
