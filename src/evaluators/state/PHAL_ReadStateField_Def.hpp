//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_AbstractDiscretization.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_Utils.hpp"
#include "PHAL_ReadStateField.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_DataLayout_MDALayout.hpp"
#include "Teuchos_TestForException.hpp"

#include "Albany_AbstractSTKFieldContainer.hpp"
#include "Albany_AbstractSTKMeshStruct.hpp"

namespace PHAL {

//
//
//
template <typename EvalT, typename Traits>
ReadStateField<EvalT, Traits>::ReadStateField(const Teuchos::ParameterList&)
{
  // States not read for generic type, only specializations
  this->setName("Read State Field");
}

//
//
//
template <typename EvalT, typename Traits>
void
ReadStateField<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData,
    PHX::FieldManager<Traits>&)
{
  // States not read for generic type, only specializations
}

//
//
//
template <typename EvalT, typename Traits>
void ReadStateField<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData /* workset */)
{
  // States not read for generic type, only specializations
}

//
//
//
template <typename Traits>
ReadStateField<PHAL::AlbanyTraits::Residual, Traits>::ReadStateField(
    const Teuchos::ParameterList& p)
{
  field_name  = p.get<std::string>("Field Name");
  state_name  = p.get<std::string>("State Name");
  auto layout = p.get<Teuchos::RCP<PHX::DataLayout>>("State Field Layout");
  field       = decltype(field)(field_name, layout);
  field_type  = layout->name(0);

  auto dummy    = Teuchos::rcp(new PHX::MDALayout<Dummy>(0));
  read_state_op = Teuchos::rcp(new PHX::Tag<ScalarT>(field_name, dummy));

  this->addEvaluatedField(*read_state_op);
  this->addEvaluatedField(field.fieldTag());

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
  if (field_type == PHX::print<Cell>()) {
    readElemState(workset);
  } else if (field_type == PHX::print<Node>()) {
    readNodalState(workset);
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(
        field_type == PHX::print<Cell>() || field_type == PHX::print<Node>(),
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

  auto disc = workset.disc;
  ALBANY_ASSERT(disc != Teuchos::null, "Null discretization");
  auto mesh = Teuchos::rcp_dynamic_cast<Albany::AbstractSTKMeshStruct>(
      disc->getMeshStruct());
  ALBANY_ASSERT(mesh != Teuchos::null, "Null STK mesh structure");

  stk::mesh::MetaData& metaData = *mesh->metaData;
  stk::mesh::BulkData& bulkData = *mesh->bulkData;

  auto const& elem_gid_wslid = disc->getElemGIDws();
  using LIDT                 = decltype(elem_gid_wslid.begin()->second.LID);
  using GIDT                 = decltype(elem_gid_wslid.begin()->first);
  std::map<LIDT, GIDT> elem_lid_2_gid;
  for (auto&& gid_wslid : elem_gid_wslid) {
    if (gid_wslid.second.ws == (int) workset.wsIndex) {
      elem_lid_2_gid.emplace(gid_wslid.second.LID, gid_wslid.first);
    }
  }

  std::vector<PHX::DataLayout::size_type> dims;
  field.dimensions(dims);

  switch (dims.size()) {
    case 2: {
      using SFT = Albany::AbstractSTKFieldContainer::ScalarFieldType;
      auto scalar_field =
          metaData.get_field<SFT>(stk::topology::ELEM_RANK, state_name);
      ALBANY_ASSERT(scalar_field != nullptr);
      for (unsigned int cell = 0; cell < workset.numCells; ++cell) {
        auto gid    = elem_lid_2_gid[cell];
        auto e      = bulkData.get_entity(stk::topology::ELEM_RANK, gid + 1);
        auto values = stk::mesh::field_data(*scalar_field, e);
        field(cell) = values[0];
      }
    } break;
    case 3: {
      using VFT = Albany::AbstractSTKFieldContainer::VectorFieldType;
      auto vector_field =
          metaData.get_field<VFT>(stk::topology::NODE_RANK, state_name);
      ALBANY_ASSERT(vector_field != nullptr);
      for (unsigned int cell = 0; cell < workset.numCells; ++cell) {
        auto gid    = elem_lid_2_gid[cell];
        auto e      = bulkData.get_entity(stk::topology::ELEM_RANK, gid + 1);
        auto values = stk::mesh::field_data(*vector_field, e);
        for (unsigned int i = 0; i < dims[2]; ++i) field(cell, 0, i) = values[i];
      }
    } break;
    default:
      ALBANY_ASSERT(false, "Unexpected dimension: only cell scalar/vector");
  }
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

  auto disc = workset.disc;
  ALBANY_ASSERT(disc != Teuchos::null, "Null discretization");
  auto mesh = Teuchos::rcp_dynamic_cast<Albany::AbstractSTKMeshStruct>(
      disc->getMeshStruct());
  ALBANY_ASSERT(mesh != Teuchos::null, "Null STK mesh structure");

  stk::mesh::MetaData& metaData = *mesh->metaData;
  stk::mesh::BulkData& bulkData = *mesh->bulkData;

  auto const& wsElgid = disc->getWsElNodeID();

  std::vector<PHX::DataLayout::size_type> dims;
  field.dimensions(dims);

  switch (dims.size()) {
    case 2: {
      using SFT = Albany::AbstractSTKFieldContainer::ScalarFieldType;
      auto scalar_field =
          metaData.get_field<SFT>(stk::topology::NODE_RANK, state_name);
      ALBANY_ASSERT(scalar_field != nullptr);
      for (unsigned int cell = 0; cell < workset.numCells; ++cell)
        for (unsigned int node = 0; node < dims[1]; ++node) {
          auto gid    = wsElgid[workset.wsIndex][cell][node];
          auto e      = bulkData.get_entity(stk::topology::NODE_RANK, gid + 1);
          auto values = stk::mesh::field_data(*scalar_field, e);
          field(cell, node) = values[0];
        }
    } break;
    case 3: {
      using VFT = Albany::AbstractSTKFieldContainer::VectorFieldType;
      auto vector_field =
          metaData.get_field<VFT>(stk::topology::NODE_RANK, state_name);
      ALBANY_ASSERT(vector_field != nullptr);
      for (unsigned int cell = 0; cell < workset.numCells; ++cell)
        for (unsigned int node = 0; node < dims[1]; ++node) {
          auto gid    = wsElgid[workset.wsIndex][cell][node];
          auto e      = bulkData.get_entity(stk::topology::NODE_RANK, gid + 1);
          auto values = stk::mesh::field_data(*vector_field, e);
          for (unsigned int i = 0; i < dims[2]; ++i) field(cell, node, i) = values[i];
        }
    } break;
    default:
      ALBANY_ASSERT(false, "Unexpected dimension: only cell scalar/vector");
  }
}

}  // namespace PHAL
