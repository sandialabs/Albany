//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "PHAL_LoadSideSetStateField.hpp"

#include "Albany_STKDiscretization.hpp"

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Shards_CellTopology.hpp"

namespace PHAL
{

template<typename EvalT, typename Traits, typename ScalarType>
LoadSideSetStateFieldBase<EvalT, Traits, ScalarType>::
LoadSideSetStateFieldBase (const Teuchos::ParameterList& p)
{
  sideSetName = p.get<std::string>("Side Set Name");

  fieldName = p.get<std::string>("Field Name");
  stateName = p.get<std::string>("State Name");

  field  = PHX::MDField<ScalarType>(fieldName, p.get<Teuchos::RCP<PHX::DataLayout> >("Field Layout") );

  this->addEvaluatedField (field);

  const auto& phx_dl = field.fieldTag().dataLayout();
  TEUCHOS_TEST_FOR_EXCEPTION(phx_dl.name(0)!=PHX::print<Side>(), std::runtime_error,
      "Error! To load a side-set state, the first tag of the layout MUST be 'Side'.\n");

  const auto rank = phx_dl.rank();
  nodalState = rank>1 ? phx_dl.name(1)==PHX::print<Node>() : false;
  if (nodalState) {
    TEUCHOS_TEST_FOR_EXCEPTION (rank!=2 && rank!=3, std::runtime_error,
        "Error! Only Scalar and Vector field supported for nodal states.\n");
    TEUCHOS_TEST_FOR_EXCEPTION(
        rank>2 && phx_dl.name(2)!=PHX::print<Dim>() && phx_dl.name(2)!=PHX::print<LayerDim>(),
        std::runtime_error,
        "Error! To load a side-set nodal state, the third tag (if present) MUST be 'Dim'.\n")
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION (rank!=1 && rank!=2, std::runtime_error,
        "Error! Only Scalar and Vector field supported for elem states.\n");
    TEUCHOS_TEST_FOR_EXCEPTION(
        rank>2 && phx_dl.name(2)!=PHX::print<Dim>() && phx_dl.name(2)!=PHX::print<LayerDim>(),
         std::runtime_error,
        "Error! To save a side-set elem state, the second tag (if present) MUST be 'Dim'.\n");
  }

  this->setName ("Load Side Set Field " + fieldName + " from Side Set State " + stateName 
    + PHX::print<EvalT>());
}

// **********************************************************************
template<typename EvalT, typename Traits, typename ScalarType>
void LoadSideSetStateFieldBase<EvalT, Traits, ScalarType>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(field,fm);

  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
  if (d.memoizer_active()) memoizer.enable_memoizer();
}

template<typename EvalT, typename Traits, typename ScalarType>
void LoadSideSetStateFieldBase<EvalT, Traits, ScalarType>::
evaluateFields(typename Traits::EvalData workset)
{
  if (memoizer.have_saved_data(workset,this->evaluatedFields())) return;

  if (this->nodalState)
    loadNodeState(workset);
  else
    loadElemState(workset);
}

template<typename EvalT, typename Traits, typename ScalarType>
void LoadSideSetStateFieldBase<EvalT, Traits, ScalarType>::
loadNodeState(typename Traits::EvalData workset)
{
  TEUCHOS_TEST_FOR_EXCEPTION (workset.sideSets==Teuchos::null, std::logic_error,
    "Error! The mesh does not store any side set.\n");

  if (workset.sideSetViews->find(sideSetName)==workset.sideSetViews->end())
    return; // Side set not present in this workset

  TEUCHOS_TEST_FOR_EXCEPTION (workset.disc==Teuchos::null, std::logic_error,
    "Error! The workset must store a valid discretization pointer.\n");

  const auto& ssDiscs = workset.disc->getSideSetDiscretizations();

  TEUCHOS_TEST_FOR_EXCEPTION (ssDiscs.size()==0, std::logic_error,
      "Error! The discretization must store side set discretizations.\n");

  TEUCHOS_TEST_FOR_EXCEPTION (ssDiscs.find(sideSetName)==ssDiscs.end(), std::logic_error,
      "Error! No discretization found for side set " << sideSetName << ".\n");

  const auto& ss_disc = ssDiscs.at(sideSetName);

  TEUCHOS_TEST_FOR_EXCEPTION (ss_disc==Teuchos::null, std::logic_error,
      "Error! Side discretization is invalid for side set " << sideSetName << ".\n");

  // Get side disc STK bulk/meta data
  const auto& metaData = Teuchos::rcp_dynamic_cast<Albany::STKDiscretization>(ss_disc)->getSTKMetaData();
  const auto& bulkData = Teuchos::rcp_dynamic_cast<Albany::STKDiscretization>(ss_disc)->getSTKBulkData();

  // Get local node numeration map from the disc
  const auto& ssNodeNumerationMaps = workset.disc->getSideNodeNumerationMap();
  TEUCHOS_TEST_FOR_EXCEPTION (ssNodeNumerationMaps.find(sideSetName)==ssNodeNumerationMaps.end(),
      std::logic_error, "Error! Sideset " << sideSetName << " has no sideNodeNumeration map.\n");

  // Establishing the kind of field layout
  std::vector<PHX::DataLayout::size_type> dims;
  field.dimensions(dims);

  // Get the stk field
  typedef Albany::AbstractSTKFieldContainer::STKFieldType SFT;
  SFT* stk_field = metaData.template get_field<double>(stk::topology::NODE_RANK,stateName);
  TEUCHOS_TEST_FOR_EXCEPTION (stk_field==nullptr, std::runtime_error,
      "Error! STK Field ptr is null.\n");
  
  int numNodes = dims[1];

  // When the 3d mesh is built online, the gid of the 3d-mesh side *should*
  // match that of the 2d-mesh cell. HOWEVER, when the mesh is saved, then
  // loaded in a future run, stk might reassign side gids. In that case,
  // the life line is given by the map in the STK discretization, which
  // associates to each 3d-side GID the corresponding 2d-cell GID
  const auto& side3d_to_cell2d = workset.disc->getSideToSideSetCellMap().at(sideSetName);

  auto sideSet = workset.sideSetViews->at(sideSetName);
  for (int sideSet_idx = 0; sideSet_idx < sideSet.size; ++sideSet_idx) {
    // Get the side GID
    const int side_GID = sideSet.side_GID.h_view(sideSet_idx);

    // Get the lid ordering map
    // Recall: map[i] = j means that the i-th node in the 3d side is the j-th node in the 2d cell
    const auto& node_map = ssNodeNumerationMaps.at(sideSetName).at(side_GID);

    // Get the cell in the 2d mesh
    const auto cell2d_GID = side3d_to_cell2d.at(side_GID);
    const auto cell2d = bulkData.get_entity(stk::topology::ELEM_RANK, cell2d_GID+1);
    const auto nodes2d = bulkData.begin_nodes(cell2d);

    if (dims.size() == 2) {
      auto field_d_view = Kokkos::subview(field.get_view(), sideSet_idx, Kokkos::ALL);
      auto field_d_view_tmp = Kokkos::View<ScalarType*, PHX::Device>("field_d_view_tmp",numNodes);
      auto field_h_mirror = Kokkos::create_mirror_view(field_d_view_tmp);
      for (int inode=0; inode<numNodes; ++inode) {
        const double* data = stk::mesh::field_data(*stk_field,nodes2d[node_map[inode]]);
        field_h_mirror(inode) = *data;
      }
      Kokkos::deep_copy(field_d_view_tmp, field_h_mirror);
      Kokkos::deep_copy(field_d_view, field_d_view_tmp);
    } else if (dims.size() == 3) {
      auto field_d_view = Kokkos::subview(field.get_view(), sideSet_idx, Kokkos::ALL, Kokkos::ALL);
      auto field_d_view_tmp = Kokkos::View<ScalarType**, PHX::Device>("field_d_view_tmp",numNodes,dims[2]);;
      auto field_h_mirror = Kokkos::create_mirror_view(field_d_view_tmp);
      for (int inode=0; inode<numNodes; ++inode) {
        const double* data = stk::mesh::field_data(*stk_field,nodes2d[node_map[inode]]);
        for (int idim=0; idim<static_cast<int>(dims[2]); ++idim) {
          field_h_mirror(inode,idim) = data[idim];
        }
      }
      Kokkos::deep_copy(field_d_view_tmp, field_h_mirror);
      Kokkos::deep_copy(field_d_view, field_d_view_tmp);
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error,
          "Error! Unsupported field dimension. However, you should have gotten an error before!\n");
    }
  }
}

template<typename EvalT, typename Traits, typename ScalarType>
void LoadSideSetStateFieldBase<EvalT, Traits, ScalarType>::
loadElemState(typename Traits::EvalData workset)
{
  TEUCHOS_TEST_FOR_EXCEPTION (workset.sideSets==Teuchos::null, std::logic_error,
    "Error! The mesh does not store any side set.\n");

  if (workset.sideSetViews->find(sideSetName)==workset.sideSetViews->end())
    return; // Side set not present in this workset

  TEUCHOS_TEST_FOR_EXCEPTION (workset.disc==Teuchos::null, std::logic_error,
    "Error! The workset must store a valid discretization pointer.\n");

  const auto& ssDiscs = workset.disc->getSideSetDiscretizations();

  TEUCHOS_TEST_FOR_EXCEPTION (ssDiscs.size()==0, std::logic_error,
      "Error! The discretization must store side set discretizations.\n");

  TEUCHOS_TEST_FOR_EXCEPTION (ssDiscs.find(sideSetName)==ssDiscs.end(), std::logic_error,
      "Error! No discretization found for side set " << sideSetName << ".\n");

  const auto& ss_disc = ssDiscs.at(sideSetName);

  TEUCHOS_TEST_FOR_EXCEPTION (ss_disc==Teuchos::null, std::logic_error,
      "Error! Side discretization is invalid for side set " << sideSetName << ".\n");

  // Get side disc STK bulk/meta data
  const auto& metaData = Teuchos::rcp_dynamic_cast<Albany::STKDiscretization>(ss_disc)->getSTKMetaData();
  const auto& bulkData = Teuchos::rcp_dynamic_cast<Albany::STKDiscretization>(ss_disc)->getSTKBulkData();

  // Establishing the kind of field layout
  std::vector<PHX::DataLayout::size_type> dims;
  field.dimensions(dims);

  // Get the stk field
  typedef Albany::AbstractSTKFieldContainer::STKFieldType SFT;
  SFT* stk_field = metaData.template get_field<double>(stk::topology::ELEM_RANK,stateName);
  TEUCHOS_TEST_FOR_EXCEPTION (stk_field==nullptr, std::runtime_error,
      "Error! STK field ptr is null.\n");

  // Loop on the sides of this sideSet that are in this workset
  auto sideSet = workset.sideSetViews->at(sideSetName);
  for (int sideSet_idx = 0; sideSet_idx < sideSet.size; ++sideSet_idx) {
    // Get the side GID
    const int side_GID = sideSet.side_GID.h_view(sideSet_idx);

    // Get the cell in the 2d mesh
    const auto cell2d = bulkData.get_entity(stk::topology::ELEM_RANK, side_GID+1);

    const double* data = stk::mesh::field_data(*stk_field,cell2d);
    if (dims.size() == 1) {
      auto field_d_view = Kokkos::subview(field.get_view(), sideSet_idx);
      Kokkos::deep_copy(field_d_view, *data);
    } else if (dims.size() == 2) {
      auto field_d_view = Kokkos::subview(field.get_view(), sideSet_idx, Kokkos::ALL);
      auto field_d_view_tmp = Kokkos::View<ScalarType*, PHX::Device>("field_d_view_tmp",dims[1]);
      auto field_h_mirror = Kokkos::create_mirror_view(field_d_view);
      for (int idim=0; idim<static_cast<int>(dims[1]); ++idim) {
        field_h_mirror(sideSet_idx,idim) = data[idim];
      }
      Kokkos::deep_copy(field_d_view_tmp, field_h_mirror);
      Kokkos::deep_copy(field_d_view, field_d_view_tmp);
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error,
          "Error! Unsupported field dimension. However, you should have gotten an error before!\n");
    }
  }
}

} // Namespace PHAL
