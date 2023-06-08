//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_GeneralPurposeFieldsNames.hpp"
#include "Albany_Layouts.hpp"
#include "PHAL_SaveSideSetStateField.hpp"
#include "Albany_STKDiscretization.hpp"

#include "Phalanx_DataLayout.hpp"
#include "Teuchos_TestForException.hpp"
#include <Teuchos_RCPDecl.hpp>
#include <stdexcept>

namespace PHAL
{

// **********************************************************************
template<typename Traits>
SaveSideSetStateField<PHAL::AlbanyTraits::Residual, Traits>::
SaveSideSetStateField (const Teuchos::ParameterList& p,
                       const Teuchos::RCP<Albany::Layouts>& dl)
{
  TEUCHOS_TEST_FOR_EXCEPTION (!dl->isSideLayouts, std::runtime_error,
      "Error! Input Layouts struct is not that of a sideset.\n");

  sideSetName   = p.get<std::string>("Side Set Name");

  fieldName = p.get<std::string>("Field Name");
  stateName = p.get<std::string>("State Name");

  rank = p.get<FRT>("Field Rank");
  loc  = p.get<FL>("Field Location");
  TEUCHOS_TEST_FOR_EXCEPTION (loc!=FL::Cell && loc!=FL::Node, std::runtime_error,
      "Error! Only Node and Cell field location supported.\n")
  TEUCHOS_TEST_FOR_EXCEPTION (loc==FL::Node && rank==FRT::Gradient, std::runtime_error,
      "Error! Gradient fields only supported if at Cell location.\n");

  auto layout = Albany::get_field_layout(rank,loc,dl);
  
  field = decltype(field)(fieldName, layout);

  savestate_operation = Teuchos::rcp(new PHX::Tag<ScalarT>(fieldName, dl->dummy));

  this->addDependentField (field.fieldTag());
  this->addEvaluatedField (*savestate_operation);

  if (rank==FRT::Gradient) {
    const auto tangents_name = Albany::tangents_name + "_" + sideSetName;
    const auto w_meas_name   = Albany::weighted_measure_name + "_" + sideSetName;

    tangents = decltype(tangents)(tangents_name, dl->qp_tensor_cd_sd);
    this->addDependentField(tangents);

    w_measure = decltype(w_measure)(tangents_name, dl->qp_tensor_cd_sd);
    this->addDependentField(w_measure);
  }

  numQPs = dl->qp_scalar->dimension(1);
  numNodes = dl->node_scalar->dimension(1);

  this->setName ("Save Side Set Field " + fieldName + " to Side Set State " + stateName + " <Residual>");
}

// **********************************************************************
template<typename Traits>
void SaveSideSetStateField<PHAL::AlbanyTraits::Residual, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(field,fm);
  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
  if (d.memoizer_active()) memoizer.enable_memoizer();
}
// **********************************************************************
template<typename Traits>
void SaveSideSetStateField<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  if (memoizer.have_saved_data(workset,this->evaluatedFields())) return;

  if (loc==FL::Node)
    saveNodeState(workset);
  else
    saveElemState(workset);
}

template<typename Traits>
void SaveSideSetStateField<PHAL::AlbanyTraits::Residual, Traits>::
saveElemState(typename Traits::EvalData workset)
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
    "Error! STK Field ptr is null.\n");

  // Loop on the sides of this sideSet that are in this workset
  auto sideSet = workset.sideSetViews->at(sideSetName);
  double tan_cell_val, meas;
  for (int sideSet_idx = 0; sideSet_idx < sideSet.size; ++sideSet_idx) {
    // Get the side GID
    const int side_GID = sideSet.side_GID.h_view(sideSet_idx);

    // Get the cell in the 2d mesh
    const auto cell2d = bulkData.get_entity(stk::topology::ELEM_RANK, side_GID+1);

    double* data = stk::mesh::field_data(*stk_field,cell2d);
    switch (rank) {
      case FRT::Scalar:
        *data = field(sideSet_idx);
        break;
      case FRT::Vector:
        for (int idim=0; idim<static_cast<int>(dims[1]); ++idim) {
          data[idim] = field(sideSet_idx,idim);
        }
        break;
      case FRT::Gradient:
        meas = 0;
        for (int qp=0; qp<numQPs; ++qp) {
          meas += w_measure(sideSet_idx,qp);
        }
        for (int idim=0; idim<static_cast<int>(dims[1]); ++idim) {
          data[idim] = 0.0;
          for (int itan=0; itan<static_cast<int>(dims[1]); ++itan) {
            tan_cell_val = 0;
            for (int qp=0; qp<numQPs; ++qp) {
              tan_cell_val += tangents(sideSet_idx,qp,idim,itan)*w_measure(sideSet_idx,qp);
            }
            data[idim] +=  (tan_cell_val/meas) * field(sideSet_idx,itan);
          }
        }
        break;
      case FRT::Tensor:
        for (int idim=0; idim<static_cast<int>(dims[1]); ++idim) {
          for (int jdim=0; jdim<static_cast<int>(dims[2]); ++jdim) {
            data[idim*dims[1]+jdim] = field(sideSet_idx,idim,jdim);
        }}
        break;
    }
  }
}

template<typename Traits>
void SaveSideSetStateField<PHAL::AlbanyTraits::Residual, Traits>::
saveNodeState(typename Traits::EvalData workset)
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
      "Error! STK field ptr is null.\n");

  // Loop on the sides of this sideSet that are in this workset
  auto sideSet = workset.sideSetViews->at(sideSetName);
  for (int sideSet_idx = 0; sideSet_idx < sideSet.size; ++sideSet_idx) {
    // Get the side GID
    const int side_GID = sideSet.side_GID.h_view(sideSet_idx);

    // Get the lid ordering map
    // Recall: map[i] = j means that the i-th node in the 3d side is the j-th node in the 2d cell
    const auto& node_map = ssNodeNumerationMaps.at(sideSetName).at(side_GID);

    // Get the cell in the 2d mesh
    const auto cell2d = bulkData.get_entity(stk::topology::ELEM_RANK, side_GID+1);
    const auto nodes2d = bulkData.begin_nodes(cell2d);

    for (int inode=0; inode<numNodes; ++inode) {
      double* data = stk::mesh::field_data(*stk_field,nodes2d[node_map[inode]]);
      switch (rank) {
        case FRT::Scalar:
          *data = field(sideSet_idx,inode);
          break;
        case FRT::Vector:
          for (int idim=0; idim<static_cast<int>(dims[2]); ++idim) {
            data[idim] = field(sideSet_idx,inode,idim);
          }
          break;
        case FRT::Gradient:
          for (int idim=0; idim<static_cast<int>(dims[2]); ++idim) {
            for (int jdim=0; jdim<static_cast<int>(dims[3]); ++jdim) {
              data[idim*dims[2]+jdim] = field(sideSet_idx,inode,idim,jdim);
          }}
          break;
        default:
          TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error,
              "Error! Unsupported field dimension. However, you should have gotten an error before!\n");
      }
    }
  }
}

} // Namespace PHAL
