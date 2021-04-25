//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Shards_CellTopology.hpp"

#include "PHAL_LoadSideSetStateField.hpp"

#include "Albany_AbstractDiscretization.hpp"

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

  TEUCHOS_TEST_FOR_EXCEPTION (workset.sideSets==Teuchos::null, std::logic_error,
                              "Error! The mesh does not store any side set.\n");

  if (workset.sideSetViews->find(sideSetName)==workset.sideSetViews->end()) return; // Side set not present in this workset

  TEUCHOS_TEST_FOR_EXCEPTION (workset.disc==Teuchos::null, std::logic_error,
                              "Error! The workset must store a valid discretization pointer.\n");

  const Albany::AbstractDiscretization::SideSetDiscretizationsType& ssDiscs = workset.disc->getSideSetDiscretizations();

  TEUCHOS_TEST_FOR_EXCEPTION (ssDiscs.size()==0, std::logic_error,
                              "Error! The discretization must store side set discretizations.\n");

  TEUCHOS_TEST_FOR_EXCEPTION (ssDiscs.find(sideSetName)==ssDiscs.end(), std::logic_error,
                              "Error! No discretization found for side set " << sideSetName << ".\n");

  Teuchos::RCP<Albany::AbstractDiscretization> ss_disc = ssDiscs.at(sideSetName);

  TEUCHOS_TEST_FOR_EXCEPTION (ss_disc==Teuchos::null, std::logic_error,
                              "Error! Side discretization is invalid for side set " << sideSetName << ".\n");

  const std::map<std::string,std::map<GO,GO> >& ss_maps = workset.disc->getSideToSideSetCellMap();

  TEUCHOS_TEST_FOR_EXCEPTION (ss_maps.find(sideSetName)==ss_maps.end(), std::logic_error,
                              "Error! Something is off: the mesh has side discretization but no sideId-to-sideSetElemId map.\n");

  const std::map<GO,GO>& ss_map = ss_maps.at(sideSetName);

  // Get states from STK mesh
  Albany::StateArrays& state_arrays = ss_disc->getStateArrays();
  Albany::StateArrayVec& esa = state_arrays.elemStateArrays;
  Albany::WsLIDList& elemGIDws3D = workset.disc->getElemGIDws();
  Albany::WsLIDList& elemGIDws2D = ss_disc->getElemGIDws();

  // Get side_node->side_set_cell_node map from discretization
  TEUCHOS_TEST_FOR_EXCEPTION (workset.disc->getSideNodeNumerationMap().find(sideSetName)==workset.disc->getSideNodeNumerationMap().end(),
                              std::logic_error, "Error! Sideset " << sideSetName << " has no sideNodeNumeration map.\n");
  const std::map<GO,std::vector<int>>& sideNodeNumerationMap = workset.disc->getSideNodeNumerationMap().at(sideSetName);

  // Establishing the kind of field layout
  std::vector<PHX::DataLayout::size_type> dims;
  field.dimensions(dims);
  int size = dims.size();
  // Check the tag of the first extent after (side,) to determine if field is nodal
  const std::string& leading_field_tag = size>1 ? field.fieldTag().dataLayout().name(1) : "";
  TEUCHOS_TEST_FOR_EXCEPTION (size>1 && leading_field_tag!=PHX::print<Node>() && leading_field_tag!=PHX::print<Dim>() && leading_field_tag!=PHX::print<VecDim>(), std::logic_error,
                              "Error! Invalid field layout in LoadSideSetStateField.\n");

  // Loop on the sides of this sideSet that are in this workset
  sideSet = workset.sideSetViews->at(sideSetName);
  for (int sideSet_idx = 0; sideSet_idx < sideSet.size; ++sideSet_idx)
  {
    // Get the data that corresponds to the side
    const int elem_GID = sideSet.elem_GID(sideSet_idx);
    const int side_GID = sideSet.side_GID(sideSet_idx);

    // Not sure if this is even possible, but just for debug pourposes
    TEUCHOS_TEST_FOR_EXCEPTION (elemGIDws3D[ elem_GID ].ws != (int) workset.wsIndex, std::logic_error,
                                "Error! This workset has a side that belongs to an element not in the workset.\n");

    // We know the side ID, so we can fetch two things:
    //    1) the 2D-wsIndex where the 2D element lies
    //    2) the LID of the 2D element

    TEUCHOS_TEST_FOR_EXCEPTION (ss_map.find(side_GID)==ss_map.end(), std::logic_error,
                                "Error! The sideId-to-sideSetElemId map does not store this side GID. Weird, should never happen.\n");

    int ss_cell_GID = ss_map.at(side_GID);
    int wsIndex2D = elemGIDws2D[ss_cell_GID].ws;
    unsigned int ss_cell = elemGIDws2D[ss_cell_GID].LID;

    // Then, after a safety check, we extract the StateArray of the desired state in the right 2D-ws
    TEUCHOS_TEST_FOR_EXCEPTION (esa[wsIndex2D].find(stateName) == esa[wsIndex2D].end(), std::logic_error,
                                "Error! Cannot locate " << stateName << " in PHAL_LoadSideSetStateField_Def.\n");
    Albany::MDArray state = esa[wsIndex2D].at(stateName);

    const std::vector<int>& nodeMap = sideNodeNumerationMap.at(side_GID);

    // Now we have the two arrays: 3D and 2D. We need to take the 2D one
    // and put it at the right place in the 3D one

    switch (size)
    {
      case 1:
        // side set cell scalar
        field(sideSet_idx) = state(ss_cell);
        break;

      case 2:
        if (leading_field_tag==PHX::print<Node>())
        {
          // side set node scalar
          for (unsigned int node=0; node<dims[1]; ++node)
          {
            field(sideSet_idx,node) = state((int) ss_cell,nodeMap[node]);
          }
        }
        else
        {
          // side set cell vector/gradient
          for (unsigned int idim=0; idim<dims[1]; ++idim)
          {
            field(sideSet_idx,idim) = state(ss_cell,idim);
          }
        }
        break;

      case 3:
        if (leading_field_tag==PHX::print<Node>())
        {
          // side set node vector/gradient
          for (unsigned int node=0; node<dims[1]; ++node)
          {
            for (unsigned int dim=0; dim<dims[2]; ++dim)
              field(sideSet_idx,node,dim) = state((int) ss_cell, nodeMap[node], (int) dim);
          }
        }
        else
        {
          // side set cell tensor
          for (unsigned int idim=0; idim<dims[1]; ++idim)
          {
            for (unsigned int jdim=0; jdim<dims[2]; ++jdim)
              field(sideSet_idx,idim,jdim) = state(ss_cell,idim,jdim);
          }
        }
        break;

      default:
        TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error,
                                    "Error! Unexpected array dimensions in LoadSideSetStateField: " << size << ".\n");
    }
  }
}

} // Namespace PHAL
