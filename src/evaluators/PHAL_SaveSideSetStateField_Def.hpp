//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <vector>
#include <string>

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

namespace PHAL
{

template<typename EvalT, typename Traits>
SaveSideSetStateField<EvalT, Traits>::
SaveSideSetStateField (const Teuchos::ParameterList& p,
                       const Teuchos::RCP<Albany::Layouts>& dl)
{
  // States Not Saved for Generic Type, only Specializations
  this->setName("Save Side Set State Field"+PHX::typeAsString<EvalT>());
}

// **********************************************************************
template<typename EvalT, typename Traits>
void SaveSideSetStateField<EvalT, Traits>::
postRegistrationSetup (typename Traits::SetupData d,
                       PHX::FieldManager<Traits>& fm)
{
  // States Not Saved for Generic Type, only Specializations
}

// **********************************************************************
template<typename EvalT, typename Traits>
void SaveSideSetStateField<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  // States Not Saved for Generic Type, only Specializations
}
// **********************************************************************

// **********************************************************************
template<typename Traits>
SaveSideSetStateField<PHAL::AlbanyTraits::Residual, Traits>::
SaveSideSetStateField (const Teuchos::ParameterList& p,
                       const Teuchos::RCP<Albany::Layouts>& dl)
{
  sideSetName   = p.get<std::string>("Side Set Name");

  fieldName = p.get<std::string>("Field Name");
  stateName = p.get<std::string>("State Name");

  PHX::MDField<ScalarT> f(fieldName, p.get<Teuchos::RCP<PHX::DataLayout> >("Field Layout") );
  field = f;

  savestate_operation = Teuchos::rcp(new PHX::Tag<ScalarT>(fieldName, dl->dummy));

  this->addDependentField (field.fieldTag());
  this->addEvaluatedField (*savestate_operation);

  this->setName ("Save Side Set Field " + fieldName + " to Side Set State " + stateName + " <Residual>");
}

// **********************************************************************
template<typename Traits>
void SaveSideSetStateField<PHAL::AlbanyTraits::Residual, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(field,fm);
}
// **********************************************************************
template<typename Traits>
void SaveSideSetStateField<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  TEUCHOS_TEST_FOR_EXCEPTION (workset.sideSets==Teuchos::null, std::logic_error,
                              "Error! The mesh does not store any side set.\n");

  if(workset.sideSets->find(sideSetName) ==workset.sideSets->end())
    return; // Side set not present in this workset

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
  const std::string& tag2 = size>2 ? field.fieldTag().dataLayout().name(2) : "";
  TEUCHOS_TEST_FOR_EXCEPTION (size>2 && tag2!="Node" && tag2!="Dim" && tag2!="VecDim", std::logic_error,
                              "Error! Invalid field layout in SaveSideSetStateField.\n");

  // Loop on the sides of this sideSet that are in this workset
  const std::vector<Albany::SideStruct>& sideSet = workset.sideSets->at(sideSetName);
  for (auto const& it_side : sideSet)
  {
    // Get the data that corresponds to the side
    const int side_GID = it_side.side_GID;
    const int cell     = it_side.elem_LID;
    const int side     = it_side.side_local_id;

    // Not sure if this is even possible, but just for debug pourposes
    TEUCHOS_TEST_FOR_EXCEPTION (elemGIDws3D[ it_side.elem_GID ].ws != workset.wsIndex, std::logic_error,
                                "Error! This workset has a side that belongs to an element not in the workset.\n");

    // We know the side ID, so we can fetch two things:
    //    1) the 2D-wsIndex where the 2D element lies
    //    2) the LID of the 2D element

    TEUCHOS_TEST_FOR_EXCEPTION (ss_map.find(side_GID)==ss_map.end(), std::logic_error,
                                "Error! The sideId-to-sideSetElemId map does not store this side GID. Weird, should never happen.\n");

    int ss_cell_GID = ss_map.at(side_GID);
    int wsIndex2D = elemGIDws2D[ss_cell_GID].ws;
    int ss_cell = elemGIDws2D[ss_cell_GID].LID;

    // Then, after a safety check, we extract the StateArray of the desired state in the right 2D-ws
    TEUCHOS_TEST_FOR_EXCEPTION (esa[wsIndex2D].find(stateName) == esa[wsIndex2D].end(), std::logic_error,
                                "Error! Cannot locate " << stateName << " in PHAL_SaveSideSetStateField_Def.\n");
    Albany::MDArray state = esa[wsIndex2D].at(stateName);

    const std::vector<int>& nodeMap = sideNodeNumerationMap.at(side_GID);

    // Now we have the two arrays: 3D and 2D. We need to take the part we need from the 3D
    // and put it in the 2D one

    std::vector<PHX::DataLayout::size_type> dims;
    field.dimensions(dims);
    int size = dims.size();

    switch (size)
    {
      case 2:
        // side set cell scalar
        state(ss_cell) = field(cell,side);
        break;

      case 3:
        if (tag2=="Node")
        {
          // side set node scalar
          for (int node=0; node<dims[2]; ++node)
          {
            state(ss_cell,nodeMap[node]) = field(cell,side,node);
          }
        }
        else
        {
          // side set cell vector/gradient
          for (int idim=0; idim<dims[2]; ++idim)
          {
            state(ss_cell,idim) = field(cell,side,idim);
          }
        }
        break;

      case 4:
        if (tag2=="Node")
        {
          // side set node vector/gradient
          for (int node=0; node<dims[2]; ++node)
          {
            for (int dim=0; dim<dims[3]; ++dim)
              state(ss_cell,nodeMap[node],dim) = field(cell,side,node,dim);
          }
        }
        else
        {
          // side set cell tensor
          for (int idim=0; idim<dims[2]; ++idim)
          {
            for (int jdim=0; jdim<dims[3]; ++jdim)
              state(ss_cell,idim,jdim) = field(cell,side,idim,jdim);
          }
        }
        break;

      default:
        TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error,
                                    "Error! Unexpected array dimensions in SaveSideSetStateField: " << size << ".\n");
    }
  }
}

} // Namespace PHAL
