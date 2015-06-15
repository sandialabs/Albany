//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <vector>
#include <string>

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Shards_CellTopology.hpp"
#include "Intrepid_FieldContainer.hpp"

namespace FELIX
{

// **********************************************************************
template<typename EvalT,typename Traits>
LoadSideSetStateField<EvalT, Traits>::
LoadSideSetStateField (const Teuchos::ParameterList& p,
                       const Albany::MeshSpecsStruct& meshSpecs)
{
  sideSetName = p.get<std::string>("Side Set Name");

  cellFieldName =  p.get<std::string>("Cell Field Name");
  sideStateName =  p.get<std::string>("Side Set State Name");

  PHX::MDField<ScalarT> f(cellFieldName, p.get<Teuchos::RCP<PHX::DataLayout> >("Cell Field Layout") );
  field = f;

  this->addEvaluatedField (field);

  this->setName ("Load Cell Field " + cellFieldName + " from Side Set State " + sideStateName);

  const CellTopologyData * const elem_top = &meshSpecs.ctd;
  cellType = Teuchos::rcp(new shards::CellTopology (elem_top));
  const CellTopologyData * const side_top = elem_top->side[0].topology;
  sideType = Teuchos::rcp(new shards::CellTopology(side_top));

  Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > intrepidBasis;
  intrepidBasis = Albany::getIntrepidBasis(*side_top);
  numSideNodes = intrepidBasis->getCardinality();
  sideDims = sideType->getDimension();
}

// **********************************************************************
template<typename EvalT, typename Traits>
void LoadSideSetStateField<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(field,fm);
}
// **********************************************************************
template<typename EvalT, typename Traits>
void LoadSideSetStateField<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  TEUCHOS_TEST_FOR_EXCEPTION (workset.sideSets==Teuchos::null, std::logic_error,
                              "Error! The mesh does not store any side set.\n");

  TEUCHOS_TEST_FOR_EXCEPTION (workset.disc==Teuchos::null, std::logic_error,
                              "Error! The workset must store a valid discretization pointer.\n");

  TEUCHOS_TEST_FOR_EXCEPTION (workset.disc->getSideSetDiscretizations()==Teuchos::null, std::logic_error,
                              "Error! The discretization must store side set discretizations.\n");

  const Albany::SideSetList& ssList = *(workset.sideSets);
  Albany::SideSetList::const_iterator it_ss = ssList.find(sideSetName);

  Teuchos::RCP<Teuchos::FancyOStream> output(Teuchos::VerboseObjectBase::getDefaultOStream());

  if(it_ss == ssList.end())
  {
    return; // Side set not present in this workset
  }

  std::map<std::string,Teuchos::RCP<Albany::AbstractDiscretization> >::iterator it_disc;
  it_disc = workset.disc->getSideSetDiscretizations()->find(sideSetName);

  TEUCHOS_TEST_FOR_EXCEPTION (it_disc==workset.disc->getSideSetDiscretizations()->end(), std::logic_error,
                              "Error! No discretization found for side set " << sideSetName << ".\n");

  // Get states from STK mesh
  Albany::StateArrays& state_arrays = it_disc->second->getStateArrays();
  Albany::StateArrayVec& elem_state_arrays = state_arrays.elemStateArrays;
  Albany::WsLIDList& elemGIDws3D = workset.disc->getElemGIDws();
  Albany::WsLIDList& elemGIDws2D = it_disc->second->getElemGIDws();

  // Loop on the sides of this sideSet that are in this workset
  const std::vector<Albany::SideStruct>& sideSet = it_ss->second;
  for (std::size_t side=0; side < sideSet.size(); ++side)
  {
    // Get the data that corresponds to the side

    const int side_GID = sideSet[side].side_GID;
    const int elem3D_GID = sideSet[side].elem_GID;
    const int elem3D_LID = sideSet[side].elem_LID;
    const int elem_side = sideSet[side].side_local_id;

    // Not sure if this is even possible, but just for debug pourposes
    TEUCHOS_TEST_FOR_EXCEPTION (elemGIDws3D[ sideSet[side].elem_GID ].ws != workset.wsIndex, std::logic_error,
                                "Error! This workset has a side that belongs to an element not in the workset.\n");

    // WARNING: we STRONGLY rely on the assumption that the sideSet ID is the same as the element ID
    //          on the boundary mesh. This is true for ExtrudedMeshStruct, with NO columnwise ordering

    // We know the side ID, so we can fetch two things:
    //    1) the 2D-wsIndex where the 2D element lies
    //    2) the LID of the 2D element
    int elem2D_GID = workset.disc->getSideIdToSideSetElemIdMap()->find(sideSetName)->second[side_GID];
    int wsIndex2D = elemGIDws2D[elem2D_GID].ws;
    int elem2D_LID = elemGIDws2D[elem2D_GID].LID;

    // Then, we extract the StateArray of the desired state in the right 2D-ws
    Albany::StateArray::const_iterator it_state = elem_state_arrays[wsIndex2D].find(sideStateName);

    TEUCHOS_TEST_FOR_EXCEPTION (it_state == elem_state_arrays[wsIndex2D].end(), std::logic_error,
                                "Error! Cannot locate " << sideStateName << " in PHAL_LoadSideSetStateField_Def.\n");

    // Now we have the two arrays: 3D and 2D. We need to take the part we need from the 3D
    // and put it at the right place in the 2D one
    Albany::MDArray state = it_state->second;

    std::vector<PHX::DataLayout::size_type> dims;
    field.dimensions(dims);
    int size = dims.size();

    // In StateManager, ElemData (1 scalar per cell) is stored as QuadPoint (1 qp),
    // so size would figure as 2 even if it is actually 1. Therefore we had to
    // call size on dims computed on field. Then, we call it on state to get
    // the correct state dimensions
    state.dimensions(dims);

    switch (size)
    {
      case 1:
      {
        field(elem3D_LID) = state(elem2D_LID);
        break;
      }
      case 2:
        for (int node = 0; node < dims[1]; ++node)
        {
          int node3D = cellType->getNodeMap(sideDims, elem_side, node);
          field(elem3D_LID,node3D) = field(elem3D_LID,node3D);
        }
        break;

      case 3:
        for (int node = 0; node < dims[1]; ++node)
        {
          int node3D = cellType->getNodeMap(sideDims, elem_side, node);
          for (int dim = 0; dim < dims[2]; ++dim)
            field(elem3D_LID,node3D,dim) = state(elem2D_LID, node, dim);
        }
        break;

      default:
        TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error,
                                    "Error! Unexpected array dimensions in LoadSideSetStateField: " << size << ".\n");
    }
  }
}

} // Namespace FELIX
