//*****************************************************************//
//    Albany 2.0:  Copyright 2013 Kitware Inc.                     //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_Catalyst_Grid.hpp"

#include "Shards_BasicTopologies.hpp"

#include "Teuchos_VerboseObject.hpp"
#include "Teuchos_TestForException.hpp"

#include <vtkIdTypeArray.h>

#include <cassert>
#include <string>
#include <vector>

using Teuchos::RCP;
using Teuchos::ArrayRCP;

namespace Albany {
namespace Catalyst {

vtkStandardNewMacro(Grid)
vtkStandardNewMacro(GridImplementation)

GridImplementation::GridImplementation()
  : Discretization(NULL),
    DegreesOfFreedom(0),
    ElementsWsLid(NULL),
    ElementOffset(0)
{
}

GridImplementation::~GridImplementation()
{
}

void GridImplementation::PrintSelf(ostream &os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

bool GridImplementation::SetDecorator(Decorator *decorator)
{
  this->Discretization = decorator;
  this->DegreesOfFreedom = decorator->getNumEq();
  this->NodeLookup = decorator->getWsElNodeEqID();

  this->ElementsWsLid = &decorator->getElemGIDws();
  // The key of the first entry in the ElementsWsLid is the offset we should
  // use when accessing the ids:
  if (this->ElementsWsLid->size() > 0) {
    WsLIDList::const_iterator first = this->ElementsWsLid->begin();
    this->ElementOffset = first->first;
  }
  else {
    this->ElementOffset = 0;
  }

  // Build topology LUT
  typedef ArrayRCP<RCP<Albany::MeshSpecsStruct> > MeshSpecsT;
  typedef MeshSpecsT::const_iterator MeshSpecsIter;
  const MeshSpecsT &meshSpecs(decorator->getMeshStruct()->getMeshSpecs());

  typedef ArrayRCP<std::string> RCPStringsT;
  const RCPStringsT &ebNames(decorator->getWsEBNames());

  this->CellTypeLookup.resize(ebNames.size(), VTK_EMPTY_CELL);

  for (size_t wsId = 0; wsId < ebNames.size(); ++wsId) {
    const std::string &elName(ebNames[wsId]);
    for (MeshSpecsIter it = meshSpecs.begin(), itEnd = meshSpecs.end();
        it != itEnd; ++it) {
      if ((*it)->ebName == elName)  {
        this->CellTypeLookup[wsId] = TopologyToCellType(&(*it)->ctd);
        break;
      }
    }
    TEUCHOS_TEST_FOR_EXCEPTION(this->CellTypeLookup[wsId] == VTK_EMPTY_CELL,
                               std::runtime_error,
                               "No CellTopologyData instance found for element "
                               "block " << elName);
  }

  return true;
}

vtkIdType GridImplementation::GetNumberOfCells()
{
  return static_cast<vtkIdType>(this->ElementsWsLid->size());
}

int GridImplementation::GetCellType(vtkIdType cellId)
{
  int ws = -1;
  int unused = -1;
  this->GetWorksetFromCellId(cellId, ws, unused);
  return this->GetCellTypeFromWorkset(ws);
}

void GridImplementation::GetCellPoints(vtkIdType cellId, vtkIdList *ptIds)
{
  int ws = -1;
  int lid = -1;
  this->GetWorksetFromCellId(cellId, ws, lid);
  typedef Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > NodesType;
  const NodesType &nodes = this->NodeLookup[ws][lid];
  vtkIdType cellSize = static_cast<vtkIdType>(nodes.size());
  ptIds->SetNumberOfIds(cellSize);

  NodesType::const_iterator nodesIter = nodes.begin();
  vtkIdType *ptIdIter = ptIds->GetPointer(0);

  while ((cellSize--) > 0) {
    *(ptIdIter++) =
        static_cast<vtkIdType>(*(nodesIter++)[0] / this->DegreesOfFreedom);
  }

  // For wedge, swap points 0 <--> 1 and 3 <--> 4. All other supported cells
  // in VTK match shards' winding.
  if (this->GetCellTypeFromWorkset(ws) == VTK_WEDGE) {
    using namespace std;
    vtkIdType *pts = ptIds->GetPointer(0);
    swap(pts[0], pts[1]);
    swap(pts[3], pts[4]);
  }
}

namespace {
// predicate for find_if in GetPointCells
struct WsLidMatcher {
  wsLid needle;
  WsLidMatcher() { needle.ws = 0; needle.LID = 0; }
  bool operator()(const WsLIDList::value_type &val) const
  {
    return (val.second.ws == needle.ws && val.second.LID== needle.LID);
  }
};
} // end anon namespace

void GridImplementation::GetPointCells(vtkIdType ptId, vtkIdList *cellIds)
{
  // This is slow. Might be worth optimizing if it becomes a bottleneck.
  using Teuchos::ArrayRCP;
  typedef ArrayRCP<ArrayRCP<int> > NodeLevel;
  typedef ArrayRCP<NodeLevel> LidLevel;

  cellIds->Reset();

  // We'll use these to keep track of our iteration progress and lookup cellids:
  WsLidMatcher matcher;
  int &ws = matcher.needle.ws;
  int &lid = matcher.needle.LID;

  // For lookups
  WsLIDList::const_iterator haystackBegin = this->ElementsWsLid->begin();
  WsLIDList::const_iterator haystackEnd = this->ElementsWsLid->end();
  WsLIDList::const_iterator result;

  // Search for matching pt ids by iterating through all cells.
  for (ws = 0; ws < this->NodeLookup.size(); ++ws) {
    const LidLevel &lids(this->NodeLookup[ws]);
    for (lid = 0; lid < lids.size(); ++lid) {
      const NodeLevel &nodes(lids[lid]);
      for (int node = 0; node < nodes.size(); ++node) {
        if (static_cast<vtkIdType>(nodes[node][0] / this->DegreesOfFreedom)
            == ptId) {
          result = std::find_if(haystackBegin, haystackEnd, matcher);
          if (result != haystackEnd)
            cellIds->InsertNextId(static_cast<vtkIdType>(result->first));
        }
      }
    }
  }
}

int GridImplementation::GetMaxCellSize()
{
  int size = 0;
  for (std::vector<VTKCellType>::const_iterator it = CellTypeLookup.begin(),
       itEnd = CellTypeLookup.end(); it != itEnd; ++it)
    size = std::max(size, static_cast<int>(GetCellSize(*it)));
  return size;
}

void GridImplementation::GetIdsOfCellsOfType(int type, vtkIdTypeArray *array)
{
  // All cells within a workset are homogeneous, so just look for the entries
  // CellTypeLookup, which is indexed by WS, then lookup all of the cell ids
  // in the global element map.

  // Find worksets:
  std::vector<int> worksets;
  {
    typedef std::vector<VTKCellType>::const_iterator TypeIterT;
    TypeIterT begin(this->CellTypeLookup.begin());
    TypeIterT end = this->CellTypeLookup.end();
    TypeIterT typeIter = begin;
    while (typeIter != end) {
      if (*typeIter == static_cast<VTKCellType>(type))
        worksets.push_back(typeIter - begin);
      ++typeIter;
    }
  }

  // There are usually about 50 cells/workset:
  array->SetNumberOfComponents(1);
  array->Allocate(static_cast<vtkIdType>(50 * worksets.size()));

  // Get cells:
  {
    WsLIDList::const_iterator iter = this->ElementsWsLid->begin();
    WsLIDList::const_iterator end = this->ElementsWsLid->end();

    typedef std::vector<int>::iterator WsIterT;
    WsIterT wsBegin = worksets.begin();
    WsIterT wsEnd = worksets.end();

    std::sort(wsBegin, wsEnd);

    while (iter != end) {
      if (std::binary_search(wsBegin, wsEnd, iter->second.ws))
        array->InsertNextValue(static_cast<vtkIdType>(iter->second.ws));
    }
  }
}

int GridImplementation::IsHomogeneous()
{
  if (!this->CellTypeLookup.empty()) {
    typedef std::vector<VTKCellType>::const_iterator CTIterT;
    CTIterT begin = this->CellTypeLookup.begin();
    CTIterT end = this->CellTypeLookup.end();
    VTKCellType type = *(begin++);
    while (begin != end) {
      if (*(begin++) != type)
        return false;
    }
  }
  return true;
}

void GridImplementation::Allocate(vtkIdType, int)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
                             "Read-only container.");
}

vtkIdType GridImplementation::InsertNextCell(int, vtkIdList*)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
                             "Read-only container.");
  return -1;
}

vtkIdType GridImplementation::InsertNextCell(int, vtkIdType, vtkIdType*)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
                             "Read-only container.");
  return -1;
}

vtkIdType GridImplementation::InsertNextCell(int, vtkIdType, vtkIdType*,
                                             vtkIdType, vtkIdType*)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
                             "Read-only container.");
  return -1;
}

void GridImplementation::ReplaceCell(vtkIdType cellId, int npts, vtkIdType *pts)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
                             "Read-only container.");
}

inline VTKCellType GridImplementation::TopologyToCellType(const CellTopologyData *ctd)
{
  /// @todo this only covers linear cells for the moment. If adding more, verify
  /// that the vertex specs are the same in VTK/shards, and update GetCellPoints
  /// appropriately.
  switch (ctd->key) {
  default:
    *Teuchos::VerboseObjectBase::getDefaultOStream()
        << "Unrecognized key for CellTopologyData name '" << ctd->name << "'.";
    return VTK_EMPTY_CELL;
  case shards::Particle::key:
    return VTK_VERTEX;
  case shards::Line<2>::key:
    return VTK_LINE;
  case shards::Triangle<3>::key:
    return VTK_TRIANGLE;
  case shards::Quadrilateral<4>::key:
    return VTK_QUAD;
  case shards::Tetrahedron<4>::key:
    return VTK_TETRA;
  case shards::Hexahedron<8>::key:
    return VTK_HEXAHEDRON;
  case shards::Wedge<6>::key:
    return VTK_WEDGE;
  case shards::Pyramid<5>::key:
    return VTK_PYRAMID;
  }
}

inline vtkIdType GridImplementation::GetCellSize(VTKCellType cellType)
{
  switch (cellType) {
  case VTK_TRIANGLE:
    return 3;
  case VTK_QUAD:
    return 4;
  case VTK_LINE:
    return 2;
  case VTK_TETRA:
    return 4;
  case VTK_VERTEX:
    return 1;
  case VTK_HEXAHEDRON:
    return 8;
  case VTK_WEDGE:
    return 6;
  case VTK_PYRAMID:
    return 5;
  default:
    *Teuchos::VerboseObjectBase::getDefaultOStream()
        << "Unhandled cell type: " << cellType;
  case VTK_EMPTY_CELL:
    return 0;
  }
}

inline void GridImplementation::GetWorksetFromCellId(vtkIdType cellId,
                                                     int &ws, int &lid)
{
  WsLIDList::const_iterator match =
      this->ElementsWsLid->find(static_cast<int>(cellId) + this->ElementOffset);
  if (match != this->ElementsWsLid->end()) {
    const wsLid &result = match->second;
    ws = result.ws;
    lid = result.LID;
  }
  else {
    ws = -1;
    lid = -1;
  }
}

VTKCellType GridImplementation::GetCellTypeFromWorkset(int ws)
{
  return (ws >= 0 && ws < this->CellTypeLookup.size())
      ? this->CellTypeLookup[ws] : VTK_EMPTY_CELL;
}

} // namespace Catalyst
} // namespace Albany
