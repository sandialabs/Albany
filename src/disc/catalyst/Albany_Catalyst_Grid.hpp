//*****************************************************************//
//    Albany 2.0:  Copyright 2013 Kitware Inc.                     //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_CATALYST_GRID
#define ALBANY_CATALYST_GRID

#include "vtkObject.h"

#include "vtkMappedUnstructuredGrid.h" // For mapped unstructured grid wrapper

#include "Albany_Catalyst_Decorator.hpp" // For decorator class

namespace Albany {
namespace Catalyst {

class GridImplementation: public vtkObject
{
public:
  static GridImplementation *New();
  virtual void PrintSelf(ostream &os, vtkIndent indent);
  vtkTypeMacro(GridImplementation, vtkObject)

  bool SetDecorator(Decorator *decorator);

  // API for vtkMappedUnstructuredGrid's implementation.
  vtkIdType GetNumberOfCells();
  int GetCellType(vtkIdType cellId);
  void GetCellPoints(vtkIdType cellId, vtkIdList *ptIds);
  void GetPointCells(vtkIdType ptId, vtkIdList *cellIds);
  int GetMaxCellSize();
  void GetIdsOfCellsOfType(int type, vtkIdTypeArray *array);
  int IsHomogeneous();

  // This container is read only -- these methods do nothing but print a
  // warning.
  void Allocate(vtkIdType numCells, int extSize = 1000);
  vtkIdType InsertNextCell(int type, vtkIdList *ptIds);
  vtkIdType InsertNextCell(int type, vtkIdType npts, vtkIdType *ptIds);
  vtkIdType InsertNextCell(int type, vtkIdType npts, vtkIdType *ptIds,
                           vtkIdType nfaces, vtkIdType *faces);
  void ReplaceCell(vtkIdType cellId, int npts, vtkIdType *pts);

protected:
  GridImplementation();
  ~GridImplementation();

private:
  GridImplementation(const GridImplementation &); // Not implemented.
  void operator=(const GridImplementation &);   // Not implemented.

  const Decorator *Discretization;

  int DegreesOfFreedom;

  // Nested ArrayRCPs -- lookup [wsId][wsElLid][elNodeInd][] / DOF = node id
  typedef Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<
            Teuchos::ArrayRCP<int> > > > WsElLid2NodeT;
  WsElLid2NodeT NodeLookup;

  // Map of all elements, global element id -> struct { int ws, int lid }
  const WsLIDList *ElementsWsLid;

  // Offset of the keys in ElementsWsLid:
  int ElementOffset;

  // Indexed by workset:
  std::vector<const CellTopologyData *> TopologyLookup;
  std::vector<VTKCellType> CellTypeLookup;
  static VTKCellType TopologyToCellType(const CellTopologyData *ctd);
  static vtkIdType GetCellSize(VTKCellType cellType);
  void GetWorksetFromCellId(vtkIdType cellId, int &ws, int &lid);
  VTKCellType GetCellTypeFromWorkset(int ws);
};

vtkMakeMappedUnstructuredGrid(Grid, GridImplementation)

} // end namespace Catalyst
} // end namespace Albany

#endif // ALBANY_CATALYST_GRID
