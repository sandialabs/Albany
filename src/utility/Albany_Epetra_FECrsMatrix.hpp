#ifndef ALBANY_EPETRA_FE_CRS_MATRIX_HPP
#define ALBANY_EPETRA_FE_CRS_MATRIX_HPP

#include <Epetra_FECrsMatrix.h>
#include <Epetra_FECrsGraph.h>

namespace Albany
{

/*
 * A version of Epetra_FECrs* classes with an additional functionality
 * 
 * The FE classes in Epetra do not expose the information about the overlapped
 * distribution. The col map could be used to retrieve it in most cases,
 * but it relies on the assumption that Range=Domain. That's not always true.
 * Therefore, we roll our own version, whish simply adds a getter for
 * the OverlapRangeMap.
 * 
 * All the other functionalities are simply inherited from their base classes,
 * and no additional method is implemented.
 */

// Extension of Epetra_FECrsGraph
class EpetraFECrsGraph : public Epetra_FECrsGraph
{
public:
  EpetraFECrsGraph (Epetra_DataAccess CV,
        const Epetra_Map& rowMap,
        const Epetra_Map& ovRowMap,
        int* NumIndicesPerRow,
        bool ignoreNonLocalEntries = false,
        bool buildNonlocalGraph = false)
    : Epetra_FECrsGraph(CV,rowMap,NumIndicesPerRow,ignoreNonLocalEntries,buildNonlocalGraph)
    , m_ovRowMap(ovRowMap)
  {
    // Nothing to do
  }

  virtual ~EpetraFECrsGraph () = default;
  EpetraFECrsGraph(const EpetraFECrsGraph&) = default;

  const Epetra_BlockMap& OverlapRangeMap () const { return m_ovRowMap; }
protected:

  Epetra_BlockMap m_ovRowMap;
};

// Extension of Epetra_FECrsMatrix
class EpetraFECrsMatrix : public Epetra_FECrsMatrix
{
public:

  EpetraFECrsMatrix (const EpetraFECrsGraph& feGraph, bool ignoreNonLocalEntries)
   : Epetra_FECrsMatrix(Copy,feGraph,ignoreNonLocalEntries)
   , m_feGraph(feGraph)
  {
    // Nothing to do
  }

  virtual ~EpetraFECrsMatrix () = default;
  EpetraFECrsMatrix (const EpetraFECrsMatrix&) = default;

  const Epetra_BlockMap& OverlapRangeMap () const {
    return m_feGraph.OverlapRangeMap();
  }

protected:

  const EpetraFECrsGraph& m_feGraph;
};

} // namespace Albany

#endif // ALBANY_EPETRA_FE_CRS_MATRIX_HPP
