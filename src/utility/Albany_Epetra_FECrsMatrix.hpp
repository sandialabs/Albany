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
 * but it can only retrieve Domain information (if Range!=Domain, we can't
 * retrieve information on the overlapped range map).
 * Therefore, we roll our own version, which simply stores the overlapped
 * range/domain maps, and exposes them with a getter.
 * 
 * All the other functionalities are simply inherited from their base classes,
 * and no additional method is implemented or overridden.
 */

// Extension of Epetra_FECrsGraph
class EpetraFECrsGraph : public Epetra_FECrsGraph
{
public:
  EpetraFECrsGraph (Epetra_DataAccess CV,
        const Epetra_Map& rowMap,
        const Epetra_Map& ovRowMap,
        const Epetra_Map& ovDomainMap,
        int* NumIndicesPerRow,
        bool ignoreNonLocalEntries = false,
        bool buildNonlocalGraph = false)
    : Epetra_FECrsGraph(CV,rowMap,NumIndicesPerRow,ignoreNonLocalEntries,buildNonlocalGraph)
    , m_ovRowMap(ovRowMap)
    , m_ovDomainMap(ovDomainMap)
  {
    // Nothing to do
  }

  virtual ~EpetraFECrsGraph () = default;
  EpetraFECrsGraph(const EpetraFECrsGraph&) = default;

  const Epetra_BlockMap& OverlapRangeMap () const { return m_ovRowMap; }
  const Epetra_BlockMap& OverlapDomainMap () const { return m_ovDomainMap; }
protected:

  Epetra_BlockMap m_ovRowMap;
  Epetra_BlockMap m_ovDomainMap;
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
  const Epetra_BlockMap& OverlapDomainMap () const {
    return m_feGraph.OverlapDomainMap();
  }

protected:

  const EpetraFECrsGraph& m_feGraph;
};

} // namespace Albany

#endif // ALBANY_EPETRA_FE_CRS_MATRIX_HPP
