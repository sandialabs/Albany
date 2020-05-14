#include "Albany_ThyraCrsMatrixFactory.hpp"

#include "Albany_ThyraUtils.hpp"
#include "Albany_TpetraThyraUtils.hpp"
#include "Albany_EpetraThyraUtils.hpp"

#ifdef ALBANY_EPETRA
#include "Epetra_FECrsGraph.h"
#include "Epetra_FECrsMatrix.h"
#include "Epetra_Export.h"
#endif
#include "Albany_TpetraTypes.hpp"

#include "Albany_Utils.hpp"
#include "Albany_Macros.hpp"

#include <set>
#include <vector>

namespace Albany {

// The implementation of the graph
struct ThyraCrsMatrixFactory::Impl {

  Impl () = default;

  std::map<GO,std::set<GO>> temp_graph;
#ifdef ALBANY_EPETRA
  Teuchos::RCP<Epetra_FECrsGraph> e_graph;
#endif
  Teuchos::RCP<Tpetra_FECrsGraph> t_graph;
};

ThyraCrsMatrixFactory::
ThyraCrsMatrixFactory (const Teuchos::RCP<const Thyra_VectorSpace> domain_vs,
                       const Teuchos::RCP<const Thyra_VectorSpace> range_vs,
                       const Teuchos::RCP<const Thyra_VectorSpace> row_vs)
 : m_graph(new Impl())
 , m_domain_vs(domain_vs)
 , m_range_vs(range_vs)
 , m_row_vs(row_vs)
 , m_filled (false)
{
  auto bt = Albany::build_type();
  TEUCHOS_TEST_FOR_EXCEPTION (bt==BuildType::None, std::logic_error, "Error! No build type set for albany.\n");

  if (m_row_vs.is_null()) {
    m_row_vs = m_range_vs;
    m_row_same_as_range = false;
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION (!isOneToOne(range_vs), std::logic_error,
      "[ThyraCrsMatrixFactory] Error! When providing a row vs, the range vs must be one-to-one.\n");
    m_row_same_as_range = true;
  }

  if (bt==BuildType::Epetra) {
#ifndef ALBANY_EPETRA
    TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Epetra is not enabled in albany.\n");
#endif
  }
}

void ThyraCrsMatrixFactory::insertGlobalIndices (const GO row, const Teuchos::ArrayView<const GO>& indices)
{
  // Indices are inserted in a temporary local graph. 
  // The actual graph is created and filled when fillComplete is called

  const auto bt = Albany::build_type();
  const bool epetra = bt==BuildType::Epetra;
#ifdef ALBANY_EPETRA
  const GO max_safe_gid = epetra ? static_cast<GO>(Teuchos::OrdinalTraits<Epetra_GO>::max())
                                 : Teuchos::OrdinalTraits<GO>::max();
  TEUCHOS_TEST_FOR_EXCEPTION(epetra && row>max_safe_gid, std::runtime_error,
      "Error! Input gids exceed Epetra_GO ranges.\n");
#else
  TEUCHOS_TEST_FOR_EXCEPTION(epetra, std::logic_error,
      "Error! Epetra is not enabled in Albany.\n");
  const GO max_safe_gid = Teuchos::OrdinalTraits<GO>::max();
#endif

  auto& row_indices = m_graph->temp_graph[row];
  const int size = indices.size();
  for (int i=0; i<size; ++i) {
    // Epetra_GO is 32 bits, while GO is 64, so check the gids fit in 32 bits.
    TEUCHOS_TEST_FOR_EXCEPTION(epetra && indices[i]>max_safe_gid, std::runtime_error,
         "Error! Input gids exceed Epetra_GO ranges.\n");
    row_indices.emplace(indices[i]);
  }
}

void ThyraCrsMatrixFactory::fillComplete () {

  // We create the CrsGraph, insert indices from the temporary local graph,
  // and call fill complete.
  // Note: we can't compute the nnz per row here, cause Epetra wants the
  //       array for the non-overlapped range map, while Tpetra wants
  //       the array for the overlapped row map.

  const auto bt = Albany::build_type();
  if (bt==BuildType::Epetra) {
#ifdef ALBANY_EPETRA
    auto e_range  = getEpetraMap(m_range_vs);
    const int numLocalRows = getLocalSubdim(m_range_vs);
    Teuchos::Array<int> nnz_per_row(numLocalRows);
    for (int lrow=0; lrow<numLocalRows; ++lrow) {
      const GO gid = e_range->GID(lrow);
      nnz_per_row[lrow] = m_graph->temp_graph.at(gid).size();
    }

    m_graph->e_graph = Teuchos::rcp(new Epetra_FECrsGraph(Copy,*e_range,nnz_per_row.data(),!m_row_same_as_range,true));

    // Insder rows.
    for (const auto& it : m_graph->temp_graph) {
      const auto& row_indices = it.second;
      const int row_size = row_indices.size();
      if(row_size>0) {
        Teuchos::Array<Epetra_GO> e_indices(row_indices.size());
        int i=0;
        for (const auto index : row_indices) {
          e_indices[i] = index;
          ++i;
        }
        const Epetra_GO row = static_cast<Epetra_GO>(it.first);
        m_graph->e_graph->InsertGlobalIndices(row,row_size,e_indices.getRawPtr());
      }
    }

    auto e_domain = getEpetraMap(m_domain_vs);
    m_graph->e_graph->GlobalAssemble(*e_domain,*e_range);
    m_graph->e_graph->OptimizeStorage();

    // Cleanup temporaries
    m_graph->temp_graph.clear();
#else
    TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Epetra is not enabled in albany.\n");
#endif
  } else {
    auto t_range = getTpetraMap(m_range_vs);
    auto t_row   = getTpetraMap(m_row_vs);

    // For some reason Tpetra_FECrsGraph does not have a ctor that takes an Teuchos::ArrayView,
    // so we must create a DualView.
    using exec_space = Tpetra_CrsGraph::execution_space;
    using DView = Kokkos::DualView<size_t*, exec_space>;
    DView nnz_per_row("nnz",getLocalSubdim(m_row_vs));
    for (const auto& it : m_graph->temp_graph) {
      LO lrow = t_row->getLocalElement(static_cast<Tpetra_GO>(it.first));
      nnz_per_row.h_view[lrow] = it.second.size();
    }

    m_graph->t_graph = Teuchos::rcp(new Tpetra_FECrsGraph(t_range,t_row,nnz_per_row));

    for (const auto& it : m_graph->temp_graph) {
      const auto& row_indices = it.second;
      if(row_indices.size()>0) {
        Teuchos::Array<Tpetra_GO> t_indices(row_indices.size());
        int i=0;
        for (const auto &index : row_indices) {
          t_indices[i] = index;
          ++i;
        }
        m_graph->t_graph->insertGlobalIndices(static_cast<Tpetra_GO>(it.first),t_indices);
      }
    }

    auto t_domain = getTpetraMap(m_domain_vs);
    m_graph->t_graph->fillComplete(t_domain,t_range);

    // Cleanup temporaries
    m_graph->temp_graph.clear();
  }

  m_filled = true;
}

Teuchos::RCP<Thyra_LinearOp> ThyraCrsMatrixFactory::createOp (const bool ignoreNonLocalRows) const {
  TEUCHOS_TEST_FOR_EXCEPTION (!is_filled(), std::logic_error, "Error! Cannot create a linear operator if the graph is not filled.\n");

  auto const zero = Teuchos::ScalarTraits<ST>::zero();
  Teuchos::RCP<Thyra_LinearOp> op;
  const auto bt = Albany::build_type();
  if (bt==BuildType::Epetra) {
#ifdef ALBANY_EPETRA
    Teuchos::RCP<Epetra_CrsMatrix> matrix;
    if (ignoreNonLocalRows) {
      // FECrsGraph *is* a CrsGraph. CrsMatrix will only deal with CrsGraph stuff
      matrix = Teuchos::rcp(new Epetra_CrsMatrix(Copy, *m_graph->e_graph));
    } else {
      matrix = Teuchos::rcp(new Epetra_FECrsMatrix(Copy, *m_graph->e_graph, !m_row_same_as_range)); 
    }
    matrix->PutScalar(zero); 
    op = createThyraLinearOp(Teuchos::rcp_implicit_cast<Epetra_Operator>(matrix));
#else
    TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Epetra is not enabled in albany.\n");
#endif
  } else {
    Teuchos::RCP<Tpetra_CrsMatrix> matrix;
    if (ignoreNonLocalRows) {
      // FECrsGraph *is* a CrsGraph. CrsMatrix will only deal with CrsGraph stuff
      matrix = Teuchos::rcp(new Tpetra_CrsMatrix(m_graph->t_graph));
    } else {
      auto fe_matrix = Teuchos::rcp (new Tpetra_FECrsMatrix(m_graph->t_graph));
      // Tpetra creates FECrsMatrix already in assembly mode.
      // We want linear ops to be in assembly mode *only when explicitly requested*.
      // If the Thyra LinearOp is created with the matrix in assembly mode,
      // its range/domain vs's would be the overlapped ones.
      fe_matrix->setAllToScalar(zero);
      fe_matrix->endFill();
      matrix = fe_matrix;
    }
    op = createThyraLinearOp(Teuchos::rcp_implicit_cast<Tpetra_Operator>(matrix));
  }

  return op;
}

} // namespace Albany
