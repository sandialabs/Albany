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
  Teuchos::RCP<const Epetra_BlockMap> e_range;
  Teuchos::RCP<const Epetra_BlockMap> e_row;
#endif
  Teuchos::RCP<Tpetra_FECrsGraph> t_graph;
  Teuchos::RCP<const Tpetra_Map> t_range;
  Teuchos::RCP<const Tpetra_Map> t_row;
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
#ifdef ALBANY_EPETRA
    m_graph->e_range  = getEpetraBlockMap(range_vs);
    m_graph->e_row    = getEpetraBlockMap(row_vs);
#else
    TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Epetra is not enabled in albany.\n");
#endif
  } else {
    m_graph->t_range = getTpetraMap(range_vs);
    m_graph->t_row   = getTpetraMap(row_vs);
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

  // We created the CrsGraph,
  // insert indices from the temporary local graph,
  // and call fill complete.
  // For some reason Tpetra_FECrsGraph does not have a ctor that takes an Teuchos::ArrayView,
  // so we must create a DualView.
  using exec_space = Tpetra_CrsGraph::execution_space;
  using DView = Kokkos::DualView<size_t*, exec_space>;
  DView nnz_per_row("nnz",getLocalSubdim(m_range_vs));
  for (int lrow=0; lrow<nnz_per_row.extent_int(0); ++lrow) {
    nnz_per_row.h_view[lrow] = m_graph->temp_graph[lrow].size();
  }

  const auto bt = Albany::build_type();
  if (bt==BuildType::Epetra) {
#ifdef ALBANY_EPETRA
    int* nnz_per_row_ptr = reinterpret_cast<int*>(nnz_per_row.h_view.data());
    m_graph->e_graph = Teuchos::rcp(new Epetra_FECrsGraph(Copy,*m_graph->e_range,nnz_per_row_ptr,!m_row_same_as_range));

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

    auto e_domain = getEpetraBlockMap(m_domain_vs);
    m_graph->e_graph->FillComplete(*e_domain,*m_graph->e_range);
    m_graph->e_graph->OptimizeStorage();

    // Cleanup temporaries
    m_graph->temp_graph.clear();
    m_graph->e_range = m_graph->e_row = Teuchos::null;
#else
    TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Epetra is not enabled in albany.\n");
#endif
  } else {
    m_graph->t_graph = Teuchos::rcp(new Tpetra_FECrsGraph(m_graph->t_range,m_graph->t_row,nnz_per_row));

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
    m_graph->t_graph->fillComplete(t_domain,m_graph->t_range);

    // Cleanup temporaries
    m_graph->temp_graph.clear();
    m_graph->t_range = m_graph->t_row = Teuchos::null;
  }

  m_filled = true;
}

Teuchos::RCP<Thyra_LinearOp> ThyraCrsMatrixFactory::createOp (const bool ignoreNonLocalRows) const {
  TEUCHOS_TEST_FOR_EXCEPTION (!m_filled, std::logic_error, "Error! Cannot create a linear operator if the graph is not filled.\n");

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
      matrix = Teuchos::rcp (new Tpetra_FECrsMatrix(m_graph->t_graph));
    }
    matrix->resumeFill(); 
    matrix->setAllToScalar(zero);
    matrix->fillComplete(); 
    op = createThyraLinearOp(Teuchos::rcp_implicit_cast<Tpetra_Operator>(matrix));
  }

  return op;
}

} // namespace Albany
