#include "Albany_ThyraCrsMatrixFactory.hpp"

#ifdef ALBANY_EPETRA
#include "Epetra_CrsGraph.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_Export.h"
#endif
#include "Albany_TpetraTypes.hpp"

#include "Albany_Utils.hpp"
#include "Albany_Macros.hpp"

namespace Albany {

// The implementation of the graph
struct ThyraCrsMatrixFactory::Impl {

  Impl () = default;

#ifdef ALBANY_EPETRA
  Teuchos::RCP<Epetra_CrsGraph> e_graph;
#endif
  Teuchos::RCP<Tpetra_CrsGraph> t_graph;
};

ThyraCrsMatrixFactory::
ThyraCrsMatrixFactory (const Teuchos::RCP<const Thyra_VectorSpace> domain_vs,
                       const Teuchos::RCP<const Thyra_VectorSpace> range_vs)
 : m_graph(new Impl())
 , m_domain_vs(domain_vs)
 , m_range_vs(range_vs)
 , m_filled (false)
{
  auto bt = Albany::build_type();
  TEUCHOS_TEST_FOR_EXCEPTION (bt==BuildType::None, std::logic_error, "Error! No build type set for albany.\n");

  if (bt==BuildType::Epetra) {
#ifdef ALBANY_EPETRA
    e_range  = getEpetraBlockMap(range_vs);
    e_local_graph.resize(e_range->NumMyElements());
#else
    TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Epetra is not enabled in albany.\n");
#endif
  } else {
    t_range = getTpetraMap(range_vs);
    t_local_graph.resize(t_range->getNodeNumElements());
  }
}

ThyraCrsMatrixFactory::
ThyraCrsMatrixFactory (const Teuchos::RCP<const Thyra_VectorSpace> domain_vs,
                       const Teuchos::RCP<const Thyra_VectorSpace> range_vs,
                       const Teuchos::RCP<const ThyraCrsMatrixFactory> overlap_src)
 : m_domain_vs(domain_vs)
 , m_range_vs(range_vs)
{
  TEUCHOS_TEST_FOR_EXCEPTION (!overlap_src->is_filled(), std::logic_error,
                              "Error! Can only build a graph from an overlapped source if source has been filled already.\n");
  m_graph = Teuchos::rcp(new Impl());

  auto bt = Albany::build_type();
  TEUCHOS_TEST_FOR_EXCEPTION (bt==BuildType::None, std::logic_error, "Error! No build type set for albany.\n");
  if (bt==BuildType::Epetra) {
#ifdef ALBANY_EPETRA
    e_range = getEpetraBlockMap(range_vs);
    auto e_overlap_range = getEpetraBlockMap(overlap_src->m_range_vs);
    auto e_overlap_graph = overlap_src->m_graph->e_graph;

    m_graph->e_graph = Teuchos::rcp(new Epetra_CrsGraph(Copy,*e_range,e_overlap_graph->GlobalMaxNumIndices()));

    Epetra_Export exporter (*e_overlap_range,*e_range);
    m_graph->e_graph->Export(*e_overlap_graph,exporter,Insert);

    auto e_domain = getEpetraBlockMap(domain_vs);
    m_graph->e_graph->FillComplete(*e_domain,*e_range);
    m_graph->e_graph->OptimizeStorage();
#else
    TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Epetra is not enabled in albany.\n");
#endif
  } else {
    t_range = getTpetraMap(range_vs);
    auto t_overlap_range = getTpetraMap(overlap_src->m_range_vs);
    auto t_overlap_graph = overlap_src->m_graph->t_graph;

    //Creating an empty graph. The graph will be automatically resized when exported.
    m_graph->t_graph = createCrsGraph(t_range);

    Tpetra_Export exporter(t_overlap_range,t_range);
    m_graph->t_graph->doExport(*t_overlap_graph,exporter,Tpetra::INSERT);

    auto t_domain = getTpetraMap(domain_vs);
    m_graph->t_graph->fillComplete(t_domain,t_range);
  }

  m_filled = true;
}

void ThyraCrsMatrixFactory::insertGlobalIndices (const GO row, const Teuchos::ArrayView<const GO>& indices) {
  
  // Indices are inserted in a temporary local graph. 
  // The actual graph is created and filled when fillComplete is called

  auto bt = Albany::build_type();
  if (bt==BuildType::Epetra) {
#ifdef ALBANY_EPETRA
    const GO max_safe_gid = static_cast<GO>(Teuchos::OrdinalTraits<Epetra_GO>::max());
    ALBANY_EXPECT(row<=max_safe_gid, "Error! Input gids exceed Epetra_GO ranges.\n");
    (void) max_safe_gid;

    // Epetra expects pointers to non-const, and Epetra_GO may differ from GO.
    const Epetra_GO e_row = static_cast<Epetra_GO>(row);
    const int e_size = indices.size();

    int e_lrow = e_range->LID(e_row);

    //ignore indices that are not owned by the this processor
    if(e_lrow < 0) return;

    auto& e_row_indices = e_local_graph[e_lrow];
    for (int i=0; i<e_size; ++i) {
      ALBANY_EXPECT(indices[i]<=max_safe_gid, "Error! Input gids exceed Epetra_GO ranges.\n");
      e_row_indices.emplace(static_cast<Epetra_GO>(indices[i]));
    }
#else
    TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Epetra is not enabled in albany.\n");
#endif
  } else {
    // Despite being both 64 bits, GO and Tpetra_GO *may* be different *types*.
    int lrow = t_range->getLocalElement(static_cast<Tpetra_GO>(row));

    //ignore indices that are not owned by the this processor
    if(lrow < 0) return;
    
    auto& row_indices = t_local_graph[lrow];
    for (int i=0; i<indices.size(); ++i) {
      row_indices.emplace(static_cast<Tpetra_GO>(indices[i]));
    }
  }
}

void ThyraCrsMatrixFactory::fillComplete () {

  // We created the CrsGraph,
  // insert indices from the temporary local graph,
  // and call fill complete.

  auto bt = Albany::build_type();
  if (bt==BuildType::Epetra) {
#ifdef ALBANY_EPETRA
    Teuchos::ArrayRCP<int> nonzeros_per_row_array(e_range->NumMyElements());
    for (int lrow=0; lrow<nonzeros_per_row_array.size(); ++lrow)
      nonzeros_per_row_array[lrow] = e_local_graph[lrow].size();

    m_graph->e_graph = Teuchos::rcp(new Epetra_CrsGraph(Copy,*e_range,nonzeros_per_row_array.getRawPtr(),true));

    for (int lrow=0; lrow<nonzeros_per_row_array.size(); ++lrow) {
      auto& row_indices = e_local_graph[lrow];
      if(row_indices.size()>0) {
        Teuchos::Array<Epetra_GO> e_indices(row_indices.size());
        int i=0;
        for (const auto &index : row_indices)
          e_indices[i++] = index;
        auto row = e_range->GID(lrow);
        m_graph->e_graph->InsertGlobalIndices(row,e_indices.size(),e_indices.getRawPtr());
      }
    }

    e_local_graph.clear();
    auto e_domain = getEpetraBlockMap(m_domain_vs);
    m_graph->e_graph->FillComplete(*e_domain,*e_range);
    m_graph->e_graph->OptimizeStorage();
    e_range.reset();
#else
    TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Epetra is not enabled in albany.\n");
#endif
  } else {

    Teuchos::ArrayRCP<size_t> nonzeros_per_row_array(t_range->getNodeNumElements());

    for (int lrow=0; lrow<nonzeros_per_row_array.size(); ++lrow) {
      nonzeros_per_row_array[lrow] = t_local_graph[lrow].size();
    }

    m_graph->t_graph = Teuchos::rcp(new Tpetra_CrsGraph(t_range,nonzeros_per_row_array()));

    for (int lrow=0; lrow<nonzeros_per_row_array.size(); ++lrow) {
      auto& row_indices = t_local_graph[lrow];
      if(row_indices.size()>0) {
        Teuchos::Array<Tpetra_GO> t_indices(row_indices.size());
        int i=0;
        for (const auto &index : row_indices)
          t_indices[i++] = index;
        auto row = t_range->getGlobalElement(lrow);

        m_graph->t_graph->insertGlobalIndices(row,t_indices);
      }
    }

    t_local_graph.clear();
    auto t_domain = getTpetraMap(m_domain_vs);
    m_graph->t_graph->fillComplete(t_domain,t_range);
    t_range.reset();
  }

  m_filled = true;
}

Teuchos::RCP<Thyra_LinearOp> ThyraCrsMatrixFactory::createOp () const {
  TEUCHOS_TEST_FOR_EXCEPTION (!m_filled, std::logic_error, "Error! Cannot create a linear operator if the graph is not filled.\n");

  Teuchos::RCP<Thyra_LinearOp> op;
  auto bt = Albany::build_type();
  if (bt==BuildType::Epetra) {
#ifdef ALBANY_EPETRA
    Teuchos::RCP<Epetra_CrsMatrix> matrix = Teuchos::rcp(new Epetra_CrsMatrix(Copy, *m_graph->e_graph)); 
    matrix->PutScalar(0.0); 
    Teuchos::RCP<Epetra_Operator> mat = Teuchos::rcp_implicit_cast<Epetra_Operator>(matrix); 
    op = createThyraLinearOp(mat);
#else
    TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Epetra is not enabled in albany.\n");
#endif
  } else {
    Teuchos::RCP<Tpetra_CrsMatrix> mat = Teuchos::rcp (new Tpetra_CrsMatrix(m_graph->t_graph));
    auto const zero = Teuchos::ScalarTraits<ST>::zero();
    mat->resumeFill(); 
    mat->setAllToScalar(zero);
    mat->fillComplete(); 
    op = createThyraLinearOp(Teuchos::rcp_implicit_cast<Tpetra_Operator>(mat));
  }

  return op;
}

} // namespace Albany
