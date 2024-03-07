#include "Albany_ThyraCrsMatrixFactory.hpp"

#include "Albany_ThyraUtils.hpp"
#include "Albany_TpetraThyraUtils.hpp"
#include "Albany_TpetraTypes.hpp"

#include "Tpetra_FEMultiVector.hpp"
#include "Albany_Utils.hpp"
#include "Albany_Macros.hpp"

#include <set>
#include <vector>

namespace Albany {

// The implementation of the graph
struct ThyraCrsMatrixFactory::Impl {

  Impl () = default;

  std::map<GO,std::set<GO>> temp_graph;
  Teuchos::RCP<Tpetra_FECrsGraph> t_graph;
};

ThyraCrsMatrixFactory::
ThyraCrsMatrixFactory (const Teuchos::RCP<const Thyra_VectorSpace> domain_vs,
                       const Teuchos::RCP<const Thyra_VectorSpace> range_vs)
 : ThyraCrsMatrixFactory(domain_vs,range_vs,domain_vs,range_vs)
{
  // Nothing to do here
}

ThyraCrsMatrixFactory::
ThyraCrsMatrixFactory (const Teuchos::RCP<const Thyra_VectorSpace> domain_vs,
                       const Teuchos::RCP<const Thyra_VectorSpace> range_vs,
                       const Teuchos::RCP<const Thyra_VectorSpace> ov_domain_vs,
                       const Teuchos::RCP<const Thyra_VectorSpace> ov_range_vs)
 : m_graph(new Impl())
 , m_domain_vs(domain_vs)
 , m_range_vs(range_vs)
 , m_ov_domain_vs(ov_domain_vs)
 , m_ov_range_vs(ov_range_vs)
 , m_filled (false)
{
  TEUCHOS_TEST_FOR_EXCEPTION (domain_vs.is_null(), std::runtime_error,
                              "Error! Input domain vs is null.\n");
  TEUCHOS_TEST_FOR_EXCEPTION (range_vs.is_null(), std::runtime_error,
                              "Error! Input range vs is null.\n");
  TEUCHOS_TEST_FOR_EXCEPTION (ov_domain_vs.is_null(), std::runtime_error,
                              "Error! Input overlapped domain vs is null.\n");
  TEUCHOS_TEST_FOR_EXCEPTION (ov_range_vs.is_null(), std::runtime_error,
                              "Error! Input overlapped range vs is null.\n");

  if (sameAs(m_range_vs,m_ov_range_vs)) {
    // Restrict this test for square matrices:
    if(sameAs(m_range_vs,m_domain_vs))
      TEUCHOS_TEST_FOR_EXCEPTION (!sameAs(m_domain_vs,m_ov_domain_vs), std::runtime_error,
                                  "Error! Range and overlapped range vs coincide, but domain and overlapped domain vs do not.\n");
    m_fe_crs = false;
  } else {
    // When building a FECrs matrix, we REQUIRE the overlapped domain vs.
    // This is because if we let Tpetra build the column map, even if the
    // the owned gids come first, the remaining ones would be in an order that is
    // different from what we have in the overlapped maps in the discretization.
    // This would cause the GID<->LID mapping inside Epetra to be different
    // from the one we get using the overlapped maps from the discretization.
    // See Trilinos issue 7455 and PR 7572 for more details
    TEUCHOS_TEST_FOR_EXCEPTION (!isOneToOne(m_domain_vs), std::logic_error,
      "[ThyraCrsMatrixFactory] Error! When providing an overlapped domain vs, the domain vs must be one-to-one.\n");
    TEUCHOS_TEST_FOR_EXCEPTION (!isOneToOne(m_range_vs), std::logic_error,
      "[ThyraCrsMatrixFactory] Error! When providing an overlapped range vs, the range vs must be one-to-one.\n");
    m_fe_crs = true;
  }

  m_ov_range_indexer = createGlobalLocalIndexer(m_ov_range_vs);
}

void ThyraCrsMatrixFactory::
insertGlobalIndices (const GO row, const GO col, const bool symmetric)
{
  insertGlobalIndices(row,Teuchos::ArrayView<const GO>(&col,1));
  if (symmetric) {
    insertGlobalIndices(col,Teuchos::ArrayView<const GO>(&row,1));
  }
}
void ThyraCrsMatrixFactory::
insertGlobalIndices (const Teuchos::ArrayView<const GO>& rows,
                     const Teuchos::ArrayView<const GO>& cols,
                     const bool symmetric)
{
  for (const GO row : rows) {
    insertGlobalIndices(row,cols);
  }
  if (symmetric) {
    for (const GO col : cols) {
      insertGlobalIndices(col,rows);
    }
  }
}

void ThyraCrsMatrixFactory::insertGlobalIndices (const GO row, const Teuchos::ArrayView<const GO>& indices)
{
  // Indices are inserted in a temporary local graph. 
  // The actual graph is created and filled when fillComplete is called,
  // so that we have an actual count of the non-zeros, to properly
  // allocate the Tpetra static graph.
  // Note: the alternative would be to have user do two loops: during the first,
  //       the non-zeros are counted, then the graph is created with the exact
  //       nnz count, and in the second loop the indices are inserted.
  //       Keeping indices in a temp auxiliary structure, allowing a single loop,
  //       seems the easiest solution, and not too bad, considering graphs are
  //       usually created once during simulation setup.
  const GO max_safe_gid = Teuchos::OrdinalTraits<GO>::max();
  ALBANY_EXPECT (m_ov_range_indexer->isLocallyOwnedElement(row),
                 "Error! Row " + std::to_string(row) + " is not in the overlap range map.\n");
  auto& row_indices = m_graph->temp_graph[row];
  const int size = indices.size();
  for (int i=0; i<size; ++i) {
    row_indices.emplace(indices[i]);
  }
}

void ThyraCrsMatrixFactory::fillComplete () {

  // We create the CrsGraph, insert indices from the temporary local graph,
  // and call fill complete.
  // Note: now that we dropped support of Epetra we could compute the nnz here.
  // Note: the nnz per row needs to be the GLOBAL one. That is, for each
  //       row, we need to combine nnz coming from all ranks.

 {
    auto t_range  = getTpetraMap(m_range_vs);
    auto t_domain = getTpetraMap(m_domain_vs);
    auto t_ov_range  = getTpetraMap(m_ov_range_vs);
    auto t_ov_domain = getTpetraMap(m_ov_domain_vs);

    // Compute the number of nnz per row *for the globally assembled matrix*
    // Note: we cannot use int as ST for nnz, cause we don't know if it is enabled
    //       in Tpetra. Besides, we later need a DualView storing size_t, so we'd
    //       have to copy anyways. And ST=size_t is also bad, since it is likely
    //       not enabled in Tpetra. Therefore, use ST to store the nnz count,
    //       then copy the result into a dual view of size_t.
    // Note: you could use two Tpetra_Vector's, doing the import/export manually.
    //       However, notice that you'd have to do a combine AND a scatter, since
    //       for the FECrsGraph we need the nnz to have the overlapped map.
    //       FE multivector does all the work for us, so just use that.
    Teuchos::RCP<Tpetra_Import> importer(new Tpetra_Import (t_range,t_ov_range));
    Tpetra::FEMultiVector<ST,LO,Tpetra_GO,KokkosNode> nnz(t_range,importer,1);
    nnz.beginAssembly();
    for (const auto& it : m_graph->temp_graph) {
      LO lrow = t_ov_range->getLocalElement(static_cast<Tpetra_GO>(it.first));
      nnz.sumIntoLocalValue(lrow,0,it.second.size());
    }
    // Add up nnz from different ranks.
    nnz.endAssembly();

    // Switch back to the overlapped vector
    nnz.switchActiveMultiVector();

    // For some reason Tpetra_FECrsGraph does not have a ctor that takes an Teuchos::ArrayView,
    // so we must create a DualView.
    using exec_space = Tpetra_CrsGraph::execution_space;
    using DView = Kokkos::DualView<size_t*, exec_space>;
    LO numOvRows = getLocalSubdim(m_ov_range_vs);
    DView nnz_per_row("nnz",numOvRows);
    auto ov_nnz_data = nnz.getData(0);
    for (LO i=0; i<numOvRows; ++i) {
      nnz_per_row.view_host()[i] = static_cast<size_t>(ov_nnz_data[i]);
    }
    // Make sure it is synced to device
    nnz_per_row.modify_host();
    nnz_per_row.sync_device();

    // Tpetra::FECrsGraph has a check to ensure that at least one of the col gids in a given row
    // is owned in the unique map (meaning "if you own an element, you should own at least one of its nodes").
    // This check is too restrictive for albany. Giving it up, means that we will have a column map
    // that may contain more ids than it is actually necessary (in the owned graph), but since there
    // is no way around it right now, we have to accept it.
    // TODO: A better way to do this would be to fix node sharing (or, better, element sharing) in the stk bulk data.
    Teuchos::RCP<Teuchos::ParameterList> pl(new Teuchos::ParameterList());
    pl->set<bool>("Check Col GIDs In At Least One Owned Row",false);

    // Now that we have the exact count of nnz for the unique graph, we can create the FECrs graph.
    m_graph->t_graph = Teuchos::rcp(new Tpetra_FECrsGraph(t_range,t_ov_range,nnz_per_row,t_ov_domain,Teuchos::null,t_domain));

    // Now we can set the parameter list
    // Note: you CAN'T set it in the c-tor, since the ctor of the base class (CrsGraph) calls getValidParameters.
    //       Since you're in the c-tor, you DON'T get virtual dispatch, and you only get the valid parameters
    //       of CrsGraph, which do NOT include the one we need, causing an error during the input pl call to
    //       validateParametersAndSetDefaults.
    m_graph->t_graph->setParameterList(pl);

    // Loop over the temp auxiliary structure, and fill the actual Tpetra graph
    m_graph->t_graph->beginAssembly();
    for (const auto& it : m_graph->temp_graph) {
      const auto& row_indices = it.second;
      if(row_indices.size()>0) {
        Teuchos::Array<Tpetra_GO> t_indices(row_indices.size());
        int i=0;
        for (const auto &index : row_indices) {
          t_indices[i] = index;
          ++i;
        }
        m_graph->t_graph->insertGlobalIndices(static_cast<Tpetra_GO>(it.first),t_indices());
      }
    }

    // Global assemble the graph
    m_graph->t_graph->endAssembly();

    // Cleanup temporaries
    m_graph->temp_graph.clear();
  }

  m_filled = true;
}

Teuchos::RCP<Thyra_LinearOp> ThyraCrsMatrixFactory::createOp (const bool ignoreNonLocalRows) const {
  TEUCHOS_TEST_FOR_EXCEPTION (!is_filled(), std::logic_error, "Error! Cannot create a linear operator if the graph is not filled.\n");

  auto const zero = Teuchos::ScalarTraits<ST>::zero();
  Teuchos::RCP<Thyra_LinearOp> op;

  Teuchos::RCP<Tpetra_CrsMatrix> matrix;
  if (ignoreNonLocalRows) {
    // FECrsGraph *is* a CrsGraph. CrsMatrix will only deal with CrsGraph stuff
    matrix = Teuchos::rcp(new Tpetra_CrsMatrix(m_graph->t_graph));
    matrix->setAllToScalar(zero);
    matrix->fillComplete();
  } else {
    auto fe_matrix = Teuchos::rcp (new Tpetra_FECrsMatrix(m_graph->t_graph));
    // Tpetra creates FECrsMatrix already in assembly mode.
    // We want linear ops to be in assembly mode *only when explicitly requested*.
    // If the Thyra LinearOp is created with the matrix in assembly mode,
    // its range/domain vs's would be the overlapped ones.
    fe_matrix->beginAssembly();
    fe_matrix->setAllToScalar(zero);
    fe_matrix->endAssembly();
    matrix = fe_matrix;
  }
  op = createThyraLinearOp(Teuchos::rcp_implicit_cast<Tpetra_Operator>(matrix));


  return op;
}

} // namespace Albany
