#ifndef ALBANY_THYRA_BLOCKED_CRS_MATRIX_FACTORY_HPP
#define ALBANY_THYRA_BLOCKED_CRS_MATRIX_FACTORY_HPP

#include "Teuchos_RCP.hpp"
#include "Albany_ThyraTypes.hpp"

#include "Albany_TpetraThyraUtils.hpp"

#include "Albany_ThyraCrsMatrixFactory.hpp"

#include <set>

namespace Albany
{

  /*
 * A class to setup a crs graph and then build an empty operator
 * 
 * Thyra does not have the concept of 'Graph', since it is designed to abstract
 * at a mathematical level, in a world of vector spaces, operators, and vectors.
 * The concept of graph has to do with the particular implementation of an operator,
 * namely its storage. From the computational point of view it is an important object,
 * hence we need to have access to its functionalities. Since we can't get it
 * in Thyra, and we don't want to expose particular implementations (e.g., Tpetra),
 * we implement a very light-weight structure, that can create and setup
 * a graph, and, upon request, create a linear operator associated with that graph.
 * The implementation details of the graph are hidden, as is the concrete linear
 * algebra package underneath.
 */

  struct ThyraBlockedCrsMatrixFactory
  {

    // Create an empty graph, that needs to be filled later
    ThyraBlockedCrsMatrixFactory(const Teuchos::RCP<const Thyra_ProductVectorSpace> domain_vs,
                                 const Teuchos::RCP<const Thyra_ProductVectorSpace> range_vs,
                                 const int nonzeros_per_row = -1); //currently not used

    ThyraBlockedCrsMatrixFactory(const Teuchos::RCP<const Thyra_ProductVectorSpace> domain_vs,
                                 const Teuchos::RCP<const Thyra_ProductVectorSpace> range_vs,
                                 const Teuchos::RCP<const Thyra_ProductVectorSpace> ov_domain_vs,
                                 const Teuchos::RCP<const Thyra_ProductVectorSpace> ov_range_vs,
                                 const int nonzeros_per_row = -1); //currently not used

    // Inserts global indices in a temporary local graph.
    // Indices that are not owned by calling processor are ignored
    // The actual graph is created when FillComplete is called
    void insertGlobalIndices(const GO row, const Teuchos::ArrayView<const GO> &indices,
                             const size_t i_block = 0, const size_t j_block = 0);

    // Creates the CrsGraph,
    // inserting indices from the temporary local graph,
    // and calls fillComplete.
    void fillComplete();

    Teuchos::RCP<const Thyra_ProductVectorSpace> getDomainVectorSpace() const { return m_domain_vs; }
    Teuchos::RCP<const Thyra_ProductVectorSpace> getRangeVectorSpace() const { return m_range_vs; }
    Teuchos::RCP<Albany::ThyraCrsMatrixFactory> getBlockFactory(const size_t i_block, const size_t j_block) { return block_factories[i_block][j_block]; }
    void setBlockFactory(const size_t i_block, const size_t j_block, Teuchos::RCP<Albany::ThyraCrsMatrixFactory> factory) { block_factories[i_block][j_block] = factory; }

    bool is_filled() const { return m_filled; }

    Teuchos::RCP<Thyra_BlockedLinearOp> createOp() const;

  private:
    Teuchos::RCP<const Thyra_ProductVectorSpace> m_domain_vs;
    Teuchos::RCP<const Thyra_ProductVectorSpace> m_range_vs;

    bool m_filled;

    Teuchos::Array<Teuchos::Array<Teuchos::RCP<Albany::ThyraCrsMatrixFactory>>> block_factories;
    const size_t n_m_blocks;
  };

} // namespace Albany

#endif // ALBANY_THYRA_BLOCKED_CRS_MATRIX_FACTORY_HPP
