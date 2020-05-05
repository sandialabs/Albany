#ifndef ALBANY_THYRA_CRS_MATRIX_FACTORY_HPP
#define ALBANY_THYRA_CRS_MATRIX_FACTORY_HPP

#include "Teuchos_RCP.hpp"
#include "Albany_ThyraTypes.hpp"

namespace Albany {

/*
 * A class to setup a crs graph and then build an empty operator
 * 
 * Thyra does not have the concept of 'Graph', since it is designed to abstract
 * at a mathematical level, in a world of vector spaces, operators, and vectors.
 * The concept of graph has to do with the particular implementation of an operator,
 * namely its storage. From the computational point of view it is an important object,
 * hence we need to have access to its functionalities. Since we can't get it
 * in Thyra, and we don't want to expose particular implementations (e.g., Epetra
 * vs Tpetra), we implementa a very light-weight structure, that can create and setup
 * a graph, and, upon request, create a linear operator associated with that graph.
 * The implementation details of the graph are hidden, as is the concrete linear
 * algebra package underneath. The global function 'Albany::build_type' is used
 * to determine in which format the graph has to be stored.
 */

struct ThyraCrsMatrixFactory {

  // Prepares the factory for the creation of a graph with given domain/range
  // vector spaces. If row_vs is null, off-rank row insertions will be ignored.
  // If row_vs is not null, this class will assume insertion in a FE style.
  // In particular, it will be assumed (and checked) that range_vs is a one-to-one vs,
  // and we will also assume (checked only in Tpetra builds) that row_vs is a
  // superset of range_vs, containing owned and shared dofs.
  ThyraCrsMatrixFactory (const Teuchos::RCP<const Thyra_VectorSpace> domain_vs,
                         const Teuchos::RCP<const Thyra_VectorSpace> range_vs,
                         const Teuchos::RCP<const Thyra_VectorSpace> row_vs = Teuchos::null);

  // Inserts global indices in a temporary local structure. 
  // The actual graph is created when fillComplete is called
  void insertGlobalIndices (const GO row, const Teuchos::ArrayView<const GO>& indices);

  // Fills the actual graph optimizing storage (exact count of nnz per row).
  void fillComplete ();

  Teuchos::RCP<const Thyra_VectorSpace> getDomainVectorSpace () const { return m_domain_vs; }
  Teuchos::RCP<const Thyra_VectorSpace> getRangeVectorSpace  () const { return m_range_vs; }

  bool is_filled () const { return m_filled; }

  // Creates an operator after the graph has been created
  //  PreCondition: is_filled() == true
  // Notes:
  //  - If ignoreNonLocalRows is true, the operator's graph will only contain
  //    rows corresponding to range_vs. Only relevant if row_vs != range_vs
  //  - The operator is guaranteed to have a static filled graph, have its
  //    storage allocated (and optimized, if possible), and all entries set to 0.
  Teuchos::RCP<Thyra_LinearOp>  createOp (const bool ignoreNonLocalRows = false) const;

private:

  // Struct hiding the concrete implementation. This is an implementation
  // detail of this class, so it's private and its implementation is not in the header.
  struct Impl;
  Teuchos::RCP<Impl> m_graph;

  Teuchos::RCP<const Thyra_VectorSpace> m_domain_vs;
  Teuchos::RCP<const Thyra_VectorSpace> m_range_vs;
  Teuchos::RCP<const Thyra_VectorSpace> m_row_vs;

  bool m_filled;              // Whether fill of the graph has happened
  bool m_row_same_as_range;   // Whether row_vs and range_vs are the same
};

} // namespace Albany

#endif // ALBANY_THYRA_CRS_MATRIX_FACTORY_HPP
