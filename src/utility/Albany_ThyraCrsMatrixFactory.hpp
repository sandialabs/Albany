#ifndef ALBANY_THYRA_CRS_MATRIX_FACTORY_HPP
#define ALBANY_THYRA_CRS_MATRIX_FACTORY_HPP

#include "Teuchos_RCP.hpp"
#include "Albany_ThyraTypes.hpp"

#include "Albany_TpetraThyraUtils.hpp"
#include "Albany_EpetraThyraUtils.hpp"

#include <set>

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

  // Create an empty graph, that needs to be filled later
  ThyraCrsMatrixFactory (const Teuchos::RCP<const Thyra_VectorSpace> domain_vs,
                         const Teuchos::RCP<const Thyra_VectorSpace> range_vs);

  // Create a graph from an overlapped one
  ThyraCrsMatrixFactory (const Teuchos::RCP<const Thyra_VectorSpace> domain_vs,
                         const Teuchos::RCP<const Thyra_VectorSpace> range_vs,
                         const Teuchos::RCP<const ThyraCrsMatrixFactory> overlap_src);

  // Inserts global indices in a temporary local graph. 
  // Indices that are not owned by callig processor are ignored
  // The actual graph is created when FillComplete is called
  void insertGlobalIndices (const GO row, const Teuchos::ArrayView<const GO>& indices);

  // Creates the CrsGraph,
  // inserting indices from the temporary local graph,
  // and calls fillComplete.
  void fillComplete ();

  Teuchos::RCP<const Thyra_VectorSpace> getDomainVectorSpace () const { return m_domain_vs; }
  Teuchos::RCP<const Thyra_VectorSpace> getRangeVectorSpace  () const { return m_range_vs; }

  bool is_filled () const { return m_filled; }

  Teuchos::RCP<Thyra_LinearOp>  createOp () const;

private:

  // Struct hiding the concrete implementation. This is an implementation
  // detail of this class, so it's private and its implementation is not in the header.
  struct Impl;
  Teuchos::RCP<Impl> m_graph;

  Teuchos::RCP<const Thyra_VectorSpace> m_domain_vs;
  Teuchos::RCP<const Thyra_VectorSpace> m_range_vs;

#ifdef ALBANY_EPETRA
  std::vector<std::set<Epetra_GO>> e_local_graph;
  Teuchos::RCP<const Epetra_BlockMap> e_range;
#endif

  std::vector<std::set<Tpetra_GO>> t_local_graph;
  Teuchos::RCP<const Tpetra_Map> t_range;

  bool m_filled;
};

} // namespace Albany

#endif // ALBANY_THYRA_CRS_MATRIX_FACTORY_HPP
