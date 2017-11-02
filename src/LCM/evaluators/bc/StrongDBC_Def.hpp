//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "Teuchos_TestForException.hpp"
#include "Albany_Utils.hpp"
#include <Tpetra_MultiVectorFiller.hpp>

//#define DAN_VERBOSE
//#define RUN_IKT_CODE


namespace LCM {

//
// Specialization: Residual
//
template <typename Traits>
StrongDBC<PHAL::AlbanyTraits::Residual, Traits>::StrongDBC(
    Teuchos::ParameterList &p)
    : PHAL::DirichletBase<PHAL::AlbanyTraits::Residual, Traits>(p) {
  return;
}

//
//
//
template <typename Traits>
void
StrongDBC<PHAL::AlbanyTraits::Residual, Traits>::evaluateFields(
    typename Traits::EvalData dirichlet_workset) {
  Teuchos::RCP<Tpetra_Vector>
  f = dirichlet_workset.fT;

  Teuchos::RCP<Tpetra_Vector>
  x = Teuchos::rcpFromRef(const_cast<Tpetra_Vector &>(*dirichlet_workset.xT));

  Teuchos::ArrayRCP<ST>
  f_view = f->get1dViewNonConst();

  Teuchos::ArrayRCP<ST>
  x_view = x->get1dViewNonConst();

  // Grab the vector of node GIDs for this Node Set ID
  std::vector<std::vector<int>> const &
  ns_nodes = dirichlet_workset.nodeSets->find(this->nodeSetID)->second;

  for (size_t ns_node = 0; ns_node < ns_nodes.size(); ns_node++) {
    int const
    dof = ns_nodes[ns_node][this->offset];

    f_view[dof] = 0.0;
    x_view[dof] = this->value;

#if defined(ALBANY_LCM)
    // Record DOFs to avoid setting Schwarz BCs on them.
    dirichlet_workset.fixed_dofs_.insert(dof);
#endif
  }

  return;
}

//
// Specialization: Jacobian
//
template <typename Traits>
StrongDBC<PHAL::AlbanyTraits::Jacobian, Traits>::StrongDBC(
    Teuchos::ParameterList &p)
    : PHAL::DirichletBase<PHAL::AlbanyTraits::Jacobian, Traits>(p) {
  return;
}

//
//
//
template <typename Traits>
void
StrongDBC<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData dirichlet_workset) {
  auto f = dirichlet_workset.fT;
  auto x = Teuchos::rcpFromRef(const_cast<Tpetra_Vector &>(*dirichlet_workset.xT));
  auto J = dirichlet_workset.JacT;

  auto row_map = J->getRowMap();
  auto col_map = J->getColMap();
  // we make this assumption, which lets us use both local row and column
  // indices into a single is_dbc vector
  ALBANY_ASSERT(col_map->isLocallyFitted(*row_map));

  auto& ns_nodes = dirichlet_workset.nodeSets->find(this->nodeSetID)->second;

  bool const fill_residual = f != Teuchos::null;

  auto f_view = fill_residual == true ? f->get1dViewNonConst() : Teuchos::null;
  auto x_view = fill_residual == true ? x->get1dViewNonConst() : Teuchos::null;

  Teuchos::Array<GO>
  global_index(1);

  Teuchos::Array<LO>
  index(1);

  Teuchos::Array<ST>
  entry(1);

  Teuchos::Array<ST>
  entries;

  Teuchos::Array<LO>
  indices;

  using MV = Tpetra_MultiVector;
  using MVF = Tpetra::MultiVectorFiller<MV>;

  MVF is_dbc_filler(row_map, 1);

  for (size_t ns_node = 0; ns_node < ns_nodes.size(); ns_node++) {
    int const
    dof = ns_nodes[ns_node][this->offset];
    global_index[0] = row_map->getGlobalElement(dof);
    entry[0] = 1.0;
    is_dbc_filler.sumIntoGlobalValues(global_index, 0, entry);
  }
  MV is_dbc(col_map, 1);
  is_dbc_filler.globalAssemble(is_dbc);
  auto is_dbc_view = is_dbc.get1dView();

  size_t const
  num_local_rows = J->getNodeNumRows();
  auto min_local_row = row_map->getMinLocalIndex();
  auto max_local_row = row_map->getMaxLocalIndex();
  for (auto local_row = min_local_row; local_row <= max_local_row; ++local_row) {
    auto num_row_entries = J->getNumEntriesInLocalRow(local_row);

    entries.resize(num_row_entries);
    indices.resize(num_row_entries);

    J->getLocalRowCopy(local_row, indices(), entries(), num_row_entries);

    auto row_is_dbc = is_dbc_view[local_row] > 0.0;

    if (row_is_dbc && fill_residual == true) {
      f_view[local_row] = 0.0;
      x_view[local_row] = this->value.val();
    }

    for (size_t row_entry = 0; row_entry < num_row_entries; ++row_entry) {
      auto local_col = indices[row_entry];
      auto is_diagonal_entry = local_col == local_row;
      if (is_diagonal_entry) continue;
      ALBANY_ASSERT(local_col >= col_map->getMinLocalIndex());
      ALBANY_ASSERT(local_col <= col_map->getMaxLocalIndex());
      auto col_is_dbc = is_dbc_view[local_col] > 0.0;
      if (row_is_dbc || col_is_dbc) {
        entries[row_entry] = 0.0;
      }
    }
    J->replaceLocalValues(local_row, indices(), entries());
  }
  return;
}

//
// Specialization: Tangent
//
template <typename Traits>
StrongDBC<PHAL::AlbanyTraits::Tangent, Traits>::StrongDBC(
    Teuchos::ParameterList &p)
    : PHAL::DirichletBase<PHAL::AlbanyTraits::Tangent, Traits>(p) {
  return;
}

//
//
//
template <typename Traits>
void
StrongDBC<PHAL::AlbanyTraits::Tangent, Traits>::evaluateFields(
    typename Traits::EvalData dirichlet_workset) {
  return;
}

//
// Specialization: DistParamDeriv
//
template <typename Traits>
StrongDBC<PHAL::AlbanyTraits::DistParamDeriv, Traits>::StrongDBC(
    Teuchos::ParameterList &p)
    : PHAL::DirichletBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p) {
  return;
}

//
//
//
template <typename Traits>
void
StrongDBC<PHAL::AlbanyTraits::DistParamDeriv, Traits>::evaluateFields(
    typename Traits::EvalData dirichlet_workset) {
  return;
}

}  // namespace LCM
