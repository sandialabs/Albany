//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_SDIRICHLET_DEF_HPP
#define PHAL_SDIRICHLET_DEF_HPP

#include "PHAL_SDirichlet.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "Teuchos_TestForException.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ThyraUtils.hpp"

// TODO: remove this include when you manage to abstract away from Tpetra the Jacobian impl.
#include "Albany_TpetraThyraUtils.hpp"

namespace PHAL {

//
// Specialization: Residual
//
template<typename Traits>
SDirichlet<PHAL::AlbanyTraits::Residual, Traits>::SDirichlet(
    Teuchos::ParameterList& p)
    : PHAL::DirichletBase<PHAL::AlbanyTraits::Residual, Traits>(p)
{
  return;
}

//
//
//
template<typename Traits>
void
SDirichlet<PHAL::AlbanyTraits::Residual, Traits>::evaluateFields(
    typename Traits::EvalData dirichlet_workset)
{
  Teuchos::RCP<Thyra_Vector> x = Teuchos::rcp_const_cast<Thyra_Vector>(dirichlet_workset.x);
  Teuchos::RCP<Thyra_Vector> f = dirichlet_workset.f;

  Teuchos::ArrayRCP<ST> f_view = Albany::getNonconstLocalData(f);
  Teuchos::ArrayRCP<ST> x_view = Albany::getNonconstLocalData(x);

  // Grab the vector of node GIDs for this Node Set ID
  std::vector<std::vector<int>> const& ns_nodes = dirichlet_workset.nodeSets->find(this->nodeSetID)->second;

  for (size_t ns_node = 0; ns_node < ns_nodes.size(); ns_node++) {
    int const dof = ns_nodes[ns_node][this->offset];

    f_view[dof] = 0.0;
    x_view[dof] = this->value;

#if defined(ALBANY_LCM)
    // Record DOFs to avoid setting Schwarz BCs on them.
    dirichlet_workset.fixed_dofs_.insert(dof);
#endif
  }
}

//
// Specialization: Jacobian
//
template<typename Traits>
SDirichlet<PHAL::AlbanyTraits::Jacobian, Traits>::SDirichlet(
    Teuchos::ParameterList& p)
    : PHAL::DirichletBase<PHAL::AlbanyTraits::Jacobian, Traits>(p)
{
  scale = p.get<RealType>("SDBC Scaling", 1.0);  
}

//
//
//
template<typename Traits>
void
SDirichlet<PHAL::AlbanyTraits::Jacobian, Traits>::evaluateFields(
    typename Traits::EvalData dirichlet_workset)
{
  Teuchos::RCP<Thyra_Vector> f = dirichlet_workset.f;
  Teuchos::RCP<Thyra_Vector> x = Teuchos::rcp_const_cast<Thyra_Vector>(dirichlet_workset.x);

  // TODO: abstract away the tpetra interface
  Teuchos::RCP<Tpetra_CrsMatrix> J = Albany::getTpetraMatrix(dirichlet_workset.Jac);

  auto row_map = J->getRowMap();
  auto col_map = J->getColMap();
  // we make this assumption, which lets us use both local row and column
  // indices into a single is_dbc vector
  ALBANY_ASSERT(col_map->isLocallyFitted(*row_map));

  auto& ns_nodes = dirichlet_workset.nodeSets->find(this->nodeSetID)->second;

  bool const fill_residual = f != Teuchos::null;

  auto f_view = fill_residual ? Albany::getNonconstLocalData(f) : Teuchos::null;
  auto x_view = fill_residual ? Albany::getNonconstLocalData(x) : Teuchos::null;

  Teuchos::Array<Tpetra_GO> global_index(1);

  Teuchos::Array<LO> index(1);

  Teuchos::Array<ST> entry(1);

  Teuchos::Array<ST> entries;

  Teuchos::Array<LO> indices;

  using IntVec = Tpetra::Vector<int, Tpetra_LO, Tpetra_GO, KokkosNode>;
  using Import = Tpetra::Import<Tpetra_LO, Tpetra_GO, KokkosNode>;
  Teuchos::RCP<const Import> import;

  auto domain_map = row_map;  // we are assuming this!

  // in theory we should use the importer from the CRS graph, although
  // I saw a segfault in one of the tests when doing this...
  // if (J->getCrsGraph()->isFillComplete()) {
  //  import = J->getCrsGraph()->getImporter();
  //} else {
  // this construction is expensive!
  import = Teuchos::rcp(new Import(domain_map, col_map));
  //}

  IntVec row_is_dbc(row_map);
  IntVec col_is_dbc(col_map);

  int const spatial_dimension = dirichlet_workset.spatial_dimension_;

#if defined(ALBANY_LCM)
  auto const& fixed_dofs = dirichlet_workset.fixed_dofs_;
#endif

  row_is_dbc.template modify<Kokkos::HostSpace>();
  {
    auto row_is_dbc_data =
        row_is_dbc.template getLocalView<Kokkos::HostSpace>();
    ALBANY_ASSERT(row_is_dbc_data.extent(1) == 1);
#if defined(ALBANY_LCM)
    if (dirichlet_workset.is_schwarz_bc_ == false) {  // regular SDBC
#endif
      for (size_t ns_node = 0; ns_node < ns_nodes.size(); ns_node++) {
        auto dof                = ns_nodes[ns_node][this->offset];
        row_is_dbc_data(dof, 0) = 1;
      }
#if defined(ALBANY_LCM)
    } else {  // special case for Schwarz SDBC
      for (size_t ns_node = 0; ns_node < ns_nodes.size(); ns_node++) {
        for (int offset = 0; offset < spatial_dimension; ++offset) {
          auto dof = ns_nodes[ns_node][offset];
          // If this DOF already has a DBC, skip it.
          if (fixed_dofs.find(dof) != fixed_dofs.end()) continue;
          row_is_dbc_data(dof, 0) = 1;
        }
      }
    }
#endif
  }
  col_is_dbc.doImport(row_is_dbc, *import, Tpetra::ADD);
  auto col_is_dbc_data = col_is_dbc.template getLocalView<Kokkos::HostSpace>();

  size_t const num_local_rows = J->getNodeNumRows();
  auto         min_local_row  = row_map->getMinLocalIndex();
  auto         max_local_row  = row_map->getMaxLocalIndex();
  for (auto local_row = min_local_row; local_row <= max_local_row;
       ++local_row) {
    auto num_row_entries = J->getNumEntriesInLocalRow(local_row);

    entries.resize(num_row_entries);
    indices.resize(num_row_entries);

    J->getLocalRowCopy(local_row, indices(), entries(), num_row_entries);

    auto row_is_dbc = col_is_dbc_data(local_row, 0) > 0;

    if (row_is_dbc && fill_residual == true) {
      f_view[local_row] = 0.0;
      x_view[local_row] = this->value.val();
    }
    

    for (size_t row_entry = 0; row_entry < num_row_entries; ++row_entry) {
      auto local_col         = indices[row_entry];
      auto is_diagonal_entry = local_col == local_row;
      //IKT, 4/5/18: scale diagonal entries by provided scaling 
      if (is_diagonal_entry && row_is_dbc) {
        entries[row_entry] *= scale;   
      }
      if (is_diagonal_entry) continue;
      ALBANY_ASSERT(local_col >= col_map->getMinLocalIndex());
      ALBANY_ASSERT(local_col <= col_map->getMaxLocalIndex());
      auto col_is_dbc = col_is_dbc_data(local_col, 0) > 0;
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
template<typename Traits>
SDirichlet<PHAL::AlbanyTraits::Tangent, Traits>::SDirichlet(
    Teuchos::ParameterList& p)
    : PHAL::DirichletBase<PHAL::AlbanyTraits::Tangent, Traits>(p)
{
  return;
}

//
//
//
template<typename Traits>
void
SDirichlet<PHAL::AlbanyTraits::Tangent, Traits>::evaluateFields(
    typename Traits::EvalData dirichlet_workset)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
      true,
      Teuchos::Exceptions::InvalidParameter,
      std::endl
          << "Error!  Tangent specialization for PHAL::SDirichlet "
             "is not implemented!\n");
  return;
}

//
// Specialization: DistParamDeriv
//
template<typename Traits>
SDirichlet<PHAL::AlbanyTraits::DistParamDeriv, Traits>::SDirichlet(
    Teuchos::ParameterList& p)
    : PHAL::DirichletBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p)
{
  return;
}

//
//
//
template<typename Traits>
void
SDirichlet<PHAL::AlbanyTraits::DistParamDeriv, Traits>::evaluateFields(
    typename Traits::EvalData dirichlet_workset)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
      true,
      Teuchos::Exceptions::InvalidParameter,
      std::endl
          << "Error!  DistParamDeriv specialization for PHAL::SDirichlet "
             "is not implemented!\n");
  return;
}

}  // namespace PHAL

#endif
