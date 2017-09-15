//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "Teuchos_TestForException.hpp"

//#define DEBUG

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

#if defined(DEBUG)
  Teuchos::FancyOStream &fos = *Teuchos::VerboseObjectBase::getDefaultOStream();
  fos << "\n*** RESIDUAL ***\n";
  f->describe(fos, Teuchos::VERB_EXTREME);
  fos << "\n*** RESIDUAL ***\n";
  fos << "\n*** SOLUTION ***\n";
  x->describe(fos, Teuchos::VERB_EXTREME);
  fos << "\n*** SOLUTION ***\n";
#endif  // DEBUG
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
StrongDBC<PHAL::AlbanyTraits::Jacobian, Traits>::evaluateFields(
    typename Traits::EvalData dirichlet_workset) {
  Teuchos::RCP<Tpetra_Vector>
  f = dirichlet_workset.fT;

  Teuchos::RCP<Tpetra_Vector>
  x = Teuchos::rcpFromRef(const_cast<Tpetra_Vector &>(*dirichlet_workset.xT));

  Teuchos::RCP<Tpetra_CrsMatrix>
  J = dirichlet_workset.JacT;

  Teuchos::RCP<const Tpetra_Map>
  jac_map = J->getMap();

  auto const
  global_length = x->getGlobalLength();

  std::vector<ST>
  marker(global_length, 0.0);

  std::vector<std::vector<int>> const &
  ns_nodes = dirichlet_workset.nodeSets->find(this->nodeSetID)->second;

  bool const
  fill_residual = f != Teuchos::null;

  Teuchos::ArrayRCP<ST>
  f_view = fill_residual == true ? f->get1dViewNonConst() : Teuchos::null;

  Teuchos::ArrayRCP<ST>
  x_view = fill_residual == true ? x->get1dViewNonConst() : Teuchos::null;

  Teuchos::Array<LO>
  index(1);

  Teuchos::Array<ST>
  entry(1);

  Teuchos::Array<ST>
  entries;

  Teuchos::Array<LO>
  indices;

  for (size_t ns_node = 0; ns_node < ns_nodes.size(); ns_node++) {
    int const
    dof = ns_nodes[ns_node][this->offset];

    GO const
    global_dof = jac_map->getGlobalElement(dof);

    marker[global_dof] += 1.0;

    if (fill_residual == true) {
      f_view[dof] = 0.0;
      x_view[dof] = this->value.val();
    }

    size_t const
    num_rows = J->getNodeNumRows();

    for (size_t row = 0; row < num_rows; ++row) {
      size_t
      num_cols = J->getNumEntriesInLocalRow(row);

      entries.resize(num_cols);
      indices.resize(num_cols);

      index[0] = dof;
      entry[0] = 0.0;

      J->getLocalRowCopy(row, indices(), entries(), num_cols);

      if (row == dof) {
        // Set entries other than the diagonal to zero
        for (size_t col = 0; col < num_cols; ++col) {
          auto const
          col_index = indices[col];

          if (col_index != dof) entries[col] = 0.0;
        }
        J->replaceLocalValues(dof, indices(), entries());
      }
    }
  }

  std::vector<ST>
  global_marker(global_length, 0.0);

  for (int i = 0; i < global_length; i++) {
    Teuchos::reduceAll(
        *(jac_map->getComm()), Teuchos::REDUCE_SUM,
        /*numvals=*/1, &marker[i], &global_marker[i]);
  }

  auto const
  num_global_cols = J->getGlobalNumCols();

  auto const
  num_global_rows = J->getGlobalNumRows();

  // loop over global columns
  for (auto gcol = 0; gcol < num_global_cols; ++gcol) {
    // check if gcol dof is dirichlet dof
    ST const
    is_dir_dof = global_marker[gcol];
    // if gcol is dirichlet dof, zero out all (global) rows corresponding to
    // global column gcol
    if (is_dir_dof != 0.0) {
#ifdef DEBUG
      auto const proc_num = jac_map->getComm()->getRank();

      std::cout << "IKT proc, zeroeing out column = " << proc_num << ", "
                << gcol << std::endl;
#endif
      // loop over global rows
      for (auto grow = 0; grow < num_global_rows; ++grow) {
        if (grow != gcol) {
          Teuchos::Array<GO>
          gcol_array(1);

          gcol_array[0] = gcol;

          Teuchos::Array<ST>
          value(1);

          value[0] = 0.0;

          J->replaceGlobalValues(grow, gcol_array(), value());
        }
      }
    }
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
