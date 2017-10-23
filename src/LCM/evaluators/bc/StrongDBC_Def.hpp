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

//#define DAN_DEBUG
//#define RUN_ITK_CODE

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

#if defined(DAN_DEBUG)
  Teuchos::FancyOStream &fos = *Teuchos::VerboseObjectBase::getDefaultOStream();
  fos << "\n*** RESIDUAL ***\n";
  f->describe(fos, Teuchos::VERB_EXTREME);
  fos << "\n*** RESIDUAL ***\n";
  fos << "\n*** SOLUTION ***\n";
  x->describe(fos, Teuchos::VERB_EXTREME);
  fos << "\n*** SOLUTION ***\n";
#endif  // DAN_DEBUG
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

#ifdef RUN_ITK_CODE
  auto const
  global_length = x->getGlobalLength();

  auto const 
  max_global_index = x->getMap()->getMaxAllGlobalIndex(); 

  auto const 
  min_global_index = x->getMap()->getMinAllGlobalIndex(); 
 
#ifdef DAN_DEBUG
  Teuchos::FancyOStream &fos = *Teuchos::VerboseObjectBase::getDefaultOStream();
  fos << "IKT global_length, max_global_index, min_global_index = " << global_length << ", " << 
                max_global_index << ", " << min_global_index << std::endl; 
#endif

  std::vector<ST>
  marker(max_global_index+1, 0.0);
#endif // RUN_ITK_CODE

  std::vector<std::vector<int>> const &
  ns_nodes = dirichlet_workset.nodeSets->find(this->nodeSetID)->second;

  bool const
  fill_residual = f != Teuchos::null;

  Teuchos::ArrayRCP<ST>
  f_view = fill_residual == true ? f->get1dViewNonConst() : Teuchos::null;

  Teuchos::ArrayRCP<ST>
  x_view = fill_residual == true ? x->get1dViewNonConst() : Teuchos::null;

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

#ifdef RUN_ITK_CODE
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
#ifdef DAN_DEBUG
        auto const proc_num = jac_map->getComm()->getRank();
        auto grow = jac_map->getGlobalElement(row);
  
        std::cout << "IKT proc, zeroeing out row = " << proc_num << ", "
                  << grow << std::endl;
#endif
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
  global_marker(max_global_index+1, 0.0);

  for (int i = 0; i < max_global_index+1; i++) {
    Teuchos::reduceAll(
        *(jac_map->getComm()), Teuchos::REDUCE_SUM,
        /*numvals=*/1, &marker[i], &global_marker[i]);
  }

  // loop over global columns
  for (auto gcol = min_global_index; gcol < max_global_index+1; ++gcol) {
    // check if gcol dof is dirichlet dof
    ST const
    is_dir_dof = global_marker[gcol];
    // if gcol is dirichlet dof, zero out all (global) rows corresponding to
    // global column gcol
    if (is_dir_dof != 0.0) {
#ifdef DAN_DEBUG
      auto const proc_num = jac_map->getComm()->getRank();

      std::cout << "IKT proc, zeroeing out column = " << proc_num << ", "
                << gcol << std::endl;
#endif
      // loop over global rows
      for (auto grow = min_global_index; grow < max_global_index+1; ++grow) {
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
#endif // RUN_ITK_CODE

  using MV = Tpetra::MultiVector<>;
  using MVF = Tpetra::MultiVectorFiller<MV>;

  MVF is_dbc_filler(jac_map, 1);

  for (size_t ns_node = 0; ns_node < ns_nodes.size(); ns_node++) {
    int const
    dof = ns_nodes[ns_node][this->offset];
    global_index[0] = jac_map->getGlobalElement(dof);
    entry[0] = 1.0;
    is_dbc_filler.sumIntoGlobalValues(global_index, 0, entry);
  }
  MV is_dbc(jac_map, 1);
  is_dbc_filler.globalAssemble(is_dbc);
  auto is_dbc_view = is_dbc.get1dView();

  size_t const
  num_local_rows = J->getNodeNumRows();
  for (size_t local_row = 0; local_row < num_local_rows; ++local_row) {
    size_t
    num_row_entries = J->getNumEntriesInLocalRow(local_row);

    entries.resize(num_row_entries);
    indices.resize(num_row_entries);

    J->getLocalRowCopy(local_row, indices(), entries(), num_row_entries);

    auto row_is_dbc = is_dbc_view[local_row] > 0.0;

    if (row_is_dbc && fill_residual == true) {
    //f_view[local_row] = 0.0;
    //x_view[local_row] = this->value.val();
#ifdef DAN_DEBUG
      auto grow = jac_map->getGlobalElement(local_row);
      auto const proc_num = jac_map->getComm()->getRank();
      
      std::cout << "DAI proc, zeroeing out row = " << proc_num << ", "
                << grow << std::endl;
#endif
    }

    for (size_t row_entry = 0; row_entry < num_row_entries; ++row_entry) {
      auto const
      local_col = indices[row_entry];
      auto is_diagonal_entry = local_col == local_row;
      if (is_diagonal_entry) continue;
      auto col_is_dbc = is_dbc_view[local_col] > 0.0;
      if (row_is_dbc || col_is_dbc) {
        entries[row_entry] = 0.0;
      }
#ifdef DAN_DEBUG
      if (col_is_dbc) {
        auto gcol = jac_map->getGlobalElement(local_col);
        auto const proc_num = jac_map->getComm()->getRank();
        
        std::cout << "DAI proc, zeroeing out column = " << proc_num << ", "
                  << gcol << std::endl;
      }
#endif
    }

  //J->replaceLocalValues(local_row, indices(), entries());
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
