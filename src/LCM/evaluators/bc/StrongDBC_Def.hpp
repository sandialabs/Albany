//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "Teuchos_TestForException.hpp"

//
// Generic Template Code for Constructor and PostRegistrationSetup
//

namespace LCM {

//
//
//
template<typename EvalT, typename Traits>
StrongDBC_Base<EvalT, Traits>::
StrongDBC_Base(Teuchos::ParameterList & p) :
    PHAL::DirichletBase<EvalT, Traits>(p)
{
  return;
}

//
// Specialization: Residual
//
template<typename Traits>
StrongDBC<PHAL::AlbanyTraits::Residual, Traits>::
StrongDBC(Teuchos::ParameterList & p) :
    StrongDBC_Base<PHAL::AlbanyTraits::Residual, Traits>(p)
{
  return;
}

//
//
//
template<typename Traits>
void
StrongDBC<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData dirichlet_workset)
{
  Teuchos::RCP<Tpetra_Vector>
  f = dirichlet_workset.fT;

  Teuchos::ArrayRCP<ST>
  f_view = f->get1dViewNonConst();

  // Grab the vector of node GIDs for this Node Set ID
  std::vector<std::vector<int>> const &
  ns_nodes = dirichlet_workset.nodeSets->find(this->nodeSetID)->second;

  for (size_t node = 0; node < ns_nodes.size(); node++) {

    int const
    dof = ns_nodes[node][this->offset];

    f_view[dof] = 0.0;

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
template<typename Traits>
StrongDBC<PHAL::AlbanyTraits::Jacobian, Traits>::
StrongDBC(Teuchos::ParameterList & p) :
    StrongDBC_Base<PHAL::AlbanyTraits::Jacobian, Traits>(p)
{
  return;
}

//
//
//
template<typename Traits>
void StrongDBC<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData dirichlet_workset)
{
  Teuchos::RCP<Tpetra_Vector>
  f = dirichlet_workset.fT;

  Teuchos::RCP<Tpetra_CrsMatrix>
  J = dirichlet_workset.JacT;

  RealType const
  j_coeff = dirichlet_workset.j_coeff;

  std::vector<std::vector<int>> const &
  ns_nodes = dirichlet_workset.nodeSets->find(this->nodeSetID)->second;

  bool
  fill_residual = f != Teuchos::null;

  Teuchos::ArrayRCP<ST>
  f_view = fill_residual == true ? f->get1dViewNonConst() : Teuchos::null;

  Teuchos::Array<LO>
  index(1);

  Teuchos::Array<ST>
  entry(1);

  Teuchos::Array<ST>
  entries;

  Teuchos::Array<LO>
  indices;

  for (size_t node = 0; node < ns_nodes.size(); node++) {

      int const
      dof = ns_nodes[node][this->offset];

      if (fill_residual == true) f_view[dof] = 0.0;

      size_t const
      num_rows = J->getNodeNumRows();

      for (size_t row = 0; row < num_rows; ++row) {

        size_t
        num_cols = J->getNumEntriesInLocalRow(row);

        entries.resize(num_cols);
        indices.resize(num_cols);

        index[0] = row;
        entry[0] = 0.0;

        J->getLocalRowCopy(row, indices(), entries(), num_cols);

        if (row == dof) {
          // Set entries other than the diagonal to zero
          for (size_t col = 0; col < num_cols; ++col) {
            if (col != dof) entries[col] = 0.0;
          }
          J->replaceLocalValues(dof, indices(), entries());
        } else {
          J->replaceLocalValues(dof, index(), entry());
        }

      }

  }
  return;
}

//
// Specialization: Tangent
//
template<typename Traits>
StrongDBC<PHAL::AlbanyTraits::Tangent, Traits>::
StrongDBC(Teuchos::ParameterList & p) :
    StrongDBC_Base<PHAL::AlbanyTraits::Tangent, Traits>(p)
{
  return;
}

//
//
//
template<typename Traits>
void StrongDBC<PHAL::AlbanyTraits::Tangent, Traits>::
evaluateFields(typename Traits::EvalData dirichlet_workset)
{
  return;
}

//
// Specialization: DistParamDeriv
//
template<typename Traits>
StrongDBC<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
StrongDBC(Teuchos::ParameterList & p) :
    StrongDBC_Base<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p)
{
  return;
}

//
//
//
template<typename Traits>
void StrongDBC<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
evaluateFields(typename Traits::EvalData dirichlet_workset)
{
  return;
}

} // namespace LCM
