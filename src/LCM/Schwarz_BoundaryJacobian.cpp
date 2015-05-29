//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Schwarz_BoundaryJacobian.hpp"
#include "Teuchos_ParameterListExceptions.hpp"
#include "Teuchos_TestForException.hpp"

#include "Albany_GenericSTKMeshStruct.hpp"
#include "Albany_STKDiscretization.hpp"
#include "Albany_Utils.hpp"
//#include "Tpetra_LocalMap.h"

//#define WRITE_TO_MATRIX_MARKET
//#define OUTPUT_TO_SCREEN
//#define DEBUG_LCM_SCHWARZ

#ifdef WRITE_TO_MATRIX_MARKET
static int
mm_counter = 0;
#endif // WRITE_TO_MATRIX_MARKET

//#define APPLY_GENERAL_IMPLICIT
//#define APPLY_H27_H8_EXPLICIT
//#define APPLY_H8_H8_EXPLICIT

LCM::
Schwarz_BoundaryJacobian::
Schwarz_BoundaryJacobian(
    Teuchos::RCP<Teuchos_Comm const> const & comm,
    Teuchos::ArrayRCP<Teuchos::RCP<Albany::Application> > const & ca,
    Teuchos::Array<Teuchos::RCP<Tpetra_CrsMatrix> > jacs,
    int const this_app_index,
    int const coupled_app_index) :
        commT_(comm),
        coupled_apps_(ca),
        jacs_(jacs),
        this_app_index_(this_app_index),
        coupled_app_index_(coupled_app_index),
        b_use_transpose_(false),
        b_initialized_(false),
        n_models_(0)
{
  assert(0 <= this_app_index && this_app_index < ca.size());
  assert(0 <= coupled_app_index && coupled_app_index < ca.size());
  domain_map_ = ca[coupled_app_index]->getMapT();
  range_map_ = ca[this_app_index]->getMapT();
}

LCM::
Schwarz_BoundaryJacobian::
~Schwarz_BoundaryJacobian()
{
}

// Initialize the operator with everything needed to apply it
void
LCM::
Schwarz_BoundaryJacobian::
initialize()
{
}

// Returns explicit matrix representation of operator if available.
Teuchos::RCP<Tpetra_CrsMatrix>
LCM::
Schwarz_BoundaryJacobian::
getExplicitOperator() const
{
  auto const
  max_num_cols = getDomainMap()->getNodeNumElements();

  Teuchos::RCP<Tpetra_CrsMatrix>
  K = Teuchos::rcp(
      new Tpetra_CrsMatrix(getRangeMap(), getDomainMap(), max_num_cols));

  auto const
  zero = Teuchos::ScalarTraits<ST>::zero();

  K->setAllToScalar(zero);

  K->fillComplete();

  return K;
}

// Returns the result of a Tpetra_Operator applied to a
// Tpetra_MultiVector X in Y.
void
LCM::
Schwarz_BoundaryJacobian::
apply(
    Tpetra_MultiVector const & X,
    Tpetra_MultiVector & Y,
    Teuchos::ETransp mode,
    ST alpha,
    ST beta) const
{
  auto const
  zero = Teuchos::ScalarTraits<ST>::zero();

  Y.putScalar(zero);
}

#if defined(APPLY_H27_H8_EXPLICIT)
// Returns the result of a Tpetra_Operator applied to a
// Tpetra_MultiVector X in Y.
void
LCM::
Schwarz_BoundaryJacobian::
apply(
    Tpetra_MultiVector const & X,
    Tpetra_MultiVector & Y,
    Teuchos::ETransp mode,
    ST alpha,
    ST beta) const
{
  auto const
  num_dof = Y.getGlobalLength();

  auto const
  dim = 3;

  auto const
  num_nodes = num_dof / dim;

  // Initialize Y vector.
  auto const
  zero = Teuchos::ScalarTraits<ST>::zero();

  Y.putScalar(zero);

  Teuchos::ArrayRCP<ST>
  Y_view = Y.get1dViewNonConst();

  Teuchos::ArrayRCP<ST const>
  X_view = X.get1dView();

  std::vector<int> const
  ns_coarse = {4,5,6,7};

  std::vector<int> const
  ns_fine = {0,1,2,3,4,5,6,7,8};

  std::vector<std::vector<int>> const
  from_fine = {{23}, {19}, {18}, {21}};

  std::vector<std::vector<int>> const
  from_coarse = {{0,5}, {1,4}, {0,1,4,5}, {2,7}, {1,2,4,7}, {3,6},
      {2,3,6,7}, {0,3,5,6}, {0,1,2,3,4,5,6,7}};

  auto const
  this_app_index = getThisAppIndex();

  std::vector<int> const &
  ns_nodes = this_app_index == 1 ? ns_coarse : ns_fine;

  std::vector<std::vector<int>> const &
  from_nodes = this_app_index == 1 ? from_fine : from_coarse;

  for (auto node = 0; node < num_nodes; ++node) {

    for (auto i = 0; i < ns_nodes.size(); ++i) {
      if (node != ns_nodes[i]) continue;

      Intrepid::Vector<double, dim>
      value(Intrepid::ZEROS);

      std::vector<int> const &
      inodes = from_nodes[i];

      auto const
      num_inodes = inodes.size();

      auto const
      weight = 1.0 / num_inodes;

      for (auto j = 0; j < num_inodes; ++j) {

        auto const
        inode = inodes[j];

        for (auto d = 0; d < dim; ++d) {
          value(d) += weight * X_view[dim * inode + d];
        }
      }
      for (auto d = 0; d < dim; ++d) {
        Y_view[dim * node + d] = -value(d);
      }
    }
  }

#ifdef WRITE_TO_MATRIX_MARKET
  char name[100];
  sprintf(name, "X_%04d.mm", mm_counter);
  Tpetra_MatrixMarket_Writer::writeDenseFile(name, X);
#endif  // WRITE_TO_MATRIX_MARKET

#ifdef WRITE_TO_MATRIX_MARKET
  sprintf(name, "Y_%04d.mm", mm_counter);
  Tpetra_MatrixMarket_Writer::writeDenseFile(name, Y);
#endif  // WRITE_TO_MATRIX_MARKET

#ifdef WRITE_TO_MATRIX_MARKET
  sprintf(name, "Jac%04d_%04d.mm", this_app_index, mm_counter);
  Tpetra_MatrixMarket_Writer::writeSparseFile(name, jacs_[this_app_index]);
  mm_counter++;
#endif // WRITE_TO_MATRIX_MARKET
}
#endif // APPLY_H27_H8_EXPLICIT

#if defined(APPLY_GENERAL_IMPLICIT)
// Returns the result of a Tpetra_Operator applied to a
// Tpetra_MultiVector X in Y.
void
LCM::
Schwarz_BoundaryJacobian::
apply(
    Tpetra_MultiVector const & X,
    Tpetra_MultiVector & Y,
    Teuchos::ETransp mode,
    ST alpha,
    ST beta) const
{
#ifdef OUTPUT_TO_SCREEN
  std::cout << __PRETTY_FUNCTION__ << "\n";
#endif

  auto const
  this_app_index = getThisAppIndex();

  Albany::Application const &
  this_app = getApplication(this_app_index);

  auto const
  coupled_app_index = getCoupledAppIndex();

  // Initialize Y vector.
  auto const
  zero = Teuchos::ScalarTraits<ST>::zero();

  Y.putScalar(zero);

  // If they are not coupled get out.
  if (this_app.isCoupled(coupled_app_index) == false) return;

  Albany::Application const &
  coupled_app = getApplication(coupled_app_index);

  std::string const &
  coupled_block_name = this_app.getCoupledBlockName(coupled_app_index);

  std::string const &
  this_nodeset_name = this_app.getNodesetName(coupled_app_index);

  // Get DOFs associated with node set.
  Teuchos::RCP<Albany::AbstractDiscretization>
  this_disc = this_app.getDiscretization();

  auto *
  this_stk_disc = static_cast<Albany::STKDiscretization *>(this_disc.get());

  Teuchos::RCP<Albany::AbstractDiscretization>
  coupled_disc = coupled_app.getDiscretization();

  auto *
  coupled_stk_disc =
      static_cast<Albany::STKDiscretization *>(coupled_disc.get());

  Teuchos::ArrayRCP<double> const &
  coupled_coordinates = coupled_stk_disc->getCoordinates();

  auto const
  dimension = this_stk_disc->getNumDim();

#if defined(DEBUG_LCM_SCHWARZ)
  std::cout << "COORDINATES:\n";

  auto const
  length = X.getLocalLength();

  auto const
  num_nodes = length / dimension;

  for (auto n = 0; n < num_nodes; ++n) {
    auto const
    dof_x = dimension * n;

    auto const
    dof_y = dof_x + 1;

    auto const
    dof_z = dof_x + 2;

    Intrepid::Vector<double, 3>
    x(coupled_coordinates[dof_x],
      coupled_coordinates[dof_y],
      coupled_coordinates[dof_z]);

    std::cout << std::setw(4) << n << " " << x << '\n';
  }
#endif // DEBUG_LCM_SCHWARZ

  Albany::NodeSetList const &
  nodesets = this_stk_disc->getNodeSets();

  std::vector<std::vector<int>> const &
  ns_dof = nodesets.find(this_nodeset_name)->second;

  auto const
  ns_number_nodes = ns_dof.size();

  Teuchos::ArrayRCP<ST>
  Y_view = Y.get1dViewNonConst();

  for (auto ns_node = 0; ns_node < ns_number_nodes; ++ns_node) {

    Intrepid::Vector<ST> const
    bc_value = computeBC(this_app, coupled_app, X, dimension, ns_node);

#if defined(DEBUG_LCM_SCHWARZ)
    std::cout << "DIMENSION    DOF\n";
    std::cout << "--------------------------------------------------------\n";
#endif // DEBUG_LCM_SCHWARZ

  for (auto i = 0; i < dimension; ++i) {
      auto const
      dof = ns_dof[ns_node][i];

#if defined(DEBUG_LCM_SCHWARZ)
      std::cout << std::setw(8) << i;
      std::cout << std::setw(8) << dof;
      std::cout << '\n';
#endif // DEBUG_LCM_SCHWARZ
      auto const
      value = bc_value(i);

      Y_view[dof] = -value;
    }

#if defined(DEBUG_LCM_SCHWARZ)
    std::cout << "--------------------------------------------------------\n";
#endif // DEBUG_LCM_SCHWARZ

  } // node in node set loop

#ifdef WRITE_TO_MATRIX_MARKET
  char name[100];
  sprintf(name, "X_%04d.mm", mm_counter);
  Tpetra_MatrixMarket_Writer::writeDenseFile(name, X);
#endif  // WRITE_TO_MATRIX_MARKET

#ifdef WRITE_TO_MATRIX_MARKET
  sprintf(name, "Y_%04d.mm", mm_counter);
  Tpetra_MatrixMarket_Writer::writeDenseFile(name, Y);
#endif  // WRITE_TO_MATRIX_MARKET

#ifdef WRITE_TO_MATRIX_MARKET
  sprintf(name, "Jac%04d_%04d.mm", this_app_index, mm_counter);
  Tpetra_MatrixMarket_Writer::writeSparseFile(name, jacs_[this_app_index]);
  mm_counter++;
#endif // WRITE_TO_MATRIX_MARKET
}
#endif //APPLY_GENERAL_IMPLICIT
