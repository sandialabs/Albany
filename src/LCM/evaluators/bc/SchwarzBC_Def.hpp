//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_Application.hpp"
#include "Albany_GenericSTKMeshStruct.hpp"
#include "Albany_STKDiscretization.hpp"
#include "Intrepid_MiniTensor.h"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "Teuchos_TestForException.hpp"

//define DEBUG_LCM_SCHWARZ

//
// Generic Template Code for Constructor and PostRegistrationSetup
//

namespace LCM {

template<typename EvalT, typename Traits>
SchwarzBC_Base<EvalT, Traits>::
SchwarzBC_Base(Teuchos::ParameterList & p) :
    PHAL::DirichletBase<EvalT, Traits>(p),
    app_(p.get<Teuchos::RCP<Albany::Application>>(
        "Application", Teuchos::null)),
    coupled_app_name_(p.get<std::string>("Coupled Application", "self")),
    coupled_block_name_(p.get<std::string>("Coupled Block"))
{
  std::string const &
  nodeset_name = this->nodeSetID;

  app_->setCoupledAppBlockNodeset(
      coupled_app_name_,
      coupled_block_name_,
      nodeset_name);
}

//
//
//
template<typename EvalT, typename Traits>
void
SchwarzBC_Base<EvalT, Traits>::
computeBCs(
    typename Traits::EvalData dirichlet_workset,
    size_t const ns_node,
    ScalarT & x_val,
    ScalarT & y_val,
    ScalarT & z_val)
{
  // Schwarz BC should be zero.
  // Smith, Bjorstad & Gropp, Domain Decomposition, 1994, page 7
  x_val = 0.0;
  y_val = 0.0;
  z_val = 0.0;
  return;
}

//
// Specialization: Residual
//
template<typename Traits>
SchwarzBC<PHAL::AlbanyTraits::Residual, Traits>::
SchwarzBC(Teuchos::ParameterList & p) :
    SchwarzBC_Base<PHAL::AlbanyTraits::Residual, Traits>(p)
{
}

//
//
//
template<typename Traits>
void
SchwarzBC<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData dirichlet_workset)
{
  // Solution
  Teuchos::RCP<const Tpetra_Vector>
  xT = dirichlet_workset.xT;

  Teuchos::ArrayRCP<const ST>
  xT_const_view = xT->get1dView();

  // Residual
  Teuchos::RCP<Tpetra_Vector>
  fT = dirichlet_workset.fT;

  Teuchos::ArrayRCP<ST>
  fT_view = fT->get1dViewNonConst();

  std::vector<std::vector<int> > const &
  ns_dof = dirichlet_workset.nodeSets->find(this->nodeSetID)->second;

  auto const
  ns_number_nodes = ns_dof.size();

  for (auto ns_node = 0; ns_node < ns_number_nodes; ++ns_node) {

    ScalarT
    x_val, y_val, z_val;

    this->computeBCs(dirichlet_workset, ns_node, x_val, y_val, z_val);

    auto const
    x_dof = ns_dof[ns_node][0];

    auto const
    y_dof = ns_dof[ns_node][1];

    auto const
    z_dof = ns_dof[ns_node][2];

    fT_view[x_dof] = xT_const_view[x_dof] - x_val;
    fT_view[y_dof] = xT_const_view[y_dof] - y_val;
    fT_view[z_dof] = xT_const_view[z_dof] - z_val;

  } // node in node set loop

  return;
}

//
// Specialization: Jacobian
//
template<typename Traits>
SchwarzBC<PHAL::AlbanyTraits::Jacobian, Traits>::
SchwarzBC(Teuchos::ParameterList & p) :
    SchwarzBC_Base<PHAL::AlbanyTraits::Jacobian, Traits>(p)
{
}

//
//
//
template<typename Traits>
void SchwarzBC<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData dirichlet_workset)
{
  Teuchos::RCP<Tpetra_Vector>
  fT = dirichlet_workset.fT;

  Teuchos::ArrayRCP<ST>
  fT_view;

  Teuchos::RCP<Tpetra_CrsMatrix>
  jacT = dirichlet_workset.JacT;

  Teuchos::RCP<const Tpetra_Vector>
  xT = dirichlet_workset.xT;

  Teuchos::ArrayRCP<const ST>
  xT_const_view = xT->get1dView();

  RealType const
  j_coeff = dirichlet_workset.j_coeff;

  std::vector<std::vector<int> > const &
  ns_nodes = dirichlet_workset.nodeSets->find(this->nodeSetID)->second;

  Teuchos::Array<LO>
  index(1);

  Teuchos::Array<ST>
  value(1);

  value[0] = j_coeff;

  Teuchos::Array<ST>
  matrix_entries;

  Teuchos::Array<LO>
  matrix_indices;

  bool const
  fill_residual = (fT != Teuchos::null);

  if (fill_residual == true) {
    fT_view = fT->get1dViewNonConst();
  }

  for (auto ns_node = 0; ns_node < ns_nodes.size(); ++ns_node) {

    auto const
    x_dof = ns_nodes[ns_node][0];

    auto const
    y_dof = ns_nodes[ns_node][1];

    auto const
    z_dof = ns_nodes[ns_node][2];

    ScalarT
    x_val, y_val, z_val;

    this->computeBCs(dirichlet_workset, ns_node, x_val, y_val, z_val);

    // replace jac values for the X dof
    auto
    num_entries = jacT->getNumEntriesInLocalRow(x_dof);

    matrix_entries.resize(num_entries);
    matrix_indices.resize(num_entries);

    jacT->getLocalRowCopy(
        x_dof,
        matrix_indices(),
        matrix_entries(),
        num_entries);

    for (auto i = 0; i < num_entries; ++i) {
      matrix_entries[i] = 0;
    }

    jacT->replaceLocalValues(x_dof, matrix_indices(), matrix_entries());
    index[0] = x_dof;
    jacT->replaceLocalValues(x_dof, index(), value());

    // replace jac values for the y dof
    num_entries = jacT->getNumEntriesInLocalRow(y_dof);

    matrix_entries.resize(num_entries);
    matrix_indices.resize(num_entries);

    jacT->getLocalRowCopy(
        y_dof,
        matrix_indices(),
        matrix_entries(),
        num_entries);

    for (auto i = 0; i < num_entries; ++i) {
      matrix_entries[i] = 0;
    }

    jacT->replaceLocalValues(y_dof, matrix_indices(), matrix_entries());
    index[0] = y_dof;
    jacT->replaceLocalValues(y_dof, index(), value());

    // replace jac values for the z dof
    num_entries = jacT->getNumEntriesInLocalRow(z_dof);

    matrix_entries.resize(num_entries);
    matrix_indices.resize(num_entries);

    jacT->getLocalRowCopy(
        z_dof,
        matrix_indices(),
        matrix_entries(),
        num_entries);

    for (auto i = 0; i < num_entries; ++i) {
      matrix_entries[i] = 0;
    }

    jacT->replaceLocalValues(z_dof, matrix_indices(), matrix_entries());
    index[0] = z_dof;
    jacT->replaceLocalValues(z_dof, index(), value());

    if (fill_residual == true) {
      fT_view[x_dof] = xT_const_view[x_dof] - x_val.val();
      fT_view[y_dof] = xT_const_view[y_dof] - y_val.val();
      fT_view[z_dof] = xT_const_view[z_dof] - z_val.val();
    }
  }
}

//
// Specialization: Tangent
//
template<typename Traits>
SchwarzBC<PHAL::AlbanyTraits::Tangent, Traits>::
SchwarzBC(Teuchos::ParameterList & p) :
    SchwarzBC_Base<PHAL::AlbanyTraits::Tangent, Traits>(p)
{
}

//
//
//
template<typename Traits>
void SchwarzBC<PHAL::AlbanyTraits::Tangent, Traits>::
evaluateFields(typename Traits::EvalData dirichlet_workset)
{
  Teuchos::RCP<Tpetra_Vector>
  fT = dirichlet_workset.fT;

  Teuchos::RCP<Tpetra_MultiVector>
  fpT = dirichlet_workset.fpT;

  Teuchos::RCP<Tpetra_MultiVector>
  JVT = dirichlet_workset.JVT;

  Teuchos::RCP<const Tpetra_Vector>
  xT = dirichlet_workset.xT;

  Teuchos::RCP<const Tpetra_MultiVector>
  VxT = dirichlet_workset.VxT;

  RealType const
  j_coeff = dirichlet_workset.j_coeff;

  std::vector<std::vector<int> > const &
  ns_nodes = dirichlet_workset.nodeSets->find(this->nodeSetID)->second;

  Teuchos::ArrayRCP<const ST>
  VxT_const_view;

  Teuchos::ArrayRCP<ST>
  fT_view;

  Teuchos::ArrayRCP<const ST>
  xT_const_view = xT->get1dView();

  if (fT != Teuchos::null) {
    fT_view = fT->get1dViewNonConst();
  }

  for (auto ns_node = 0; ns_node < ns_nodes.size(); ++ns_node) {
    auto const
    x_dof = ns_nodes[ns_node][0];

    auto const
    y_dof = ns_nodes[ns_node][1];

    auto const
    z_dof = ns_nodes[ns_node][2];

    ScalarT
    x_val, y_val, z_val;

    this->computeBCs(dirichlet_workset, ns_node, x_val, y_val, z_val);

    if (fT != Teuchos::null) {
      fT_view[x_dof] = xT_const_view[x_dof] - x_val.val();
      fT_view[y_dof] = xT_const_view[y_dof] - y_val.val();
      fT_view[z_dof] = xT_const_view[z_dof] - z_val.val();
    }

    if (JVT != Teuchos::null) {
      Teuchos::ArrayRCP<ST>
      JVT_view;

      for (auto i = 0; i < dirichlet_workset.num_cols_x; ++i) {
        JVT_view = JVT->getDataNonConst(i);
        VxT_const_view = VxT->getData(i);
        JVT_view[x_dof] = j_coeff * VxT_const_view[x_dof];
        JVT_view[y_dof] = j_coeff * VxT_const_view[y_dof];
        JVT_view[z_dof] = j_coeff * VxT_const_view[z_dof];
      }
    }

    if (fpT != Teuchos::null) {
      Teuchos::ArrayRCP<ST>
      fpT_view;

      for (auto i = 0; i < dirichlet_workset.num_cols_p; ++i) {
        fpT_view = fpT->getDataNonConst(i);
        fpT_view[x_dof] = -x_val.dx(dirichlet_workset.param_offset + i);
        fpT_view[y_dof] = -y_val.dx(dirichlet_workset.param_offset + i);
        fpT_view[z_dof] = -z_val.dx(dirichlet_workset.param_offset + i);
      }
    }
  }
}

//
// Specialization: DistParamDeriv
//
template<typename Traits>
SchwarzBC<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
SchwarzBC(Teuchos::ParameterList & p) :
    SchwarzBC_Base<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p)
{
}

//
//
//
template<typename Traits>
void SchwarzBC<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
evaluateFields(typename Traits::EvalData dirichlet_workset)
{
  Teuchos::RCP<Tpetra_MultiVector>
  fpVT = dirichlet_workset.fpVT;

  Teuchos::ArrayRCP<ST>
  fpVT_view;

  bool const
  trans = dirichlet_workset.transpose_dist_param_deriv;

  auto const
  num_cols = fpVT->getNumVectors();

  //
  // We're currently assuming Dirichlet BC's can't be distributed parameters.
  // Thus we don't need to actually evaluate the BC's here.  The code to do
  // so is still here, just commented out for future reference.
  //

  std::vector<std::vector<int> > const &
  ns_nodes = dirichlet_workset.nodeSets->find(this->nodeSetID)->second;

  std::vector<double *> const &
  ns_coord = dirichlet_workset.nodeSetCoords->find(this->nodeSetID)->second;

  // global and local indices into unknown vector
  // double *
  // coord;

  // ScalarT
  // x_val, y_val, z_val;

  // For (df/dp)^T*V we zero out corresponding entries in V
  if (trans == true) {
    Teuchos::RCP<Tpetra_MultiVector>
    VpT = dirichlet_workset.Vp_bcT;

    Teuchos::ArrayRCP<ST>
    VpT_view;

    for (auto inode = 0; inode < ns_nodes.size(); ++inode) {

      auto const
      x_dof = ns_nodes[inode][0];

      auto const
      y_dof = ns_nodes[inode][1];

      auto const
      z_dof = ns_nodes[inode][2];
      // coord = ns_coord[inode];

      // this->computeBCs(coord, x_val, y_val, z_val);

      for (auto col = 0; col < num_cols; ++col) {
        //(*Vp)[col][x_dof] = 0.0;
        //(*Vp)[col][y_dof] = 0.0;
        //(*Vp)[col][z_dof] = 0.0;
        VpT_view = VpT->getDataNonConst(col);
        VpT_view[x_dof] = 0.0;
        VpT_view[y_dof] = 0.0;
        VpT_view[z_dof] = 0.0;
      }
    }
  }

  // for (df/dp)*V we zero out corresponding entries in df/dp
  else {
    for (auto inode = 0; inode < ns_nodes.size(); ++inode) {

      auto const
      x_dof = ns_nodes[inode][0];

      auto const
      y_dof = ns_nodes[inode][1];

      auto const
      z_dof = ns_nodes[inode][2];
      // coord = ns_coord[inode];

      // this->computeBCs(coord, x_val, y_val, z_val);

      for (auto col = 0; col < num_cols; ++col) {
        //(*fpV)[col][x_dof] = 0.0;
        //(*fpV)[col][y_dof] = 0.0;
        //(*fpV)[col][z_dof] = 0.0;
        fpVT_view = fpVT->getDataNonConst(col);
        fpVT_view[x_dof] = 0.0;
        fpVT_view[y_dof] = 0.0;
        fpVT_view[z_dof] = 0.0;
      }
    }
  }
}

//
// Specialization: Stochastic Galerkin Residual
//
#ifdef ALBANY_SG_MP
template<typename Traits>
SchwarzBC<PHAL::AlbanyTraits::SGResidual, Traits>::
SchwarzBC(Teuchos::ParameterList & p) :
SchwarzBC_Base<PHAL::AlbanyTraits::SGResidual, Traits>(p)
{
}

//
//
//
template<typename Traits>
void SchwarzBC<PHAL::AlbanyTraits::SGResidual, Traits>::
evaluateFields(typename Traits::EvalData dirichlet_workset)
{
  Teuchos::RCP<Stokhos::EpetraVectorOrthogPoly>
  f = dirichlet_workset.sg_f;

  Teuchos::RCP<const Stokhos::EpetraVectorOrthogPoly>
  x = dirichlet_workset.sg_x;

  std::vector<std::vector<int> > const &
  ns_nodes = dirichlet_workset.nodeSets->find(this->nodeSetID)->second;

  std::vector<double*> const &
  ns_coord = dirichlet_workset.nodeSetCoords->find(this->nodeSetID)->second;

  // global and local indices into unknown vector
  int
  x_dof, y_dof, z_dof;

  double *
  coord;

  ScalarT
  x_val, y_val, z_val;

  int const
  nblock = x->size();

  for (size_t ns_node = 0; ns_node < ns_nodes.size(); ++ns_node) {
    x_dof = ns_nodes[ns_node][0];
    y_dof = ns_nodes[ns_node][1];
    z_dof = ns_nodes[ns_node][2];
    coord = ns_coord[ns_node];

    this->computeBCs(dirichlet_workset, ns_node, x_val, y_val, z_val);

    for (int block = 0; block < nblock; ++block) {
      (*f)[block][x_dof] = (*x)[block][x_dof] - x_val.coeff(block);
      (*f)[block][y_dof] = (*x)[block][y_dof] - y_val.coeff(block);
      (*f)[block][z_dof] = (*x)[block][z_dof] - z_val.coeff(block);
    }
  }
}

//
// Specialization: Stochastic Galerkin Jacobian
//
template<typename Traits>
SchwarzBC<PHAL::AlbanyTraits::SGJacobian, Traits>::
SchwarzBC(Teuchos::ParameterList & p) :
SchwarzBC_Base<PHAL::AlbanyTraits::SGJacobian, Traits>(p)
{
}

//
//
//
template<typename Traits>
void SchwarzBC<PHAL::AlbanyTraits::SGJacobian, Traits>::
evaluateFields(typename Traits::EvalData dirichlet_workset)
{
  Teuchos::RCP< Stokhos::EpetraVectorOrthogPoly>
  f = dirichlet_workset.sg_f;

  Teuchos::RCP< Stokhos::VectorOrthogPoly<Epetra_CrsMatrix> >
  jac = dirichlet_workset.sg_Jac;

  Teuchos::RCP<const Stokhos::EpetraVectorOrthogPoly>
  x = dirichlet_workset.sg_x;

  RealType const
  j_coeff = dirichlet_workset.j_coeff;

  std::vector<std::vector<int> > const &
  ns_nodes = dirichlet_workset.nodeSets->find(this->nodeSetID)->second;

  std::vector<double*> const &
  ns_coord = dirichlet_workset.nodeSetCoords->find(this->nodeSetID)->second;

  RealType *
  matrix_entries;

  int *
  matrix_indices;

  int
  num_entries;

  RealType
  diag = j_coeff;

  bool
  fill_residual = (f != Teuchos::null);

  int
  nblock = 0;

  if (f != Teuchos::null) {
    nblock = f->size();
  }

  int const
  nblock_jac = jac->size();

  // local indices into unknown vector
  int
  x_dof, y_dof, z_dof;

  double *
  coord;

  ScalarT
  x_val, y_val, z_val;

  for (size_t ns_node = 0; ns_node < ns_nodes.size(); ++ns_node) {
    x_dof = ns_nodes[ns_node][0];
    y_dof = ns_nodes[ns_node][1];
    z_dof = ns_nodes[ns_node][2];
    coord = ns_coord[ns_node];

    this->computeBCs(dirichlet_workset, ns_node, x_val, y_val, z_val);

    // replace jac values for the X dof
    for (int block = 0; block < nblock_jac; ++block) {
      (*jac)[block].ExtractMyRowView(x_dof, num_entries, matrix_entries,
          matrix_indices);
      for (int i = 0; i < num_entries; ++i) matrix_entries[i] = 0;

      // replace jac values for the y dof
      (*jac)[block].ExtractMyRowView(y_dof, num_entries, matrix_entries,
          matrix_indices);
      for (int i = 0; i < num_entries; ++i) matrix_entries[i] = 0;

      // replace jac values for the z dof
      (*jac)[block].ExtractMyRowView(z_dof, num_entries, matrix_entries,
          matrix_indices);
      for (int i = 0; i < num_entries; ++i) matrix_entries[i] = 0;
    }

    (*jac)[0].ReplaceMyValues(x_dof, 1, &diag, &x_dof);
    (*jac)[0].ReplaceMyValues(y_dof, 1, &diag, &y_dof);
    (*jac)[0].ReplaceMyValues(z_dof, 1, &diag, &z_dof);

    if (fill_residual == true) {

      for (int block = 0; block < nblock; ++block) {
        (*f)[block][x_dof] = (*x)[block][x_dof] - x_val.val().coeff(block);
        (*f)[block][y_dof] = (*x)[block][y_dof] - y_val.val().coeff(block);
        (*f)[block][z_dof] = (*x)[block][z_dof] - z_val.val().coeff(block);
      }
    }
  }
}

//
// Specialization: Stochastic Galerkin Tangent
//
template<typename Traits>
SchwarzBC<PHAL::AlbanyTraits::SGTangent, Traits>::
SchwarzBC(Teuchos::ParameterList & p) :
SchwarzBC_Base<PHAL::AlbanyTraits::SGTangent, Traits>(p)
{
}

//
//
//
template<typename Traits>
void SchwarzBC<PHAL::AlbanyTraits::SGTangent, Traits>::
evaluateFields(typename Traits::EvalData dirichlet_workset)
{
  Teuchos::RCP<Stokhos::EpetraVectorOrthogPoly>
  f = dirichlet_workset.sg_f;

  Teuchos::RCP<Stokhos::EpetraMultiVectorOrthogPoly>
  fp = dirichlet_workset.sg_fp;

  Teuchos::RCP<Stokhos::EpetraMultiVectorOrthogPoly>
  JV = dirichlet_workset.sg_JV;

  Teuchos::RCP<const Stokhos::EpetraVectorOrthogPoly>
  x = dirichlet_workset.sg_x;

  Teuchos::RCP<Epetra_MultiVector const>
  Vx = dirichlet_workset.Vx;

  RealType const
  j_coeff = dirichlet_workset.j_coeff;

  std::vector<std::vector<int> > const &
  ns_nodes = dirichlet_workset.nodeSets->find(this->nodeSetID)->second;

  std::vector<double*> const &
  ns_coord = dirichlet_workset.nodeSetCoords->find(this->nodeSetID)->second;

  int const
  nblock = x->size();

  // global and local indices into unknown vector
  int
  x_dof, y_dof, z_dof;

  double *
  coord;

  ScalarT
  x_val, y_val, z_val;

  for (size_t ns_node = 0; ns_node < ns_nodes.size(); ++ns_node) {
    x_dof = ns_nodes[ns_node][0];
    y_dof = ns_nodes[ns_node][1];
    z_dof = ns_nodes[ns_node][2];
    coord = ns_coord[ns_node];

    this->computeBCs(dirichlet_workset, ns_node, x_val, y_val, z_val);

    if (f != Teuchos::null) {

      for (int block = 0; block < nblock; ++block) {
        (*f)[block][x_dof] = (*x)[block][x_dof] - x_val.val().coeff(block);
        (*f)[block][y_dof] = (*x)[block][y_dof] - y_val.val().coeff(block);
        (*f)[block][z_dof] = (*x)[block][z_dof] - z_val.val().coeff(block);
      }
    }

    if (JV != Teuchos::null) {
      for (int i = 0; i < dirichlet_workset.num_cols_x; ++i) {
        (*JV)[0][i][x_dof] = j_coeff*(*Vx)[i][x_dof];
        (*JV)[0][i][y_dof] = j_coeff*(*Vx)[i][y_dof];
        (*JV)[0][i][z_dof] = j_coeff*(*Vx)[i][z_dof];
      }
    }

    if (fp != Teuchos::null) {
      for (int i=0; i<dirichlet_workset.num_cols_p; ++i) {
        for (int block = 0; block < nblock; ++block) {
          (*fp)[block][i][x_dof] =
          -x_val.dx(dirichlet_workset.param_offset+i).coeff(block);
          (*fp)[block][i][y_dof] =
          -y_val.dx(dirichlet_workset.param_offset+i).coeff(block);
          (*fp)[block][i][z_dof] =
          -z_val.dx(dirichlet_workset.param_offset+i).coeff(block);
        }
      }
    }

  }
}

//
// Specialization: Multi-point Residual
//
template<typename Traits>
SchwarzBC<PHAL::AlbanyTraits::MPResidual, Traits>::
SchwarzBC(Teuchos::ParameterList & p) :
SchwarzBC_Base<PHAL::AlbanyTraits::MPResidual, Traits>(p)
{
}

//
//
//
template<typename Traits>
void SchwarzBC<PHAL::AlbanyTraits::MPResidual, Traits>::
evaluateFields(typename Traits::EvalData dirichlet_workset)
{
  Teuchos::RCP<Stokhos::ProductEpetraVector>
  f = dirichlet_workset.mp_f;

  Teuchos::RCP<Stokhos::ProductEpetraVector const>
  x = dirichlet_workset.mp_x;

  std::vector<std::vector<int> > const &
  ns_nodes = dirichlet_workset.nodeSets->find(this->nodeSetID)->second;

  std::vector<double*> const &
  ns_coord = dirichlet_workset.nodeSetCoords->find(this->nodeSetID)->second;

  // global and local indices into unknown vector
  int
  x_dof, y_dof, z_dof;

  double *
  coord;

  ScalarT
  x_val, y_val, z_val;

  int const
  nblock = x->size();

  for (size_t ns_node = 0; ns_node < ns_nodes.size(); ++ns_node) {
    x_dof = ns_nodes[ns_node][0];
    y_dof = ns_nodes[ns_node][1];
    z_dof = ns_nodes[ns_node][2];
    coord = ns_coord[ns_node];

    this->computeBCs(dirichlet_workset, ns_node, x_val, y_val, z_val);

    for (int block = 0; block < nblock; ++block) {
      (*f)[block][x_dof] = (*x)[block][x_dof] - x_val.coeff(block);
      (*f)[block][y_dof] = (*x)[block][y_dof] - y_val.coeff(block);
      (*f)[block][z_dof] = (*x)[block][z_dof] - z_val.coeff(block);
    }
  }
}

//
// Specialization: Multi-point Jacobian
//
template<typename Traits>
SchwarzBC<PHAL::AlbanyTraits::MPJacobian, Traits>::
SchwarzBC(Teuchos::ParameterList & p) :
SchwarzBC_Base<PHAL::AlbanyTraits::MPJacobian, Traits>(p)
{
}

//
//
//
template<typename Traits>
void SchwarzBC<PHAL::AlbanyTraits::MPJacobian, Traits>::
evaluateFields(typename Traits::EvalData dirichlet_workset)
{
  Teuchos::RCP<Stokhos::ProductEpetraVector>
  f = dirichlet_workset.mp_f;

  Teuchos::RCP< Stokhos::ProductContainer<Epetra_CrsMatrix> >
  jac = dirichlet_workset.mp_Jac;

  Teuchos::RCP<Stokhos::ProductEpetraVector const>
  x = dirichlet_workset.mp_x;

  RealType const
  j_coeff = dirichlet_workset.j_coeff;

  std::vector<std::vector<int> > const &
  ns_nodes = dirichlet_workset.nodeSets->find(this->nodeSetID)->second;

  std::vector<double*> const &
  ns_coord = dirichlet_workset.nodeSetCoords->find(this->nodeSetID)->second;

  RealType *
  matrix_entries;

  int *
  matrix_indices;

  int
  num_entries;

  RealType
  diag = j_coeff;

  bool
  fill_residual = (f != Teuchos::null);

  int
  nblock = 0;

  if (f != Teuchos::null) {
    nblock = f->size();
  }

  int const
  nblock_jac = jac->size();

  // local indices into unknown vector
  int
  x_dof, y_dof, z_dof;

  double *
  coord;

  ScalarT
  x_val, y_val, z_val;

  for (size_t ns_node = 0; ns_node < ns_nodes.size(); ++ns_node) {
    x_dof = ns_nodes[ns_node][0];
    y_dof = ns_nodes[ns_node][1];
    z_dof = ns_nodes[ns_node][2];
    coord = ns_coord[ns_node];

    this->computeBCs(dirichlet_workset, ns_node, x_val, y_val, z_val);

    // replace jac values for the X dof
    for (int block=0; block<nblock_jac; ++block) {
      (*jac)[block].ExtractMyRowView(x_dof, num_entries, matrix_entries,
          matrix_indices);
      for (int i = 0; i < num_entries; ++i) matrix_entries[i] = 0;
      (*jac)[block].ReplaceMyValues(x_dof, 1, &diag, &x_dof);

      // replace jac values for the y dof
      (*jac)[block].ExtractMyRowView(y_dof, num_entries, matrix_entries,
          matrix_indices);
      for (int i = 0; i < num_entries; ++i) matrix_entries[i] = 0;
      (*jac)[block].ReplaceMyValues(y_dof, 1, &diag, &y_dof);

      // replace jac values for the z dof
      (*jac)[block].ExtractMyRowView(z_dof, num_entries, matrix_entries,
          matrix_indices);
      for (int i = 0; i < num_entries; ++i) matrix_entries[i] = 0;
      (*jac)[block].ReplaceMyValues(z_dof, 1, &diag, &z_dof);
    }

    if (fill_residual == true) {

      for (int block = 0; block < nblock; ++block) {
        (*f)[block][x_dof] = (*x)[block][x_dof] - x_val.val().coeff(block);
        (*f)[block][y_dof] = (*x)[block][y_dof] - y_val.val().coeff(block);
        (*f)[block][z_dof] = (*x)[block][z_dof] - z_val.val().coeff(block);
      }
    }
  }
}

//
// Specialization: Multi-point Tangent
//
template<typename Traits>
SchwarzBC<PHAL::AlbanyTraits::MPTangent, Traits>::
SchwarzBC(Teuchos::ParameterList & p) :
SchwarzBC_Base<PHAL::AlbanyTraits::MPTangent, Traits>(p)
{
}

//
//
//
template<typename Traits>
void SchwarzBC<PHAL::AlbanyTraits::MPTangent, Traits>::
evaluateFields(typename Traits::EvalData dirichlet_workset)
{
  Teuchos::RCP<Stokhos::ProductEpetraVector>
  f = dirichlet_workset.mp_f;

  Teuchos::RCP<Stokhos::ProductEpetraMultiVector>
  fp = dirichlet_workset.mp_fp;

  Teuchos::RCP<Stokhos::ProductEpetraMultiVector>
  JV = dirichlet_workset.mp_JV;

  Teuchos::RCP<Stokhos::ProductEpetraVector const>
  x = dirichlet_workset.mp_x;

  Teuchos::RCP<Epetra_MultiVector const>
  Vx = dirichlet_workset.Vx;

  RealType const
  j_coeff = dirichlet_workset.j_coeff;

  std::vector<std::vector<int> > const &
  ns_nodes = dirichlet_workset.nodeSets->find(this->nodeSetID)->second;

  std::vector<double*> const &
  ns_coord = dirichlet_workset.nodeSetCoords->find(this->nodeSetID)->second;

  int const
  nblock = x->size();

  // global and local indices into unknown vector
  int
  x_dof, y_dof, z_dof;

  double *
  coord;

  ScalarT
  x_val, y_val, z_val;

  for (size_t ns_node = 0; ns_node < ns_nodes.size(); ++ns_node) {
    x_dof = ns_nodes[ns_node][0];
    y_dof = ns_nodes[ns_node][1];
    z_dof = ns_nodes[ns_node][2];
    coord = ns_coord[ns_node];

    this->computeBCs(dirichlet_workset, ns_node, x_val, y_val, z_val);

    if (f != Teuchos::null) {

      for (int block = 0; block < nblock; ++block) {
        (*f)[block][x_dof] = (*x)[block][x_dof] - x_val.val().coeff(block);
        (*f)[block][y_dof] = (*x)[block][y_dof] - y_val.val().coeff(block);
        (*f)[block][z_dof] = (*x)[block][z_dof] - z_val.val().coeff(block);
      }
    }

    if (JV != Teuchos::null) {
      for (int i = 0; i<dirichlet_workset.num_cols_x; ++i) {
        for (int block = 0; block < nblock; ++block) {
          (*JV)[block][i][x_dof] = j_coeff*(*Vx)[i][x_dof];
          (*JV)[block][i][y_dof] = j_coeff*(*Vx)[i][y_dof];
          (*JV)[block][i][z_dof] = j_coeff*(*Vx)[i][z_dof];
        }
      }
    }

    if (fp != Teuchos::null) {

      for (int i = 0; i < dirichlet_workset.num_cols_p; ++i) {
        for (int block = 0; block < nblock; ++block) {
          (*fp)[block][i][x_dof] =
          -x_val.dx(dirichlet_workset.param_offset+i).coeff(block);
          (*fp)[block][i][y_dof] =
          -y_val.dx(dirichlet_workset.param_offset+i).coeff(block);
          (*fp)[block][i][z_dof] =
          -z_val.dx(dirichlet_workset.param_offset+i).coeff(block);
        }
      }
    }

  }
}
#endif //ALBANY_SG_MP

} // namespace LCM
