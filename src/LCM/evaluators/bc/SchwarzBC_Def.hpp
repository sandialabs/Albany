//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "AAdapt_STKAdapt.hpp"
#include "Intrepid_MiniTensor.h"
#include "PerceptMesh.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "Teuchos_TestForException.hpp"

//
// Genereric Template Code for Constructor and PostRegistrationSetup
//

namespace LCM {

template <typename EvalT, typename Traits>
SchwarzBC_Base<EvalT, Traits>::
SchwarzBC_Base(Teuchos::ParameterList & p) :
  PHAL::DirichletBase<EvalT, Traits>(p),
  coupled_block_(p.get<std::string>("Coupled Block")),
  disc_(Teuchos::null)
{
}

//
//
//
template<typename EvalT, typename Traits>
void
SchwarzBC_Base<EvalT, Traits>::
computeBCs(double * coord, ScalarT & x_val, ScalarT & y_val, ScalarT & z_val)
{
  // Do the real work here.
  // Placeholder for now.
  x_val = 0;
  y_val = 0;
  z_val = 0;
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
  //
  // Fetch data structures and corresponding info that is needed
  //
  Teuchos::RCP<Epetra_Vector>
  f = dirichlet_workset.f;

  Teuchos::RCP<Epetra_Vector const>
  x = dirichlet_workset.x;

  Teuchos::RCP<Albany::AbstractDiscretization>
  disc = dirichlet_workset.disc;

  Albany::STKDiscretization *
  stk_discretization = static_cast<Albany::STKDiscretization *>(disc.get());

  this->setDiscretization(disc);

  Teuchos::RCP<Albany::GenericSTKMeshStruct>
  gms = Teuchos::rcp_dynamic_cast<Albany::GenericSTKMeshStruct>(
      stk_discretization->getSTKMeshStruct()
  );

  Teuchos::ArrayRCP<double> &
  coordinates = stk_discretization->getCoordinates();

  Albany::WorksetArray<std::string>::type const &
  ws_eb_names = disc->getWsEBNames();

  std::string const
  coupled_block = this->getCoupledBlock();

  Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >
  mesh_specs = gms->getMeshSpecs();

  CellTopologyData const
  cell_topology_data = mesh_specs[0]->ctd;

  size_t const
  dimension = cell_topology_data.dimension;

  size_t const
  vertex_count = cell_topology_data.vertex_count;

  Intrepid::ELEMENT::Type const
  element_type = Intrepid::find_type(dimension, vertex_count);

  //
  // Collect nodal coordinates of BC nodes
  //
  std::vector<std::vector<int> > const &
  ns_dof =  dirichlet_workset.nodeSets->find(this->nodeSetID)->second;

  size_t const
  ns_number_nodes = ns_dof.size();

  std::vector< Intrepid::Vector<double> >
  ns_points(ns_number_nodes);

  std::vector<double *> const &
  ns_coord = dirichlet_workset.nodeSetCoords->find(this->nodeSetID)->second;

  typedef
  Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > >::type
  WSELND;

  WSELND const &
  ws_el_2_nd = stk_discretization->getWsElNodeID();

  std::vector< Intrepid::Vector<double> >
  element_vertices(vertex_count);

  for (size_t i = 0; i < vertex_count; ++i) {
    element_vertices[i].set_dimension(dimension);
  }

  typedef std::pair<size_t, size_t> WorksetElement;

  std::vector<WorksetElement>
  WorksetElements(ns_number_nodes);

  double const
  tolerance = 1.0e-4;

  for (size_t ns_node = 0; ns_node < ns_number_nodes; ++ns_node) {
    double * const
    coord = ns_coord[ns_node];

    Intrepid::Vector<double> &
    point = ns_points[ns_node];

    point.set_dimension(dimension);

    point.fill(coord);

    // Determine the element that cointains this point.
    bool
    found = false;

    for (size_t workset = 0; workset < ws_el_2_nd.size(); ++workset) {

      std::string const &
      element_block = ws_eb_names[workset];

      if (element_block != coupled_block) continue;

      size_t const
      elements_per_workset = ws_el_2_nd[workset].size();

      for (size_t element = 0; element < elements_per_workset; ++element) {

        for (size_t node = 0; node < vertex_count; ++node) {

          size_t const
          node_id = ws_el_2_nd[workset][element][node];

          double * const
          pcoord = &(coordinates[dimension * node_id]);

          element_vertices[node].fill(pcoord);
        }

        bool
        in_element = false;

        switch (element_type) {

        default:
          std::cerr << "ERROR: " << __PRETTY_FUNCTION__ << '\n';
          std::cerr << "Unknown element type: " << element_type << '\n';
          exit(1);
          break;

        case Intrepid::ELEMENT::TETRAHEDRAL:
          in_element = Intrepid::in_tetrahedron(
              point,
              element_vertices[0],
              element_vertices[1],
              element_vertices[2],
              element_vertices[3],
              tolerance);
          break;

        case Intrepid::ELEMENT::HEXAHEDRAL:
          in_element = Intrepid::in_hexahedron(
              point,
              element_vertices[0],
              element_vertices[1],
              element_vertices[2],
              element_vertices[3],
              element_vertices[4],
              element_vertices[5],
              element_vertices[6],
              element_vertices[7],
              tolerance);
          break;

        }

        if (in_element == true) {
          found = true;
          WorksetElements.push_back(std::make_pair(workset, element));
          std::cout << "NS node: " << ns_node << ' ';
          std::cout << "Workset: " << workset << ' ';
          std::cout << "Element: " << element << '\n';
          std::cout << "Point: " << point << '\n';
          for (size_t n = 0; n < element_vertices.size(); ++n) {
            std::cout << "Vertex " << n << " :" << element_vertices[n] << '\n';
          }
          break;
        }

      } // element loop

      if (found == true) {
        break;
      }

    } // workset loop

    assert(found == true);

  } // node in node set loop

  for (size_t ns_node = 0; ns_node < ns_number_nodes; ++ns_node) {

    int
    x_dof = ns_dof[ns_node][0];

    int
    y_dof = ns_dof[ns_node][1];

    int
    z_dof = ns_dof[ns_node][2];

    double *
    coord = ns_coord[ns_node];

    double &
    x = coord[0];

    double &
    y = coord[1];

    double &
    z = coord[2];

    ScalarT
    x_val, y_val, z_val;

    this->computeBCs(coord, x_val, y_val, z_val);

    (*f)[x_dof] = x_val;
    (*f)[y_dof] = y_val;
    (*f)[z_dof] = z_val;
  }

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
  Teuchos::RCP<Epetra_Vector>
  f = dirichlet_workset.f;

  Teuchos::RCP<Epetra_CrsMatrix>
  jac = dirichlet_workset.Jac;

  Teuchos::RCP<Epetra_Vector const>
  x = dirichlet_workset.x;

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

    this->computeBCs(coord, x_val, y_val, z_val);

    // replace jac values for the X dof
    jac->ExtractMyRowView(x_dof, num_entries, matrix_entries, matrix_indices);
    for (int i = 0; i < num_entries; ++i) matrix_entries[i] = 0;
    jac->ReplaceMyValues(x_dof, 1, &diag, &x_dof);

    // replace jac values for the y dof
    jac->ExtractMyRowView(y_dof, num_entries, matrix_entries, matrix_indices);
    for (int i = 0; i < num_entries; ++i) matrix_entries[i] = 0;
    jac->ReplaceMyValues(y_dof, 1, &diag, &y_dof);

    // replace jac values for the z dof
    jac->ExtractMyRowView(z_dof, num_entries, matrix_entries, matrix_indices);
    for (int i = 0; i < num_entries; ++i) matrix_entries[i] = 0;
    jac->ReplaceMyValues(z_dof, 1, &diag, &z_dof);

    if (fill_residual == true) {
      (*f)[x_dof] = x_val.val();
      (*f)[y_dof] = y_val.val();
      (*f)[z_dof] = z_val.val();
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
  Teuchos::RCP<Epetra_Vector>
  f = dirichlet_workset.f;

  Teuchos::RCP<Epetra_MultiVector>
  fp = dirichlet_workset.fp;

  Teuchos::RCP<Epetra_MultiVector>
  JV = dirichlet_workset.JV;

  Teuchos::RCP<Epetra_Vector const>
  x = dirichlet_workset.x;

  Teuchos::RCP<Epetra_MultiVector const>
  Vx = dirichlet_workset.Vx;

  RealType const
  j_coeff = dirichlet_workset.j_coeff;

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

  for (size_t ns_node = 0; ns_node < ns_nodes.size(); ++ns_node) {
    x_dof = ns_nodes[ns_node][0];
    y_dof = ns_nodes[ns_node][1];
    z_dof = ns_nodes[ns_node][2];
    coord = ns_coord[ns_node];

    this->computeBCs(coord, x_val, y_val, z_val);

    if (f != Teuchos::null) {
      (*f)[x_dof] = x_val.val();
      (*f)[y_dof] = y_val.val();
      (*f)[z_dof] = z_val.val();
    }

    if (JV != Teuchos::null) {
      for (int i = 0; i < dirichlet_workset.num_cols_x; ++i) {
        (*JV)[i][x_dof] = j_coeff * (*Vx)[i][x_dof];
        (*JV)[i][y_dof] = j_coeff * (*Vx)[i][y_dof];
        (*JV)[i][z_dof] = j_coeff * (*Vx)[i][z_dof];
      }
    }

    if (fp != Teuchos::null) {
      for (int i = 0; i < dirichlet_workset.num_cols_p; ++i) {
        (*fp)[i][x_dof] = -x_val.dx(dirichlet_workset.param_offset + i);
        (*fp)[i][y_dof] = -y_val.dx(dirichlet_workset.param_offset + i);
        (*fp)[i][z_dof] = -z_val.dx(dirichlet_workset.param_offset + i);
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

    this->computeBCs(coord, x_val, y_val, z_val);

    for (int block = 0; block < nblock; ++block) {
      (*f)[block][x_dof] = x_val.coeff(block);
      (*f)[block][y_dof] = y_val.coeff(block);
      (*f)[block][z_dof] = z_val.coeff(block);
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

    this->computeBCs(coord, x_val, y_val, z_val);

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
        (*f)[block][x_dof] = x_val.val().coeff(block);
        (*f)[block][y_dof] = y_val.val().coeff(block);
        (*f)[block][z_dof] = z_val.val().coeff(block);
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

    this->computeBCs(coord, x_val, y_val, z_val);

    if (f != Teuchos::null) {
      for (int block = 0; block < nblock; ++block) {
        (*f)[block][x_dof] = x_val.val().coeff(block);
        (*f)[block][y_dof] = y_val.val().coeff(block);
        (*f)[block][z_dof] = z_val.val().coeff(block);
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
          x_val.dx(dirichlet_workset.param_offset+i).coeff(block);
          (*fp)[block][i][y_dof] =
          y_val.dx(dirichlet_workset.param_offset+i).coeff(block);
          (*fp)[block][i][z_dof] =
          z_val.dx(dirichlet_workset.param_offset+i).coeff(block);
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

    this->computeBCs(coord, x_val, y_val, z_val);

    for (int block = 0; block < nblock; ++block) {
      (*f)[block][x_dof] = x_val.coeff(block);
      (*f)[block][y_dof] = y_val.coeff(block);
      (*f)[block][z_dof] = z_val.coeff(block);
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

    this->computeBCs(coord, x_val, y_val, z_val);

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
        (*f)[block][x_dof] = x_val.val().coeff(block);
        (*f)[block][y_dof] = y_val.val().coeff(block);
        (*f)[block][z_dof] = z_val.val().coeff(block);
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

    this->computeBCs(coord, x_val, y_val, z_val);

    if (f != Teuchos::null) {
      for (int block = 0; block < nblock; ++block) {
        (*f)[block][x_dof] = x_val.val().coeff(block);
        (*f)[block][y_dof] = y_val.val().coeff(block);
        (*f)[block][z_dof] = z_val.val().coeff(block);
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
          x_val.dx(dirichlet_workset.param_offset+i).coeff(block);
          (*fp)[block][i][y_dof] =
          y_val.dx(dirichlet_workset.param_offset+i).coeff(block);
          (*fp)[block][i][z_dof] =
          z_val.dx(dirichlet_workset.param_offset+i).coeff(block);
        }
      }
    }

  }
}
#endif //ALBANY_SG_MP

}
 // namespace LCM

