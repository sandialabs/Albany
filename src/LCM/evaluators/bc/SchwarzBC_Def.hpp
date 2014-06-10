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
computeBCs(
    typename Traits::EvalData dirichlet_workset,
    size_t const ns_node,
    ScalarT & x_val,
    ScalarT & y_val,
    ScalarT & z_val)
{
  Teuchos::RCP<Albany::AbstractDiscretization>
  disc = dirichlet_workset.disc;

  assert(disc != Teuchos::null);

  Albany::STKDiscretization *
  stk_discretization = static_cast<Albany::STKDiscretization *>(disc.get());

  Teuchos::RCP<Epetra_Vector>
  solution = stk_discretization->getSolutionField();

  Albany::GenericSTKMeshStruct &
  gms = dynamic_cast<Albany::GenericSTKMeshStruct &>(
      *(stk_discretization->getSTKMeshStruct())
  );

  Teuchos::ArrayRCP<double> &
  coordinates = stk_discretization->getCoordinates();

  std::string const
  coupled_block = this->getCoupledBlock();

  Albany::WorksetArray<std::string>::type const &
  ws_eb_names = disc->getWsEBNames();

  Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >
  mesh_specs = gms.getMeshSpecs();

  CellTopologyData const
  cell_topology_data = mesh_specs[0]->ctd;

  shards::CellTopology
  cell_topology(&cell_topology_data);

  size_t const
  dimension = cell_topology_data.dimension;

  size_t const
  vertex_count = cell_topology_data.vertex_count;

  Intrepid::ELEMENT::Type const
  element_type = Intrepid::find_type(dimension, vertex_count);

  std::vector<double *> const &
  ns_coord = dirichlet_workset.nodeSetCoords->find(this->nodeSetID)->second;

  typedef
  Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > >::type
  WSELND;

  WSELND const &
  ws_el_2_nd = stk_discretization->getWsElNodeID();

  std::vector< Intrepid::Vector<double> >
  element_vertices(vertex_count);

  std::vector< Intrepid::Vector<double> >
  element_solution(vertex_count);

  for (size_t i = 0; i < vertex_count; ++i) {
    element_vertices[i].set_dimension(dimension);
    element_solution[i].set_dimension(dimension);
  }

  double const
  tolerance = 1.0e-4;

  double * const
  coord = ns_coord[ns_node];

  Intrepid::Vector<double>
  point;

  point.set_dimension(dimension);

  point.fill(coord);

  // Determine the element that cointains this point.
  bool
  found = false;

  size_t
  parametric_dimension = 0;

  Teuchos::RCP<Intrepid::Basis<double, Intrepid::FieldContainer<double> > >
  basis;

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

        for (size_t i = 0; i < dimension; ++i) {
          element_solution[node](i) = (*solution)[dimension * node_id + i];
        }
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
        parametric_dimension = 3;

        basis = Teuchos::rcp(new Intrepid::Basis_HGRAD_TET_C1_FEM<
            double, Intrepid::FieldContainer<double> >());

        in_element = Intrepid::in_tetrahedron(
            point,
            element_vertices[0],
            element_vertices[1],
            element_vertices[2],
            element_vertices[3],
            tolerance);
        break;

      case Intrepid::ELEMENT::HEXAHEDRAL:
        parametric_dimension = 3;

        basis = Teuchos::rcp(new Intrepid::Basis_HGRAD_HEX_C1_FEM<
            double, Intrepid::FieldContainer<double> >());

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
        break;
      }

    } // element loop

    if (found == true) {
      break;
    }

  } // workset loop

  assert(found == true);

  // We do this element by element
  size_t const
  number_cells = 1;

  // Container for the parametric coordinates
  Intrepid::FieldContainer<double>
  parametric_point(number_cells, parametric_dimension);

  for (size_t j = 0; j < parametric_dimension; ++j) {
    parametric_point(0, j) = 0.0;
  }

  // Container for the physical point
  Intrepid::FieldContainer<double>
  physical_coordinates(number_cells, dimension);

  for (size_t i = 0; i < dimension; ++i) {
    physical_coordinates(0, i) = point(i);
  }

  // Container for the physical nodal coordinates
  // TODO: matToReference more general, accepts more topologies.
  // Use it to find if point is contained in element as well.
  Intrepid::FieldContainer<double>
  nodal_coordinates(number_cells, vertex_count, dimension);

  for (size_t i = 0; i < vertex_count; ++i) {
    for (size_t j = 0; j < dimension; ++j) {
      nodal_coordinates(0,i,j) = element_vertices[i](j);
    }
  }

  // Get parametric coordinates
  Intrepid::CellTools<double>::mapToReferenceFrame(
      parametric_point,
      physical_coordinates,
      nodal_coordinates,
      cell_topology,
      0
  );

  // Evaluate shape functions at parametric point.
  size_t const number_points = 1;

  Intrepid::FieldContainer<double>
  basis_values(vertex_count, number_points);

  basis->getValues(basis_values, parametric_point, Intrepid::OPERATOR_VALUE);

  // Evaluate solution at parametric point using values of shape
  // functions just computed.
  Intrepid::Vector<double>
  value(dimension, Intrepid::ZEROS);

  std::cout << "Coupling to block: " << coupled_block << '\n';

  for (size_t i = 0; i < vertex_count; ++i) {
    value += basis_values(i, 0) * element_solution[i];
    std::cout << std::scientific << std::setprecision(16);
    std::cout << basis_values(i, 0) << "    " << element_solution[i] << '\n';
  }

  std::cout << " ==> " << value << '\n';

  x_val = value(0);
  y_val = value(1);
  z_val = value(2);
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

  // Coordinates
  Teuchos::RCP<Epetra_Vector const>
  x = dirichlet_workset.x;

  // Solution
  Teuchos::RCP<Epetra_Vector>
  f = dirichlet_workset.f;

  std::cout << "\n*** RESIDUAL ***\n";
  std::cout << "\n*** X BEFORE ***\n";
  x->Print(std::cout);
  std::cout << "\n*** F BEFORE ***\n";
  f->Print(std::cout);

  Teuchos::RCP<Albany::AbstractDiscretization>
  disc = dirichlet_workset.disc;

  Albany::STKDiscretization *
  stk_discretization = static_cast<Albany::STKDiscretization *>(disc.get());

  this->setDiscretization(disc);

  //
  // Collect nodal coordinates of nodeset (BC) nodes
  //
  std::vector<std::vector<int> > const &
  ns_dof =  dirichlet_workset.nodeSets->find(this->nodeSetID)->second;

  std::cout << '\n';
  for (size_t i = 0; i < ns_dof.size(); ++i) {
    for (size_t j = 0; j < ns_dof[i].size(); ++j) {
      std::cout << ' ' << ns_dof[i][j];
    }
    std::cout << '\n';
  }

  size_t const
  ns_number_nodes = ns_dof.size();

  ScalarT
  x_val, y_val, z_val;

  for (size_t ns_node = 0; ns_node < ns_number_nodes; ++ns_node) {

    this->computeBCs(dirichlet_workset, ns_node, x_val, y_val, z_val);

    size_t const
    dof_x = ns_dof[ns_node][0];

    size_t const
    dof_y = ns_dof[ns_node][1];

    size_t const
    dof_z = ns_dof[ns_node][2];

    (*f)[dof_x] = (*x)[dof_x] - x_val;
    (*f)[dof_y] = (*x)[dof_y] - y_val;
    (*f)[dof_z] = (*x)[dof_z] - z_val;

  } // node in node set loop

  std::cout << "\n*** X AFTER ***\n";
  x->Print(std::cout);
  std::cout << "\n*** F AFTER ***\n";
  f->Print(std::cout);

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
  std::cout << "\n*** JACOBIAN ***\n";

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

    this->computeBCs(dirichlet_workset, ns_node, x_val, y_val, z_val);

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
      (*f)[x_dof] = (*x)[x_dof] - x_val.val();
      (*f)[y_dof] = (*x)[y_dof] - y_val.val();
      (*f)[z_dof] = (*x)[z_dof] - z_val.val();
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
  std::cout << "\n*** TANGENT ***\n";

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

    this->computeBCs(dirichlet_workset, ns_node, x_val, y_val, z_val);

    if (f != Teuchos::null) {
      (*f)[x_dof] = (*x)[x_dof] - x_val.val();
      (*f)[y_dof] = (*x)[y_dof] - y_val.val();
      (*f)[z_dof] = (*x)[z_dof] - z_val.val();
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
  Teuchos::RCP<Epetra_MultiVector> fpV = dirichlet_workset.fpV;
  bool trans = dirichlet_workset.transpose_dist_param_deriv;
  int num_cols = fpV->NumVectors();

  //
  // We're currently assuming Dirichlet BC's can't be distributed parameters.
  // Thus we don't need to actually evaluate the BC's here.  The code to do
  // so is still here, just commented out for future reference.
  //

  std::vector<std::vector<int> > const &
  ns_nodes = dirichlet_workset.nodeSets->find(this->nodeSetID)->second;

  std::vector<double*> const &
  ns_coord = dirichlet_workset.nodeSetCoords->find(this->nodeSetID)->second;

  // global and local indices into unknown vector
  int
  xlunk, ylunk, zlunk;

  // double *
  // coord;

  // ScalarT
  // x_val, y_val, z_val;

  // For (df/dp)^T*V we zero out corresponding entries in V
  if (trans) {
    Teuchos::RCP<Epetra_MultiVector> Vp = dirichlet_workset.Vp_bc;
    for (size_t inode = 0; inode < ns_nodes.size(); ++inode) {
      xlunk = ns_nodes[inode][0];
      ylunk = ns_nodes[inode][1];
      zlunk = ns_nodes[inode][2];
      // coord = ns_coord[inode];

      // this->computeBCs(coord, x_val, y_val, z_val);

      for (int col=0; col<num_cols; ++col) {
        (*Vp)[col][xlunk] = 0.0;
        (*Vp)[col][ylunk] = 0.0;
        (*Vp)[col][zlunk] = 0.0;
      }
    }
  }

  // for (df/dp)*V we zero out corresponding entries in df/dp
  else {
    for (size_t inode = 0; inode < ns_nodes.size(); ++inode) {
      xlunk = ns_nodes[inode][0];
      ylunk = ns_nodes[inode][1];
      zlunk = ns_nodes[inode][2];
      // coord = ns_coord[inode];

      // this->computeBCs(coord, x_val, y_val, z_val);

      for (int col=0; col<num_cols; ++col) {
        (*fpV)[col][xlunk] = 0.0;
        (*fpV)[col][ylunk] = 0.0;
        (*fpV)[col][zlunk] = 0.0;
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

}
 // namespace LCM

