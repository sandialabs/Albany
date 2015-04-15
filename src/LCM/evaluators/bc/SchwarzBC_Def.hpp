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

template <typename EvalT, typename Traits>
SchwarzBC_Base<EvalT, Traits>::
SchwarzBC_Base(Teuchos::ParameterList & p) :
  PHAL::DirichletBase<EvalT, Traits>(p),
  app_(p.get<Teuchos::RCP<Albany::Application>>("Application", Teuchos::null)),
  coupled_app_name_(p.get<std::string>("Coupled Application", "self")),
  coupled_block_name_(p.get<std::string>("Coupled Block")),
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
  Teuchos::ArrayRCP<Teuchos::RCP<Albany::Application> >
  coupled_apps = dirichlet_workset.apps_;

  Teuchos::RCP<Albany::AbstractDiscretization>
  coupled_disc = Teuchos::null;

  Teuchos::RCP<Albany::AbstractDiscretization>
  disc = dirichlet_workset.disc;

  Albany::STKDiscretization *
  stk_disc = static_cast<Albany::STKDiscretization *>(disc.get());

  Teuchos::RCP<const Tpetra_Vector>
  solution = stk_disc->getSolutionFieldT();

  Teuchos::ArrayRCP<const ST>
  solution_view = solution->get1dView();

  const Teuchos::ArrayRCP<double> &
  coordinates = stk_disc->getCoordinates();

  auto const
  num_coupled_apps = coupled_apps.size();

  bool const
  self_coupled = num_coupled_apps == 0;

  if (self_coupled == true) {

    // Mainly just for testing the Schwarz BC
    coupled_disc = disc;

  } else {

    std::string const
    coupled_app_name = this->getCoupledAppName();

    int const
    coupled_app_index = appIndexFromName(coupled_app_name);

    if (coupled_app_index >= coupled_apps.size()) {
      std::cerr << "\nERROR: " << __PRETTY_FUNCTION__ << '\n';
      std::cerr << "Application index out of range: " << coupled_app_index;
      std::cerr << '\n';
      std::cerr << "Number of coupled applications: " << coupled_apps.size();
      std::cerr << '\n';
      exit(1);
    }

    Teuchos::RCP<Albany::Application>
    coupled_app = coupled_apps[coupled_app_index];

    coupled_disc = coupled_app->getDiscretization();

  }

  Albany::STKDiscretization *
  coupled_stk_disc =
      static_cast<Albany::STKDiscretization *>(coupled_disc.get());

  Albany::GenericSTKMeshStruct &
  coupled_gms = dynamic_cast<Albany::GenericSTKMeshStruct &>
    (*(coupled_stk_disc->getSTKMeshStruct()));

  Albany::WorksetArray<std::string>::type const &
  coupled_ws_eb_names = coupled_disc->getWsEBNames();

  Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >
  coupled_mesh_specs = coupled_gms.getMeshSpecs();

  // Get cell topology of the application and block to which this node set
  // is coupled.
  std::map<std::string, int> const &
  coupled_block_name_2_index = coupled_gms.ebNameToIndex;

  std::string const
  coupled_block_name = this->getCoupledBlockName();

  std::map<std::string, int>::const_iterator
  it = coupled_block_name_2_index.find(coupled_block_name);

  if (it == coupled_block_name_2_index.end()) {
    std::cerr << "\nERROR: " << __PRETTY_FUNCTION__ << '\n';
    std::cerr << "Unknown coupled block: " << coupled_block_name << '\n';
    exit(1);
  }

  int const
  coupled_block_index = it->second;

  CellTopologyData const
  coupled_cell_topology_data = coupled_mesh_specs[coupled_block_index]->ctd;

  shards::CellTopology
  coupled_cell_topology(&coupled_cell_topology_data);

  size_t const
  coupled_dimension = coupled_cell_topology_data.dimension;

  size_t const
  coupled_vertex_count = coupled_cell_topology_data.vertex_count;

  Intrepid::ELEMENT::Type const
  coupled_element_type =
      Intrepid::find_type(coupled_dimension, coupled_vertex_count);

  std::vector<double *> const &
  ns_coord = dirichlet_workset.nodeSetCoords->find(this->nodeSetID)->second;

  typedef
  Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> > >::type
  WSELND;

  WSELND const &
  ws_el_2_nd = stk_disc->getWsElNodeID();

  std::vector< Intrepid::Vector<double> >
  coupled_element_vertices(coupled_vertex_count);

  std::vector< Intrepid::Vector<double> >
  coupled_element_solution(coupled_vertex_count);

  for (size_t i = 0; i < coupled_vertex_count; ++i) {
    coupled_element_vertices[i].set_dimension(coupled_dimension);
    coupled_element_solution[i].set_dimension(coupled_dimension);
  }

  double const
  tolerance = 5.0e-2;

  double * const
  coord = ns_coord[ns_node];

  Intrepid::Vector<double>
  point;

  point.set_dimension(coupled_dimension);

  point.fill(coord);

  // Determine the element that contains this point.
  bool
  found = false;

  size_t
  parametric_dimension = 0;

  Teuchos::RCP<Intrepid::Basis<double, Intrepid::FieldContainer<double> > >
  basis;

  for (size_t workset = 0; workset < ws_el_2_nd.size(); ++workset) {

    std::string const &
    coupled_element_block = coupled_ws_eb_names[workset];

    if (coupled_element_block != coupled_block_name) continue;

    size_t const
    elements_per_workset = ws_el_2_nd[workset].size();

    for (size_t element = 0; element < elements_per_workset; ++element) {

      for (size_t node = 0; node < coupled_vertex_count; ++node) {

        size_t const
        node_id = ws_el_2_nd[workset][element][node];

        double * const
        pcoord = &(coordinates[coupled_dimension * node_id]);

        coupled_element_vertices[node].fill(pcoord);

        for (size_t i = 0; i < coupled_dimension; ++i) {
          coupled_element_solution[node](i) =
              solution_view[coupled_dimension * node_id + i];
        }
      }

      bool
      in_element = false;

      switch (coupled_element_type) {

      default:
        std::cerr << "\nERROR: " << __PRETTY_FUNCTION__ << '\n';
        std::cerr << "Unknown element type: " << coupled_element_type << '\n';
        exit(1);
        break;

      case Intrepid::ELEMENT::TETRAHEDRAL:
        parametric_dimension = 3;

        basis = Teuchos::rcp(new Intrepid::Basis_HGRAD_TET_C1_FEM<
            double, Intrepid::FieldContainer<double> >());

        in_element = Intrepid::in_tetrahedron(
            point,
            coupled_element_vertices[0],
            coupled_element_vertices[1],
            coupled_element_vertices[2],
            coupled_element_vertices[3],
            tolerance);
        break;

      case Intrepid::ELEMENT::HEXAHEDRAL:
        parametric_dimension = 3;

        basis = Teuchos::rcp(new Intrepid::Basis_HGRAD_HEX_C1_FEM<
            double, Intrepid::FieldContainer<double> >());

        in_element = Intrepid::in_hexahedron(
            point,
            coupled_element_vertices[0],
            coupled_element_vertices[1],
            coupled_element_vertices[2],
            coupled_element_vertices[3],
            coupled_element_vertices[4],
            coupled_element_vertices[5],
            coupled_element_vertices[6],
            coupled_element_vertices[7],
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
  physical_coordinates(number_cells, coupled_dimension);

  for (size_t i = 0; i < coupled_dimension; ++i) {
    physical_coordinates(0, i) = point(i);
  }

  // Container for the physical nodal coordinates
  // TODO: matToReference more general, accepts more topologies.
  // Use it to find if point is contained in element as well.
  Intrepid::FieldContainer<double>
  nodal_coordinates(number_cells, coupled_vertex_count, coupled_dimension);

  for (size_t i = 0; i < coupled_vertex_count; ++i) {
    for (size_t j = 0; j < coupled_dimension; ++j) {
      nodal_coordinates(0,i,j) = coupled_element_vertices[i](j);
    }
  }

  // Get parametric coordinates
  Intrepid::CellTools<double>::mapToReferenceFrame(
      parametric_point,
      physical_coordinates,
      nodal_coordinates,
      coupled_cell_topology,
      0
  );

  // Evaluate shape functions at parametric point.
  size_t const number_points = 1;

  Intrepid::FieldContainer<double>
  basis_values(coupled_vertex_count, number_points);

  basis->getValues(basis_values, parametric_point, Intrepid::OPERATOR_VALUE);

  // Evaluate solution at parametric point using values of shape
  // functions just computed.
  Intrepid::Vector<double>
  value(coupled_dimension, Intrepid::ZEROS);

#if defined(DEBUG_LCM_SCHWARZ)
  std::cout << "Coupling to block: " << coupled_app_name << '\n';
#endif // DEBUG_LCM_SCHWARZ

  for (size_t i = 0; i < coupled_vertex_count; ++i) {
    value += basis_values(i, 0) * coupled_element_solution[i];

#if defined(DEBUG_LCM_SCHWARZ)
    std::cout << std::scientific << std::setprecision(16);
    std::cout << basis_values(i, 0) << "    " << coupled_element_solution[i] << '\n';
#endif // DEBUG_LCM_SCHWARZ

  }

#if defined(DEBUG_LCM_SCHWARZ)
  std::cout << " ==> " << value << '\n';
#endif // DEBUG_LCM_SCHWARZ

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
  Teuchos::RCP<const Tpetra_Vector> xT = dirichlet_workset.xT;
  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();

  // Solution
  Teuchos::RCP<Tpetra_Vector> fT = dirichlet_workset.fT;
  Teuchos::ArrayRCP<ST> fT_nonconstView = fT->get1dViewNonConst();


#if defined(DEBUG_LCM_SCHWARZ)
  std::cout << "\n*** RESIDUAL ***\n";
  std::cout << "\n*** X BEFORE COMPUTE BC ***\n";
  xT->print(std::cout);
  std::cout << "\n*** F BEFORE COMPUTE BC ***\n";
  fT->print(std::cout);
#endif // DEBUG_LCM_SCHWARZ

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

#if defined(DEBUG_LCM_SCHWARZ)
  std::cout << "CONSTRAINED DOFS:\n";
  for (size_t i = 0; i < ns_dof.size(); ++i) {
    for (size_t j = 0; j < ns_dof[i].size(); ++j) {
      std::cout << ' ' << ns_dof[i][j];
    }
    std::cout << '\n';
  }
#endif // DEBUG_LCM_SCHWARZ

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

    fT_nonconstView[dof_x] = xT_constView[dof_x] - x_val;
    fT_nonconstView[dof_y] = xT_constView[dof_y] - y_val;
    fT_nonconstView[dof_z] = xT_constView[dof_z] - z_val;

  } // node in node set loop

#if defined(DEBUG_LCM_SCHWARZ)
  std::cout << "\n*** X AFTER COMPUTE BC ***\n";
  xT->print(std::cout);
  std::cout << "\n*** F AFTER COMPUTE BC ***\n";
  fT->print(std::cout);
#endif // DEBUG_LCM_SCHWARZ

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
#if defined(DEBUG_LCM_SCHWARZ)
  std::cout << "\n*** JACOBIAN ***\n";
#endif // DEBUG_LCM_SCHWARZ

  Teuchos::RCP<Tpetra_Vector> fT = dirichlet_workset.fT;
  Teuchos::ArrayRCP<ST> fT_nonconstView;

  Teuchos::RCP<Tpetra_CrsMatrix> jacT = dirichlet_workset.JacT;

  Teuchos::RCP<const Tpetra_Vector> xT = dirichlet_workset.xT;
  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();

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


  Teuchos::Array<LO> index(1);
  Teuchos::Array<ST> value(1);
  size_t numEntriesT;
  value[0] = j_coeff;
  Teuchos::Array<ST> matrixEntriesT;
  Teuchos::Array<LO> matrixIndicesT;

  bool fill_residual = (fT != Teuchos::null);
  if (fill_residual) fT_nonconstView = fT->get1dViewNonConst();

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

    numEntriesT = jacT->getNumEntriesInLocalRow(x_dof);
    matrixEntriesT.resize(numEntriesT);
    matrixIndicesT.resize(numEntriesT);

    // replace jac values for the X dof
    jacT->getLocalRowCopy(x_dof, matrixIndicesT(), matrixEntriesT(), numEntriesT);
    for (int i = 0; i < num_entries; ++i) matrixEntriesT[i] = 0;
    index[0] = x_dof;
    jacT->replaceLocalValues(x_dof, index(), value());

    // replace jac values for the y dof
    jacT->getLocalRowCopy(y_dof, matrixIndicesT(), matrixEntriesT(), numEntriesT);
    for (int i = 0; i < num_entries; ++i) matrixEntriesT[i] = 0;
    index[0] = y_dof;
    jacT->replaceLocalValues(y_dof, index(), value());

    // replace jac values for the z dof
    jacT->getLocalRowCopy(z_dof, matrixIndicesT(), matrixEntriesT(), numEntriesT);
    for (int i = 0; i < num_entries; ++i) matrixEntriesT[i] = 0;
    index[0] = z_dof;
    jacT->replaceLocalValues(z_dof, index(), value());

    if (fill_residual == true) {
      fT_nonconstView[x_dof] = xT_constView[x_dof] - x_val.val();
      fT_nonconstView[y_dof] = xT_constView[y_dof] - y_val.val();
      fT_nonconstView[z_dof] = xT_constView[z_dof] - z_val.val();
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
#if defined(DEBUG_LCM_SCHWARZ)
  std::cout << "\n*** TANGENT ***\n";
#endif // DEBUG_LCM_SCHWARZ

  Teuchos::RCP<Tpetra_Vector>  fT = dirichlet_workset.fT;

  Teuchos::RCP<Tpetra_MultiVector>  fpT = dirichlet_workset.fpT;

  Teuchos::RCP<Tpetra_MultiVector> JVT = dirichlet_workset.JVT;

  Teuchos::RCP<const Tpetra_Vector> xT = dirichlet_workset.xT;

  Teuchos::RCP<const Tpetra_MultiVector> VxT = dirichlet_workset.VxT;

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

  Teuchos::ArrayRCP<const ST> VxT_constView;
  Teuchos::ArrayRCP<ST> fT_nonconstView;
  if (fT != Teuchos::null) fT_nonconstView = fT->get1dViewNonConst();
  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();

  for (size_t ns_node = 0; ns_node < ns_nodes.size(); ++ns_node) {
    x_dof = ns_nodes[ns_node][0];
    y_dof = ns_nodes[ns_node][1];
    z_dof = ns_nodes[ns_node][2];
    coord = ns_coord[ns_node];

    this->computeBCs(dirichlet_workset, ns_node, x_val, y_val, z_val);

    if (fT != Teuchos::null) {
      fT_nonconstView[x_dof] = xT_constView[x_dof] - x_val.val();
      fT_nonconstView[y_dof] = xT_constView[y_dof] - y_val.val();
      fT_nonconstView[z_dof] = xT_constView[z_dof] - z_val.val();
    }

    if (JVT != Teuchos::null) {
      Teuchos::ArrayRCP<ST> JVT_nonconstView;
      for (int i = 0; i < dirichlet_workset.num_cols_x; ++i) {
        JVT_nonconstView = JVT->getDataNonConst(i); 
        VxT_constView = VxT->getData(i); 
        JVT_nonconstView[x_dof] = j_coeff * VxT_constView[x_dof];
        JVT_nonconstView[y_dof] = j_coeff * VxT_constView[y_dof];
        JVT_nonconstView[z_dof] = j_coeff * VxT_constView[z_dof];
      }
    }

    if (fpT != Teuchos::null) {
      Teuchos::ArrayRCP<ST> fpT_nonconstView;
      for (int i = 0; i < dirichlet_workset.num_cols_p; ++i) {
        fpT_nonconstView = fpT->getDataNonConst(i); 
        fpT_nonconstView[x_dof] = -x_val.dx(dirichlet_workset.param_offset + i);
        fpT_nonconstView[y_dof] = -y_val.dx(dirichlet_workset.param_offset + i);
        fpT_nonconstView[z_dof] = -z_val.dx(dirichlet_workset.param_offset + i);
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

  Teuchos::RCP<Tpetra_MultiVector> fpVT = dirichlet_workset.fpVT;
  Teuchos::ArrayRCP<ST> fpVT_nonconstView; 
  bool trans = dirichlet_workset.transpose_dist_param_deriv;
  int num_cols = fpVT->getNumVectors();

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
    Teuchos::RCP<Tpetra_MultiVector> VpT = dirichlet_workset.Vp_bcT;
    Teuchos::ArrayRCP<ST> VpT_nonconstView; 
    for (size_t inode = 0; inode < ns_nodes.size(); ++inode) {
      xlunk = ns_nodes[inode][0];
      ylunk = ns_nodes[inode][1];
      zlunk = ns_nodes[inode][2];
      // coord = ns_coord[inode];

      // this->computeBCs(coord, x_val, y_val, z_val);

      for (int col=0; col<num_cols; ++col) {
        //(*Vp)[col][xlunk] = 0.0;
        //(*Vp)[col][ylunk] = 0.0;
        //(*Vp)[col][zlunk] = 0.0;
        VpT_nonconstView = VpT->getDataNonConst(col); 
        VpT_nonconstView[xlunk] = 0.0; 
        VpT_nonconstView[ylunk] = 0.0; 
        VpT_nonconstView[zlunk] = 0.0; 
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
        //(*fpV)[col][xlunk] = 0.0;
        //(*fpV)[col][ylunk] = 0.0;
        //(*fpV)[col][zlunk] = 0.0;
        fpVT_nonconstView = fpVT->getDataNonConst(col); 
        fpVT_nonconstView[xlunk] = 0.0; 
        fpVT_nonconstView[ylunk] = 0.0; 
        fpVT_nonconstView[zlunk] = 0.0; 
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

