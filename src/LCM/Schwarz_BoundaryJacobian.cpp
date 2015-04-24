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

#define WRITE_TO_MATRIX_MARKET
//#define OUTPUT_TO_SCREEN
#define DEBUG_LCM_SCHWARZ

#ifdef WRITE_TO_MATRIX_MARKET
static int
mm_counter = 0;
#endif // WRITE_TO_MATRIX_MARKET


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

  int const
  this_app_index = getThisAppIndex();

  Albany::Application const &
  this_app = getApplication(this_app_index);

  int const
  coupled_app_index = getCoupledAppIndex();

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

  Albany::STKDiscretization *
  this_stk_disc = static_cast<Albany::STKDiscretization *>(this_disc.get());

  int const
  dimension = this_stk_disc->getNumDim();

  Albany::NodeSetList const &
  nodesets = this_stk_disc->getNodeSets();

  std::vector<std::vector<int>> const &
  ns_dof = nodesets.find(this_nodeset_name)->second;

  auto const
  ns_number_nodes = ns_dof.size();

  // Initialize Y vector.
  auto const
  zero = Teuchos::ScalarTraits<ST>::zero();

  Y.putScalar(zero);

  Teuchos::ArrayRCP<ST>
  Y_view = Y.get1dViewNonConst();

  for (auto ns_node = 0; ns_node < ns_number_nodes; ++ns_node) {

    Intrepid::Vector<ST> const
    bc_value = computeBC(this_app, coupled_app, X, dimension, ns_node);

    for (auto i = 0; i < dimension; ++i) {
      auto const
      dof = ns_dof[ns_node][i];

      auto const
      value = bc_value(i);

      Y_view[dof] = value;
    }

  } // node in node set loop

  // Create scratch space
  Tpetra_MultiVector
  Z(Y, Teuchos::DataAccess::Copy);

  // Multiply with the corresponding diagonal Jacobian.
  jacs_[this_app_index]->apply(Z, Y);

  // FIXME: Temporary to test.
  //Y.putScalar(zero);

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
  sprintf(name, "Z_%04d.mm", mm_counter);
  Tpetra_MatrixMarket_Writer::writeDenseFile(name, Z);
#endif  // WRITE_TO_MATRIX_MARKET

#ifdef WRITE_TO_MATRIX_MARKET
  sprintf(name, "Jac%04d_%04d.mm", this_app_index, mm_counter);
  Tpetra_MatrixMarket_Writer::writeSparseFile(name, jacs_[this_app_index]);
  mm_counter++;
#endif // WRITE_TO_MATRIX_MARKET
}

Intrepid::Vector<double>
LCM::
Schwarz_BoundaryJacobian::
computeBC(
    Albany::Application const & this_app,
    Albany::Application const & coupled_app,
    Tpetra_MultiVector const & coupled_solution,
    int const dimension,
    size_t const ns_node) const
{
  Teuchos::RCP<Albany::AbstractDiscretization>
  this_disc = this_app.getDiscretization();

  Albany::STKDiscretization *
  this_stk_disc =
      static_cast<Albany::STKDiscretization *>(this_disc.get());

  Teuchos::RCP<Albany::AbstractDiscretization>
  coupled_disc = coupled_app.getDiscretization();

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
  int const
  this_app_index = this_app.getAppIndex();

  std::string const &
  this_app_name = this_app.getAppName();

  std::string const &
  coupled_app_name = coupled_app.getAppName();

  int const
  coupled_app_index = coupled_app.getAppIndex();

  std::string const
  coupled_block_name = this_app.getCoupledBlockName(coupled_app_index);

  std::map<std::string, int> const &
  coupled_block_name_2_index = coupled_gms.ebNameToIndex;

  auto
  it = coupled_block_name_2_index.find(coupled_block_name);

  if (it == coupled_block_name_2_index.end()) {
    std::cerr << "\nERROR: " << __PRETTY_FUNCTION__ << '\n';
    std::cerr << "Unknown coupled block: " << coupled_block_name << '\n';
    std::cerr << "Coupling application : " << this_app_name << '\n';
    std::cerr << "To application       : " << coupled_app_name << '\n';
    exit(1);
  }

  int const
  coupled_block_index = it->second;

  CellTopologyData const
  coupled_cell_topology_data = coupled_mesh_specs[coupled_block_index]->ctd;

  shards::CellTopology
  coupled_cell_topology(&coupled_cell_topology_data);

  auto const
  coupled_dimension = coupled_cell_topology_data.dimension;

  // FIXME: Generalize element topology.
  auto const
  coupled_vertex_count = coupled_cell_topology_data.vertex_count;

  Intrepid::ELEMENT::Type const
  coupled_element_type =
      Intrepid::find_type(coupled_dimension, coupled_vertex_count);

  std::string const &
  coupled_nodeset_name = this_app.getNodesetName(coupled_app_index);

  std::vector<double *> const &
  ns_coord =
      this_stk_disc->getNodeSetCoords().find(coupled_nodeset_name)->second;

  auto const &
  ws_elem_2_node_id = coupled_stk_disc->getWsElNodeID();

  std::vector<Intrepid::Vector<double>>
  coupled_element_vertices(coupled_vertex_count);

  std::vector<Intrepid::Vector<double>>
  coupled_element_solution(coupled_vertex_count);

  for (auto i = 0; i < coupled_vertex_count; ++i) {
    coupled_element_vertices[i].set_dimension(coupled_dimension);
    coupled_element_solution[i].set_dimension(coupled_dimension);
  }

  // This tolerance is used for geometric approximations. It will be used
  // to determine whether a node of this_app is inside an element of
  // coupled_app within that tolerance.
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

  Teuchos::RCP<Intrepid::Basis<double, Intrepid::FieldContainer<double>>>
  basis;

  Teuchos::ArrayRCP<ST const>
  coupled_solution_view = coupled_solution.get1dView();

  Teuchos::ArrayRCP<double> const &
  coupled_coordinates = coupled_stk_disc->getCoordinates();

  for (auto workset = 0; workset < ws_elem_2_node_id.size(); ++workset) {

    std::string const &
    coupled_element_block = coupled_ws_eb_names[workset];

    if (coupled_element_block != coupled_block_name) continue;

    auto const
    elements_per_workset = ws_elem_2_node_id[workset].size();

    for (auto element = 0; element < elements_per_workset; ++element) {

      for (auto node = 0; node < coupled_vertex_count; ++node) {

        auto const
        node_id = ws_elem_2_node_id[workset][element][node];

        double * const
        pcoord = &(coupled_coordinates[coupled_dimension * node_id]);

        coupled_element_vertices[node].fill(pcoord);

        for (auto i = 0; i < coupled_dimension; ++i) {
          coupled_element_solution[node](i) =
              coupled_solution_view[coupled_dimension * node_id + i];
        } // dimension loop

      } // node loop

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
            double, Intrepid::FieldContainer<double>>());

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
            double, Intrepid::FieldContainer<double>>());

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

      } // switch

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
  auto const
  number_cells = 1;

  // Container for the parametric coordinates
  Intrepid::FieldContainer<double>
  parametric_point(number_cells, parametric_dimension);

  for (auto j = 0; j < parametric_dimension; ++j) {
    parametric_point(0, j) = 0.0;
  }

  // Container for the physical point
  Intrepid::FieldContainer<double>
  physical_coordinates(number_cells, coupled_dimension);

  for (auto i = 0; i < coupled_dimension; ++i) {
    physical_coordinates(0, i) = point(i);
  }

  // Container for the physical nodal coordinates
  // TODO: matToReference more general, accepts more topologies.
  // Use it to find if point is contained in element as well.
  Intrepid::FieldContainer<double>
  nodal_coordinates(number_cells, coupled_vertex_count, coupled_dimension);

  for (auto i = 0; i < coupled_vertex_count; ++i) {
    for (auto j = 0; j < coupled_dimension; ++j) {
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
  auto const
  number_points = 1;

  Intrepid::FieldContainer<double>
  basis_values(coupled_vertex_count, number_points);

  basis->getValues(basis_values, parametric_point, Intrepid::OPERATOR_VALUE);

  // Evaluate solution at parametric point using values of shape
  // functions just computed.
  Intrepid::Vector<double>
  value(coupled_dimension, Intrepid::ZEROS);

#if defined(DEBUG_LCM_SCHWARZ)
  std::cout << "Coupling to app  : " << coupled_app_name << '\n';
  std::cout << "Coupling to block: " << coupled_block_name << '\n';
#endif // DEBUG_LCM_SCHWARZ

  for (auto i = 0; i < coupled_vertex_count; ++i) {
    value += basis_values(i, 0) * coupled_element_solution[i];

#if defined(DEBUG_LCM_SCHWARZ)
    std::cout << std::scientific << std::setprecision(16);
    std::cout << basis_values(i, 0) << "    ";
    std::cout << coupled_element_solution[i] << '\n';
#endif // DEBUG_LCM_SCHWARZ

  }

#if defined(DEBUG_LCM_SCHWARZ)
  std::cout << " ==> " << value << '\n';
#endif // DEBUG_LCM_SCHWARZ

  return value;
}
