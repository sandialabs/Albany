//
// Simple VTK visualization of composite tetrahedron.
// Requires two file names as arguments for compact and exploded views.
//
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

int
main(int ac, char * av[])
{
  std::string
  intact_filename(av[1]);

  std::ofstream
  intact_fs;

  intact_fs.open(intact_filename.c_str(), std::ios::out);

  if (intact_fs.is_open() == false) {
    std::cout << "Unable to open intact output file: ";
    std::cout << intact_filename << '\n';
    return 1;
  }

  std::cout << "Write intact VTK file: ";
  std::cout << intact_filename << '\n';

  // Header
  intact_fs << "# vtk DataFile Version 3.0\n";
  intact_fs << "Albany/LCM\n";
  intact_fs << "ASCII\n";
  intact_fs << "DATASET UNSTRUCTURED_GRID\n";

  //std::vector<std::vector<double>> const
  //coordinates = {{0,0,0}, {1,0,0}, {0,1,0}, {0,0,1}};

  //std::vector<std::vector<double>> const
  //xi = {{1,0,0,0}, {0,1,0,0}, {0,0,1,0}, {0,0,0,1}};

  //std::vector<std::vector<int>> const
  //connectivity = {{0,1,2,3}};

  //std::vector<std::vector<double>> const
  //coordinates = {
  //  {0.00, 0.00, 0.00},
  //  {1.00, 0.00, 0.00},
  //  {0.00, 1.00, 0.00},
  //  {0.00, 0.00, 1.00},
  //  {0.50, 0.00, 0.00},
  //  {0.50, 0.50, 0.00},
  //  {0.00, 0.50, 0.00},
  //  {0.00, 0.00, 0.50},
  //  {0.50, 0.00, 0.50},
  //  {0.00, 0.50, 0.50},
  //  {0.25, 0.25, 0.25}
  //};

  auto const
  c = sqrt(2.0) / 2.0;

  std::vector<std::vector<double>> const
  coordinates = {
    {-1.00,  0.00, -c},
    {+1.00,  0.00, -c},
    { 0.00, +1.00, +c},
    { 0.00, -1.00, +c},
    { 0.00,  0.00, -c},
    {+0.50, +0.50,  0.00},
    {-0.50, +0.50,  0.00},
    {-0.50, -0.50,  0.00},
    {+0.50, -0.50,  0.00},
    { 0.00,  0.00, +c},
    { 0.00,  0.00,  0.00}
  };

  std::vector<std::vector<double>> const
  xi = {
    {1.00, 0.00, 0.00, 0.00},
    {0.00, 1.00, 0.00, 0.00},
    {0.00, 0.00, 1.00, 0.00},
    {0.00, 0.00, 0.00, 1.00},
    {0.50, 0.50, 0.00, 0.00},
    {0.00, 0.50, 0.50, 0.00},
    {0.50, 0.00, 0.50, 0.00},
    {0.50, 0.00, 0.00, 0.50},
    {0.00, 0.50, 0.00, 0.50},
    {0.00, 0.00, 0.50, 0.50},
    {0.25, 0.25, 0.25, 0.25}
  };

  std::vector<std::vector<int>> const
  connectivity = {
    {0,4,6,7},
    {1,5,4,8},
    {2,6,5,9},
    {3,8,7,9},
    {4,8,5,10},
    {5,8,9,10},
    {9,8,7,10},
    {7,8,4,10},
    {4,5,6,10},
    {5,9,6,10},
    {9,7,6,10},
    {7,4,6,10}
  };

  auto
  num_nodes = coordinates.size();

  auto const
  num_functions = num_nodes;

  intact_fs << "POINTS " << num_nodes << " double\n";

  for (auto node = coordinates.begin(); node != coordinates.end(); ++node) {

    std::vector<double> const &
    X = *node;

    for (auto dim = X.begin(); dim != X.end(); ++dim) {
      intact_fs << std::setw(24) << std::scientific << std::setprecision(16);
      intact_fs << *dim;
    }
    intact_fs << '\n';
  }

  auto const
  num_cells = connectivity.size();

  int
  cell_list_size = 0;

  for (auto cell = connectivity.begin(); cell != connectivity.end(); ++cell) {
    cell_list_size += cell->size() + 1;
  }

  // Cell connectivity
  intact_fs << "CELLS " << num_cells << " " << cell_list_size << '\n';
  for (auto cell = connectivity.begin(); cell != connectivity.end(); ++cell) {
    auto const
    num_cell_nodes = cell->size();

    intact_fs << num_cell_nodes;

    for (auto cell_node = cell->begin(); cell_node != cell->end(); ++cell_node) {
      intact_fs << ' ' << *cell_node;
    }
    intact_fs << '\n';
  }

  intact_fs << "CELL_TYPES " << num_cells << '\n';
  for (auto cell = connectivity.begin(); cell != connectivity.end(); ++cell) {
    auto const
    num_cell_nodes = cell->size();

    int
    vtk_cell_type = -1;

    // Simplices only.
    switch (num_cell_nodes) {
    default:
      std::cerr << "ERROR: " << __PRETTY_FUNCTION__;
      std::cerr << '\n';
      std::cerr << "Invalid number of nodes in cell: ";
      std::cerr << num_cell_nodes;
      std::cerr << '\n';
      return(1);
      break;

    case 1:
      vtk_cell_type = 1;
      break;

    case 2:
      vtk_cell_type = 3;
      break;

    case 3:
      vtk_cell_type = 5;
      break;

    case 4:
      vtk_cell_type = 10;
      break;

    }
    intact_fs << vtk_cell_type << '\n';
  }

  // Point values
  auto const
  num_params = xi[0].size();

  intact_fs << "POINT_DATA " << num_nodes << "\n";

  for (size_t param = 0; param < num_params; ++param) {

    intact_fs << "SCALARS xi_" << param << " double 1\n";
    intact_fs << "LOOKUP_TABLE default\n";

    for (size_t node = 0; node < num_nodes; ++node) {

      intact_fs << std::setw(24) << std::scientific << std::setprecision(16);
      intact_fs << xi[node][param];

      intact_fs << '\n';
    }

  }

  for (size_t N = 0; N < num_functions; ++N) {

    intact_fs << "SCALARS N_" << N << " double 1\n";
    intact_fs << "LOOKUP_TABLE default\n";

    for (size_t node = 0; node < num_nodes; ++node) {

      intact_fs << std::setw(24) << std::scientific << std::setprecision(16);
      intact_fs << (N == node ? 1.0 : 0.0);

      intact_fs << '\n';
    }

  }

  intact_fs.close();

  //
  // Split tetrahedra
  //
  std::string
  split_filename(av[2]);

  std::ofstream
  split_fs;

  split_fs.open(split_filename.c_str(), std::ios::out);

  if (split_fs.is_open() == false) {
    std::cout << "Unable to open split output file: ";
    std::cout << split_filename << '\n';
    return 1;
  }

  std::cout << "Write split VTK file: ";
  std::cout << split_filename << '\n';

  // Header
  split_fs << "# vtk DataFile Version 3.0\n";
  split_fs << "Albany/LCM\n";
  split_fs << "ASCII\n";
  split_fs << "DATASET UNSTRUCTURED_GRID\n";

  std::vector<std::vector<double>>
  centroids;

  auto const
  nodes_cell = connectivity[0].size();

  for (size_t cell = 0; cell < connectivity.size(); ++cell) {
    std::vector<double>
    centroid = coordinates[connectivity[cell][0]];

    for (size_t node = 1; node < nodes_cell; ++node) {

      std::vector<double> const &
      X = coordinates[connectivity[cell][node]];

      for (size_t dim = 0; dim < centroid.size(); ++dim) {
	centroid[dim] += X[dim];
      }
    }

    for (size_t dim = 0; dim < centroid.size(); ++dim) {
      centroid[dim] /= nodes_cell;
    }

    centroids.push_back(centroid);

  }

  std::vector<size_t>
  map;

  num_nodes = 0;

  for (size_t cell = 0; cell < connectivity.size(); ++cell) {
    for (size_t node = 0; node < connectivity[cell].size(); ++node) {
      ++num_nodes;
      map.push_back(connectivity[cell][node]);
    }
  }

  split_fs << "POINTS " << num_nodes << " double\n";

  for (size_t node = 0; node < num_nodes; ++node) {

    auto const
    index = map[node];

    auto const
    cell = node / nodes_cell;

    std::vector<double> const &
    centroid = centroids[cell];

    std::vector<double> const &
    X = coordinates[index];

    for (size_t dim = 0; dim < X.size(); ++dim) {
      split_fs << std::setw(24) << std::scientific << std::setprecision(16);
      split_fs << X[dim] + centroid[dim];
    }
    split_fs << '\n';
  }

  int
  node_number = 0;

  // Cell connectivity
  split_fs << "CELLS " << num_cells << " " << cell_list_size << '\n';
  for (size_t cell = 0; cell < num_cells; ++cell) {
    size_t const
    num_cell_nodes = connectivity[cell].size();

    split_fs << num_cell_nodes;

    for (size_t cell_node = 0; cell_node < num_cell_nodes; ++cell_node) {
      split_fs << ' ' << node_number;
      ++node_number;
    }
    split_fs << '\n';
  }

  // Cell types
  split_fs << "CELL_TYPES " << num_cells << '\n';
  for (size_t cell = 0; cell < num_cells; ++cell) {
    size_t const
    num_cell_nodes = connectivity[cell].size();

    int
    vtk_cell_type = -1;

    // Simplices only.
    switch (num_cell_nodes) {
    default:
      std::cerr << "ERROR: " << __PRETTY_FUNCTION__;
      std::cerr << '\n';
      std::cerr << "Invalid number of nodes in cell: ";
      std::cerr << num_cell_nodes;
      std::cerr << '\n';
      return(1);
      break;

    case 1:
      vtk_cell_type = 1;
      break;

    case 2:
      vtk_cell_type = 3;
      break;

    case 3:
      vtk_cell_type = 5;
      break;

    case 4:
      vtk_cell_type = 10;
      break;

    }
    split_fs << vtk_cell_type << '\n';
  }

  // Point values
  split_fs << "POINT_DATA " << num_nodes << "\n";

  for (size_t param = 0; param < num_params; ++param) {

    split_fs << "SCALARS xi_" << param << " double 1\n";
    split_fs << "LOOKUP_TABLE default\n";

    for (size_t node = 0; node < num_nodes; ++node) {

      size_t const
      index = map[node];

      split_fs << std::setw(24) << std::scientific << std::setprecision(16);
      split_fs << xi[index][param];

      split_fs << '\n';
    }

  }

  for (size_t N = 0; N < num_functions; ++N) {

    split_fs << "SCALARS N_" << N << " double 1\n";
    split_fs << "LOOKUP_TABLE default\n";

    for (size_t node = 0; node < num_nodes; ++node) {

      split_fs << std::setw(24) << std::scientific << std::setprecision(16);
      split_fs << (N == map[node] ? 1.0 : 0.0);

      split_fs << '\n';
    }

  }

  split_fs.close();

  return 0;
}
