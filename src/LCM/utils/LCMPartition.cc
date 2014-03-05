//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

// Define only if Zoltan is enabled
#if defined (ALBANY_LCM) && defined(ALBANY_ZOLTAN)

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <sstream>
#include <string>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>

#include "LCMPartition.h"

using Intrepid::Index;
using Intrepid::Vector;
using Teuchos::ArrayRCP;

namespace LCM {

//
// Anonymous namespace for helper functions
//
namespace {

//
// Print parameters and partitions computed by Zoltan.
// Used for debugging.
//
void PrintPartitionInfo(
    std::ostream & output_stream,
    int &changes,
    int &num_gid_entries,
    int &num_lid_entries,
    int &num_import,
    ZOLTAN_ID_PTR &import_global_ids,
    ZOLTAN_ID_PTR &import_local_ids,
    int * &import_procs,
    int * &import_to_part,
    int &num_export,
    ZOLTAN_ID_PTR &export_global_ids,
    ZOLTAN_ID_PTR &export_local_ids,
    int * &export_procs,
    int * &export_to_part )
{

  output_stream << "Changes           : " << changes << '\n';
  output_stream << "Number GID entries: " << num_gid_entries << '\n';
  output_stream << "Number LID entries: " << num_lid_entries << '\n';
  output_stream << "Number to import  : " << num_import << '\n';
  output_stream << "Number to export  : " << num_export << '\n';

  output_stream << "Import GIDs:" << '\n';
  for (int i = 0; i < num_import; ++i) {
    output_stream << import_global_ids[i] << '\n';
  }

  output_stream << "Import LIDs:" << '\n';
  for (int i = 0; i < num_import; ++i) {
    output_stream << import_local_ids[i] << '\n';
  }

  output_stream << "Import procs:" << '\n';
  for (int i = 0; i < num_import; ++i) {
    output_stream << import_procs[i] << '\n';
  }

  output_stream << "Import parts:" << '\n';
  for (int i = 0; i < num_import; ++i) {
    output_stream << import_to_part[i] << '\n';
  }

  output_stream << "Export GIDs:" << '\n';
  for (int i = 0; i < num_export; ++i) {
    output_stream << export_global_ids[i] << '\n';
  }

  output_stream << "Export LIDs:" << '\n';
  for (int i = 0; i < num_export; ++i) {
    output_stream << export_local_ids[i] << '\n';
  }

  output_stream << "Export procs:" << '\n';
  for (int i = 0; i < num_export; ++i) {
    output_stream << export_procs[i] << '\n';
  }

  output_stream << "Export parts:" << '\n';
  for (int i = 0; i < num_export; ++i) {
    output_stream << export_to_part[i] << '\n';
  }

}

//
// Given a vector of points and a set of indices to this vector:
// 1) Find the bounding box of the indexed points.
// 2) Compute the vector sum of the indexed points.
//
boost::tuple< Vector<double>, Vector<double>, Vector<double> >
bounds_and_sum_subset(
    std::vector< Vector<double> > const & points,
    std::set<Index> const & indices)
{
  assert(points.size() > 0);
  assert(indices.size() > 0);

  Index const
  first = *indices.begin();

  Vector<double>
  sum = points[first];

  Vector<double>
  lower_corner = sum;

  Vector<double>
  upper_corner = sum;

  Index const
  N = sum.get_dimension();

  for (std::set<Index>::const_iterator it = ++indices.begin();
      it != indices.end();
      ++it) {

    Index const
    index = *it;

    Vector<double> const &
    p = points[index];

    sum += p;

    for (Index i = 0; i < N; ++i) {
      lower_corner(i) = std::min(lower_corner(i), p(i));
      upper_corner(i) = std::max(upper_corner(i), p(i));
    }

  }

  return boost::make_tuple(lower_corner, upper_corner, sum);
}

//
// Given point, a vector of centers and a set of indices into this vector:
// Return the index of the center closest to the point among the
// indexed centers.
//
Index
closest_subset(
    Vector<double> const & point,
    std::vector< ClusterCenter > const & centers,
    std::set<Index> const & indices)
{
  assert(centers.size() > 0);
  assert(indices.size() > 0);

  Index const
  first = *indices.begin();

  double
  minimum = norm_square(centers[first].position - point);

  Index
  index_minimum = first;

  for (std::set<Index>::const_iterator it = ++indices.begin();
      it != indices.end();
      ++it) {

    Index const
    index = *it;

    Vector<double> const &
    p = centers[index].position;

    double const
    s = norm_square(p - point);

    if (s < minimum) {
      minimum = s;
      index_minimum = index;
    }

  }

  return index_minimum;
}

//
// Given a vector of points and a set of indices:
// 1) Find the bounding box of the indexed points.
// 2) Divide the bounding box along its largest dimension, using median.
// 3) Assign points to one side or the other, and return index sets.
//
std::pair< std::set<Index>, std::set<Index> >
split_box(
    std::vector< Vector<double> > const & points,
    std::set<Index> const & indices)
{
  assert(points.size() > 0);
  assert(indices.size() > 0);

  //
  // Compute bounding box
  //
  Index const
  first = *indices.begin();

  Vector<double>
  lower_corner = points[first];

  Vector<double>
  upper_corner = lower_corner;

  Index const
  N = lower_corner.get_dimension();

  for (std::set<Index>::const_iterator it = ++indices.begin();
      it != indices.end();
      ++it) {

    Index const
    index = *it;

    Vector<double> const &
    p = points[index];

    for (Index i = 0; i < N; ++i) {
      lower_corner(i) = std::min(lower_corner(i), p(i));
      upper_corner(i) = std::max(upper_corner(i), p(i));
    }

  }

  //
  // Find largest dimension
  //
  Vector<double> const
  span = upper_corner - lower_corner;

  assert(norm_square(span) > 0.0);

  double
  maximum_span = span(0);

  Index
  largest_dimension = 0;

  for (Index i = 1; i < N; ++i) {

    double const
    s = span(i);

    if (s > maximum_span) {
      maximum_span = s;
      largest_dimension = i;
    }

  }

  //
  // Find median coordinate along largest dimension
  //
  std::vector<double>
  coordinates;

  for (std::set<Index>::const_iterator it = indices.begin();
      it != indices.end();
      ++it) {

    Index const
    index = *it;

    Vector<double> const &
    p = points[index];

    coordinates.push_back(p(largest_dimension));

  }

  std::sort(coordinates.begin(), coordinates.end());

  double
  split_coordinate =
      Intrepid::median<double>(coordinates.begin(), coordinates.end());

  //
  // Check whether splitting the box will result in one box of
  // the same volume as the original and another one of zero volume.
  // If so, split the original box into two of equal volume.
  //
  bool const
  box_unchanged =
      split_coordinate == lower_corner(largest_dimension) ||
      split_coordinate == upper_corner(largest_dimension);

  if (box_unchanged == true) {

    std::sort(coordinates.begin(), coordinates.end());

    split_coordinate =
        0.5 * (coordinates[0] + coordinates[coordinates.size() - 1]);

  }

  //
  // Assign points to lower or upper half.
  //
  std::set<Index>
  indices_lower;

  std::set<Index>
  indices_upper;

  Vector<double>
  split_limit = upper_corner;

  split_limit(largest_dimension) = split_coordinate;

  for (std::set<Index>::const_iterator it = indices.begin();
      it != indices.end();
      ++it) {

    Index const
    index = *it;

    Vector<double> const &
    p = points[index];

    if (in_box(p, lower_corner, split_limit) == true) {
      indices_lower.insert(index);
    } else {
      indices_upper.insert(index);
    }

  }

  return std::make_pair(indices_lower, indices_upper);
}

} // anonymous namespace

//
// Build KD tree of list of points.
// \param point list
// \return Boost shared pointer to root node of tree.
//
template<typename Node>
boost::shared_ptr<Node>
BuildKDTree(std::vector< Vector<double> > const & points)
{

  //
  // Initially all points are in the index set.
  //
  Index const
  number_points = points.size();

  std::set<Index>
  points_indices;

  for (Index i = 0; i < number_points; ++i) {
    points_indices.insert(i);
  }

  boost::shared_ptr<Node>
  dummy;

  std::string
  name = "0";

  boost::shared_ptr<Node>
  root = CreateKDTreeNode(name, dummy, points, points_indices);

  return root;
}

//
// Create KD tree node.
// \param point list
// \return Boost shared pointer to node of tree if created, 0 otherwise.
//
template<typename Node>
boost::shared_ptr<Node>
CreateKDTreeNode(
    std::string const & name,
    boost::shared_ptr<Node> parent,
    std::vector< Vector<double> > const & points,
    std::set<Index> const & points_indices)
{
  if (name.length() >=64) {
    std::cout << "Name is too long: " << name << '\n';
  }

  //
  // Create and fill in node.
  //
  boost::shared_ptr<Node>
  node(new Node);

  node->name = name;

  node->parent = parent;

  Index const
  count = points_indices.size();

  node->count = count;
  node->cell_points = points_indices;

  switch (count) {

  // Empty node
  case 0:
    break;

    // Leaf node
  case 1:
  {
    Vector<double> const &
    p = points[*points_indices.begin()];
    node->lower_corner = p;
    node->upper_corner = p;
    node->weighted_centroid = p;
  }
  break;

  default:
  {
    boost::tie(
        node->lower_corner,
        node->upper_corner,
        node->weighted_centroid) =
            bounds_and_sum_subset(points, points_indices);

    std::set<Index>
    indices_left;

    std::set<Index>
    indices_right;

    boost::tie(indices_left, indices_right) =
        split_box(points, points_indices);

    std::string
    name_left = name + "0";

    std::string
    name_right = name + "1";

    node->left =
        CreateKDTreeNode(name_left, node, points, indices_left);

    node->right =
        CreateKDTreeNode(name_right, node, points, indices_right);
  }
  break;

  }

  return node;
}

//
// KdTree constructor with list of points.
//
template<typename Node>
KDTree<Node>::KDTree(
    std::vector<Vector<double> > const & points,
    Index const number_centers)
    {
  root_ = BuildKDTree<Node>(points);

  // Set candidate centers to all
  std::set<Index>
  candidate_centers;

  for (Index i = 0; i < number_centers; ++i) {
    candidate_centers.insert(i);
  }

  root_->candidate_centers = candidate_centers;

  return;
    }

//
// Visit Tree nodes recursively and
// perform the action defined by the Visitor object.
//
template<typename Node, typename Visitor>
void
VisitTreeNode(Node & node, Visitor const & visitor)
{
  if (visitor.pre_stop(node) == true) return;

  visitor(node);

  if (visitor.post_stop(node) == true) return;

  VisitTreeNode(node->left, visitor);
  VisitTreeNode(node->right, visitor);

  return;
}

//
// Traverse a Tree and perform the action defined by the Visitor object.
//
template<typename Tree, typename Visitor>
void
TraverseTree(Tree & tree, Visitor const & visitor)
{
  VisitTreeNode(tree.get_root(), visitor);
  return;
}

//
// Output visitor for KDTree node.
//
template<typename Node>
void
OutputVisitor<Node>::operator()(Node const & node) const
{
  std::cout << "Node        : " << node->name << '\n';
  std::cout << "Count       : " << node->count << '\n';
  std::cout << "Lower corner: " << node->lower_corner << '\n';
  std::cout << "Upper corner: " << node->upper_corner << '\n';

  Vector<double>
  centroid = node->weighted_centroid / node->count;

  std::cout << "Centroid    : " << centroid << '\n';

  return;
}

//
// Pre-visit stopping criterion for traversing the tree.
//
template<typename Node>
bool
OutputVisitor<Node>::pre_stop(Node const & node) const
{
  return node.get() == NULL;
}

//
// Post-visit stopping criterion for traversing the tree.
//
template<typename Node>
bool
OutputVisitor<Node>::post_stop(Node const & node) const
{
  return false;
}

//
// Constructor for filtering visitor
//
template<typename Node, typename Center>
FilterVisitor<Node, Center>::FilterVisitor(
    std::vector<Vector<double> > & p,
    std::vector<Center> & c) :
    points(p),
    centers(c)
{
}

namespace {

template<typename Center, typename Iterator>
Index
closest_center_from_subset(
    Vector<double> const & point,
    std::vector<Center> const & centers,
    Iterator begin,
    Iterator end)
{
  assert(std::distance(begin, end) > 0);

  Index
  closest_index = *begin;

  double
  minimum_distance = norm_square(point - centers[closest_index].position);

  for (Iterator it = ++begin; it != end; ++it) {

    Index const
    i = *it;

    double const
    s = norm_square(point - centers[i].position);

    if (s < minimum_distance) {
      closest_index = i;
      minimum_distance = s;
    }

  }

  return closest_index;

}

//
// Given the corners of a box, a vector of centers and
// a subset of indices to the centers:
// Determine the closest center among the subset to the midcell.
// For the remaining centers, define hyperplanes that are
// equidistant to them and the closest center to the midcell.
// Determine whether the box lies entirely on the side of the hyperplane
// where the closest center to the midcell lies as well.
//
template<typename Center>
std::pair<Index, std::set<Index> >
box_proximity_to_centers(
    Vector<double> const & lower_corner,
    Vector<double> const & upper_corner,
    std::vector<Center> const & centers,
    std::set<Index> const & index_subset)
{
  assert(centers.size() > 0);
  assert(index_subset.size() > 0);

  Vector<double> const
  midcell = 0.5 * (lower_corner + upper_corner);

  // Determine the closest point to box center only among those
  // listed in the index subset.
  Index
  index_closest = *index_subset.begin();

  double
  minimum = norm_square(midcell - centers[index_closest].position);

  for (std::set<Index>::const_iterator
      index_iterator = ++index_subset.begin();
      index_iterator != index_subset.end();
      ++index_iterator) {

    Index const
    i = *index_iterator;

    double const
    s = norm_square(midcell - centers[i].position);

    if (s < minimum) {
      index_closest = i;
      minimum = s;
    }

  }

  Vector<double> const &
  closest_to_midcell = centers[index_closest].position;

  std::set<Index>
  indices_candidates;

  // Determine where the box lies
  for (std::set<Index>::const_iterator index_iterator = index_subset.begin();
      index_iterator != index_subset.end();
      ++index_iterator) {

    Index const
    i = *index_iterator;

    if (i == index_closest) {
      indices_candidates.insert(i);
      continue;
    }

    Vector<double> const &
    p = centers[i].position;

    Vector<double> const
    u = p - closest_to_midcell;

    Index const
    N = u.get_dimension();

    Vector<double>
    v(N);

    for (Index j = 0; j < N; ++j) {

      v(j) = u(j) >= 0.0 ? upper_corner(j) : lower_corner(j);

    }

    if (norm_square(p - v) < norm_square(closest_to_midcell - v)) {
      indices_candidates.insert(i);
    }

  }

  return std::make_pair(index_closest, indices_candidates);
}

} // anonymous namespace

//
// Filtering visitor for KDTree node
//
template<typename Node, typename Center>
void
FilterVisitor<Node, Center>::operator()(Node const & node) const
{
  bool const
  node_is_empty = node->count == 0;

  if (node_is_empty == true) {
    return;
  }

  bool const
  node_is_leaf = node->count == 1;

  if (node_is_leaf == true) {

    // Get point
    Index const
    point_index = *(node->cell_points.begin());

    Vector<double> const &
    point = points[point_index];

    // Find closest center to it
    Index
    index_closest =
        closest_center_from_subset(
            point,
            centers,
            node->candidate_centers.begin(),
            node->candidate_centers.end());

    // Update closest center
    Center &
    closest_center = centers[index_closest];

    closest_center.weighted_centroid += point;
    ++closest_center.count;

    node->closest_center_to_midcell = index_closest;

  } else { // node_is_leaf == false

    // Get midpoint of cell
    Index
    index_closest_midcell = 0;

    std::set<Index>
    candidate_indices;

    boost::tie(index_closest_midcell, candidate_indices) =
        box_proximity_to_centers(
            node->lower_corner,
            node->upper_corner,
            centers,
            node->candidate_centers);

    node->candidate_centers = candidate_indices;

    if (candidate_indices.size() == 1) {

      Center &
      center = centers[index_closest_midcell];

      center.weighted_centroid += node->weighted_centroid;
      center.count += node->count;
    } else {

      // Update children
      node->left->candidate_centers = candidate_indices;
      node->right->candidate_centers = candidate_indices;
    }

    node->closest_center_to_midcell = index_closest_midcell;

  }

  return;
}

//
// Pre-visit stopping criterion for traversing the tree.
//
template<typename Node, typename Center>
bool
FilterVisitor<Node, Center>::pre_stop(Node const & node) const
{
  bool const
  has_no_centers = node->candidate_centers.size() == 0;

  return has_no_centers == true;
}

//
// Post-visit stopping criterion for traversing the tree.
//
template<typename Node, typename Center>
bool
FilterVisitor<Node, Center>::post_stop(Node const & node) const
{
  bool const
  node_is_leaf = node->count <= 1;

  bool const
  has_single_center = node->candidate_centers.size() == 1;

  bool const
  stop_traversal = node_is_leaf || has_single_center;

  return stop_traversal == true;
}

//
// Default constructor for Connectivity Array
//
ConnectivityArray::ConnectivityArray() :
        type_(Intrepid::ELEMENT::UNKNOWN),
        dimension_(0),
        discretization_ptr_(Teuchos::null),
        tolerance_(0.0),
        requested_cell_size_(0.0),
        maximum_iterations_(0),
        initializer_scheme_(PARTITION::HYPERGRAPH)
{
  return;
}

//
// Build array specifying input and output
// \param input_file Exodus II input fine name
// \param output_file Exodus II output fine name
//
ConnectivityArray::ConnectivityArray(
    std::string const & input_file,
    std::string const & output_file) :
          type_(Intrepid::ELEMENT::UNKNOWN),
          dimension_(0),
          discretization_ptr_(Teuchos::null),
          tolerance_(0.0),
          requested_cell_size_(0.0),
          maximum_iterations_(0),
          initializer_scheme_(PARTITION::HYPERGRAPH)
{

  using Albany::StateStruct;

  //Teuchos::GlobalMPISession mpiSession(&argc,&argv);

  Teuchos::RCP<Teuchos::ParameterList> params =
      rcp(new Teuchos::ParameterList("params"));

  // Create discretization object
  Teuchos::RCP<Teuchos::ParameterList> disc_params =
      Teuchos::sublist(params, "Discretization");

  //set Method to Exodus and set input file name
  disc_params->set<std::string>("Method", "Exodus");
  disc_params->set<std::string>("Exodus Input File Name", input_file);
  disc_params->set<std::string>("Exodus Output File Name", output_file);
  // Max of 10000 workset size -- automatically resized down
  disc_params->set<int>("Workset Size", 10000);

  Teuchos::RCP<Epetra_Comm>
  communicator = Albany::createEpetraCommFromMpiComm(Albany_MPI_COMM_WORLD);

  Albany::DiscretizationFactory
  disc_factory(params, communicator);

  ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >
  mesh_specs = disc_factory.createMeshSpecs();

  int
  workset_size = mesh_specs[0]->worksetSize;

  // Create a state field in stick named Partition on elements
  // 1 DOF per node
  // 1 internal variable (partition number)
  Teuchos::RCP<Albany::StateInfoStruct>
  state_info = Teuchos::rcp(new Albany::StateInfoStruct());

  StateStruct::FieldDims dims;
  // State has 1 quad point (i.e. element variable)
  dims.push_back(workset_size); dims.push_back(1);

  state_info->push_back(Teuchos::rcp(
      new StateStruct("Partition", StateStruct::QuadPoint, dims, "scalar")));

  // The default fields
  Albany::AbstractFieldContainer::FieldContainerRequirements
  req;

  discretization_ptr_ = disc_factory.createDiscretization(1, state_info, req);

  dimension_ = mesh_specs[0]->numDim;

  // Dimensioned: Workset, Cell, Local Node
  Albany::WorksetArray<ArrayRCP<ArrayRCP<ArrayRCP<int> > > >::type const &
  element_connectivity = discretization_ptr_->getWsElNodeEqID();

  ArrayRCP<double>
  coordinates = discretization_ptr_->getCoordinates();

  // For higher-order elements, mid-nodes are ignored and only
  // the nodes at the corners of the element are considered
  // to define the topology.
  const CellTopologyData
  cell_topology = mesh_specs[0]->ctd;

  Index const
  dimension = cell_topology.dimension;

  assert(dimension == dimension_);

  int const
  vertices_per_element = cell_topology.vertex_count;

  type_ = Intrepid::find_type(dimension, vertices_per_element);

  // Assume all the elements have the same number of nodes and eqs
  ArrayRCP<int>::size_type
  nodes_per_element = element_connectivity[0][0].size();

  // Do some logic so we can get from unknown ID to node ID
  int const number_equations = element_connectivity[0][0][0].size();
  int stride = 1;
  if (number_equations > 1) {
    if (element_connectivity[0][0][0][0] + 1 ==
        element_connectivity[0][0][0][1]) {
      // usual interleaved unknowns case
      stride = number_equations;
    }
  }


  // Build coordinate array.
  // Assume that local numbering of nodes is contiguous.
  ArrayRCP<double>::size_type
  number_nodes = coordinates.size() / dimension;

  for (ArrayRCP<double>::size_type
      node = 0;
      node < number_nodes;
      ++node) {

    Vector<double> point(0.0, 0.0, 0.0);

    for (Index j = 0; j < dimension; ++j) {
      point(j) = coordinates[node * dimension + j];
    }

    nodes_.insert(std::make_pair(node, point));
  }

  // Build connectivity array.
  // Assume that local numbering of elements is contiguous.
  // Ignore extra nodes in higher-order elements
  ArrayRCP<ArrayRCP<int> >::size_type
  element_number = 0;

  ArrayRCP<ArrayRCP<ArrayRCP<int> > >::size_type
  workset = 0;

  for (workset = 0; workset < element_connectivity.size(); ++workset) {

    for (ArrayRCP<ArrayRCP<int> >::size_type
        cell = 0;
        cell < element_connectivity[workset].size();
        ++cell, ++element_number) {

      IDList
      nodes_element(nodes_per_element);

      for (ArrayRCP<int>::size_type
          node = 0;
          node < vertices_per_element;
          ++node) {

        // Get node ID from first unknown ID by dividing by stride
        nodes_element[node] =
            element_connectivity[workset][cell][node][0] / stride;

      }

      connectivity_.insert(std::make_pair(element_number, nodes_element));

    }

  }

  return;
}


//
// \return Number of nodes on the array
//
Index
ConnectivityArray::GetNumberNodes() const
{
  return nodes_.size();
}

//
// \return Number of elements in the array
//
Index
ConnectivityArray::GetNumberElements() const
{
  return connectivity_.size();
}

//
// \return Space dimension
//
Index
ConnectivityArray::GetDimension() const
{
  return dimension_;
}

//
// \return K-means tolerance
//
double
ConnectivityArray::GetTolerance() const
{
  return tolerance_;
}

//
// \return requested cell size for voxelization
//
double
ConnectivityArray::GetCellSize() const
{
  return requested_cell_size_;
}

//
// \return maximum iterations for K-means
//
Index
ConnectivityArray::GetMaximumIterations() const
{
  return maximum_iterations_;
}

//
// \param K-means tolerance
//
void
ConnectivityArray::SetTolerance(double tolerance)
{
  tolerance_ = tolerance;
}

//
// \return requested cell size for voxelization
//
void
ConnectivityArray::SetCellSize(double requested_cell_size)
{
  requested_cell_size_ = requested_cell_size;
}

//
// \param maximum itearions for K-means
//
void
ConnectivityArray::SetMaximumIterations(Index maximum_iterarions)
{
  maximum_iterations_ = maximum_iterarions;
}

//
// \param Initializer scheme
//
void
ConnectivityArray::SetInitializerScheme(PARTITION::Scheme initializer_scheme)
{
  initializer_scheme_ = initializer_scheme;
}

//
// \return Initializer scheme
//
PARTITION::Scheme
ConnectivityArray::GetInitializerScheme() const
{
  return initializer_scheme_;
}

//
// \return Type of finite element in the array
// (assume same type for all elements)
//
Intrepid::ELEMENT::Type
ConnectivityArray::GetType() const
{
  return type_;
}

//
// \return Node ID and associated point in space
//
PointMap
ConnectivityArray::GetNodeList() const
{
  return nodes_;
}

//
// \return Element - nodes connectivity
//
AdjacencyMap
ConnectivityArray::GetConnectivity() const
{
  return connectivity_;
}

//
// \return Albany abstract discretization corresponding to array
//
Albany::AbstractDiscretization &
ConnectivityArray::GetDiscretization()
{
  return (*discretization_ptr_.get());
}

//
// \return Number of nodes that define element topology
// (assume same type for all elements)
//
Index
ConnectivityArray::GetNodesPerElement() const
{
  Index
  nodes_per_element;

  switch (GetType()) {

  default:
    std::cerr << "ERROR: Unknown element type in GetNodesPerElement()";
    std::cerr << '\n';
    exit(1);
    break;

  case Intrepid::ELEMENT::SEGMENTAL:
    nodes_per_element = 2;
    break;

  case Intrepid::ELEMENT::TRIANGULAR:
    nodes_per_element = 3;
    break;

  case Intrepid::ELEMENT::QUADRILATERAL:
    nodes_per_element = 4;
    break;

  case Intrepid::ELEMENT::TETRAHEDRAL:
    nodes_per_element = 4;
    break;

  case Intrepid::ELEMENT::HEXAHEDRAL:
    nodes_per_element = 8;
    break;

  }

  return nodes_per_element;
}


//
// \return Volume for each element
//
ScalarMap
ConnectivityArray::GetVolumes() const
{
  ScalarMap volumes;
  for (AdjacencyMap::const_iterator
      elements_iter = connectivity_.begin();
      elements_iter != connectivity_.end();
      ++elements_iter) {

    int const &
    element = (*elements_iter).first;

    IDList const &
    node_list = (*elements_iter).second;

    std::vector< Vector<double> >
    points;

    for (IDList::size_type
        i = 0;
        i < node_list.size();
        ++i) {

      PointMap::const_iterator
      nodes_iter = nodes_.find(node_list[i]);

      assert(nodes_iter != nodes_.end());
      points.push_back((*nodes_iter).second);

    }

    double volume = 0.0;

    switch (type_) {

    case Intrepid::ELEMENT::SEGMENTAL:
      volume = Intrepid::length(points[0], points[1]);
      break;

    case Intrepid::ELEMENT::TRIANGULAR:
      volume = Intrepid::area(points[0], points[1], points[2]);
      break;

    case Intrepid::ELEMENT::QUADRILATERAL:
      volume = Intrepid::area(points[0], points[1], points[2], points[3]);
      break;

    case Intrepid::ELEMENT::TETRAHEDRAL:
      volume = Intrepid::volume(points[0], points[1], points[2], points[3]);
      break;

    case Intrepid::ELEMENT::HEXAHEDRAL:
      volume = Intrepid::volume(points[0], points[1], points[2], points[3],
          points[4], points[5], points[6], points[7]);
      break;

    default:
      std::cerr << "Unknown element type in calculating volume." << '\n';
      exit(1);
      break;

    }

    volumes.insert(std::make_pair(element, volume));

  }

  return volumes;
}

//
// \return Total volume of the array
//
double
ConnectivityArray::GetVolume() const
{
  double volume = 0.0;

  const ScalarMap
  volumes = GetVolumes();

  for (ScalarMap::const_iterator
      volumes_iter = volumes.begin();
      volumes_iter != volumes.end();
      ++volumes_iter) {

    volume += (*volumes_iter).second;

  }

  return volume;
}

//
// \return Partitions when partitioned
//
std::map<int, int>
ConnectivityArray::GetPartitions() const
{
  return partitions_;
}

//
// \return Volume for each partition when partitioned
//
ScalarMap
ConnectivityArray::GetPartitionVolumes() const
{
  std::map<int, int>
  partitions = GetPartitions();

  ScalarMap
  volumes = GetVolumes();

  ScalarMap
  partition_volumes;

  for (std::map<int, int>::const_iterator part_iter = partitions.begin();
      part_iter != partitions.end();
      ++part_iter) {

    int element = (*part_iter).first;
    int partition = (*part_iter).second;

    ScalarMap::const_iterator
    volumes_iterator = volumes.find(element);

    if (volumes_iterator == volumes.end()) {
      std::cerr << "Cannot find volume for element " << element << '\n';
      exit(1);
    }

    double volume = (*volumes_iterator).second;

    ScalarMap::const_iterator
    partition_volumes_iter = partition_volumes.find(partition);

    if (partition_volumes_iter == partition_volumes.end()) {
      partition_volumes[partition] = volume;
    } else {
      partition_volumes[partition] += volume;
    }

  }

  return partition_volumes;
}

//
// \return Partition centroids
//
std::vector< Vector<double> >
ConnectivityArray::GetPartitionCentroids() const
{
  std::map<int, int>
  partitions = GetPartitions();

  ScalarMap
  element_volumes = GetVolumes();

  ScalarMap
  partition_volumes = GetPartitionVolumes();

  Index const
  number_partitions = partition_volumes.size();

  std::vector< Vector<double> >
  partition_centroids(number_partitions);

  for (Index i = 0; i < number_partitions; ++i) {
    partition_centroids[i].set_dimension(GetDimension());
    partition_centroids[i].clear();
  }

  // Determine number of nodes that define element topology
  Index const
  nodes_per_element = GetNodesPerElement();

  for (std::map<int, int>::const_iterator partitions_iterator =
      partitions.begin();
      partitions_iterator != partitions.end();
      ++partitions_iterator) {

    int
    element = (*partitions_iterator).first;

    int
    partition = (*partitions_iterator).second;

    AdjacencyMap::const_iterator
    elements_iterator = connectivity_.find(element);

    if (elements_iterator == connectivity_.end()) {
      std::cerr << "Cannot find element in partition centroids." << element;
      std::cerr << '\n';
      exit(1);
    }

    IDList const &
    node_list = (*elements_iterator).second;

    std::vector< Vector<double> >
    element_nodes;

    for (IDList::size_type i = 0; i < nodes_per_element; ++i) {

      PointMap::const_iterator
      nodes_iterator = nodes_.find(node_list[i]);

      assert(nodes_iterator != nodes_.end());

      element_nodes.push_back((*nodes_iterator).second);

    }

    Vector<double> const
    element_centroid = centroid(element_nodes);

    ScalarMap::const_iterator
    volumes_iterator = element_volumes.find(element);

    if (volumes_iterator == element_volumes.end()) {
      std::cerr << "Cannot find volume for element " << element;
      std::cerr << '\n';
      exit(1);
    }

    double
    element_volume = (*volumes_iterator).second;

    partition_centroids[partition] += element_volume * element_centroid;

  }

  for (Index i = 0; i < number_partitions; ++i) {
    partition_centroids[i] = partition_centroids[i] / partition_volumes[i];
  }

  return partition_centroids;
}

//
// \return Centroids for each element
//
PointMap
ConnectivityArray::GetCentroids() const
{
  PointMap
  centroids;

  for (AdjacencyMap::const_iterator
      elements_iter = connectivity_.begin();
      elements_iter != connectivity_.end();
      ++elements_iter) {

    // Get an element
    int const &
    element = (*elements_iter).first;

    IDList const &
    node_list = (*elements_iter).second;

    std::vector< Vector<double> >
    points;

    // Collect element nodes
    for (IDList::size_type
        i = 0;
        i < node_list.size();
        ++i) {

      int const
      node = node_list[i];

      PointMap::const_iterator
      nodes_iter = nodes_.find(node);

      assert(nodes_iter != nodes_.end());

      Vector<double> const
      point = (*nodes_iter).second;

      points.push_back(point);

    }

    Vector<double> const
    centroid = Intrepid::centroid(points);

    centroids.insert(std::make_pair(element, centroid));

  }

  return centroids;

}

///
/// \return Bounding box for all nodes
///
std::pair<Vector<double>, Vector<double> >
ConnectivityArray::BoundingBox() const
{
  PointMap::const_iterator
  it = nodes_.begin();

  Vector<double>
  min = (*it).second;

  Vector<double>
  max = min;

  Index const
  N = min.get_dimension();

  ++it;

  for (; it != nodes_.end(); ++it) {

    Vector<double> const &
    node = (*it).second;

    for (Index i = 0; i < N; ++i) {
      min(i) = std::min(min(i), node(i));
      max(i) = std::max(max(i), node(i));
    }

  }

  return std::make_pair(min, max);
}

namespace {

boost::tuple<Index, double, double>
parametric_limits(Intrepid::ELEMENT::Type const element_type)
{
  Index
  parametric_dimension = 3;

  double
  parametric_size = 1.0;

  double
  lower_limit = 0.0;

  switch (element_type) {

  default:
    std::cerr << "ERROR: Unknown element type in paramtetric_limits";
    std::cerr << '\n';
    exit(1);
    break;

  case Intrepid::ELEMENT::TRIANGULAR:
    lower_limit = 0.0;
    parametric_size = 1.0;
    parametric_dimension = 3;
    break;

  case Intrepid::ELEMENT::QUADRILATERAL:
    lower_limit = -1.0;
    parametric_size = 2.0;
    parametric_dimension = 2;
    break;

  case Intrepid::ELEMENT::TETRAHEDRAL:
    lower_limit = 0.0;
    parametric_size = 1.0;
    parametric_dimension = 4;
    break;

  case Intrepid::ELEMENT::HEXAHEDRAL:
    lower_limit = -1.0;
    parametric_size = 2.0;
    parametric_dimension = 3;
    break;

  }

  return boost::make_tuple(
      parametric_dimension,
      parametric_size,
      lower_limit);
}

} // namespace anonymous

//
// Background of the domain for fast determination
// of points being inside or outside the domain.
// \return points inside the domain.
//
std::vector< Vector<double> >
ConnectivityArray::CreateGrid()
{
  std::cout << '\n';
  std::cout << "Creating background mesh ..." << '\n';

  //
  // First determine the maximum dimension of the bounding box.
  //
  Vector<double>
  lower_corner;

  Vector<double>
  upper_corner;

  boost::tie(lower_corner, upper_corner) = BoundingBox();

  Vector<double> const
  bounding_box_span = upper_corner - lower_corner;

  Index const
  dimension = lower_corner.get_dimension();

  double
  maximum_dimension = 0.0;

  for (Index i = 0; i < dimension; ++i) {
    maximum_dimension = std::max(maximum_dimension, bounding_box_span(i));
  }

  double const
  delta = GetCellSize();

  //
  // Determine number of cells for each dimension.
  //
  Vector<Index>
  cells_per_dimension(dimension);

  cell_size_.set_dimension(dimension);

  for (Index i = 0; i < dimension; ++i) {

    Index const
    number_cells = std::ceil((bounding_box_span(i)) / delta);

    cells_per_dimension(i) = number_cells;
    cell_size_(i) = bounding_box_span(i) / number_cells;

  }

  //
  // Set up the cell array.
  // Generalization to N dimensions fails here.
  // This is specific to 3D.
  //
  cells_.resize(cells_per_dimension(0));
  for (Index i = 0; i < cells_per_dimension(0); ++i) {
    cells_[i].resize(cells_per_dimension(1));
    for (Index j = 0; j < cells_per_dimension(1); ++j) {
      cells_[i][j].resize(cells_per_dimension(2));
      for (Index k = 0; k < cells_per_dimension(2); ++k) {
        cells_[i][j][k] = false;
      }
    }
  }

  // Iterate through elements to set array.
  Index const
  nodes_per_element = GetNodesPerElement();

  Index const
  number_of_elements = connectivity_.size();

  for (AdjacencyMap::const_iterator
      elements_iter = connectivity_.begin();
      elements_iter != connectivity_.end();
      ++elements_iter) {

    int const
    element = (*elements_iter).first;

    if ((element + 1) % 10000 == 0) {
      std::cout << "Processing element: " << element + 1;
      std::cout << "/" << number_of_elements << '\n';
    }

    IDList const &
    node_list = (*elements_iter).second;

    std::vector< Vector<double> >
    element_nodes;

    for (IDList::size_type i = 0;
        i < nodes_per_element;
        ++i) {

      PointMap::const_iterator
      nodes_iter = nodes_.find(node_list[i]);

      assert(nodes_iter != nodes_.end());

      element_nodes.push_back((*nodes_iter).second);

    }

    Vector<double>
    min;

    Vector<double>
    max;

    boost::tie(min, max) =
        Intrepid::bounding_box<double>(element_nodes.begin(),
            element_nodes.end());

    Vector<double> const
    element_span = max - min;

    Vector<Index>
    divisions(dimension);

    // Determine number of divisions on each dimension.
    // One division if voxel is large.
    for (Index i = 0; i < dimension; ++i) {
      divisions(i) =
          cell_size_(i) > element_span(i) ?
              1 :
              2.0 * element_span(i) / cell_size_(i) + 0.5;
    }

    // Generate points inside the element according to
    // the divisions and mark the corresponding voxel
    // as being inside the domain.
    Intrepid::ELEMENT::Type
    element_type = GetType();

    Index
    parametric_dimension = 3;

    double
    parametric_size = 1.0;

    double
    lower_limit = 0.0;

    boost::tie(parametric_dimension, parametric_size, lower_limit) =
        parametric_limits(element_type);

    Vector<double>
    origin(parametric_dimension);

    for (Index i = 0; i < dimension; ++i) {
      origin(i) = lower_limit;
    }

    Vector<double>
    xi(parametric_dimension);

    for (Index i = 0; i <= divisions(0); ++i) {
      xi(0) = origin(0) + double(i) / divisions(0) * parametric_size;
      for (Index j = 0; j <= divisions(1); ++j) {
        xi(1) = origin(1) + double(j) / divisions(1) * parametric_size;
        for (Index k = 0; k <= divisions(2); ++k) {
          xi(2) = origin(2) + double(k) / divisions(2) * parametric_size;
          Vector<double>
          p = interpolate_element(element_type, xi, element_nodes);
          for (Index l = 0; l < dimension; ++l) {
            p(l) = std::max(p(l), lower_corner(l));
            p(l) = std::min(p(l), upper_corner(l));
          }

          Vector<int>
          index = PointToIndex(p);

          for (Index l = 0; l < dimension; ++l) {
            assert(index(l) >= 0);
            assert(index(l) <= int(cells_per_dimension(l)));

            if (index(l) == int(cells_per_dimension(l))) {
              --index(l);
            }
          }

          cells_[index(0)][index(1)][index(2)] = true;
        }
      }
    }
  }

  std::cout << connectivity_.size() << " elements processed." << '\n';

  // Create points and output voxelization for debugging
  std::vector< Vector<double> >
  domain_points;

  std::ofstream ofs("cells.csv");
  ofs << "X, Y, Z, I" << '\n';
  Vector<double> p(dimension);

  for (Index i = 0; i < cells_per_dimension(0); ++i) {
    p(0) = (i + 0.5) * bounding_box_span(0) / cells_per_dimension(0) +
        lower_corner(0);
    for (Index j = 0; j < cells_per_dimension(1); ++j) {
      p(1) = (j + 0.5) * bounding_box_span(1) / cells_per_dimension(1) +
          lower_corner(1);
      for (Index k = 0; k < cells_per_dimension(2); ++k) {
        p(2) = (k + 0.5) * bounding_box_span(2) / cells_per_dimension(2) +
            lower_corner(2);

        if (cells_[i][j][k] == true) {
          domain_points.push_back(p);
        }

        ofs << p << "," << cells_[i][j][k] << '\n';
      }
    }
  }

  Index const
  number_generated_points =
      cells_per_dimension(0) *
      cells_per_dimension(1) *
      cells_per_dimension(2);

  Index const
  number_points_in_domain = domain_points.size();

  double const
  ratio = double(number_points_in_domain) / double(number_generated_points);

  std::cout << "Number of cells inside domain: ";
  std::cout << number_points_in_domain;
  std::cout << '\n';
  std::cout << "Number of generated cells    : ";
  std::cout << number_generated_points;
  std::cout << '\n';
  std::cout << "Ratio                        : ";
  std::cout << ratio;
  std::cout << '\n';

  return domain_points;
}

//
// Convert point to index into voxel array
//
Vector<int>
ConnectivityArray::PointToIndex(Vector<double> const & point) const
{
  int const
  i = (point(0) - lower_corner_(0)) / cell_size_(0);

  int const
  j = (point(1) - lower_corner_(1)) / cell_size_(1);

  int const
  k = (point(2) - lower_corner_(2)) / cell_size_(2);

  return Vector<int>(i, j, k);
}

//
// Determine if a given point is inside the mesh.
// 3D only for now.
//
bool
ConnectivityArray::IsInsideMesh(Vector<double> const & point) const
{
  int
  i = (point(0) - lower_corner_(0)) / cell_size_(0);

  int
  j = (point(1) - lower_corner_(1)) / cell_size_(1);

  int
  k = (point(2) - lower_corner_(2)) / cell_size_(2);


  int const
  x_size = cells_.size();

  int const
  y_size = cells_[0].size();

  int const
  z_size = cells_[0][0].size();


  if (i < 0 || i > x_size) {
    return false;
  }

  if (j < 0 || j > y_size) {
    return false;
  }

  if (k < 0 || k > z_size) {
    return false;
  }


  if (i == x_size) --i;
  if (j == y_size) --j;
  if (k == z_size) --k;

  return cells_[i][j][k];
}

//
// Determine is a given point is inside the mesh
// doing it element by element. Slow but useful
// to set up an initial data structure that will
// be used on a faster method.
//
bool
ConnectivityArray::IsInsideMeshByElement(Vector<double> const & point) const
{

  // Check bounding box first
  if (in_box(point, lower_corner_, upper_corner_) == false) {
    return false;
  }

  // Now check element by element
  for (AdjacencyMap::const_iterator
      elements_iter = connectivity_.begin();
      elements_iter != connectivity_.end();
      ++elements_iter) {

    IDList const &
    node_list = (*elements_iter).second;

    std::vector< Vector<double> >
    node;

    for (IDList::size_type
        i = 0;
        i < node_list.size();
        ++i) {

      PointMap::const_iterator
      nodes_iter = nodes_.find(node_list[i]);

      assert(nodes_iter != nodes_.end());
      node.push_back((*nodes_iter).second);

    }

    switch (type_) {

    case Intrepid::ELEMENT::TETRAHEDRAL:
      if (in_tetrahedron(point, node[0], node[1], node[2], node[3]) == true) {
        return true;
      }
      break;

    case Intrepid::ELEMENT::HEXAHEDRAL:
      if (in_hexahedron(point, node[0], node[1], node[2], node[3],
          node[4], node[5], node[6], node[7])) {
        return true;
      }
      break;

    default:
      std::cerr << "Unknown element type in K-means partition." << '\n';
      exit(1);
      break;

    }

  }

  return false;
}

//
// \param length_scale Length scale for partitioning for
// variational non-local regularization
// \return Number of partitions defined as total volume
// of the array divided by the cube of the length scale
//
Index
ConnectivityArray::GetNumberPartitions(double const length_scale) const
{
  double const
  ball_volume = length_scale * length_scale * length_scale;

  Index const
  number_partitions =
      static_cast<Index>(round(GetVolume() / ball_volume));

  return number_partitions;
}

//
// Anonymous namespace for helper functions
//
namespace {

//
// Helper function that return a deterministic pseudo random
// sequence 0,..,N-1 for visualization purposes.
//
std::vector<int>
shuffled_sequence(int number_elements)
{
  std::vector<int>
  shuffled;

  std::vector<int>
  unshuffled;

  for (int i = 0; i < number_elements; ++i) {
    unshuffled.push_back(i);
  }

  // Happens to be a Mersenne prime for int_32
  const int
  prime = std::numeric_limits<int>::max();

  for (int i = 0; i < number_elements; ++i) {
    const int
    remainder = number_elements - i;

    const int
    index = prime % remainder;

    const int
    selection = unshuffled[index];

    shuffled.push_back(selection);

    std::vector<int>::iterator
    delete_position = unshuffled.begin();

    std::advance(delete_position, index);

    unshuffled.erase(delete_position);
  }

  return shuffled;
}

//
// Helper function to renumber partitions to avoid gaps in numbering.
// Also for better color contrast in visualization programs, shuffle
// the partition number so that it is less likely that partitions
// with very close numbers are next to each other, leading to almost
// the same color in output.
//
std::map<int, int>
RenumberPartitions(std::map<int, int> const & old_partitions) {

  std::set<int>
  partitions_set;

  for (std::map<int, int>::const_iterator it = old_partitions.begin();
      it != old_partitions.end();
      ++it) {

    int const
    partition = (*it).second;

    partitions_set.insert(partition);
  }

  std::set<int>::size_type
  number_partitions = partitions_set.size();

  std::vector<int>
  partition_shuffle = shuffled_sequence(number_partitions);

  std::map<int, int>
  partition_map;

  int
  partition_index = 0;

  for (std::set<int>::const_iterator it = partitions_set.begin();
      it != partitions_set.end();
      ++it) {

    int const
    partition = (*it);

    partition_map[partition] = partition_index;

    ++partition_index;

  }

  std::map<int, int>
  new_partitions;

  for (std::map<int, int>::const_iterator it = old_partitions.begin();
      it != old_partitions.end();
      ++it) {

    int const
    element = (*it).first;

    int const
    old_partition = (*it).second;

    int const
    partition_index = partition_map[old_partition];

    int const
    new_partition = partition_shuffle[partition_index];

    new_partitions[element] = new_partition;

  }

  return new_partitions;
}

} // anonymous namespace

void
ConnectivityArray::CheckNullVolume() const
{
  ScalarMap const
  partition_volumes = GetPartitionVolumes();

  std::vector<Index>
  zero_volume;

  for (ScalarMap::const_iterator it = partition_volumes.begin();
      it != partition_volumes.end();
      ++it) {

    Index const
    partition = (*it).first;

    double const
    volume = (*it).second;

    if (volume == 0.0) {
      zero_volume.push_back(partition);
    }

  }

  Index const
  number_null_partitions = zero_volume.size();

  if (number_null_partitions > 0) {
    std::cerr << "ERROR: The following partitions have zero volume.";
    std::cerr << '\n';
    std::cerr << "Length scale may be too small:";
    std::cerr << '\n';

    for (Index i = 0; i < number_null_partitions; ++i) {
      std::cerr << " " << zero_volume[i];
    }

    std::cerr << '\n';

    exit(1);
  }

  return;
}

//
// Partition mesh according to the specified algorithm and length scale
// \param partition_scheme The partition algorithm to use
// \param length_scale The length scale for variational nonlocal
// regularization
// \return Partition number for each element
//
std::map<int, int>
ConnectivityArray::Partition(
    const PARTITION::Scheme partition_scheme,
    double const length_scale)
{

  std::map<int, int>
  partitions;

  switch (partition_scheme) {

  case PARTITION::RANDOM:
    partitions = PartitionRandom(length_scale);
    break;

  case PARTITION::HYPERGRAPH:
    partitions = PartitionHyperGraph(length_scale);
    break;

  case PARTITION::GEOMETRIC:
    partitions = PartitionGeometric(length_scale);
    break;

  case PARTITION::KMEANS:
    partitions = PartitionKMeans(length_scale);
    break;

  case PARTITION::SEQUENTIAL:
    partitions = PartitionSequential(length_scale);
    break;

  case PARTITION::KDTREE:
    partitions = PartitionKDTree(length_scale);
    break;

  default:
    std::cerr << "Unknown partitioning scheme." << '\n';
    exit(1);
    break;

  }

  CheckNullVolume();

  // Store for use by other methods
  partitions_ = RenumberPartitions(partitions);

  return partitions_;

}

//
// \param Collection of centers
// \return Partition map that assigns each element to the
// closest center to its centroid
//
std::map<int, int>
ConnectivityArray::PartitionByCenters(
    std::vector< Vector<double> > const & centers)
{
  Index const
  number_partitions = centers.size();

  // Partition map.
  std::map<int, int>
  partitions;

  // Keep track of which partitions have been assigned elements.
  std::set<Index>
  unassigned_partitions;

  for (Index partition = 0; partition < number_partitions; ++partition) {
    unassigned_partitions.insert(partition);
  }

  // Determine number of nodes that define element topology
  Index const
  nodes_per_element = GetNodesPerElement();

  std::ofstream centroids_ofs("centroids.csv");

  centroids_ofs << "X,Y,Z" << '\n';
  for (AdjacencyMap::const_iterator
      elements_iter = connectivity_.begin();
      elements_iter != connectivity_.end();
      ++elements_iter) {

    int const &
    element = (*elements_iter).first;

    IDList const &
    node_list = (*elements_iter).second;

    std::vector< Vector<double> >
    element_nodes;

    for (IDList::size_type i = 0; i < nodes_per_element; ++i) {

      PointMap::const_iterator
      nodes_iter = nodes_.find(node_list[i]);

      assert(nodes_iter != nodes_.end());

      element_nodes.push_back((*nodes_iter).second);

    }

    Vector<double> const
    element_centroid = centroid(element_nodes);

    centroids_ofs << element_centroid << '\n';

    Index const
    partition = closest_point(element_centroid, centers);

    partitions[element] = partition;

    std::set<Index>::const_iterator
    it = unassigned_partitions.find(partition);

    if (it != unassigned_partitions.end()) {
      unassigned_partitions.erase(it);
    }

  }

  if (unassigned_partitions.size() > 0) {
    std::cout << "WARNING: The following partitions were not" << '\n';
    std::cout << "assigned any elements (mesh too coarse?):" << '\n';

    for (std::set<Index>::const_iterator it = unassigned_partitions.begin();
        it != unassigned_partitions.end();
        ++it) {
      std::cout << (*it) << '\n';
    }

  }

  std::ofstream generators_ofs("centers.csv");
  generators_ofs << "X,Y,Z" << '\n';
  for (Index i = 0; i < centers.size(); ++i) {
    generators_ofs << centers[i] << '\n';
  }

  return partitions;

}

//
// Partition mesh with Zoltan Hypergraph algortithm
// \param length_scale The length scale for variational nonlocal
// regularization
// \return Partition number for each element
//
std::map<int, int>
ConnectivityArray::PartitionHyperGraph(double const length_scale)
{
  // Zoltan setup
  int const
  number_partitions = GetNumberPartitions(length_scale);

  std::stringstream
  ioss;

  ioss << number_partitions;

  std::string
  zoltan_number_parts;

  ioss >> zoltan_number_parts;

  Zoltan
  zoltan(MPI_COMM_SELF);

  zoltan.Set_Param("LB_METHOD", "HYPERGRAPH");
  zoltan.Set_Param("LB_APPROACH", "PARTITION");
  zoltan.Set_Param("DEBUG_LEVEL", "0");
  zoltan.Set_Param("OBJ_WEIGHT_DIM", "1");
  zoltan.Set_Param("NUM_GLOBAL_PARTS", zoltan_number_parts.c_str());
  zoltan.Set_Param("REMAP", "0");
  zoltan.Set_Param("HYPERGRAPH_PACKAGE", "PHG");
  zoltan.Set_Param("PHG_MULTILEVEL", "1");
  zoltan.Set_Param("PHG_EDGE_WEIGHT_OPERATION", "ERROR");
  zoltan.Set_Param("IMBALANCE_TOL", "1.01");
  zoltan.Set_Param("PHG_CUT_OBJECTIVE", "HYPEREDGES");

  //
  // Partition
  //
  DualGraph
  dual_graph(*this);

  ZoltanHyperGraph
  zoltan_hypergraph(dual_graph);

  // Set up hypergraph
  zoltan.Set_Num_Obj_Fn(
      LCM::ZoltanHyperGraph::GetNumberOfObjects,
      &zoltan_hypergraph);

  zoltan.Set_Obj_List_Fn(
      LCM::ZoltanHyperGraph::GetObjectList,
      &zoltan_hypergraph);

  zoltan.Set_HG_Size_CS_Fn(
      LCM::ZoltanHyperGraph::GetHyperGraphSize,
      &zoltan_hypergraph);

  zoltan.Set_HG_CS_Fn(
      LCM::ZoltanHyperGraph::GetHyperGraph,
      &zoltan_hypergraph);

  int changes;
  int num_gid_entries;
  int num_lid_entries;
  int num_import;
  ZOLTAN_ID_PTR import_global_ids;
  ZOLTAN_ID_PTR import_local_ids;
  int* import_procs;
  int* import_to_part;
  int num_export;
  ZOLTAN_ID_PTR export_global_ids;
  ZOLTAN_ID_PTR export_local_ids;
  int* export_procs;
  int* export_to_part;

  int rc =
      zoltan.LB_Partition(
          changes,
          num_gid_entries,
          num_lid_entries,
          num_import,
          import_global_ids,
          import_local_ids,
          import_procs,
          import_to_part,
          num_export,
          export_global_ids,
          export_local_ids,
          export_procs,
          export_to_part);

  if (rc != ZOLTAN_OK) {
    std::cerr << "Partitioning failed" << '\n';
    exit(1);
  }

  // Set up partition map initializing all partitions to zero
  std::map<int, int> partitions;

  // Initialize with zeros the partition map for all elements.
  const ScalarMap
  vertex_weights = zoltan_hypergraph.GetVertexWeights();

  // Fill up with results from Zoltan, which returns partitions for all
  // elements that belong to a partition > 0
  for (ScalarMap::const_iterator
      weights_iter = vertex_weights.begin();
      weights_iter != vertex_weights.end();
      ++weights_iter) {
    int const vertex = (*weights_iter).first;
    partitions[vertex] = 0;
  }

  // Fill up with results from Zoltan
  for (int i = 0; i < num_import; ++i) {
    int const vertex = static_cast<int>(import_local_ids[i]);
    partitions[vertex] = import_to_part[i];
  }

  // cleanup zoltan pointers
  // this will free all memory associated with the in and output data
  // to zoltan
  zoltan.LB_Free_Part(
      &import_global_ids,
      &import_local_ids,
      &import_procs,
      &import_to_part );

  zoltan.LB_Free_Part(
      &export_global_ids,
      &export_local_ids,
      &export_procs,
      &export_to_part );

  return partitions;

}

//
/// Partition mesh with Zoltan Recursive Inertial Bisection algortithm
// \param length_scale The length scale for variational nonlocal
// regularization
// \return Partition number for each element
//
std::map<int, int>
ConnectivityArray::PartitionGeometric(double const length_scale)
{
  // Zoltan setup
  int const
  number_partitions = GetNumberPartitions(length_scale);

  std::stringstream
  ioss;

  ioss << number_partitions;

  std::string
  zoltan_number_parts;

  ioss >> zoltan_number_parts;

  Zoltan
  zoltan(MPI_COMM_SELF);

  zoltan.Set_Param("LB_METHOD", "RCB");
  zoltan.Set_Param("RCB_RECOMPUTE_BOX", "1");
  zoltan.Set_Param("LB_APPROACH", "PARTITION");
  zoltan.Set_Param("DEBUG_LEVEL", "0");
  zoltan.Set_Param("OBJ_WEIGHT_DIM", "1");
  zoltan.Set_Param("NUM_GLOBAL_PARTS", zoltan_number_parts.c_str());
  zoltan.Set_Param("REMAP", "0");
  zoltan.Set_Param("IMBALANCE_TOL", "1.10");
  zoltan.Set_Param("CHECK_GEOM", "1");
  zoltan.Set_Param("AVERAGE_CUTS", "1");
  zoltan.Set_Param("REDUCE_DIMENSIONS", "1");
  zoltan.Set_Param("DEGENERATE_RATIO", "10");

  //
  // Partition
  //

  // Set up recursive inertial bisection (RIB)
  zoltan.Set_Num_Obj_Fn(LCM::ConnectivityArray::GetNumberOfObjects, this);
  zoltan.Set_Obj_List_Fn(LCM::ConnectivityArray::GetObjectList, this);
  zoltan.Set_Num_Geom_Fn(LCM::ConnectivityArray::GetNumberGeometry, this);
  zoltan.Set_Geom_Multi_Fn(LCM::ConnectivityArray::GetGeometry, this);

  int changes;
  int num_gid_entries;
  int num_lid_entries;
  int num_import;
  ZOLTAN_ID_PTR import_global_ids;
  ZOLTAN_ID_PTR import_local_ids;
  int* import_procs;
  int* import_to_part;
  int num_export;
  ZOLTAN_ID_PTR export_global_ids;
  ZOLTAN_ID_PTR export_local_ids;
  int* export_procs;
  int* export_to_part;

  int rc =
      zoltan.LB_Partition(
          changes,
          num_gid_entries,
          num_lid_entries,
          num_import,
          import_global_ids,
          import_local_ids,
          import_procs,
          import_to_part,
          num_export,
          export_global_ids,
          export_local_ids,
          export_procs,
          export_to_part);

  if (rc != ZOLTAN_OK) {
    std::cerr << "Partitioning failed" << '\n';
    exit(1);
  }

  // Set up partition map initializing all partitions to zero
  std::map<int, int> partitions;

  const ScalarMap
  element_volumes = GetVolumes();

  // Initialize with zeros the partition map for all elements.
  for (ScalarMap::const_iterator
      volumes_iter = element_volumes.begin();
      volumes_iter != element_volumes.end();
      ++volumes_iter) {
    int const element = (*volumes_iter).first;
    partitions[element] = 0;
  }

  // Fill up with results from Zoltan, which returns partitions for all
  // elements that belong to a partition > 0
  for (int i = 0; i < num_import; ++i) {
    int const element = static_cast<int>(import_local_ids[i]);
    partitions[element] = import_to_part[i];
  }

  return partitions;

}

//
/// Partition mesh with K-means algortithm
// \param length_scale The length scale for variational nonlocal
// regularization
// \return Partition number for each element
//
std::map<int, int>
ConnectivityArray::PartitionKMeans(double const length_scale)
{
  //
  // Create initial centers
  //
  std::cout << '\n';
  std::cout << "Partition with initializer ..." << '\n';

  // Partition with initializer
  PARTITION::Scheme const
  initializer_scheme = GetInitializerScheme();

  Partition(initializer_scheme, length_scale);

  // Compute partition centroids and use those as initial centers

  std::vector< Vector<double> >
  centers = GetPartitionCentroids();

  Index const
  number_partitions = centers.size();

  Vector<double>
  lower_corner;

  Vector<double>
  upper_corner;

  boost::tie(lower_corner, upper_corner) = BoundingBox();

  lower_corner_ = lower_corner;
  upper_corner_ = upper_corner;

  std::vector< Vector<double> >
  domain_points = CreateGrid();

  //
  // K-means iteration
  //
  std::cout << "Main K-means Iteration." << '\n';

  Index const
  max_iterations = GetMaximumIterations();

  Index
  number_iterations = 0;

  double const
  diagonal_distance = norm(upper_corner - lower_corner);

  double const
  tolerance = GetTolerance() * diagonal_distance;

  double
  step_norm = diagonal_distance;

  std::vector<double>
  steps(number_partitions);

  for (Index i = 0; i < number_partitions; ++i) {
    steps[i] = diagonal_distance;
  }

  Index const
  number_points = domain_points.size();

  while (step_norm >= tolerance && number_iterations < max_iterations) {

    // Assign points to closest generators
    std::vector<double>
    point_to_generator(number_points);

    for (Index i = 0; i < domain_points.size(); ++i) {
      point_to_generator[i] = closest_point(domain_points[i], centers);
    }

    // Determine cluster of points for each generator
    std::vector<std::vector<Vector<double> > >
    clusters;

    clusters.resize(number_partitions);

    for (Index p = 0; p < point_to_generator.size(); ++p) {

      Index const
      c = point_to_generator[p];

      clusters[c].push_back(domain_points[p]);

    }

    // Compute centroids of each cluster and set generators to
    // these centroids.
    for (Index i = 0; i < clusters.size(); ++i) {

      // If center is empty then generator does not move.
      if (clusters[i].size() == 0) {
        steps[i] = 0.0;
        std::cout << "Iteration: " << number_iterations;
        std::cout << ", center " << i << " has zero points." << '\n';
        continue;
      }

      Vector<double> const
      cluster_centroid = centroid(clusters[i]);

      // Update the generator
      Vector<double> const
      old_generator = centers[i];

      centers[i] = cluster_centroid;

      steps[i] = norm(centers[i] - old_generator);
    }

    step_norm = norm(Vector<double>(number_partitions, &steps[0]));

    std::cout << "Iteration: " << number_iterations;
    std::cout << ". Step: " << step_norm << ". Tol: " << tolerance;
    std::cout << '\n';

    ++number_iterations;

  }

  // Partition map.
  std::map<int, int>
  partitions = PartitionByCenters(centers);

  return partitions;
}

//
/// Partition mesh with K-means algortithm and triangle inequality
// \param length_scale The length scale for variational nonlocal
// regularization
// \return Partition number for each element
//
std::map<int, int>
ConnectivityArray::PartitionKDTree(double const length_scale)
{
  //
  // Create initial centers
  //
  std::cout << '\n';
  std::cout << "Partition with initializer ..." << '\n';

  // Partition with initializer
  PARTITION::Scheme const
  initializer_scheme = GetInitializerScheme();

  Partition(initializer_scheme, length_scale);

  // Compute partition centroids and use those as initial centers

  std::vector< Vector<double> >
  center_positions = GetPartitionCentroids();

  Index const
  number_partitions = center_positions.size();

  // Initialize centers
  std::cout << "Main K-means Iteration." << '\n';

  std::vector<ClusterCenter>
  centers(number_partitions);

  for (Index i = 0; i < number_partitions; ++i) {
    centers[i].position = center_positions[i];
    centers[i].weighted_centroid = 0.0 * center_positions[i];
  }

  Vector<double>
  lower_corner;

  Vector<double>
  upper_corner;

  boost::tie(lower_corner, upper_corner) = BoundingBox();

  lower_corner_ = lower_corner;
  upper_corner_ = upper_corner;

  std::vector< Vector<double> >
  domain_points = CreateGrid();

  //
  // Create KDTree
  //
  KDTree<KDTreeNode>
  kdtree(domain_points, number_partitions);

  //TraverseTree(kdtree, OutputVisitor<boost::shared_ptr<KDTreeNode> >());

  FilterVisitor<boost::shared_ptr<KDTreeNode>, ClusterCenter>
  filter_visitor(domain_points, centers);

  //
  // K-means iteration
  //
  Index const
  max_iterations = GetMaximumIterations();

  Index
  number_iterations = 0;

  double const
  diagonal_distance = norm(upper_corner - lower_corner);

  double const
  tolerance = GetTolerance() * GetCellSize();

  double
  step_norm = diagonal_distance;

  std::vector<double>
  steps(number_partitions);

  for (Index i = 0; i < number_partitions; ++i) {
    steps[i] = diagonal_distance;
  }

  while (step_norm >= tolerance && number_iterations < max_iterations) {

    // Initialize centers
    for (Index i = 0; i < number_partitions; ++i) {
      ClusterCenter &
      center = centers[i];

      center.weighted_centroid.clear();
      center.count = 0;
    }

    TraverseTree(kdtree, filter_visitor);

    // Update centers
    for (Index i = 0; i < centers.size(); ++i) {

      ClusterCenter &
      center = centers[i];

      // If cluster is empty then center does not move.
      if (center.count == 0) {
        steps[i] = 0.0;
        std::cout << "Iteration: " << number_iterations;
        std::cout << ", center " << i << " has zero points." << '\n';
        continue;
      }

      Vector<double> const
      new_position = center.weighted_centroid / center.count;

      steps[i] = norm(new_position - center.position);

      center.position = new_position;

    }

    step_norm = norm(Vector<double>(number_partitions, &steps[0]));

    std::cout << "Iteration: " << number_iterations;
    std::cout << ". Step: " << step_norm << ". Tol: " << tolerance;
    std::cout << '\n';

    ++number_iterations;

  }

  for (Index i = 0; i < number_partitions; i++) {
    center_positions[i] = centers[i].position;
  }

  // Partition map.
  std::map<int, int>
  partitions = PartitionByCenters(center_positions);

  return partitions;

}

//
// Partition mesh with sequential K-means algortithm
// \param length_scale The length scale for variational nonlocal
// regularization
// \return Partition number for each element
//
std::map<int, int>
ConnectivityArray::PartitionSequential(double const length_scale)
{
  int const
  number_partitions = GetNumberPartitions(length_scale);

  Vector<double>
  lower_corner;

  Vector<double>
  upper_corner;

  boost::tie(lower_corner, upper_corner) = BoundingBox();

  lower_corner_ = lower_corner;
  upper_corner_ = upper_corner;

  //
  // Create initial centers
  //

  // Partition with initializer
  const PARTITION::Scheme
  initializer_scheme = GetInitializerScheme();

  Partition(initializer_scheme, length_scale);

  // Compute partition centroids and use those as initial centers

  std::vector< Vector<double> >
  centers = GetPartitionCentroids();

  std::vector<Index>
  weights(number_partitions);

  for (int i = 0; i < number_partitions; ++i) {
    weights[i] = 1;
  }

  // K-means sequential iteration
  Index const
  number_random_points = GetMaximumIterations() * number_partitions;

  Index const
  max_iterations = number_random_points;

  Index
  number_iterations = 0;

  double const
  diagonal_distance = norm(upper_corner - lower_corner);

  double const
  tolerance = GetTolerance() * diagonal_distance;

  std::vector<double>
  steps(number_partitions);

  for (int i = 0; i < number_partitions; ++i) {
    steps[i] = diagonal_distance;
  }

  double
  step_norm = diagonal_distance;

  std::cout << "K-means Sequential." << '\n';

  while (step_norm >= tolerance && number_iterations < max_iterations) {

    // Create a random point, find closest generator
    bool
    is_point_in_domain = false;

    Vector<double>
    random_point(lower_corner.get_dimension());

    while (is_point_in_domain == false) {
      random_point = random_in_box(lower_corner, upper_corner);
      is_point_in_domain = IsInsideMesh(random_point);
    }

    // Determine index to closest generator
    Index const
    i = closest_point(random_point, centers);

    // Update the generator and the weight
    Vector<double> const
    old_generator = centers[i];

    centers[i] =
        (weights[i] * centers[i] + random_point) / (weights[i] + 1);

    weights[i] += 1;

    steps[i] = norm(centers[i] - old_generator);
    step_norm = norm(Vector<double>(number_partitions, &steps[0]));

    if (number_iterations % 10000 == 0) {
      std::cout << "Random point: " << number_iterations;
      std::cout << ". Step: " << step_norm << ". ";
      std::cout << "Tol: " << tolerance << '\n';
    }

    ++number_iterations;

  }

  std::cout << "Random point: " << number_iterations;
  std::cout << ". Step: " << step_norm << ". ";
  std::cout << "Tol: " << tolerance << '\n';

  // Partition map.
  std::map<int, int>
  partitions = PartitionByCenters(centers);

  return partitions;
}

//
// Partition mesh with randomly generated centers.
// Mostly used to initialize other schemes.
// \param length_scale The length scale for variational nonlocal
// regularization
// \return Partition number for each element
//
std::map<int, int>
ConnectivityArray::PartitionRandom(double const length_scale)
{
  int const
  number_partitions = GetNumberPartitions(length_scale);

  Vector<double>
  lower_corner;

  Vector<double>
  upper_corner;

  boost::tie(lower_corner, upper_corner) = BoundingBox();

  lower_corner_ = lower_corner;
  upper_corner_ = upper_corner;

  //
  // Create initial centers
  //
  int
  number_generators = 0;

  std::vector< Vector<double> >
  centers;

  while (number_generators < number_partitions) {

    Vector<double>
    p = random_in_box(lower_corner, upper_corner);

    if (IsInsideMesh(p) == true) {
      centers.push_back(p);
      ++number_generators;
    }

  }

  // Partition map.
  std::map<int, int>
  partitions = PartitionByCenters(centers);

  return partitions;
}

//
// Zoltan interface query function that returns the number of values
// needed to express the geometry of an object.
// For a three-dimensional object, the return value should be three.
//
// \param   data  Pointer to user-defined data.
//
// \param   ierr  Error code to be set by function.
//
// \return  The number of values needed to express the
// geometry of an object.
//
int
ConnectivityArray::GetNumberGeometry(
    void* data,
    int* ierr)
{

  ConnectivityArray &
  connectivity_array = *(static_cast<ConnectivityArray*>(data));

  *ierr = ZOLTAN_OK;

  int dimension = connectivity_array.GetDimension();

  return dimension;
}

//
// Zoltan interface, return number of objects
//
int
ConnectivityArray::GetNumberOfObjects(void* data, int* ierr)
{

  ConnectivityArray &
  connectivity_array = *(static_cast<ConnectivityArray*>(data));

  *ierr = ZOLTAN_OK;

  int num_objects = connectivity_array.GetConnectivity().size();

  return num_objects;
}

//
// Zoltan interface, return relevant object properties
//
void
ConnectivityArray::GetObjectList(
    void* data,
    int sizeGID,
    int sizeLID,
    ZOLTAN_ID_PTR globalID,
    ZOLTAN_ID_PTR localID,
    int wgt_dim,
    float* obj_wgts,
    int* ierr)
{

  ConnectivityArray &
  connectivity_array = *(static_cast<ConnectivityArray*>(data));

  *ierr = ZOLTAN_OK;

  ScalarMap
  element_volumes = connectivity_array.GetVolumes();

  ZOLTAN_ID_PTR
  global_id_ptr = globalID;

  ZOLTAN_ID_PTR
  local_id_ptr = localID;

  float*
  weight_ptr = obj_wgts;

  for (ScalarMap::const_iterator
      volumes_iter = element_volumes.begin();
      volumes_iter != element_volumes.end();
      ++volumes_iter) {

    int element = (*volumes_iter).first;
    double volume = (*volumes_iter).second;

    // Beware of this evil pointer manipulation
    (*global_id_ptr) = element;
    (*local_id_ptr) = element;
    (*weight_ptr) = volume;
    global_id_ptr++;
    local_id_ptr++;
    weight_ptr++;

  }

  return;
}

//
// Zoltan interface query function that returns a vector of geometry
// values for a list of given objects. The geometry vector is allocated
// by Zoltan to be of size num_obj * num_dim;
// its format is described below.
//
// \param data Pointer to user-defined data.
//
// \param sizeGID The number of array entries used to describe a
// single global ID.  This value is the maximum value over all processors
// of the parameter NUM_GID_ENTRIES.
//
// \param sizeLID The number of array entries used to describe a
// single local ID.  This value is the maximum value over all processors
// of the parameter NUM_LID_ENTRIES. (It should be zero if local ids
// are not used.)
//
// \param num_obj The number of object IDs in arrays
// globalID and localID
//
// \param globalID  Upon return, an array of unique global IDs for
// all objects assigned to the processor.
//
// \param localID Upon return, an array of local IDs, the meaning
// of which can be determined by the application, for all objects
// assigned to the processor. (Optional.)
//
// \param num_dim Number of coordinate entries per object
// (typically 1, 2, or 3).
//
// \param geom_vec  Upon return, an array containing geometry values.
// For object i (specified by globalID[i*sizeGID] and
// localID[i*sizeLID], i=0,1,...,num_obj-1),
// coordinate values should be stored in
// geom_vec[i*num_dim:(i+1)*num_dim-1].
//
// \param ierr Error code to be set by function.
//
void
ConnectivityArray::GetGeometry(
    void* data,
    int sizeGID,
    int sizeLID,
    int num_obj,
    ZOLTAN_ID_PTR globalID,
    ZOLTAN_ID_PTR localID,
    int num_dim,
    double* geom_vec,
    int* ierr)
{

  ConnectivityArray &
  connectivity_array = *(static_cast<ConnectivityArray*>(data));

  *ierr = ZOLTAN_OK;

  PointMap
  centroids = connectivity_array.GetCentroids();

  // Transfer the centroid coordinates to the Zoltan array
  int
  index_geom_vec = 0;

  for (PointMap::const_iterator
      centroids_iter = centroids.begin();
      centroids_iter != centroids.end();
      ++centroids_iter) {

    Vector<double> const
    centroid = (*centroids_iter).second;

    for (Index i = 0; i < 3; ++i) {

      geom_vec[index_geom_vec] = centroid(i);
      ++index_geom_vec;

    }

  }

  return;
}

//
// Write a Connectivity Array to an output stream
//
std::ostream &
operator<<(
    std::ostream & output_stream,
    ConnectivityArray const & connectivity_array)
{
  output_stream << std::setw(12) << connectivity_array.GetNumberNodes();
  output_stream << std::setw(12) << connectivity_array.GetNumberElements();
  output_stream << std::setw(12) << connectivity_array.GetType();
  output_stream << '\n';

  // Node list
  const PointMap
  nodes = connectivity_array.GetNodeList();

  int const
  dimension = connectivity_array.GetDimension();

  for (PointMap::const_iterator
      nodes_iter = nodes.begin();
      nodes_iter != nodes.end();
      ++nodes_iter) {

    int const
    node = (*nodes_iter).first;

    output_stream << std::setw(12) << node;

    Vector<double> const &
    point = (*nodes_iter).second;

    for (int j = 0; j < dimension; ++j) {
      output_stream << std::scientific;
      output_stream << std::setw(16) << std::setprecision(8);
      output_stream << point(j);
    }

    output_stream << '\n';

  }

  // Output element volumes as well
  const ScalarMap
  volumes = connectivity_array.GetVolumes();

  // Element connectivity
  const AdjacencyMap
  connectivity = connectivity_array.GetConnectivity();

  for (AdjacencyMap::const_iterator
      connectivity_iter = connectivity.begin();
      connectivity_iter != connectivity.end();
      ++connectivity_iter) {

    int const element = (*connectivity_iter).first;

    output_stream << std::setw(12) << element;

    IDList const &
    node_list = (*connectivity_iter).second;

    for (IDList::size_type j = 0; j < node_list.size(); ++j) {
      output_stream << std::setw(12) << node_list[j];
    }

    // Element volume
    ScalarMap::const_iterator
    volumes_iter = volumes.find(element);

    assert(volumes_iter != volumes.end());

    double const
    volume = (*volumes_iter).second;

    output_stream << std::scientific << std::setw(16) << std::setprecision(8);
    output_stream << volume;

    output_stream << '\n';
  }

  return output_stream;
}

//
// Default constructor for dual graph
//
DualGraph::DualGraph() : number_edges_(0)
{
  return;
}

//
// Build dual graph from connectivity array
// The term face is used as in "proper face" in algebraic topology
//
DualGraph::DualGraph(ConnectivityArray const & connectivity_array)
{

  const std::vector< std::vector<int> >
  face_connectivity = GetFaceConnectivity(connectivity_array.GetType());

  const AdjacencyMap
  connectivity = connectivity_array.GetConnectivity();

  std::map<std::set<int>, int>
  face_nodes_faceID_map;

  int
  face_count = 0;

  graph_.clear();

  AdjacencyMap
  faceID_element_map;

  // Go element by element
  for (AdjacencyMap::const_iterator
      connectivity_iter = connectivity.begin();
      connectivity_iter != connectivity.end();
      ++connectivity_iter) {

    int const
    element = (*connectivity_iter).first;

    const std::vector<int>
    element_nodes = (*connectivity_iter).second;

    // All elements go into graph, regardless of number of internal faces
    // attached to them. This clearing will allocate space for all of them.
    graph_[element].clear();

    // Determine the (generalized) faces for each element
    for (std::vector< std::vector<int> >::size_type
        i = 0;
        i < face_connectivity.size();
        ++i) {

      std::set<int>
      face_nodes;

      for (std::vector<int>::size_type
          j = 0;
          j < face_connectivity[i].size();
          ++j) {
        face_nodes.insert(element_nodes[face_connectivity[i][j]]);
      }

      // Determine whether this face is new (not found in face map)
      std::map<std::set<int>, int>::const_iterator
      face_map_iter = face_nodes_faceID_map.find(face_nodes);

      const bool
      face_is_new = face_map_iter == face_nodes_faceID_map.end();

      // If face is new then assign new ID to it and add to face map
      int
      faceID = -1;

      if (face_is_new == true) {
        faceID = face_count;
        face_nodes_faceID_map.insert(std::make_pair(face_nodes, faceID));
        ++face_count;
      } else {
        faceID = (*face_map_iter).second;
      }

      // List this element as attached to this face
      faceID_element_map[faceID].push_back(element);

    }

  }

  // Identify internal faces
  IDList
  internal_faces;

  for (AdjacencyMap::const_iterator
      face_element_iter = faceID_element_map.begin();
      face_element_iter != faceID_element_map.end();
      ++face_element_iter) {

    int const
    faceID = (*face_element_iter).first;

    int const
    number_elements_per_face = ((*face_element_iter).second).size();

    switch (number_elements_per_face) {

    case 1:
      // Do nothing
      break;

    case 2:
      internal_faces.push_back(faceID);
      break;

    default:
      std::cerr << "Bad number of faces adjacent to element." << '\n';
      exit(1);
      break;

    }

  }

  // Build dual graph
  for (IDList::size_type
      i = 0;
      i < internal_faces.size();
      ++i) {

    int const
    faceID = internal_faces[i];

    const IDList
    elements_face = faceID_element_map[faceID];

    assert(elements_face.size() == 2);

    for (IDList::size_type
        j = 0;
        j < elements_face.size();
        ++j) {

      int const
      element = elements_face[j];

      graph_[element].push_back(faceID);

    }

  }

  number_edges_ = internal_faces.size();
  vertex_weights_ = connectivity_array.GetVolumes();

  return;
}

int
DualGraph::GetNumberVertices() const
{
  return graph_.size();
}

int
DualGraph::GetNumberEdges() const
{
  return number_edges_;
}

void
DualGraph::SetGraph(AdjacencyMap & graph)
{
  graph_ = graph;
  return;
}

AdjacencyMap
DualGraph::GetGraph() const
{
  return graph_;
}

//
// \return Edge list to create boost graph
//
AdjacencyMap
DualGraph::GetEdgeList() const
{
  AdjacencyMap edge_list;

  for (AdjacencyMap::const_iterator graph_iter = graph_.begin();
      graph_iter != graph_.end();
      ++graph_iter) {
    int const vertex = (*graph_iter).first;
    const IDList edges = (*graph_iter).second;

    for (IDList::const_iterator edges_iter = edges.begin();
        edges_iter != edges.end();
        ++edges_iter) {

      int const edge = (*edges_iter);

      IDList & vertices = edge_list[edge];

      vertices.push_back(vertex);

    }

  }

  return edge_list;
}

void
DualGraph::SetVertexWeights(ScalarMap & vertex_weights)
{
  vertex_weights_ = vertex_weights;
  return;
}

ScalarMap
DualGraph::GetVertexWeights() const
{
  return vertex_weights_;
}

//
// \return Connected components in the dual graph
//
int
DualGraph::GetConnectedComponents(std::vector<int> & components) const
{
  // Create boost graph from edge list
  typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS>
  UndirectedGraph;

  typedef boost::graph_traits<UndirectedGraph>::vertex_descriptor Vertex;
  typedef boost::graph_traits<UndirectedGraph>::edge_descriptor Edge;

  UndirectedGraph graph;

  // Add vertices
  std::map<int, Vertex> dual_2_boost;
  AdjacencyMap dual_graph = GetGraph();

  for (AdjacencyMap::const_iterator graph_iter = dual_graph.begin();
      graph_iter != dual_graph.end();
      ++graph_iter) {

    Vertex boost_vertex = boost::add_vertex(graph);
    int dual_vertex = (*graph_iter).first;

    dual_2_boost.insert(std::make_pair(dual_vertex, boost_vertex));

  }

  // Add edges
  AdjacencyMap
  edge_list = GetEdgeList();

  for (AdjacencyMap::const_iterator edges_iter = edge_list.begin();
      edges_iter != edge_list.end();
      ++edges_iter) {

    const IDList vertices = (*edges_iter).second;

    int source_vertex = vertices[0];
    int target_vertex = vertices[1];

    Vertex source_boost_vertex = dual_2_boost[source_vertex];
    Vertex target_boost_vertex = dual_2_boost[target_vertex];

    boost::add_edge(source_boost_vertex, target_boost_vertex, graph);

  }

  int const number_vertices = GetNumberVertices();
  components.resize(number_vertices);

  int number_components =
      boost::connected_components(graph, &components[0]);

  return number_components;
}

//
// Print graph for debugging
//
void
DualGraph::Print() const
{

  ScalarMap
  vertex_weights = GetVertexWeights();

  AdjacencyMap
  graph = GetGraph();

  int const
  number_vertices = GetNumberVertices();

  int const
  number_edges = GetNumberEdges();

  std::cout << '\n';
  std::cout << "Vertex - Edge Format:" << '\n';
  std::cout << '\n';
  std::cout << "============================================================";
  std::cout << '\n';
  std::cout << "Number of Vertices : " << number_vertices << '\n';
  std::cout << "Number of Edges    : " << number_edges << '\n';
  std::cout << "------------------------------------------------------------";
  std::cout << '\n';
  std::cout << "Vertex  Weight          Edges" << '\n';
  std::cout << "------------------------------------------------------------";
  std::cout << '\n';

  for (ScalarMap::const_iterator vw_iter = vertex_weights.begin();
      vw_iter != vertex_weights.end();
      ++vw_iter) {

    int const vertex = (*vw_iter).first;
    double const weight = (*vw_iter).second;

    std::cout << std::setw(8) << vertex;
    std::cout << std::scientific << std::setw(16) << std::setprecision(8);
    std::cout << weight;

    AdjacencyMap::const_iterator
    graph_iter = graph.find(vertex);

    if (graph_iter == graph.end()) {
      std::cerr << "Cannot find vertex " << vertex << '\n';
      exit(1);
    }

    IDList
    edges = graph[vertex];

    for (IDList::const_iterator edges_iter = edges.begin();
        edges_iter != edges.end();
        ++edges_iter) {
      int const edge = *edges_iter;
      std::cout << std::setw(8) << edge;
    }

    std::cout << '\n';

  }

  std::cout << "============================================================";
  std::cout << '\n';

  std::cout << '\n';
  std::cout << "Edge - Vertex Format:" << '\n';
  std::cout << '\n';

  AdjacencyMap
  edge_list = GetEdgeList();

  std::cout << "------------------------------------------------------------";
  std::cout << '\n';
  std::cout << "Edge    Vertices" << '\n';
  std::cout << "------------------------------------------------------------";
  std::cout << '\n';

  for (AdjacencyMap::const_iterator edges_iter = edge_list.begin();
      edges_iter != edge_list.end();
      ++edges_iter) {

    int const edge = (*edges_iter).first;
    std::cout << std::setw(8) << edge;
    const IDList vertices = (*edges_iter).second;

    for (IDList::const_iterator vertices_iter = vertices.begin();
        vertices_iter != vertices.end();
        ++vertices_iter) {
      int const vertex = (*vertices_iter);
      std::cout << std::setw(8) << vertex;
    }

    std::cout << '\n';

  }

  std::cout << "============================================================";
  std::cout << '\n';

  return;
}

//
//
//
std::vector< std::vector<int> >
DualGraph::GetFaceConnectivity(Intrepid::ELEMENT::Type const type) const
{

  std::vector< std::vector<int> >
  face_connectivity;

  // Ugly initialization, but cannot rely on compilers
  // supporting #include <initializer_list> for the time being.
  int
  number_faces = 0;

  int
  nodes_per_face = 0;

  switch (type) {

  case Intrepid::ELEMENT::SEGMENTAL:
    number_faces = 2;
    nodes_per_face = 1;
    break;

  case Intrepid::ELEMENT::TRIANGULAR:
    number_faces = 3;
    nodes_per_face = 2;
    break;

  case Intrepid::ELEMENT::QUADRILATERAL:
    number_faces = 4;
    nodes_per_face = 2;
    break;

  case Intrepid::ELEMENT::TETRAHEDRAL:
    number_faces = 4;
    nodes_per_face = 3;
    break;

  case Intrepid::ELEMENT::HEXAHEDRAL:
    number_faces = 6;
    nodes_per_face = 4;
    break;

  default:
    std::cerr << "Unknown element type in face connectivity." << '\n';
    exit(1);
    break;

  }

  face_connectivity.resize(number_faces);
  for (int i = 0; i < number_faces; ++i) {
    face_connectivity[i].resize(nodes_per_face);
  }

  // Just for abbreviation
  std::vector< std::vector<int> > &
  f = face_connectivity;

  switch (type) {

  case Intrepid::ELEMENT::SEGMENTAL:
    f[0][0] = 0;
    f[1][0] = 1;
    break;

  case Intrepid::ELEMENT::TRIANGULAR:
    f[0][0] = 0; f[0][1] = 1;
    f[1][0] = 1; f[1][1] = 2;
    f[2][0] = 2; f[2][1] = 0;
    break;

  case Intrepid::ELEMENT::QUADRILATERAL:
    f[0][0] = 0; f[0][1] = 1;
    f[1][0] = 1; f[1][1] = 2;
    f[2][0] = 2; f[2][1] = 3;
    f[3][0] = 3; f[3][1] = 0;
    break;

  case Intrepid::ELEMENT::TETRAHEDRAL:
    f[0][0] = 0; f[0][1] = 1; f[0][2] = 2;
    f[1][0] = 0; f[1][1] = 3; f[1][2] = 1;
    f[2][0] = 1; f[2][1] = 3; f[2][2] = 2;
    f[3][0] = 2; f[3][1] = 3; f[3][2] = 0;
    break;

  case Intrepid::ELEMENT::HEXAHEDRAL:
    f[0][0] = 0; f[0][1] = 1; f[0][2] = 2; f[0][3] = 3;
    f[1][0] = 0; f[1][1] = 4; f[1][2] = 5; f[1][3] = 1;
    f[2][0] = 1; f[2][1] = 5; f[2][2] = 6; f[2][3] = 2;
    f[3][0] = 2; f[3][1] = 6; f[3][2] = 7; f[3][3] = 3;
    f[4][0] = 3; f[4][1] = 7; f[4][2] = 4; f[4][3] = 0;
    f[5][0] = 4; f[5][1] = 7; f[5][2] = 6; f[5][3] = 5;
    break;

  default:
    std::cerr << "Unknown element type in face connectivity." << '\n';
    exit(1);
    break;

  }

  return face_connectivity;
}

//
// Default constructor for Zoltan hyperedge graph (or hypergraph)
//
ZoltanHyperGraph::ZoltanHyperGraph() :
          number_vertices_(0),
          number_hyperedges_(0)
{
  return;
}

//
// Build Zoltan Hypergraph from FE mesh Dual Graph
//
ZoltanHyperGraph::ZoltanHyperGraph(DualGraph const & dual_graph)
{
  graph_ = dual_graph.GetGraph();
  vertex_weights_ = dual_graph.GetVertexWeights();
  number_vertices_ = dual_graph.GetNumberVertices();
  number_hyperedges_ = dual_graph.GetNumberEdges();
  return;
}

int
ZoltanHyperGraph::GetNumberVertices() const
{
  return graph_.size();
}

void
ZoltanHyperGraph::SetNumberHyperedges(int number_hyperedges)
{
  number_hyperedges_ = number_hyperedges;
  return;
}

int
ZoltanHyperGraph::GetNumberHyperedges() const
{
  return number_hyperedges_;
}

void
ZoltanHyperGraph::SetGraph(AdjacencyMap & graph)
{
  graph_ = graph;
  return;
}

AdjacencyMap
ZoltanHyperGraph::GetGraph() const
{
  return graph_;
}

void
ZoltanHyperGraph::SetVertexWeights(ScalarMap & vertex_weights)
{
  vertex_weights_ = vertex_weights;
  return;
}

ScalarMap
ZoltanHyperGraph::GetVertexWeights() const
{
  return vertex_weights_;
}

//
// Vector with edge IDs
//
std::vector<ZOLTAN_ID_TYPE>
ZoltanHyperGraph::GetEdgeIDs() const
{

  std::vector<ZOLTAN_ID_TYPE>
  edges;

  for (AdjacencyMap::const_iterator
      graph_iter = graph_.begin();
      graph_iter != graph_.end();
      ++graph_iter) {

    IDList
    hyperedges = (*graph_iter).second;


    for (IDList::const_iterator
        hyperedges_iter = hyperedges.begin();
        hyperedges_iter != hyperedges.end();
        ++hyperedges_iter) {

      int const hyperedge = (*hyperedges_iter);
      edges.push_back(hyperedge);

    }
  }

  return edges;
}

//
// Vector with edge pointers
//
std::vector<int>
ZoltanHyperGraph::GetEdgePointers() const
{

  std::vector<int>
  pointers;

  int
  pointer = 0;

  for (AdjacencyMap::const_iterator
      graph_iter = graph_.begin();
      graph_iter != graph_.end();
      ++graph_iter) {

    pointers.push_back(pointer);

    IDList
    hyperedges = (*graph_iter).second;


    for (IDList::const_iterator
        hyperedges_iter = hyperedges.begin();
        hyperedges_iter != hyperedges.end();
        ++hyperedges_iter) {

      ++pointer;

    }

  }

  return pointers;
}

//
// Vector with vertex IDs
//
std::vector<ZOLTAN_ID_TYPE>
ZoltanHyperGraph::GetVertexIDs() const
{

  std::vector<ZOLTAN_ID_TYPE>
  vertices;

  for (AdjacencyMap::const_iterator
      graph_iter = graph_.begin();
      graph_iter != graph_.end();
      ++graph_iter) {

    int vertex = (*graph_iter).first;
    vertices.push_back(vertex);

  }

  return vertices;
}

//
// Zoltan interface, return number of objects
//
int
ZoltanHyperGraph::GetNumberOfObjects(void* data, int* ierr)
{

  ZoltanHyperGraph &
  zoltan_hypergraph = *(static_cast<ZoltanHyperGraph*>(data));

  *ierr = ZOLTAN_OK;

  int
  num_objects = zoltan_hypergraph.GetGraph().size();

  return num_objects;
}

//
// Zoltan interface, return relevant object properties
//
void
ZoltanHyperGraph::GetObjectList(
    void* data,
    int sizeGID,
    int sizeLID,
    ZOLTAN_ID_PTR globalID,
    ZOLTAN_ID_PTR localID,
    int wgt_dim,
    float* obj_wgts,
    int* ierr)
{

  ZoltanHyperGraph &
  zoltan_hypergraph = *(static_cast<ZoltanHyperGraph*>(data));

  *ierr = ZOLTAN_OK;

  ScalarMap
  vertex_weights = zoltan_hypergraph.GetVertexWeights();

  ZOLTAN_ID_PTR
  global_id_ptr = globalID;

  ZOLTAN_ID_PTR
  local_id_ptr = localID;

  float*
  weight_ptr = obj_wgts;

  for (ScalarMap::const_iterator
      weights_iter = vertex_weights.begin();
      weights_iter != vertex_weights.end();
      ++weights_iter) {

    int vertex = (*weights_iter).first;
    double vertex_weight = (*weights_iter).second;

    // Beware of this evil pointer manipulation
    (*global_id_ptr) = vertex;
    (*local_id_ptr) = vertex;
    (*weight_ptr) = vertex_weight;
    global_id_ptr++;
    local_id_ptr++;
    weight_ptr++;

  }

  return;
}

//
// Zoltan interface, get size of hypergraph
//
void
ZoltanHyperGraph::GetHyperGraphSize(
    void* data,
    int* num_lists,
    int* num_pins,
    int* format,
    int* ierr)
{

  ZoltanHyperGraph &
  zoltan_hypergraph = *(static_cast<ZoltanHyperGraph*>(data));

  *ierr = ZOLTAN_OK;

  // Number of vertices
  *num_lists = zoltan_hypergraph.GetVertexIDs().size();

  // Numbers of pins, i.e. size of list of hyperedges attached to vertices
  *num_pins = zoltan_hypergraph.GetEdgeIDs().size();

  *format = ZOLTAN_COMPRESSED_VERTEX;

  return;
}

//
// Zoltan interface, get the hypergraph itself
//
void
ZoltanHyperGraph::GetHyperGraph(
    void* data,
    int num_gid_entries,
    int num_vtx_edge,
    int num_pins,
    int format,
    ZOLTAN_ID_PTR vtxedge_GID,
    int* vtxedge_ptr,
    ZOLTAN_ID_PTR pin_GID,
    int* ierr)
{

  ZoltanHyperGraph &
  zoltan_hypergraph = *(static_cast<ZoltanHyperGraph*>(data));

  *ierr = ZOLTAN_OK;

  // Validate
  assert(num_vtx_edge ==
      static_cast<int>(zoltan_hypergraph.GetVertexIDs().size()));

  assert(num_pins ==
      static_cast<int>(zoltan_hypergraph.GetEdgeIDs().size()));

  assert(format == ZOLTAN_COMPRESSED_VERTEX);

  // Copy hypergraph data
  std::vector<ZOLTAN_ID_TYPE>
  vertex_IDs = zoltan_hypergraph.GetVertexIDs();

  std::vector<ZOLTAN_ID_TYPE>
  edge_IDs = zoltan_hypergraph.GetEdgeIDs();

  std::vector<int>
  edge_pointers = zoltan_hypergraph.GetEdgePointers();

  for (std::vector<ZOLTAN_ID_TYPE>::size_type
      i = 0;
      i < vertex_IDs.size();
      ++i) {

    vtxedge_GID[i] = vertex_IDs[i];

  }

  for (std::vector<ZOLTAN_ID_TYPE>::size_type
      i = 0;
      i < edge_IDs.size();
      ++i) {

    pin_GID[i] = edge_IDs[i];

  }

  for (std::vector<int>::size_type
      i = 0;
      i < edge_pointers.size();
      ++i) {

    vtxedge_ptr[i] = edge_pointers[i];

  }

  return;
}

//
// Read a Zoltan Hyperedge Graph from an input stream
//
std::istream &
operator>>(std::istream & input_stream, ZoltanHyperGraph & zoltan_hypergraph)
{
  //
  // First line must contain the number of vertices and hyperedges
  //
  const std::vector<char>::size_type
  MaxChar = 256;

  char line[MaxChar];
  input_stream.getline(line, MaxChar);

  std::stringstream header(line);
  std::string token;

  // Number of vertices
  header >> token;
  int number_vertices = atoi(token.c_str());

  // Number of hyperegdes
  header >> token;
  int number_hyperedges = atoi(token.c_str());

  AdjacencyMap
  graph;

  ScalarMap
  vertex_weights;

  // Read list of hyperedge IDs adjacent to given vertex
  for (int i = 0; i < number_vertices; ++i) {

    input_stream.getline(line, MaxChar);
    std::stringstream input_line(line);

    // First entry should be vertex ID
    input_line >> token;
    int vertex = atoi(token.c_str());

    // Second entry should be vertex weight
    input_line >> token;
    double vw = atof(token.c_str());
    vertex_weights[vertex] = vw;

    // Read the hyperedges
    IDList hyperedges;
    while (input_line >> token) {
      int hyperedge = atoi(token.c_str());
      hyperedges.push_back(hyperedge);
    }

    graph[vertex] = hyperedges;

  }

  zoltan_hypergraph.SetGraph(graph);
  zoltan_hypergraph.SetVertexWeights(vertex_weights);
  zoltan_hypergraph.SetNumberHyperedges(number_hyperedges);

  return input_stream;
}

//
// Write a Zoltan Hyperedge Graph to an output stream
//
std::ostream &
operator<<(
    std::ostream & output_stream,
    ZoltanHyperGraph const & zoltan_hypergraph)
{

  output_stream << std::setw(12) << zoltan_hypergraph.GetNumberVertices();
  output_stream << std::setw(12) << zoltan_hypergraph.GetNumberHyperedges();
  output_stream << '\n';

  AdjacencyMap const &
  graph = zoltan_hypergraph.GetGraph();

  ScalarMap
  vertex_weights = zoltan_hypergraph.GetVertexWeights();

  for (AdjacencyMap::const_iterator
      graph_iter = graph.begin();
      graph_iter != graph.end();
      ++graph_iter) {

    // Vertex ID
    int const
    vertex = (*graph_iter).first;

    double const
    vertex_weight = vertex_weights[vertex];

    output_stream << std::setw(12) << vertex;
    output_stream << std::scientific;
    output_stream << std::setw(16) << std::setprecision(8);
    output_stream << vertex_weight;

    const IDList
    hyperedges = (*graph_iter).second;

    for (IDList::const_iterator
        hyperedges_iter = hyperedges.begin();
        hyperedges_iter != hyperedges.end();
        ++hyperedges_iter) {

      int const
      hyperedge = (*hyperedges_iter);

      output_stream << std::setw(12) << hyperedge;

    }

    output_stream << '\n';

  }

  return output_stream;
}

} // namespace LCM

#endif // #if defined (ALBANY_LCM) && defined(ALBANY_ZOLTAN)
