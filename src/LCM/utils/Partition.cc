//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

// Define only if Zoltan is enabled
#if defined (ALBANY_LCM) && defined(ALBANY_ZOLTAN)

#include <cassert>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <string>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>

#include "Partition.h"

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

      output_stream << "Changes           : " << changes << std::endl;
      output_stream << "Number GID entries: " << num_gid_entries << std::endl;
      output_stream << "Number LID entries: " << num_lid_entries << std::endl;
      output_stream << "Number to import  : " << num_import << std::endl;
      output_stream << "Number to export  : " << num_export << std::endl;

      output_stream << "Import GIDs:" << std::endl;
      for (int i = 0; i < num_import; ++i) {
        output_stream << import_global_ids[i] << std::endl;
      }

      output_stream << "Import LIDs:" << std::endl;
      for (int i = 0; i < num_import; ++i) {
        output_stream << import_local_ids[i] << std::endl;
      }

      output_stream << "Import procs:" << std::endl;
      for (int i = 0; i < num_import; ++i) {
        output_stream << import_procs[i] << std::endl;
      }

      output_stream << "Import parts:" << std::endl;
      for (int i = 0; i < num_import; ++i) {
        output_stream << import_to_part[i] << std::endl;
      }

      output_stream << "Export GIDs:" << std::endl;
      for (int i = 0; i < num_export; ++i) {
        output_stream << export_global_ids[i] << std::endl;
      }

      output_stream << "Export LIDs:" << std::endl;
      for (int i = 0; i < num_export; ++i) {
        output_stream << export_local_ids[i] << std::endl;
      }

      output_stream << "Export procs:" << std::endl;
      for (int i = 0; i < num_export; ++i) {
        output_stream << export_procs[i] << std::endl;
      }

      output_stream << "Export parts:" << std::endl;
      for (int i = 0; i < num_export; ++i) {
        output_stream << export_to_part[i] << std::endl;
      }

    }

  } // anonymous namespace

  //
  // Default constructor for Connectivity Array
  //
  ConnectivityArray::ConnectivityArray() :
    type_(ConnectivityArray::UNKNOWN),
    dimension_(0),
    discretization_ptr_(Teuchos::null),
    voxel_size_(0.0)
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
      type_(ConnectivityArray::UNKNOWN),
      dimension_(0),
      discretization_ptr_(Teuchos::null),
      voxel_size_(0.0)
  {
    //Teuchos::GlobalMPISession mpiSession(&argc,&argv);

    Teuchos::RCP<Teuchos::ParameterList>
    disc_params = rcp(new Teuchos::ParameterList("params"));

    //set Method to Exodus and set input file name
    disc_params->set<std::string>("Method", "Exodus");
    disc_params->set<std::string>("Exodus Input File Name", input_file);
    disc_params->set<std::string>("Exodus Output File Name", output_file);
    // Max of 10000 workset size -- automatically resized down
    disc_params->set<int>("Workset Size", 10000);
    //disc_params->print(std::cout);

    Teuchos::RCP<Epetra_Comm>
    communicator = Albany::createEpetraCommFromMpiComm(Albany_MPI_COMM_WORLD);

    Albany::DiscretizationFactory
    disc_factory(disc_params, communicator);

    Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >
    meshSpecs = disc_factory.createMeshSpecs();

    int 
    worksetSize = meshSpecs[0]->worksetSize;

    // Create a state field in stick named Partition on elements
    // 1 DOF per node
    // 1 internal variable (partition number)
    Teuchos::RCP<Albany::StateInfoStruct>
    stateInfo = Teuchos::rcp(new Albany::StateInfoStruct());

    stateInfo->push_back(Teuchos::rcp(new Albany::StateStruct("Partition")));
    Albany::StateStruct& stateRef = *stateInfo->back();
    stateRef.entity = "QuadPoint"; //Tag, should be Node or QuadPoint
    // State has 1 quad point (i.e. element variable)
    stateRef.dim.push_back(worksetSize); stateRef.dim.push_back(1);

    discretization_ptr_ = disc_factory.createDiscretization(1, stateInfo);

    dimension_ = meshSpecs[0]->numDim;

    // Dimensioned: Workset, Cell, Local Node
    Teuchos::ArrayRCP<
      Teuchos::ArrayRCP<
        Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > > >
    element_connectivity = discretization_ptr_->getWsElNodeEqID();

    Teuchos::ArrayRCP<double>
    coordinates = discretization_ptr_->getCoordinates();

    // For higher-order elements, mid-nodes are ignored and only
    // the nodes at the corners of the element are considered
    // to define the topology.
    const CellTopologyData
    cell_topology = meshSpecs[0]->ctd;

    const int
    dimension = cell_topology.dimension;

    assert(dimension == dimension_);

    const int
    vertices_per_element = cell_topology.vertex_count;

    type_ = FindType(dimension, vertices_per_element);

    // Assume all the elements have the same number of nodes and eqs
    Teuchos::ArrayRCP<int>::size_type
    nodes_per_element = element_connectivity[0][0].size();

    // Do some logic so we can get from unknown ID to node ID
    const int number_equations = element_connectivity[0][0][0].size();
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
    Teuchos::ArrayRCP<double>::size_type
    number_nodes = coordinates.size() / dimension;

    for (Teuchos::ArrayRCP<double>::size_type
        node = 0;
        node < number_nodes;
        ++node) {

      LCM::Vector<double> point(0.0, 0.0, 0.0);

      for (int j = 0; j < dimension; ++j) {
        point(j) = coordinates[node * dimension + j];
      }

      nodes_.insert(std::make_pair(node, point));
    }

    // Build connectivity array.
    // Assume that local numbering of elements is contiguous.
    // Ignore extra nodes in higher-order elements
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >::size_type
    element_number = 0;

    Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > >::size_type
    workset = 0;

    for (workset = 0; workset < element_connectivity.size(); ++workset) {

      for (Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >::size_type
          cell = 0;
          cell < element_connectivity[workset].size();
          ++cell, ++element_number) {

        IDList
        nodes_element(nodes_per_element);

        for (Teuchos::ArrayRCP<int>::size_type
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
  int
  ConnectivityArray::GetNumberNodes() const
  {
    return nodes_.size();
  }

  //
  // \return Number of elements in the array
  //
  int
  ConnectivityArray::GetNumberElements() const
  {
    return connectivity_.size();
  }

  //
  // \return Space dimension
  //
  int
  ConnectivityArray::GetDimension() const
  {
    return dimension_;
  }

  //
  // \return Type of finite element in the array
  // (assume same type for all elements)
  //
  ConnectivityArray::Type
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

      std::vector< LCM::Vector<double> >
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

      case SEGMENTAL:
        volume = LCM::length(points[0], points[1]);
        break;

      case TRIANGULAR:
        volume = LCM::area(points[0], points[1], points[2]);
        break;

      case QUADRILATERAL:
        volume = LCM::area(points[0], points[1], points[2], points[3]);
        break;

      case TETRAHEDRAL:
        volume = LCM::volume(points[0], points[1], points[2], points[3]);
        break;

      case HEXAHEDRAL:
        volume = LCM::volume(points[0], points[1], points[2], points[3],
            points[4], points[5], points[6], points[7]);
        break;

      default:
        std::cerr << "Unknown element type in calculating volume." << std::endl;
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
        std::cerr << "Cannot find volume for element " << element << std::endl;
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

      std::vector< LCM::Vector<double> >
      points;

      // Collect element nodes
      for (IDList::size_type
          i = 0;
          i < node_list.size();
          ++i) {

        const int
        node = node_list[i];

        PointMap::const_iterator
        nodes_iter = nodes_.find(node);

        assert(nodes_iter != nodes_.end());

        const LCM::Vector<double>
        point = (*nodes_iter).second;

        points.push_back(point);

      }

      const LCM::Vector<double>
      centroid = LCM::centroid(points);

      centroids.insert(std::make_pair(element, centroid));

    }

    return centroids;

  }

  ///
  /// \return Bounding box for all nodes
  ///
  std::pair<LCM::Vector<double>, LCM::Vector<double> >
  ConnectivityArray::BoundingBox() const
  {
    PointMap::const_iterator
    it = nodes_.begin();

    LCM::Vector<double>
    min = (*it).second;

    LCM::Vector<double>
    max = min;

    const Index
    N = min.get_dimension();

    ++it;

    for (; it != nodes_.end(); ++it) {

      LCM::Vector<double> const &
      node = (*it).second;

      for (Index i = 0; i < N; ++i) {
        min(i) = std::min(min(i), node(i));
        max(i) = std::max(max(i), node(i));
      }

    }

    return std::make_pair(min, max);

  }

  //
  // Voxelization of the domain for fast determination
  // of points being inside or outside the domain.
  //
  void
  ConnectivityArray::Voxelize()
  {

    //
    // First determine the maximum dimension of the bounding box.
    //
    const Index
    maximum_divisions = 16;

    LCM::Vector<double>
    min;

    LCM::Vector<double>
    max;

    boost::tie(min, max) = BoundingBox();

    const Index
    N = min.get_dimension();

    double
    maximum_dimension = 0.0;

    for (Index i = 0; i < N; ++i) {

      maximum_dimension = std::max(maximum_dimension, max(i) - min(i));

    }

    const double
    delta = maximum_dimension / maximum_divisions;

    voxel_size_ = delta;

    //
    // Determine number of voxels for each dimension.
    //
    LCM::Vector<Index>
    voxels_dimension(N);

    for (Index i = 0; i < N; ++i) {
      const Index
      number_voxels = std::ceil((max(i) - min(i)) / delta);
      voxels_dimension(i) = number_voxels;
    }

    //
    // Set up the voxels array.
    // Generalization to N dimensions fails here.
    // This is specific to 3D.
    //
    voxels_.resize(voxels_dimension(0));

    for (Index i = 0; i < voxels_dimension(0); ++i) {

      voxels_[i].resize(voxels_dimension(1));

      for (Index j = 0; j < voxels_dimension(1); ++j) {

        voxels_[i][j].resize(voxels_dimension(2));

      }

    }

    // Fill array
    LCM::Vector<double> p(N);

    for (Index i = 0; i < voxels_dimension(0); ++i) {
      p(0) = i * delta + delta / 2.0 + min(0);
      for (Index j = 0; j < voxels_dimension(1); ++j) {
        p(1) = j * delta + delta / 2.0 + min(1);
        for (Index k = 0; k < voxels_dimension(2); ++k) {
          p(2) = k * delta + delta / 2.0 + min(2);

          voxels_[i][j][k] = IsInsideMeshByElement(p);

          std::cout << i << "/" << voxels_dimension(0) << "-";
          std::cout << j << "/" << voxels_dimension(1) << "-";
          std::cout << k << "/" << voxels_dimension(2) << std::endl;

        }

      }

    }

    return;
  }

  //
  // Determine is a given point is inside the mesh.
  // 3D only for now.
  //
  bool
  ConnectivityArray::IsInsideMesh(Vector<double> const & point) const
  {
    const Index i = (point(0) - lower_corner_(0)) / voxel_size_ + 0.5;

    if (i < 0 || i >= voxels_.size()) {
      return false;
    }

    const Index j = (point(1) - lower_corner_(1)) / voxel_size_ + 0.5;

    if (j < 0 || j >= voxels_[0].size()) {
      return false;
    }

    const Index k = (point(2) - lower_corner_(2)) / voxel_size_ + 0.5;

    if (k < 0 || k >= voxels_[0][0].size()) {
      return false;
    }

    return voxels_[i][j][k];
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

      std::vector< LCM::Vector<double> >
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

      case TETRAHEDRAL:
        if (in_tetrahedron(point, node[0], node[1], node[2], node[3]) == true) {
          return true;
        }
        break;

      case HEXAHEDRAL:
        if (in_hexahedron(point, node[0], node[1], node[2], node[3],
            node[4], node[5], node[6], node[7])) {
          return true;
        }
        break;

      default:
        std::cerr << "Unknown element type in K-means partition." << std::endl;
        exit(1);
        break;

      }

    }

    return false;
  }

  //
  // Helper functions for determining the type of element
  //
  namespace {

    ConnectivityArray::Type
    FindType1D(int nodes)
    {
      ConnectivityArray::Type
      type = ConnectivityArray::UNKNOWN;

      switch (nodes) {
      case 2:
        type = ConnectivityArray::SEGMENTAL;
        break;
      default:
        type = ConnectivityArray::UNKNOWN;
        break;
      }
      return type;
    }

    ConnectivityArray::Type
    FindType2D(int nodes)
    {
      ConnectivityArray::Type
      type = ConnectivityArray::UNKNOWN;

      switch (nodes) {
      case 3:
        type = ConnectivityArray::TRIANGULAR;
        break;
      case 4:
        type = ConnectivityArray::QUADRILATERAL;
        break;
      default:
        type = ConnectivityArray::UNKNOWN;
        break;
      }
      return type;
    }

    ConnectivityArray::Type
    FindType3D(int nodes)
    {
      ConnectivityArray::Type
      type = ConnectivityArray::UNKNOWN;

      switch (nodes) {
      case 4:
        type = ConnectivityArray::TETRAHEDRAL;
        break;
      case 8:
        type = ConnectivityArray::HEXAHEDRAL;
        break;
      default:
        type = ConnectivityArray::UNKNOWN;
        break;
      }
      return type;
    }

  }

  //
  // Given number of (vertex) nodes and space dimension,
  // determine the type of a finite element.
  //
  ConnectivityArray::Type
  ConnectivityArray::FindType(int dimension, int nodes) const
  {

    ConnectivityArray::Type
    type = ConnectivityArray::UNKNOWN;

    switch (dimension) {

    case 1:
      type = FindType1D(nodes);
      break;

    case 2:
      type = FindType2D(nodes);
      break;

    case 3:
      type = FindType3D(nodes);
      break;

    default:
      type = ConnectivityArray::UNKNOWN;
      break;

    }

    if (type == ConnectivityArray::UNKNOWN) {
      std::cerr << "Unknown element type" << std::endl;
      std::cerr << "Spatial dimension: ";
      std::cerr << dimension << std::endl;
      std::cerr << "Vertices per element: ";
      std::cerr << nodes << std::endl;
      exit(1);
    }

    return type;
  }

  //
  // \param length_scale Length scale for partitioning for
  // variational non-local regularization
  // \return Number of partitions defined as total volume
  // of the array divided by the cube of the length scale
  //
  int
  ConnectivityArray::GetNumberPartitions(const double length_scale) const
  {
    const double
    ball_volume = length_scale * length_scale * length_scale;

    const int
    number_partitions =
        static_cast<int>(round(GetVolume() / ball_volume));

    return number_partitions;
  }

  //
  // Anonymous namespace for helper functions
  //
  namespace {

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

        const int
        partition = (*it).second;

        partitions_set.insert(partition);
      }

      std::set<int>::size_type
      number_partitions = partitions_set.size();

      std::vector<int>
      partition_shuffle(number_partitions);

      for (std::vector<int>::size_type i = 0; i < number_partitions; ++i) {
        partition_shuffle[i] = i;
      }

      std::random_shuffle(partition_shuffle.begin(), partition_shuffle.end());

      std::map<int, int>
      partition_map;

      int
      partition_index = 0;

      for (std::set<int>::const_iterator it = partitions_set.begin();
          it != partitions_set.end();
          ++it) {

        const int
        partition = (*it);

        partition_map[partition] = partition_index;

        ++partition_index;

      }

      std::map<int, int>
      new_partitions;

      for (std::map<int, int>::const_iterator it = old_partitions.begin();
          it != old_partitions.end();
          ++it) {

        const int
        element = (*it).first;

        const int
        old_partition = (*it).second;

        const int
        partition_index = partition_map[old_partition];

        const int
        new_partition = partition_shuffle[partition_index];

        new_partitions[element] = new_partition;

      }

      return new_partitions;
    }

  } // anonymous namespace

  //
  // Partition mesh according to the specified algorithm and length scale
  // \param partition_scheme The partition algorithm to use
  // \param length_scale The length scale for variational nonlocal
  // regularization
  // \return Partition number for each element
  //
  std::map<int, int>
  ConnectivityArray::Partition(
      const LCM::PartitionScheme partition_scheme,
      const double length_scale)
  {

    std::map<int, int>
    partitions;

    switch (partition_scheme) {

    case LCM::HYPERGRAPH:
      partitions = PartitionHyperGraph(length_scale);
      break;

    case LCM::GEOMETRIC:
      partitions = PartitionGeometric(length_scale);
      break;

    case LCM::KMEANS:
      partitions = PartitionKMeans(length_scale);
      break;

    default:
      std::cerr << "Unknown partitioning scheme." << std::endl;
      exit(1);
      break;

    }

    // Store for use by other methods
    partitions_ = RenumberPartitions(partitions);

    return partitions_;

  }

  //
  // Partition mesh with Zoltan Hypergraph algortithm
  // \param length_scale The length scale for variational nonlocal
  // regularization
  // \return Partition number for each element
  //
  std::map<int, int>
  ConnectivityArray::PartitionHyperGraph(const double length_scale)
  {
    // Zoltan setup
    const int
    number_partitions = GetNumberPartitions(length_scale);

    std::stringstream
    ioss;

    ioss << number_partitions;

    std::string
    zoltan_number_parts;

    ioss >> zoltan_number_parts;

    Zoltan
    zoltan(MPI::COMM_SELF);

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
      std::cerr << "Partitioning failed" << std::endl;
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
      const int vertex = (*weights_iter).first;
      partitions[vertex] = 0;
    }

    // Fill up with results from Zoltan
    for (int i = 0; i < num_import; ++i) {
      const int vertex = static_cast<int>(import_local_ids[i]);
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
  ConnectivityArray::PartitionGeometric(const double length_scale)
  {
    // Zoltan setup
    const int
    number_partitions = GetNumberPartitions(length_scale);

    std::stringstream
    ioss;

    ioss << number_partitions;

    std::string
    zoltan_number_parts;

    ioss >> zoltan_number_parts;

    Zoltan
    zoltan(MPI::COMM_SELF);

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
      std::cerr << "Partitioning failed" << std::endl;
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
      const int element = (*volumes_iter).first;
      partitions[element] = 0;
    }

    // Fill up with results from Zoltan, which returns partitions for all
    // elements that belong to a partition > 0
    for (int i = 0; i < num_import; ++i) {
      const int element = static_cast<int>(import_local_ids[i]);
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
  ConnectivityArray::PartitionKMeans(const double length_scale)
  {
    const int
    number_partitions = GetNumberPartitions(length_scale);

    LCM::Vector<double>
    min;

    LCM::Vector<double>
    max;

    boost::tie(min, max) = BoundingBox();

    lower_corner_ = min;
    upper_corner_ = max;

    Voxelize();

    // Create initial generators
    int
    number_generators = 0;

    std::vector< Vector<double> >
    generators;

    while (number_generators < number_partitions) {

      Vector<double>
      p = random_in_box(min, max);

      if (IsInsideMesh(p) == true) {
        generators.push_back(p);
        ++number_generators;
        //std::cout << p;
      }

    }

    // K-means iteration
    const Index
    max_iterations = 64;

    const Index
    number_random_points = 64 * number_partitions;

    Index
    number_iterations = 0;

    const double
    diagonal_distance = norm(max - min);

    const double
    tolerance = 1.0e-2 * diagonal_distance;

    double
    max_step = diagonal_distance;

    while (max_step >= tolerance && number_iterations < max_iterations) {

      // Create random points and assign to closest generators
      Index
      random_point_counter = 0;

      std::vector< Vector<double> >
      random_points;

      std::map<int, int>
      point_generator_map;

      while (random_point_counter < number_random_points) {

        const Vector<double>
        random_point = random_in_box(min, max);

        const bool
        point_is_in_mesh = IsInsideMesh(random_point);

        if (point_is_in_mesh == true) {
          random_points.push_back(random_point);

          point_generator_map[random_point_counter] =
              closest_point(random_point, generators);

          //std::cout << "Random point: " << random_point_counter;
          //std::cout << "/" << number_random_points << std::endl;

          ++random_point_counter;
        }

      }

      // Determine cluster of random points for each generator
      std::vector<std::vector<Vector<double> > >
      clusters;

      clusters.resize(number_partitions);

      for (std::map<int, int>::const_iterator it = point_generator_map.begin();
          it != point_generator_map.end();
          ++it) {

        const int
        point_index = (*it).first;

        const int
        generator_index = (*it).second;

        clusters[generator_index].push_back(random_points[point_index]);

      }

      // Compute centroids of each cluster and set generators to
      // these centroids.
      max_step = 0.0;

      for (std::vector< std::vector<Vector<double> > >::size_type i = 0;
          i < clusters.size();
          ++i) {

        // If cluster is empty then generator does not move.
        if (clusters[i].size() == 0) {
          continue;
        }

        const Vector<double>
        cluster_centroid = centroid(clusters[i]);

        const double
        step = norm(cluster_centroid - generators[i]);

        if (step > max_step) {
          max_step = step;
        }

        generators[i] = cluster_centroid;
      }

      std::cout << "Iteration: " << number_iterations;
      std::cout << ". Step: " << max_step << ". Tol:" << tolerance << std::endl;

      ++number_iterations;

    }

    // Set partition number for each element.

    // Partition map.
    std::map<int, int>
    partitions;

    for (AdjacencyMap::const_iterator
        elements_iter = connectivity_.begin();
        elements_iter != connectivity_.end();
        ++elements_iter) {

      int const &
      element = (*elements_iter).first;

      IDList const &
      node_list = (*elements_iter).second;

      std::vector< LCM::Vector<double> >
      element_nodes;

      for (IDList::size_type
          i = 0;
          i < node_list.size();
          ++i) {

        PointMap::const_iterator
        nodes_iter = nodes_.find(node_list[i]);

        assert(nodes_iter != nodes_.end());
        element_nodes.push_back((*nodes_iter).second);

      }

      const Vector<double>
      element_centroid = centroid(element_nodes);

      //std::cout << element_centroid;

      partitions[element] = closest_point(element_centroid, generators);

    }

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

      const LCM::Vector<double>
      centroid = (*centroids_iter).second;

      for (LCM::Index i = 0; i < 3; ++i) {

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
    output_stream << std::endl;

    // Node list
    const PointMap
    nodes = connectivity_array.GetNodeList();

    const int
    dimension = connectivity_array.GetDimension();

    for (PointMap::const_iterator
        nodes_iter = nodes.begin();
        nodes_iter != nodes.end();
        ++nodes_iter) {

      const int
      node = (*nodes_iter).first;

      output_stream << std::setw(12) << node;

      LCM::Vector<double> const &
      point = (*nodes_iter).second;

      for (int j = 0; j < dimension; ++j) {
        output_stream << std::scientific;
        output_stream << std::setw(16) << std::setprecision(8);
        output_stream << point(j);
      }

      output_stream << std::endl;

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

      const int element = (*connectivity_iter).first;

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

      const double
      volume = (*volumes_iter).second;

      output_stream << std::scientific << std::setw(16) << std::setprecision(8);
      output_stream << volume;

      output_stream << std::endl;
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

      const int
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

      const int
      faceID = (*face_element_iter).first;

      const int
      number_elements_per_face = ((*face_element_iter).second).size();

      switch (number_elements_per_face) {

      case 1:
        // Do nothing
        break;

      case 2:
        internal_faces.push_back(faceID);
        break;

      default:
        std::cerr << "Bad number of faces adjacent to element." << std::endl;
        exit(1);
        break;

      }

    }

    // Build dual graph
    for (IDList::size_type
        i = 0;
        i < internal_faces.size();
        ++i) {

      const int
      faceID = internal_faces[i];

      const IDList
      elements_face = faceID_element_map[faceID];

      assert(elements_face.size() == 2);

      for (IDList::size_type
          j = 0;
          j < elements_face.size();
          ++j) {

        const int
        element = elements_face[j];

        graph_[element].push_back(faceID);

      }

    }

    number_edges_ = internal_faces.size();
    vertex_weights_ = connectivity_array.GetVolumes();

    return;

  }

  //
  //
  //
  int
  DualGraph::GetNumberVertices() const
  {
    return graph_.size();
  }

  //
  //
  //
  int
  DualGraph::GetNumberEdges() const
  {
    return number_edges_;
  }

  //
  //
  //
  void
  DualGraph::SetGraph(AdjacencyMap & graph)
  {
    graph_ = graph;
    return;
  }

  //
  //
  //
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
      const int vertex = (*graph_iter).first;
      const IDList edges = (*graph_iter).second;

      for (IDList::const_iterator edges_iter = edges.begin();
          edges_iter != edges.end();
          ++edges_iter) {

        const int edge = (*edges_iter);

        IDList & vertices = edge_list[edge];

        vertices.push_back(vertex);

      }

    }

    return edge_list;
  }
  //
  //
  //
  void
  DualGraph::SetVertexWeights(ScalarMap & vertex_weights)
  {
    vertex_weights_ = vertex_weights;
    return;
  }

  //
  //
  //
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

    const int number_vertices = GetNumberVertices();
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

    const int
    number_vertices = GetNumberVertices();

    const int
    number_edges = GetNumberEdges();

    std::cout << std::endl;
    std::cout << "Vertex - Edge Format:" << std::endl;
    std::cout << std::endl;
    std::cout << "============================================================";
    std::cout << std::endl;
    std::cout << "Number of Vertices : " << number_vertices << std::endl;
    std::cout << "Number of Edges    : " << number_edges << std::endl;
    std::cout << "------------------------------------------------------------";
    std::cout << std::endl;
    std::cout << "Vertex  Weight          Edges" << std::endl;
    std::cout << "------------------------------------------------------------";
    std::cout << std::endl;

    for (ScalarMap::const_iterator vw_iter = vertex_weights.begin();
        vw_iter != vertex_weights.end();
        ++vw_iter) {

      const int vertex = (*vw_iter).first;
      const double weight = (*vw_iter).second;

      std::cout << std::setw(8) << vertex;
      std::cout << std::scientific << std::setw(16) << std::setprecision(8);
      std::cout << weight;

      AdjacencyMap::const_iterator
      graph_iter = graph.find(vertex);

      if (graph_iter == graph.end()) {
        std::cerr << "Cannot find vertex " << vertex << std::endl;
        exit(1);
      }

      IDList
      edges = graph[vertex];

      for (IDList::const_iterator edges_iter = edges.begin();
           edges_iter != edges.end();
           ++edges_iter) {
        const int edge = *edges_iter;
        std::cout << std::setw(8) << edge;
      }

      std::cout << std::endl;

    }

    std::cout << "============================================================";
    std::cout << std::endl;

    std::cout << std::endl;
    std::cout << "Edge - Vertex Format:" << std::endl;
    std::cout << std::endl;

    AdjacencyMap
    edge_list = GetEdgeList();

    std::cout << "------------------------------------------------------------";
    std::cout << std::endl;
    std::cout << "Edge    Vertices" << std::endl;
    std::cout << "------------------------------------------------------------";
    std::cout << std::endl;

    for (AdjacencyMap::const_iterator edges_iter = edge_list.begin();
        edges_iter != edge_list.end();
        ++edges_iter) {

      const int edge = (*edges_iter).first;
      std::cout << std::setw(8) << edge;
      const IDList vertices = (*edges_iter).second;

      for (IDList::const_iterator vertices_iter = vertices.begin();
          vertices_iter != vertices.end();
          ++vertices_iter) {
        const int vertex = (*vertices_iter);
        std::cout << std::setw(8) << vertex;
      }

      std::cout << std::endl;

    }

    std::cout << "============================================================";
    std::cout << std::endl;

    return;
  }

  //
  //
  //
  std::vector< std::vector<int> >
  DualGraph::GetFaceConnectivity(const ConnectivityArray::Type type) const
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

    case ConnectivityArray::SEGMENTAL:
      number_faces = 2;
      nodes_per_face = 1;
      break;

    case ConnectivityArray::TRIANGULAR:
      number_faces = 3;
      nodes_per_face = 2;
      break;

    case ConnectivityArray::QUADRILATERAL:
      number_faces = 4;
      nodes_per_face = 2;
      break;

    case ConnectivityArray::TETRAHEDRAL:
      number_faces = 4;
      nodes_per_face = 3;
      break;

    case ConnectivityArray::HEXAHEDRAL:
      number_faces = 6;
      nodes_per_face = 4;
      break;

    default:
      std::cerr << "Unknown element type in face connectivity." << std::endl;
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

    case ConnectivityArray::SEGMENTAL:
      f[0][0] = 0;
      f[1][0] = 1;
      break;

    case ConnectivityArray::TRIANGULAR:
      f[0][0] = 0; f[0][1] = 1;
      f[1][0] = 1; f[1][1] = 2;
      f[2][0] = 2; f[2][1] = 0;
      break;

    case ConnectivityArray::QUADRILATERAL:
      f[0][0] = 0; f[0][1] = 1;
      f[1][0] = 1; f[1][1] = 2;
      f[2][0] = 2; f[2][1] = 3;
      f[3][0] = 3; f[3][1] = 0;
      break;

    case ConnectivityArray::TETRAHEDRAL:
      f[0][0] = 0; f[0][1] = 1; f[0][2] = 2;
      f[1][0] = 0; f[1][1] = 3; f[1][2] = 1;
      f[2][0] = 1; f[2][1] = 3; f[2][2] = 2;
      f[3][0] = 2; f[3][1] = 3; f[3][2] = 0;
      break;

    case ConnectivityArray::HEXAHEDRAL:
      f[0][0] = 0; f[0][1] = 1; f[0][2] = 2; f[0][3] = 3;
      f[1][0] = 0; f[1][1] = 4; f[1][2] = 5; f[1][3] = 1;
      f[2][0] = 1; f[2][1] = 5; f[2][2] = 6; f[2][3] = 2;
      f[3][0] = 2; f[3][1] = 6; f[3][2] = 7; f[3][3] = 3;
      f[4][0] = 3; f[4][1] = 7; f[4][2] = 4; f[4][3] = 0;
      f[5][0] = 4; f[5][1] = 7; f[5][2] = 6; f[5][3] = 5;
      break;

    default:
      std::cerr << "Unknown element type in face connectivity." << std::endl;
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

  //
  //
  //
  int
  ZoltanHyperGraph::GetNumberVertices() const
  {
    return graph_.size();
  }

  //
  //
  //
  void
  ZoltanHyperGraph::SetNumberHyperedges(int number_hyperedges)
  {
    number_hyperedges_ = number_hyperedges;
    return;
  }

  //
  //
  //
  int
  ZoltanHyperGraph::GetNumberHyperedges() const
  {
    return number_hyperedges_;
  }

  //
  //
  //
  void
  ZoltanHyperGraph::SetGraph(AdjacencyMap & graph)
  {
    graph_ = graph;
    return;
  }

  //
  //
  //
  AdjacencyMap
  ZoltanHyperGraph::GetGraph() const
  {
    return graph_;
  }

  //
  //
  //
  void
  ZoltanHyperGraph::SetVertexWeights(ScalarMap & vertex_weights)
  {
    vertex_weights_ = vertex_weights;
    return;
  }

  //
  //
  //
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

        const int hyperedge = (*hyperedges_iter);
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
    output_stream << std::endl;

    AdjacencyMap const &
    graph = zoltan_hypergraph.GetGraph();

    ScalarMap
    vertex_weights = zoltan_hypergraph.GetVertexWeights();

    for (AdjacencyMap::const_iterator
        graph_iter = graph.begin();
        graph_iter != graph.end();
        ++graph_iter) {

      // Vertex ID
      const int
      vertex = (*graph_iter).first;

      const double
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

        const int
        hyperedge = (*hyperedges_iter);

        output_stream << std::setw(12) << hyperedge;

      }

      output_stream << std::endl;

    }

    return output_stream;

  }

} // namespace LCM

#endif // #if defined (ALBANY_LCM) && defined(ALBANY_ZOLTAN)
