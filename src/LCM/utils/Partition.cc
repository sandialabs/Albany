///
/// \file Partition.cc
/// Simple Zoltan compact hyperedge graph for partitioning meshes.
/// Implementation.
/// \author Alejandro Mota
///


// Define only if Zoltan is enabled
#if defined (ALBANY_LCM) && defined(ALBANY_ZOLTAN)

#include <cassert>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <string>

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
    discretization_ptr_(Teuchos::null)
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
      std::string const & output_file)
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

    const Teuchos::RCP<Albany::MeshSpecsStruct>
    meshSpecs = disc_factory.createMeshSpecs();

    int 
    worksetSize = meshSpecs->worksetSize;

    // 1 DOF per node
    // 1 internal variable (partition number)
    discretization_ptr_ = disc_factory.createDiscretization(1, 1, worksetSize);

    dimension_ = discretization_ptr_->getNumDim();

    // Dimensioned: Workset, Cell, Local Node
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > >
    element_connectivity = discretization_ptr_->getWsElNodeID();

    Teuchos::ArrayRCP<double>
    coordinates = discretization_ptr_->getCoordinates();

    // For higher-order elements, mid-nodes are ignored and only
    // the nodes at the corners of the element are considered
    // to define the topology.
    const CellTopologyData
    cell_topology = discretization_ptr_->getCellTopologyData();

    const int
    dimension = cell_topology.dimension;

    assert(dimension == dimension_);

    const int
    vertices_per_element = cell_topology.vertex_count;

    type_ = FindType(dimension, vertices_per_element);

    // Assume all the elements have the same number of nodes
    Teuchos::ArrayRCP<int>::size_type
    nodes_per_element = element_connectivity[0][0].size();

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

          nodes_element[node] = element_connectivity[workset][cell][node];

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
        std::exit(1);

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
      std::exit(1);
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
    number_partitions = static_cast<int>(GetVolume() / ball_volume);

    return number_partitions;
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

    default:
      std::cerr << "Unknown partitioning scheme." << std::endl;
      std::exit(1);

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
    zoltan(MPI::COMM_WORLD);

    zoltan.Set_Param("LB_METHOD", "HYPERGRAPH");
    zoltan.Set_Param("LB_APPROACH", "PARTITION");
    zoltan.Set_Param("DEBUG_LEVEL", "0");
    zoltan.Set_Param("OBJ_WEIGHT_DIM", "1");
    zoltan.Set_Param("NUM_LOCAL_PARTS", zoltan_number_parts.c_str());
    zoltan.Set_Param("REMAP", "0");
    zoltan.Set_Param("HYPERGRAPH_PACKAGE", "PHG");
    zoltan.Set_Param("PHG_MULTILEVEL", "1");
    zoltan.Set_Param("PHG_EDGE_WEIGHT_OPERATION", "ERROR");

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
      std::exit(1);
    }

    // Set up partition map initializing all partitions to zero
    std::map<int, int> partitions;

    const ScalarMap
    vertex_weights = zoltan_hypergraph.GetVertexWeights();

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
    zoltan(MPI::COMM_WORLD);

    zoltan.Set_Param("LB_METHOD", "RIB");
    zoltan.Set_Param("LB_APPROACH", "PARTITION");
    zoltan.Set_Param("DEBUG_LEVEL", "0");
    zoltan.Set_Param("OBJ_WEIGHT_DIM", "1");
    zoltan.Set_Param("NUM_LOCAL_PARTS", zoltan_number_parts.c_str());
    zoltan.Set_Param("REMAP", "0");
    zoltan.Set_Param("CHECK_GEOM", "1");
    zoltan.Set_Param("AVERAGE_CUTS", "1");
    zoltan.Set_Param("REDUCE_DIMENSIONS", "1");
    zoltan.Set_Param("DEGENERATE_RATIO", "25");

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
      std::exit(1);
    }

    // Set up partition map initializing all partitions to zero
    std::map<int, int> partitions;

    const ScalarMap
    element_volumes = GetVolumes();

    for (ScalarMap::const_iterator
        volumes_iter = element_volumes.begin();
        volumes_iter != element_volumes.end();
        ++volumes_iter) {
      const int element = (*volumes_iter).first;
      partitions[element] = 0;
    }

    // Fill up with results from Zoltan
    for (int i = 0; i < num_import; ++i) {
      const int element = static_cast<int>(import_local_ids[i]);
      partitions[element] = import_to_part[i];
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

      for (LCM::Index i = 0; i < LCM::MaxDim; ++i) {

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
        std::exit(1);
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
      std::exit(1);
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
      std::exit(1);
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
