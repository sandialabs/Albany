//
// Simple Zoltan compact hyperedge graph for partitioning meshes
//

#include <cassert>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <string>

#include "Partition.h"

namespace LCM {

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
  // Create array from Exodus file
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
    disc_params->print(std::cout);

    Teuchos::RCP<Epetra_Comm>
    communicator = Albany::createEpetraCommFromMpiComm(Albany_MPI_COMM_WORLD);

    Albany::DiscretizationFactory
    disc_factory(disc_params);

    // 1 DOF per node
    // 1 internal variable (partition number)
    discretization_ptr_ = disc_factory.create(1, 1, communicator);

    dimension_ = discretization_ptr_->getNumDim();

    Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >
    element_node_list = discretization_ptr_->getElNodeID();

    Teuchos::ArrayRCP<double>
    coordinates = discretization_ptr_->getCoordinates();

    // Assume all the elements have the same number of nodes_
    Teuchos::ArrayRCP<int>::size_type
    nodes_per_element = element_node_list[0].size();

    type_ = FindType(nodes_per_element, dimension_);

    // Build coordinate array.
    // Assume that local numbering of nodes is contiguous.
    Teuchos::ArrayRCP<double>::size_type
    number_nodes = coordinates.size() / dimension_;

    for (Teuchos::ArrayRCP<double>::size_type i = 0; i < number_nodes; ++i) {

      LCM::Vector<double> point(0.0, 0.0, 0.0);

      for (int j = 0; j < dimension_; ++j) {
        point(j) = coordinates[i * dimension_ + j];
      }

      nodes_.insert(std::make_pair(i, point));
    }

    // Build connectivity array.
    // Assume that local numbering of elements is contiguous.
    for (Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >::size_type i = 0;
        i < element_node_list.size(); ++i) {

      IDList nodes_element(nodes_per_element);

      for (Teuchos::ArrayRCP<int>::size_type j = 0;
          j < nodes_per_element; ++j) {
        nodes_element[j] = element_node_list[i][j];
      }

      connectivity_.insert(std::make_pair(i, nodes_element));
    }

    return;
  }


  //
  //
  //
  int
  ConnectivityArray::GetNumberNodes() const
  {
    return nodes_.size();
  }

  //
  //
  //
  int
  ConnectivityArray::GetNumberElements() const
  {
    return connectivity_.size();
  }

  //
  //
  //
  int
  ConnectivityArray::GetDimension() const
  {
    return dimension_;
  }

  //
  //
  //
  ConnectivityArray::Type
  ConnectivityArray::GetType() const
  {
    return type_;
  }

  //
  //
  //
  PointMap
  ConnectivityArray::GetNodeList() const
  {
    return nodes_;
  }

  //
  //
  //
  AdjacencyMap
  ConnectivityArray::GetConnectivity() const
  {
    return connectivity_;
  }

  Albany::AbstractDiscretization &
  ConnectivityArray::GetDiscretization()
  {
    return (*discretization_ptr_.get());
  }

  //
  //
  //
  ScalarMap
  ConnectivityArray::GetVolumes() const
  {
    ScalarMap volumes;
    for (AdjacencyMap::const_iterator
        amci = connectivity_.begin();
        amci != connectivity_.end();
        ++amci) {

      int const &
      element = (*amci).first;

      IDList const &
      node_list = (*amci).second;

      std::vector< LCM::Vector<double> >
      points;

      for (IDList::size_type
          i = 0;
          i < node_list.size();
          ++i) {

        PointMap::const_iterator
        pmci = nodes_.find(node_list[i]);

        assert(pmci != nodes_.end());
        points.push_back((*pmci).second);

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
  //
  //
  double
  ConnectivityArray::GetVolume() const
  {
    double volume = 0.0;

    const ScalarMap
    volumes = GetVolumes();

    for (ScalarMap::const_iterator
        it = volumes.begin();
        it != volumes.end();
        ++it) {

      volume += (*it).second;

    }

    return volume;
  }

  //
  //
  //
  ConnectivityArray::Type
  ConnectivityArray::FindType(int nodes, int dim) const
  {

    Type type;

    switch (nodes) {

    case 2:
      type = SEGMENTAL;
      break;

    case 3:
      type = TRIANGULAR;
      break;

    case 4:
      switch (dim) {
      case 2:
        type = QUADRILATERAL;
        break;
      case 3:
        type = TETRAHEDRAL;
        break;
      default:
        type = UNKNOWN;
        break;
      }
      break;

    case 8:
      type = HEXAHEDRAL;
      break;

    default:
      type = UNKNOWN;
      break;

    }

    if (type == UNKNOWN) {
      std::cerr << "Unknown element type" << std::endl;
      std::cerr << "Spatial dimension: ";
      std::cerr << dim << std::endl;
      std::cerr << "Nodes per element: ";
      std::cerr << nodes << std::endl;
      std::exit(1);
    }

    return type;
  }

  //
  //
  //
  int
  ConnectivityArray::GetNumberPartitions(const double length_scale) const
  {
    const double ball_volume = length_scale * length_scale * length_scale;
    const int number_partitions = static_cast<int>(GetVolume() / ball_volume);
    return number_partitions;
  }

  //
  //
  //
  std::map<int, int>
  ConnectivityArray::PartitionHyperGraph(const double length_scale)
  {
    // Zoltan setup
    Zoltan zoltan(MPI::COMM_WORLD);

    zoltan.Set_Param("LB_METHOD", "HYPERGRAPH");
    zoltan.Set_Param("LB_APPROACH", "PARTITION");
    zoltan.Set_Param("DEBUG_LEVEL", "0");
    zoltan.Set_Param("OBJ_WEIGHT_DIM", "1");

    const int number_partitions = GetNumberPartitions(length_scale);

    std::stringstream ioss;
    ioss << number_partitions;

    std::string zoltan_number_parts;
    ioss >> zoltan_number_parts;

    zoltan.Set_Param("NUM_LOCAL_PARTS", zoltan_number_parts.c_str());
    zoltan.Set_Param("REMAP", "0");

    zoltan.Set_Param("HYPERGRAPH_PACKAGE", "PHG");

    zoltan.Set_Param("PHG_MULTILEVEL", "1");
    zoltan.Set_Param("PHG_EDGE_WEIGHT_OPERATION", "ERROR");

    //
    // Partition
    //
    DualGraph dg(*this);
    ZoltanHyperGraph zhg(dg);

    // Set up hypergraph
    zoltan.Set_Num_Obj_Fn(LCM::ZoltanHyperGraph::GetNumberOfObjects, &zhg);
    zoltan.Set_Obj_List_Fn(LCM::ZoltanHyperGraph::GetObjectList, &zhg);
    zoltan.Set_HG_Size_CS_Fn(LCM::ZoltanHyperGraph::GetHyperGraphSize, &zhg);
    zoltan.Set_HG_CS_Fn(LCM::ZoltanHyperGraph::GetHyperGraph, &zhg);

    int changes;
    int numGidEntries;
    int numLidEntries;
    int numImport;
    ZOLTAN_ID_PTR importGlobalIds;
    ZOLTAN_ID_PTR importLocalIds;
    int *importProcs;
    int *importToPart;
    int numExport;
    ZOLTAN_ID_PTR exportGlobalIds;
    ZOLTAN_ID_PTR exportLocalIds;
    int *exportProcs;
    int *exportToPart;

    int rc =
      zoltan.LB_Partition(
          changes,
          numGidEntries,
          numLidEntries,
          numImport,
          importGlobalIds,
          importLocalIds,
          importProcs,
          importToPart,
          numExport,
          exportGlobalIds,
          exportLocalIds,
          exportProcs,
          exportToPart);

#if defined(DEBUG)

    std::cout << "Changes           : " << changes << std::endl;
    std::cout << "Number GID entries: " << numGidEntries << std::endl;
    std::cout << "Number LID entries: " << numLidEntries << std::endl;
    std::cout << "Number to import  : " << numImport << std::endl;
    std::cout << "Number to export  : " << numExport << std::endl;

    std::cout << "Import GIDs:" << std::endl;
    for (int i = 0; i < numImport; ++i) {
      std::cout << importGlobalIds[i] << std::endl;
    }

    std::cout << "Import LIDs:" << std::endl;
    for (int i = 0; i < numImport; ++i) {
      std::cout << importLocalIds[i] << std::endl;
    }

    std::cout << "Import procs:" << std::endl;
    for (int i = 0; i < numImport; ++i) {
      std::cout << importProcs[i] << std::endl;
    }

    std::cout << "Import parts:" << std::endl;
    for (int i = 0; i < numImport; ++i) {
      std::cout << importToPart[i] << std::endl;
    }

    std::cout << "Export GIDs:" << std::endl;
    for (int i = 0; i < numExport; ++i) {
      std::cout << exportGlobalIds[i] << std::endl;
    }

    std::cout << "Export LIDs:" << std::endl;
    for (int i = 0; i < numExport; ++i) {
      std::cout << exportLocalIds[i] << std::endl;
    }

    std::cout << "Export procs:" << std::endl;
    for (int i = 0; i < numExport; ++i) {
      std::cout << exportProcs[i] << std::endl;
    }

    std::cout << "Export parts:" << std::endl;
    for (int i = 0; i < numExport; ++i) {
      std::cout << exportToPart[i] << std::endl;
    }

#endif // #if defined(DEBUG)

    if (rc != ZOLTAN_OK) {
      std::cerr << "Partitioning failed" << std::endl;
      std::exit(1);
    }

    // Set up partition map initializing all partitions to zero
    std::map<int, int> partitions;

    const ScalarMap
    vertex_weights = zhg.GetVertexWeights();

    for (ScalarMap::const_iterator
        it = vertex_weights.begin();
        it != vertex_weights.end();
        ++it) {
      int vertex = (*it).first;
      partitions[vertex] = 0;
    }

    // Fill up with results from Zoltan
    for (int i = 0; i < numImport; ++i) {
      const int vertex = static_cast<int>(importLocalIds[i]);
      partitions[vertex] = importToPart[i];
    }

    return partitions;

  }


  //
  // Write a Connectivity Array to an output stream
  //
  std::ostream &
  operator<<(std::ostream & os, ConnectivityArray const & ca)
  {
    os << std::setw(12) << ca.GetNumberNodes();
    os << std::setw(12) << ca.GetNumberElements();
    os << std::setw(12) << ca.GetType();
    os << std::endl;

    // Node list
    const PointMap nodes = ca.GetNodeList();
    const int dimension = ca.GetDimension();

    for (PointMap::const_iterator
        it = nodes.begin();
        it != nodes.end();
        ++it) {

      const int node = (*it).first;
      os << std::setw(12) << node;

      LCM::Vector<double> const &
      point = (*it).second;

      for (int j = 0; j < dimension; ++j) {
        os << std::scientific << std::setw(16) << std::setprecision(8);
        os << point(j);
      }

      os << std::endl;

    }

    // Output element volumes as well
    const ScalarMap
    volumes = ca.GetVolumes();

    // Element connectivity_
    const AdjacencyMap
    connectivity = ca.GetConnectivity();

    for (AdjacencyMap::const_iterator
        it = connectivity.begin();
        it != connectivity.end();
        ++it) {

      const int element = (*it).first;

      os << std::setw(12) << element;

      IDList const &
      node_list = (*it).second;

      for (IDList::size_type j = 0; j < node_list.size(); ++j) {
        os << std::setw(12) << node_list[j];
      }

      // Element volume
      ScalarMap::const_iterator
      smci = volumes.find(element);

      assert(smci != volumes.end());

      const double
      volume = (*smci).second;

      os << std::scientific << std::setw(16) << std::setprecision(8);
      os << volume;

      os << std::endl;
    }

    return os;

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
  DualGraph::DualGraph(ConnectivityArray const & ca)
  {

    const std::vector< std::vector<int> >
    face_connectivity = GetFaceConnectivity(ca);

    const AdjacencyMap
    connectivity = ca.GetConnectivity();

    std::map<std::set<int>, int>
    face_nodes_ID_map;

    int face_count = 0;

    graph_.clear();

    AdjacencyMap
    faceID_element_map;

    // Go element by element
    for (AdjacencyMap::const_iterator
        it = connectivity.begin();
        it != connectivity.end();
        ++it) {

      const int
      element = (*it).first;

      const std::vector<int>
      element_nodes = (*it).second;

      // All elements go into graph, regardless of number of internal faces
      // attached to them. This clearing will allocate space for all of them.
      graph_[element].clear();

      // Determine the (generalized) faces for each element
      for (std::vector< std::vector<int> >::size_type
          i = 0;
          i < face_connectivity.size();
          ++i) {

        std::set<int> face_nodes;

        for (std::vector<int>::size_type
            j = 0;
            j < face_connectivity[i].size();
            ++j) {
          face_nodes.insert(element_nodes[face_connectivity[i][j]]);
        }

        // Determine whether this face is new (not found in face map)
        std::map<std::set<int>, int>::const_iterator
        face_map_iter = face_nodes_ID_map.find(face_nodes);

        bool face_is_new = face_map_iter == face_nodes_ID_map.end();

        // If face is new then assign new ID to it and add to face map
        int faceID = -1;
        if (face_is_new == true) {
          faceID = face_count;
          face_nodes_ID_map.insert(std::make_pair(face_nodes, faceID));
          ++face_count;
        } else {
          faceID = (*face_map_iter).second;
        }

        // List this element as attached to this face
        faceID_element_map[faceID].push_back(element);

      }

    }

    // Identify internal faces
    IDList internal_faces;

    for (AdjacencyMap::const_iterator
        it = faceID_element_map.begin();
        it != faceID_element_map.end();
        ++it) {

      const int
      faceID = (*it).first;

      const int
      number_elements_per_face = ((*it).second).size();

      switch (number_elements_per_face) {

      case 1:
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

        const int element = elements_face[j];
        graph_[element].push_back(faceID);

      }

    }

    number_edges_ = internal_faces.size();
    vertex_weights_ = ca.GetVolumes();

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
  DualGraph::SetGraph(AdjacencyMap & g)
  {
    graph_ = g;
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
  DualGraph::SetVertexWeights(ScalarMap & vw)
  {
    vertex_weights_ = vw;
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
  DualGraph::GetFaceConnectivity(ConnectivityArray const & ca) const
  {

    std::vector< std::vector<int> >
    face_connectivity;

    ConnectivityArray::Type
    type = ca.GetType();

    // Ugly initialization, but cannot rely on compilers
    // supporting #include <initializer_list> for the time being.
    int number_faces = 0;
    int nodes_per_face = 0;

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
  ZoltanHyperGraph::ZoltanHyperGraph(DualGraph const & dg)
  {
    graph_ = dg.GetGraph();
    vertex_weights_ = dg.GetVertexWeights();
    number_vertices_ = dg.GetNumberVertices();
    number_hyperedges_ = dg.GetNumberEdges();
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
  ZoltanHyperGraph::SetNumberHyperedges(int ne)
  {
    number_hyperedges_ = ne;
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
  ZoltanHyperGraph::SetGraph(AdjacencyMap & g)
  {
    graph_ = g;
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
  ZoltanHyperGraph::SetVertexWeights(ScalarMap & vw)
  {
    vertex_weights_ = vw;
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
        amci = graph_.begin();
        amci != graph_.end();
        ++amci) {

      IDList
      hyperedges = (*amci).second;


      for (IDList::const_iterator
          ilci = hyperedges.begin();
          ilci != hyperedges.end();
          ++ilci) {

        const int hyperedge = (*ilci);
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

    int pointer = 0;

    for (AdjacencyMap::const_iterator
        amci = graph_.begin();
        amci != graph_.end();
        ++amci) {

      pointers.push_back(pointer);

      IDList
      hyperedges = (*amci).second;


      for (IDList::const_iterator
          ilci = hyperedges.begin();
          ilci != hyperedges.end();
          ++ilci) {

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
        amci = graph_.begin();
        amci != graph_.end();
        ++amci) {

      int vertex = (*amci).first;
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
    zhg = *(static_cast<ZoltanHyperGraph*>(data));

    *ierr = ZOLTAN_OK;

    int num_objects = zhg.GetGraph().size();

    return num_objects;

  }

  //
  // Zoltan interface, return relevant object properties
  //
  void
  ZoltanHyperGraph::GetObjectList(
      void *data,
      int sizeGID,
      int sizeLID,
      ZOLTAN_ID_PTR globalID,
      ZOLTAN_ID_PTR localID,
      int wgt_dim,
      float *obj_wgts,
      int *ierr)
  {

    ZoltanHyperGraph &
    zhg = *(static_cast<ZoltanHyperGraph*>(data));

    *ierr = ZOLTAN_OK;

    ScalarMap
    vertex_weights = zhg.GetVertexWeights();

    ZOLTAN_ID_PTR
    pGID = globalID;

    ZOLTAN_ID_PTR
    pLID = localID;

    float*
    pWT = obj_wgts;

    for (ScalarMap::const_iterator
        smci = vertex_weights.begin();
        smci != vertex_weights.end();
        ++smci) {

      int vertex = (*smci).first;
      double vw = (*smci).second;

      // Beware of this evil pointer manipulation
      (*pGID) = vertex;
      (*pLID) = vertex;
      (*pWT) = vw;
      pGID++;
      pLID++;
      pWT++;

    }

    return;

  }

  //
  // Zoltan interface, get size of hypergraph
  //
  void
  ZoltanHyperGraph::GetHyperGraphSize(
      void *data,
      int *num_lists,
      int *num_pins,
      int *format,
      int *ierr)
  {

    ZoltanHyperGraph &
    zhg = *(static_cast<ZoltanHyperGraph*>(data));

    *ierr = ZOLTAN_OK;

    // Number of vertices
    *num_lists = zhg.GetVertexIDs().size();

    // Numbers of pins, i.e. size of list of hyperedges attached to vertices
    *num_pins = zhg.GetEdgeIDs().size();

    *format = ZOLTAN_COMPRESSED_VERTEX;

    return;

  }

  //
  // Zoltan interface, get the hypergraph itself
  //
  void
  ZoltanHyperGraph::GetHyperGraph(
      void *data,
      int num_gid_entries,
      int num_vtx_edge,
      int num_pins,
      int format,
      ZOLTAN_ID_PTR vtxedge_GID,
      int *vtxedge_ptr,
      ZOLTAN_ID_PTR pin_GID,
      int *ierr)
  {

    ZoltanHyperGraph &
    zhg = *(static_cast<ZoltanHyperGraph*>(data));

    *ierr = ZOLTAN_OK;

    // Validate
    assert(num_vtx_edge == static_cast<int>(zhg.GetVertexIDs().size()));
    assert(num_pins == static_cast<int>(zhg.GetEdgeIDs().size()));
    assert(format == ZOLTAN_COMPRESSED_VERTEX);

    // Copy hypergraph data
    std::vector<ZOLTAN_ID_TYPE>
    vertex_IDs = zhg.GetVertexIDs();

    std::vector<ZOLTAN_ID_TYPE>
    edge_IDs = zhg.GetEdgeIDs();

    std::vector<int>
    edge_pointers = zhg.GetEdgePointers();

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
  operator>>(std::istream & is, ZoltanHyperGraph & zhg)
  {
    //
    // First line must contain the number of vertices and hyperedges
    //
    const std::vector<char>::size_type
    MaxChar = 256;

    char line[MaxChar];
    is.getline(line, MaxChar);

    std::stringstream iss_hdr(line);
    std::string token;

    // Number of vertices
    iss_hdr >> token;
    int number_vertices = atoi(token.c_str());

    // Number of hyperegdes
    iss_hdr >> token;
    int number_hyperedges = atoi(token.c_str());

    AdjacencyMap
    graph;

    ScalarMap
    vertex_weights;

    // Read list of hyperedge IDs adjacent to given vertex
    for (int i = 0; i < number_vertices; ++i) {

      is.getline(line, MaxChar);
      std::stringstream iss_vtx(line);

      // First entry should be vertex ID
      iss_vtx >> token;
      int vertex = atoi(token.c_str());

      // Second entry should be vertex weight
      iss_vtx >> token;
      double vw = atof(token.c_str());
      vertex_weights[vertex] = vw;

      // Read the hyperedges
      IDList hyperedges;
      while (iss_vtx >> token) {
        int hyperedge = atoi(token.c_str());
        hyperedges.push_back(hyperedge);
      }

      graph[vertex] = hyperedges;

    }

    zhg.SetGraph(graph);
    zhg.SetVertexWeights(vertex_weights);
    zhg.SetNumberHyperedges(number_hyperedges);

    return is;

  }

  //
  // Write a Zoltan Hyperedge Graph to an output stream
  //
  std::ostream &
  operator<<(std::ostream & os, ZoltanHyperGraph const & zhg)
  {

    os << std::setw(12) << zhg.GetNumberVertices();
    os << std::setw(12) << zhg.GetNumberHyperedges() << std::endl;

    AdjacencyMap const &
    graph = zhg.GetGraph();

    ScalarMap
    vertex_weights = zhg.GetVertexWeights();

    for (AdjacencyMap::const_iterator
        amci = graph.begin();
        amci != graph.end();
        ++amci) {

      // Vertex ID
      const int
      vertex = (*amci).first;

      const double
      vw = vertex_weights[vertex];

      os << std::setw(12) << vertex;
      os << std::scientific << std::setw(16) << std::setprecision(8) << vw;

      const IDList
      hyperedges = (*amci).second;

      for (IDList::const_iterator
          ilci = hyperedges.begin();
          ilci != hyperedges.end();
          ++ilci) {

        const int
        hyperedge = (*ilci);

        os << std::setw(12) << hyperedge;

      }

      os << std::endl;

    }

    return os;

  }

  //
  // Test the conversion
  //
  void
  TestGraphs()
  {
    return;
  }

} // namespace LCM
