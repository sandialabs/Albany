//
// Simple Zoltan compact hyperedge graph for partitioning meshes
//

#include <iostream>
#include <map>
#include <set>
#include <vector>

#include <zoltan_cpp.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_ArrayRCP.hpp>
#include <Teuchos_ParameterList.hpp>

#include <stk/Albany_AbstractDiscretization.hpp>
#include <stk/Albany_DiscretizationFactory.hpp>
#include <stk/Albany_STKDiscretization.hpp>
#include <Albany_SolverFactory.hpp>
#include <Albany_Utils.hpp>

#include "LCM/utils/Geometry.h"

#if !defined(LCM_Partition_h)
#define LCM_Partition_h

namespace LCM {

  typedef std::vector<int> IDList;
  typedef std::map<int, IDList> AdjacencyMap;
  typedef std::map<int, double> ScalarMap;
  typedef std::map<int, LCM::Vector<double> > PointMap;

  // Forward declarations;
  class ConnectivityArray;
  class DualGraph;
  class ZoltanHyperGraph;

  //
  // Simple connectivity array.
  // Holds coordinate array as well.
  //
  class ConnectivityArray {
  public:
    enum Type {UNKNOWN, SEGMENTAL, TRIANGULAR,
      QUADRILATERAL, TETRAHEDRAL, HEXAHEDRAL};

    // constructors
    ConnectivityArray();

    ConnectivityArray(
        std::string const & input_file,
        std::string const & output_file);

    int
    GetNumberNodes() const;

    int
    GetNumberElements() const;

    int
    GetDimension() const;

    Type
    GetType() const;

    PointMap
    GetNodeList() const;

    AdjacencyMap
    GetConnectivity() const;

    ScalarMap
    GetVolumes() const;

    double
    GetVolume() const;

    int
    GetNumberPartitions(const double length_scale) const;

    Albany::AbstractDiscretization &
    GetDiscretization();

    std::map<int, int>
    PartitionHyperGraph(const double length_scale);

  private:
    Type
    FindType(int nodes, int dimension) const;

  private:

    Type
    type_;

    PointMap
    nodes_;

    AdjacencyMap
    connectivity_;

    int
    dimension_;

    Teuchos::RCP<Albany::AbstractDiscretization>
    discretization_ptr_;

  };

  //
  // Read a Conectivity Array from an input stream
  //
  std::istream &
  operator>>(std::istream & is, ConnectivityArray & ca);

  //
  // Write a Connectivity Array to an output stream
  //
  std::ostream &
  operator<<(std::ostream & os, ConnectivityArray const & ca);

  //
  // Dual graph of a connectivity array
  //
  class DualGraph {

  public:

    // Constructors
    DualGraph();

    DualGraph(ConnectivityArray const & ca);

    int
    GetNumberVertices() const;

    int
    GetNumberEdges() const;

    void
    SetVertexWeights(ScalarMap & vw);

    ScalarMap
    GetVertexWeights() const;

    void
    SetGraph(AdjacencyMap & g);

    AdjacencyMap
    GetGraph() const;

  private:

    std::vector< std::vector<int> >
    GetFaceConnectivity(ConnectivityArray const & ca) const;

  private:

    int
    number_edges_;

    AdjacencyMap
    graph_;

    ScalarMap
    vertex_weights_;

  };

  //
  // Class to interface with Zoltan
  //
  class ZoltanHyperGraph {

  public:

    // constructors
    ZoltanHyperGraph();

    ZoltanHyperGraph(DualGraph const & dg);

    // Number of vertices defined by size of graph
    int
    GetNumberVertices() const;

    void
    SetNumberHyperedges(int ne);

    int
    GetNumberHyperedges() const;

    void
    SetGraph(AdjacencyMap & g);

    AdjacencyMap
    GetGraph() const;

    void
    SetVertexWeights(ScalarMap & vw);

    ScalarMap
    GetVertexWeights() const;

    std::vector<ZOLTAN_ID_TYPE>
    GetEdgeIDs() const;

    std::vector<int>
    GetEdgePointers() const;

    std::vector<ZOLTAN_ID_TYPE>
    GetVertexIDs() const;

    // Zoltan interface functions. Must be static for linking purposes
    // (i.e. no hidden *this parameter)
    // See Zoltan documentation at
    // http://www.cs.sandia.gov/Zoltan/ug_html/ug.html
    static int
    GetNumberOfObjects(void* data, int* ierr);

    static void
    GetObjectList(
        void *data,
        int sizeGID,
        int sizeLID,
        ZOLTAN_ID_PTR globalID,
        ZOLTAN_ID_PTR localID,
        int wgt_dim,
        float *obj_wgts,
        int *ierr);

    static void
    GetHyperGraphSize(
        void *data,
        int *num_lists,
        int *num_pins,
        int *format,
        int *ierr);

    static void
    GetHyperGraph(
        void *data,
        int num_gid_entries,
        int num_vtx_edge,
        int num_pins,
        int format,
        ZOLTAN_ID_PTR
        vtxedge_GID,
        int *vtxedge_ptr,
        ZOLTAN_ID_PTR pin_GID,
        int *ierr);

  private:

    int
    number_vertices_;

    int
    number_hyperedges_;

    AdjacencyMap
    graph_;

    ScalarMap
    vertex_weights_;

  };

  //
  // Read a Zoltan Hyperedge Graph from an input stream
  //
  std::istream &
  operator>>(std::istream & is, ZoltanHyperGraph & zhg);

  //
  // Write a Zoltan Hyperedge Graph to an output stream
  //
  std::ostream &
  operator<<(std::ostream & os, ZoltanHyperGraph const & zhg);

  //
  // Tests
  //
  void
  TestGraphs();

} // namespace LCM

#endif // LCM_Partition_h
