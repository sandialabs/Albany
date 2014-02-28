//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBPUMI_NODEDATA_HPP
#define ALBPUMI_NODEDATA_HPP


#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "PHAL_Dimension.hpp"
#include "Albany_AbstractNodeFieldContainer.hpp"
#include "Albany_StateInfoStruct.hpp"

#include <apfNumbering.h>

namespace AlbPUMI {

class AbstractPUMINodeFieldContainer : public Albany::AbstractNodeFieldContainer {

  public:

    AbstractPUMINodeFieldContainer(){}
    virtual ~AbstractPUMINodeFieldContainer(){}

    virtual void saveField(const Teuchos::RCP<const Epetra_Vector>& block_mv,
            int offset, int blocksize = -1) = 0;
    virtual Albany::MDArray getMDA(const std::vector<apf::Node>& buck) = 0;
    virtual void resize(const Teuchos::RCP<const Epetra_Map>& local_node_map) = 0;

};

Teuchos::RCP<Albany::AbstractNodeFieldContainer>
buildPUMINodeField(const std::string& name, const std::vector<int>& dim, const bool output);


  // Helper class for NodeData
  template<typename DataType, unsigned ArrayDim>
  struct NodeData_Traits { };

  template<typename DataType, unsigned ArrayDim, class traits = NodeData_Traits<DataType, ArrayDim> >
  class NodeData : public AbstractPUMINodeFieldContainer {

  public:

    NodeData(const std::string& name, const std::vector<int>& dim, const bool output = false);
    virtual ~NodeData(){}

    //! Type of traits class being used
    typedef traits traits_type;

    //! Define the field type
    typedef typename traits_type::field_type field_type;

    void saveField(const Teuchos::RCP<const Epetra_Vector>& block_mv, int offset, int blocksize = -1);
    void resize(const Teuchos::RCP<const Epetra_Map>& local_node_map);
    Albany::MDArray getMDA(const std::vector<apf::Node>& buck);

  protected:

    const std::string name;      // Name of data field
    const bool output;           // Is field output to disk each time step (or at end of simulation)?
    std::vector<DataType>  buffer;        // 1D array storage -> numOwnedNodes * product of dims
    std::vector<int> dims;
    int nfield_dofs;                    // total number of dofs in this field
    std::size_t beginning_index;        // Buffer starting location for the next array allocation

    Teuchos::RCP<const Epetra_Map> local_node_map;

  };

// Explicit template definitions in support of the above

  // NodeScalar
  template <typename T>
  struct NodeData_Traits<T, 1> {

    enum { size = 1 }; // One array dimension tags: number of nodes in workset
    typedef shards::Array<T, shards::NaturalOrder, Node> field_type ;
    static field_type buildArray(T *buf, unsigned nelems, std::vector<int>& dims){

      return field_type(buf, nelems);

    }

  };

  // NodeVector
  template <typename T>
  struct NodeData_Traits<T, 2> {

    enum { size = 2 }; // Two array dimension tags: Nodes and vec dim
    typedef shards::Array<T, shards::NaturalOrder, Node, Dim> field_type ;
    static field_type buildArray(T *buf, unsigned nelems, std::vector<int>& dims){

      return field_type(buf, nelems, dims[1]);

    }

  };

  // NodeTensor
  template <typename T>
  struct NodeData_Traits<T, 3> {

    enum { size = 3 }; // Three array dimension tags: Nodes, Dim and Dim
    typedef shards::Array<T, shards::NaturalOrder, Node, Dim, Dim> field_type ;
    static field_type buildArray(T *buf, unsigned nelems, std::vector<int>& dims){

      return field_type(buf, nelems, dims[1], dims[2]);

    }

  };

}

// Define macro for explicit template instantiation
#define NODEDATA_INSTANTIATE_TEMPLATE_CLASS_SCAL(name, type) \
  template class name<type, 1>;
#define NODEDATA_INSTANTIATE_TEMPLATE_CLASS_VEC(name, type) \
  template class name<type, 2>;
#define NODEDATA_INSTANTIATE_TEMPLATE_CLASS_TENS(name, type) \
  template class name<type, 3>;

#define NODEDATA_INSTANTIATE_TEMPLATE_CLASS(name) \
  NODEDATA_INSTANTIATE_TEMPLATE_CLASS_SCAL(name, double) \
  NODEDATA_INSTANTIATE_TEMPLATE_CLASS_VEC(name, double) \
  NODEDATA_INSTANTIATE_TEMPLATE_CLASS_TENS(name, double)

#endif
