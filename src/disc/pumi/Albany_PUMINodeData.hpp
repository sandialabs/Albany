//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_PUMINODEDATA_HPP
#define ALBANY_PUMINODEDATA_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "PHAL_Dimension.hpp"
#include "Albany_AbstractNodeFieldContainer.hpp"
#include "Albany_StateInfoStruct.hpp"

#include <apfNumbering.h>

namespace Albany {

class AbstractPUMINodeFieldContainer : public Albany::AbstractNodeFieldContainer {

  public:

    AbstractPUMINodeFieldContainer(){}
    virtual ~AbstractPUMINodeFieldContainer(){}

    virtual void saveFieldVector(const Teuchos::RCP<const Tpetra_MultiVector>& mv,
            int offset) = 0;
    virtual Albany::MDArray getMDA(const std::vector<apf::Node>& buck) = 0;
    virtual void resize(const Teuchos::RCP<const Tpetra_Map>& local_node_map) = 0;

};

struct PUMINodeMetaData {
  const std::string name; // Name of data field
  const bool output;      // Is field output to disk each time step (or at end of simulation)?
  const std::vector<PHX::DataLayout::size_type> dims; // Number of nodes [, dim1] [, dim2]
  int nfield_dofs; // Total number of dofs in this field

  PUMINodeMetaData(const std::string& name, const bool output,
                   const std::vector<PHX::DataLayout::size_type>& dims);
  int ndims() const;
};

template<typename DataType>
struct PUMINodeDataBase : public AbstractPUMINodeFieldContainer,
                          public PUMINodeMetaData {
  std::vector<DataType> buffer; // 1D array storage -> numOwnedNodes * product of dims
  PUMINodeDataBase (const std::string& name, const bool output,
                    const std::vector<PHX::DataLayout::size_type>& dims)
    : PUMINodeMetaData(name, output, dims) {}
};

Teuchos::RCP<Albany::AbstractNodeFieldContainer>
buildPUMINodeField(const std::string& name, const std::vector<PHX::DataLayout::size_type>& dim, const bool output);

  // Helper class for PUMINodeData
  template<typename DataType, unsigned ArrayDim>
  struct PUMINodeData_Traits { };

  template<typename DataType, unsigned ArrayDim, class traits = PUMINodeData_Traits<DataType, ArrayDim> >
  class PUMINodeData : public PUMINodeDataBase<DataType> {

  public:

    PUMINodeData(const std::string& name, const std::vector<PHX::DataLayout::size_type>& dim, const bool output = false);
    virtual ~PUMINodeData(){}

    //! Type of traits class being used
    typedef traits traits_type;

    //! Define the field type
    typedef typename traits_type::field_type field_type;

    void saveFieldVector(const Teuchos::RCP<const Tpetra_MultiVector>& mv, int offset);
    void resize(const Teuchos::RCP<const Tpetra_Map>& local_node_map);
    Albany::MDArray getMDA(const std::vector<apf::Node>& buck);

  protected:

    std::size_t beginning_index;        // Buffer starting location for the next array allocation

    Teuchos::RCP<const Tpetra_Map> local_node_map;

  };

// Explicit template definitions in support of the above

  // NodeScalar
  template <typename T>
  struct PUMINodeData_Traits<T, 1> {

    enum { size = 1 }; // One array dimension tags: number of nodes in workset
    typedef shards::Array<T, shards::NaturalOrder, Node> field_type ;
    static field_type buildArray(T *buf, unsigned nelems, const std::vector<PHX::DataLayout::size_type>& dims){

      return field_type(buf, nelems);

    }

  };

  // NodeVector
  template <typename T>
  struct PUMINodeData_Traits<T, 2> {

    enum { size = 2 }; // Two array dimension tags: Nodes and vec dim
    typedef shards::Array<T, shards::NaturalOrder, Node, Dim> field_type ;
    static field_type buildArray(T *buf, unsigned nelems, const std::vector<PHX::DataLayout::size_type>& dims){

      return field_type(buf, nelems, dims[1]);

    }

  };

  // NodeTensor
  template <typename T>
  struct PUMINodeData_Traits<T, 3> {

    enum { size = 3 }; // Three array dimension tags: Nodes, Dim and Dim
    typedef shards::Array<T, shards::NaturalOrder, Node, Dim, Dim> field_type ;
    static field_type buildArray(T *buf, unsigned nelems, const std::vector<PHX::DataLayout::size_type>& dims){

      return field_type(buf, nelems, dims[1], dims[2]);

    }

  };

}

// Define macro for explicit template instantiation
#define PUMINODEDATA_INSTANTIATE_TEMPLATE_CLASS_SCAL(name, type) \
  template class name<type, 1>;
#define PUMINODEDATA_INSTANTIATE_TEMPLATE_CLASS_VEC(name, type) \
  template class name<type, 2>;
#define PUMINODEDATA_INSTANTIATE_TEMPLATE_CLASS_TENS(name, type) \
  template class name<type, 3>;

#define PUMINODEDATA_INSTANTIATE_TEMPLATE_CLASS(name) \
  PUMINODEDATA_INSTANTIATE_TEMPLATE_CLASS_SCAL(name, double) \
  PUMINODEDATA_INSTANTIATE_TEMPLATE_CLASS_VEC(name, double) \
  PUMINODEDATA_INSTANTIATE_TEMPLATE_CLASS_TENS(name, double)

#endif
