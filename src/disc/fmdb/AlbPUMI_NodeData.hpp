//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBPUMI_NODEDATA_HPP
#define ALBPUMI_NODEDATA_HPP


#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_StateInfoStruct.hpp"
#include "PHAL_Dimension.hpp"

namespace AlbPUMI {

  // Helper class for NodeData
  template<unsigned Dim>
  struct NodeData_Traits { };

  template<unsigned Dim, class traits = NodeData_Traits<Dim> >
  class NodeData {

  public:

    NodeData(const std::string& name, const std::vector<int>& dim);
    ~NodeData();

    //! Type of traits class being used
    typedef traits traits_type;

    //! Define the field type
    typedef typename traits_type::field_type field_type;

    field_type *allocateArray(unsigned nElemsInBucket);
    field_type *allocateArray(double *buf, unsigned nElemsInBucket);

    std::string name;      // Name of data field
    std::vector<double *> buffer;        // array storage for shards::Array
    std::vector<field_type *> shArray;  // The shards::Array
    std::vector<int> dims;

  };

// Explicit template definitions in support of the above

  // Scalar value
  template <>
  struct NodeData_Traits<1> { 

    enum { size = 1 }; // One array dimension tags: Cell 
    typedef shards::Array<double, shards::NaturalOrder, Cell> field_type ;
    static field_type* buildArray(double *buf, unsigned nelems, std::vector<int>& dims){

      return new field_type(buf, nelems);

    }

  };

  // NodeScalar
  template <>
  struct NodeData_Traits<2> { 

    enum { size = 2 }; // Two array dimension tags: Cell and Node
    typedef shards::Array<double, shards::NaturalOrder, Cell, Node> field_type ;
    static field_type* buildArray(double *buf, unsigned nelems, std::vector<int>& dims){

      return new field_type(buf, nelems, dims[1]);

    }

  };

  // NodeVector
  template <>
  struct NodeData_Traits<3> { 

    enum { size = 3 }; // Three array dimension tags: Cell, Node, and Dim
    typedef shards::Array<double, shards::NaturalOrder, Cell, Node, Dim> field_type ;
    static field_type* buildArray(double *buf, unsigned nelems, std::vector<int>& dims){

      return new field_type(buf, nelems, dims[1], dims[2]);

    }

  };

  // NodeTensor
  template <>
  struct NodeData_Traits<4> { 

    enum { size = 4 }; // Four array dimension tags: Cell, Node, Dim, and Dim
    typedef shards::Array<double, shards::NaturalOrder, Cell, Node, Dim, Dim> field_type ;
    static field_type* buildArray(double *buf, unsigned nelems, std::vector<int>& dims){

      return new field_type(buf, nelems, dims[1], dims[2], dims[3]);

    }

  };


}

// Define macro for explicit template instantiation
#define NODEDATA_INSTANTIATE_TEMPLATE_CLASS_1D(name) \
  template class name<1>;
#define NODEDATA_INSTANTIATE_TEMPLATE_CLASS_2D(name) \
  template class name<2>;
#define NODEDATA_INSTANTIATE_TEMPLATE_CLASS_3D(name) \
  template class name<3>;
#define NODEDATA_INSTANTIATE_TEMPLATE_CLASS_4D(name) \
  template class name<4>;

#define NODEDATA_INSTANTIATE_TEMPLATE_CLASS(name) \
  NODEDATA_INSTANTIATE_TEMPLATE_CLASS_1D(name) \
  NODEDATA_INSTANTIATE_TEMPLATE_CLASS_2D(name) \
  NODEDATA_INSTANTIATE_TEMPLATE_CLASS_3D(name) \
  NODEDATA_INSTANTIATE_TEMPLATE_CLASS_4D(name)

#endif
