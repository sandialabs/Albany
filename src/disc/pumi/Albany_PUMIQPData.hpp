//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/12/14: no Epetra!

#ifndef ALBANY_PUMIQPDATA_HPP
#define ALBANY_PUMIQPDATA_HPP


#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "PHAL_Dimension.hpp"
#include "Albany_StateInfoStruct.hpp"

namespace Albany {

  // Helper class for PUMIQPData
  template<typename DataType, unsigned Dim>
  struct PUMIQPData_Traits { };

  template<typename DataType, unsigned Dim, class traits = PUMIQPData_Traits<DataType, Dim> >
  class PUMIQPData {

  public:

    PUMIQPData(const std::string& name, const std::vector<PHX::DataLayout::size_type>& dim,  const bool output = false);
    ~PUMIQPData(){}

    //! Type of traits class being used
    typedef traits traits_type;

    //! Define the field type
    typedef typename traits_type::field_type field_type;

    void reAllocateBuffer(const std::size_t nelems);
    Albany::MDArray getMDA(const std::size_t nElemsInBucket);

    const std::string name;      // Name of data field
    const bool output;           // Is field output to disk each time step (or at end of simulation)?
    std::vector<DataType> buffer;        // array storage for shards::Array
    std::vector<PHX::DataLayout::size_type> dims;
    int nfield_dofs;                    // total number of dofs in this field
    std::size_t beginning_index;        // Buffer starting location for the next array allocation

  };

// Explicit template definitions in support of the above

  // Scalar value
  template <typename T>
  struct PUMIQPData_Traits<T, 1> {

    enum { size = 1 }; // One array dimension tags: Cell
    typedef shards::Array<T, shards::NaturalOrder, Cell> field_type ;
    static Albany::MDArray buildArray(T *buf, unsigned nelems, std::vector<PHX::DataLayout::size_type>& dims){

      return field_type(buf, nelems);

    }

  };

  // QPScalar
  template <typename T>
  struct PUMIQPData_Traits<T, 2> {

    enum { size = 2 }; // Two array dimension tags: Cell and QuadPoint
    typedef shards::Array<T, shards::NaturalOrder, Cell, QuadPoint> field_type ;
    static Albany::MDArray buildArray(T *buf, unsigned nelems, std::vector<PHX::DataLayout::size_type>& dims){

      return field_type(buf, nelems, dims[1]);

    }

  };

  // QPVector
  template <typename T>
  struct PUMIQPData_Traits<T, 3> {

    enum { size = 3 }; // Three array dimension tags: Cell, QuadPoint, and Dim
    typedef shards::Array<T, shards::NaturalOrder, Cell, QuadPoint, Dim> field_type ;
    static Albany::MDArray buildArray(T *buf, unsigned nelems, std::vector<PHX::DataLayout::size_type>& dims){

      return field_type(buf, nelems, dims[1], dims[2]);

    }

  };

  // QPTensor
  template <typename T>
  struct PUMIQPData_Traits<T, 4> {

    enum { size = 4 }; // Four array dimension tags: Cell, QuadPoint, Dim, and Dim
    typedef shards::Array<T, shards::NaturalOrder, Cell, QuadPoint, Dim, Dim> field_type ;
    static Albany::MDArray buildArray(T *buf, unsigned nelems, std::vector<PHX::DataLayout::size_type>& dims){

      return field_type(buf, nelems, dims[1], dims[2], dims[3]);

    }

  };


}

// Define macro for explicit template instantiation
#define PUMIQPDATA_INSTANTIATE_TEMPLATE_CLASS_1D(type, name) \
  template class name<type, 1>;
#define PUMIQPDATA_INSTANTIATE_TEMPLATE_CLASS_2D(type, name) \
  template class name<type, 2>;
#define PUMIQPDATA_INSTANTIATE_TEMPLATE_CLASS_3D(type, name) \
  template class name<type, 3>;
#define PUMIQPDATA_INSTANTIATE_TEMPLATE_CLASS_4D(type, name) \
  template class name<type, 4>;

#define PUMIQPDATA_INSTANTIATE_TEMPLATE_CLASS(name) \
  PUMIQPDATA_INSTANTIATE_TEMPLATE_CLASS_1D(double, name) \
  PUMIQPDATA_INSTANTIATE_TEMPLATE_CLASS_2D(double, name) \
  PUMIQPDATA_INSTANTIATE_TEMPLATE_CLASS_3D(double, name) \
  PUMIQPDATA_INSTANTIATE_TEMPLATE_CLASS_4D(double, name)

#endif
