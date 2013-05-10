//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_QPDATA_HPP
#define ALBANY_QPDATA_HPP


#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_StateInfoStruct.hpp"
#include "PHAL_Dimension.hpp"

namespace Albany {

  // Helper class for QPData
  template<unsigned Dim>
  struct QPData_Traits { };

  template<unsigned Dim, class traits = QPData_Traits<Dim> >
  class QPData {

  public:

    QPData(const std::string& name, const std::vector<int>& dim);
    ~QPData();

    //! Type of traits class being used
    typedef traits traits_type;

    //! Define the field type
    typedef typename traits_type::field_type field_type;

//    Albany::MDArray *allocateArray(unsigned nElemsInBucket);
//    Albany::MDArray *allocateArray(double *buf, unsigned nElemsInBucket);
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
  struct QPData_Traits<1> { 

    enum { size = 1 }; // One array dimension tags: Cell 
    typedef shards::Array<double, shards::NaturalOrder, Cell> field_type ;
    static field_type* buildArray(double *buf, unsigned nelems, std::vector<int>& dims){

      return new field_type(buf, nelems);

    }

  };

  // QPScalar
  template <>
  struct QPData_Traits<2> { 

    enum { size = 2 }; // Two array dimension tags: Cell and QuadPoint
    typedef shards::Array<double, shards::NaturalOrder, Cell, QuadPoint> field_type ;
    static field_type* buildArray(double *buf, unsigned nelems, std::vector<int>& dims){

      return new field_type(buf, nelems, dims[1]);

    }

  };

  // QPVector
  template <>
  struct QPData_Traits<3> { 

    enum { size = 3 }; // Three array dimension tags: Cell, QuadPoint, and Dim
    typedef shards::Array<double, shards::NaturalOrder, Cell, QuadPoint, Dim> field_type ;
    static field_type* buildArray(double *buf, unsigned nelems, std::vector<int>& dims){

      return new field_type(buf, nelems, dims[1], dims[2]);

    }

  };

  // QPTensor
  template <>
  struct QPData_Traits<4> { 

    enum { size = 4 }; // Four array dimension tags: Cell, QuadPoint, Dim, and Dim
    typedef shards::Array<double, shards::NaturalOrder, Cell, QuadPoint, Dim, Dim> field_type ;
    static field_type* buildArray(double *buf, unsigned nelems, std::vector<int>& dims){

      return new field_type(buf, nelems, dims[1], dims[2], dims[3]);

    }

  };


}

// Define macro for explicit template instantiation
#define QPDATA_INSTANTIATE_TEMPLATE_CLASS_1D(name) \
  template class name<1>;
#define QPDATA_INSTANTIATE_TEMPLATE_CLASS_2D(name) \
  template class name<2>;
#define QPDATA_INSTANTIATE_TEMPLATE_CLASS_3D(name) \
  template class name<3>;
#define QPDATA_INSTANTIATE_TEMPLATE_CLASS_4D(name) \
  template class name<4>;

#define QPDATA_INSTANTIATE_TEMPLATE_CLASS(name) \
  QPDATA_INSTANTIATE_TEMPLATE_CLASS_1D(name) \
  QPDATA_INSTANTIATE_TEMPLATE_CLASS_2D(name) \
  QPDATA_INSTANTIATE_TEMPLATE_CLASS_3D(name) \
  QPDATA_INSTANTIATE_TEMPLATE_CLASS_4D(name)

#endif
