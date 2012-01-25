/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Glen Hansen, gahanse@sandia.gov                    *
\********************************************************************/


#ifndef ALBANY_TMPLSTKMESHSTRUCT_HPP
#define ALBANY_TMPLSTKMESHSTRUCT_HPP

#include "Albany_GenericSTKMeshStruct.hpp"

namespace Albany {

/*!
 * \brief A Template for STK mesh classes that generate their own mesh.
 */

//! Traits class for STK mesh classes that generate their own mesh
template<int Dim>
struct albany_stk_mesh_traits { };

//! Element block specs
template<int Dim>
struct EBSpecsStruct {

    EBSpecsStruct(){}

    //! Single element block initializer
    void Initialize(unsigned int nnelems[], double blLength[]);

    //! Multiple element block initializer
    void Initialize(int i, const Teuchos::RCP<Teuchos::ParameterList>& params);

    //! Query function to determine if a given i, j, k value is in this element block
    // Note that elemNo is the logical lower left corner of the element being queried
    bool inEB(const std::vector<int>& elemNo){ for(std::size_t i = 0; i < elemNo.size(); i++){
          if(elemNo[i] < min[i]) return false;
          if(elemNo[i] >= max[i]) return false;
        }
        return true;
    }

    //! Calculate the number of elements in this block on the given dimension
    int numElems(int dim){ return (max[dim] - min[dim]);}

    //! Calculate the sizes of the elements in this element block
//    void calcElemSizes(std::vector<std::vector<double> > &h){ 
    void calcElemSizes(std::vector<double> h[]){ 
//        for(std::size_t i = 0; i < h.size(); i++)
        for(std::size_t i = 0; i < Dim; i++)
          for(unsigned j = min[i]; j < max[i]; j++)
            h[i][j] = blLength[i] / double(max[i] - min[i]);
    }

    std::string name;      // Name of element block
    int min[Dim];       // Minimum logical coordinate of the block 
    int max[Dim];       // Maximum logical coordinate of the block 
    double blLength[Dim];      

};

//! Template for STK internal mesh generation classes

template<int Dim, class traits = albany_stk_mesh_traits<Dim> >

  class TmplSTKMeshStruct : public GenericSTKMeshStruct {

    public:

    //! Type of traits class being used
    typedef traits traits_type;
    //! Default and optional element types created by the class (only meaningful in 2D - quads and tris)
    typedef typename traits_type::default_element_type default_element_type;
    typedef typename traits_type::optional_element_type optional_element_type;

    //! Default constructor
    TmplSTKMeshStruct(const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const Teuchos::RCP<const Epetra_Comm>& comm);

    ~TmplSTKMeshStruct() {};

    //! Sets mesh generation parameters
    void setFieldAndBulkData(
                  const Teuchos::RCP<const Epetra_Comm>& comm,
                  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const unsigned int neq_,
                  const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                  const unsigned int worksetSize);


    private:

    //! Build the mesh
    void buildMesh();

    //! Build a parameter list that contains valid input parameters
    Teuchos::RCP<const Teuchos::ParameterList>
      getValidDiscretizationParameters() const;

    //! Build STK parts and nodesets that correspond to the dimension of the problem and input values
    void DeclareParts(std::vector<EBSpecsStruct<Dim> > ebStructArray, 
         std::vector<std::string> ssNames,
         std::vector<std::string> nsNames);

    unsigned int nelem[traits_type::size];
    double scale[traits_type::size];
    unsigned int numEB;
    std::vector<double> x[traits_type::size];
//    std::vector<std::vector<double> > x;
    Teuchos::RCP<Epetra_Map> elem_map;
    std::vector<EBSpecsStruct<Dim> > EBSpecs;

    bool periodic;
    bool triangles; // Defaults to false, meaning quad elements

  };

// Explicit template definitions in support of the above

  template <>
  struct albany_stk_mesh_traits<0> { 

    enum { size = 1 }; // stk wants one dimension
    typedef shards::Particle default_element_type;
    typedef shards::Particle optional_element_type;

  };

  template <>
  struct albany_stk_mesh_traits<1> { 

    enum { size = 1 };
    typedef shards::Line<2> default_element_type;
    typedef shards::Line<2> optional_element_type;

  };

  template <>
  struct albany_stk_mesh_traits<2> { 

    enum { size = 2 };
    typedef shards::Quadrilateral<4> default_element_type;
    typedef shards::Triangle<3> optional_element_type;

  };

  template <>
  struct albany_stk_mesh_traits<3> { 

    enum { size = 3 };
    typedef shards::Hexahedron<8> default_element_type;
    typedef shards::Hexahedron<8> optional_element_type;

  };

} // namespace Albany

// Define macro for explicit template instantiation
#define TMPLSTK_INSTANTIATE_TEMPLATE_CLASS_0D(name) \
  template class name<0>;
#define TMPLSTK_INSTANTIATE_TEMPLATE_CLASS_1D(name) \
  template class name<1>;
#define TMPLSTK_INSTANTIATE_TEMPLATE_CLASS_2D(name) \
  template class name<2>;
#define TMPLSTK_INSTANTIATE_TEMPLATE_CLASS_3D(name) \
  template class name<3>;

#define TMPLSTK_INSTANTIATE_TEMPLATE_CLASS(name) \
  TMPLSTK_INSTANTIATE_TEMPLATE_CLASS_0D(name) \
  TMPLSTK_INSTANTIATE_TEMPLATE_CLASS_1D(name) \
  TMPLSTK_INSTANTIATE_TEMPLATE_CLASS_2D(name) \
  TMPLSTK_INSTANTIATE_TEMPLATE_CLASS_3D(name)

#endif // ALBANY_TMPLSTKMESHSTRUCT_HPP
