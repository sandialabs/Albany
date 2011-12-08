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

    void Initialize(int i, const Teuchos::RCP<Teuchos::ParameterList>& params);

    bool inEB(const std::vector<double>& centroid){ for(std::size_t i = 0; i < centroid.size(); i++){
          if(centroid[i] < scale[i] * min[i]) return false;
          if(centroid[i] > scale[i] * max[i]) return false;
        }
        return true;
    }

    std::string name;      // Name of element block
    double min[Dim];       // Minimum parametric coordinate of the block (0 to 1, unscaled)
    double max[Dim];       // Maximum parametric coordinate of the block (0 to 1 unscaled)
    double scale[Dim];       // Maximum parametric coordinate of the block (0 to 1 unscaled)
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
    unsigned int numEB;
    std::vector<double> x[traits_type::size];
    Teuchos::RCP<Epetra_Map> elem_map;
    std::vector<EBSpecsStruct<Dim> > EBSpecs;

    bool periodic;
    bool triangles; // Defaults to false, meaning quad elements

  };


// Explicit template signatures needed for the above

  template <>
  struct EBSpecsStruct<0> {

    EBSpecsStruct(){}

    void Initialize(int i, const Teuchos::RCP<Teuchos::ParameterList>& params){
      // Never more than one element block in a 0D problem
      name = "Block0";
      scale[0] = 1.0;
    }

    std::string name;      // Name of element block
    double scale[0];       // Maximum parametric coordinate of the block (0 to 1 unscaled)
  };

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

// Specific template instantiations for convenience

  typedef TmplSTKMeshStruct<0> Point0DSTKMeshStruct;
  typedef TmplSTKMeshStruct<1> Line1DSTKMeshStruct;
  typedef TmplSTKMeshStruct<2> Rect2DSTKMeshStruct;
  typedef TmplSTKMeshStruct<3> Cube3DSTKMeshStruct;


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
