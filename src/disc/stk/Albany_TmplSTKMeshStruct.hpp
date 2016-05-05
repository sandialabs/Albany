//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef ALBANY_TMPLSTKMESHSTRUCT_HPP
#define ALBANY_TMPLSTKMESHSTRUCT_HPP

#include "Albany_GenericSTKMeshStruct.hpp"

namespace Albany {

/*!
 * \brief A Template for STK mesh classes that generate their own mesh.
 */

//! Traits class for STK mesh classes that generate their own mesh
template<unsigned Dim>
struct albany_stk_mesh_traits { };

//! Element block specs
template<unsigned Dim, class traits = albany_stk_mesh_traits<Dim> >
struct EBSpecsStruct {

    EBSpecsStruct(){}

    //! Type of traits class being used
    typedef traits traits_type;

    //! Single element block initializer
    void Initialize(GO nnelems[], double blLength[]);

    //! Multiple element block initializer
    void Initialize(int i, const Teuchos::RCP<Teuchos::ParameterList>& params);

    //! Query function to determine if a given i, j, k value is in this element block
    // Note that elemNo is the logical lower left corner of the element being queried
    bool inEB(const std::vector<GO>& elemNo){ for(std::size_t i = 0; i < elemNo.size(); i++){
          if(elemNo[i] < min[i]) return false;
          if(elemNo[i] >= max[i]) return false;
        }
        return true;
    }

    //! Calculate the number of elements in this block on the given dimension
    GO numElems(int dim){ return (max[dim] - min[dim]);}

    //! Calculate the sizes of the elements in this element block
//    void calcElemSizes(std::vector<std::vector<double> > &h){
    void calcElemSizes(std::vector<double> h[]){
//        for(std::size_t i = 0; i < h.size(); i++)
        for(unsigned i = 0; i < Dim; i++)
          for(GO j = min[i]; j < max[i]; j++)
            h[i][j] = blLength[i] / double(max[i] - min[i]);
    }

    std::string name;      // Name of element block
    GO min[traits_type::size];       // Minimum logical coordinate of the block
    GO max[traits_type::size];       // Maximum logical coordinate of the block
    double blLength[traits_type::size];

};

//! Template for STK internal mesh generation classes

template<unsigned Dim, class traits = albany_stk_mesh_traits<Dim> >

  class TmplSTKMeshStruct : public GenericSTKMeshStruct {

    public:

    //! Type of traits class being used
    typedef traits traits_type;
    //! Default and optional element types created by the class (only meaningful in 2D - quads and tris)
    typedef typename traits_type::default_element_type default_element_type;
    typedef typename traits_type::optional_element_type optional_element_type;
    typedef typename traits_type::default_element_side_type default_element_side_type;

    //! Default constructor
    TmplSTKMeshStruct(const Teuchos::RCP<Teuchos::ParameterList>& params,
                      const Teuchos::RCP<Teuchos::ParameterList>& adaptParams,
                      const Teuchos::RCP<const Teuchos_Comm>& commT);

    ~TmplSTKMeshStruct() {};

    //! Sets mesh generation parameters
    void setFieldAndBulkData(
                  const Teuchos::RCP<const Teuchos_Comm>& commT,
                  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const unsigned int neq_,
                  const AbstractFieldContainer::FieldContainerRequirements& req,
                  const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                  const unsigned int worksetSize,
                  const std::map<std::string,Teuchos::RCP<Albany::StateInfoStruct> >& side_set_sis = {},
                  const std::map<std::string,AbstractFieldContainer::FieldContainerRequirements>& side_set_req = {}); // empty map as default

    //! Flag if solution has a restart values -- used in Init Cond
    bool hasRestartSolution() const {return false; }

    //! If restarting, convenience function to return restart data time
    double restartDataTime() const {return -1.0; }


    private:

    //! Build the mesh
    void buildMesh(const Teuchos::RCP<const Teuchos_Comm>& commT);


    //! Build a parameter list that contains valid input parameters
    Teuchos::RCP<const Teuchos::ParameterList>
      getValidDiscretizationParameters() const;

    //! Build STK parts and nodesets that correspond to the dimension of the problem and input values
    void DeclareParts(std::vector<EBSpecsStruct<Dim, traits>  > ebStructArray,
         std::vector<std::string> ssNames,
         std::vector<std::string> nsNames);

    GO nelem[traits_type::size];
    double scale[traits_type::size];
    unsigned int numEB;
    std::vector<double> x[traits_type::size];
//    std::vector<std::vector<double> > x;
    Teuchos::RCP<Tpetra_Map> elem_map;
    std::vector<EBSpecsStruct<Dim, traits> > EBSpecs;

    bool periodic_x, periodic_y, periodic_z;
    bool triangles; // Defaults to false, meaning quad elements

  };

// Explicit template definitions in support of the above

  template <>
  struct albany_stk_mesh_traits<0> {

    enum { size = 1 }; // stk wants one dimension
    typedef shards::Particle default_element_type;
    typedef shards::Particle optional_element_type;
    typedef shards::Particle default_element_side_type; // No sides in 0D

  };

  template <>
  struct albany_stk_mesh_traits<1> {

    enum { size = 1 };
    typedef shards::Line<2> default_element_type;
    typedef shards::Line<2> optional_element_type;
    typedef shards::Particle default_element_side_type; // No sides in 1D

  };

  template <>
  struct albany_stk_mesh_traits<2> {

    enum { size = 2 };
    typedef shards::Quadrilateral<4> default_element_type;
    typedef shards::Triangle<3> optional_element_type;
    typedef shards::Line<2> default_element_side_type;

  };

  template <>
  struct albany_stk_mesh_traits<3> {

    enum { size = 3 };
    typedef shards::Hexahedron<8> default_element_type;
    typedef shards::Hexahedron<8> optional_element_type;
    typedef shards::Quadrilateral<4> default_element_side_type;

  };

// Now, the explicit template function declarations (templated on dimension)

  template<> GO  EBSpecsStruct<0>::numElems(int i);
  template<> void EBSpecsStruct<0>::calcElemSizes(std::vector<double> h[]);

  template<> void EBSpecsStruct<0>::Initialize(GO nelems[], double blLen[]);
  template<> void EBSpecsStruct<0>::Initialize(int i, const Teuchos::RCP<Teuchos::ParameterList>& params);
  template<> void EBSpecsStruct<1>::Initialize(int i, const Teuchos::RCP<Teuchos::ParameterList>& params);
  template<> void EBSpecsStruct<2>::Initialize(int i, const Teuchos::RCP<Teuchos::ParameterList>& params);
  template<> void EBSpecsStruct<3>::Initialize(int i, const Teuchos::RCP<Teuchos::ParameterList>& params);

  template<> void TmplSTKMeshStruct<0>::buildMesh(const Teuchos::RCP<const Teuchos_Comm>& commT);
  template<> void TmplSTKMeshStruct<1>::buildMesh(const Teuchos::RCP<const Teuchos_Comm>& commT);
  template<> void TmplSTKMeshStruct<2>::buildMesh(const Teuchos::RCP<const Teuchos_Comm>& commT);
  template<> void TmplSTKMeshStruct<3>::buildMesh(const Teuchos::RCP<const Teuchos_Comm>& commT);

  template<> void TmplSTKMeshStruct<0, albany_stk_mesh_traits<0> >::setFieldAndBulkData(
                  const Teuchos::RCP<const Teuchos_Comm>& commT,
                  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const unsigned int neq_,
                  const AbstractFieldContainer::FieldContainerRequirements& req,
                  const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                  const unsigned int worksetSize,
                  const std::map<std::string,Teuchos::RCP<Albany::StateInfoStruct> >& side_set_sis,
                  const std::map<std::string,AbstractFieldContainer::FieldContainerRequirements>& side_set_req);

  template<> Teuchos::RCP<const Teuchos::ParameterList> TmplSTKMeshStruct<0>::getValidDiscretizationParameters() const;
  template<> Teuchos::RCP<const Teuchos::ParameterList> TmplSTKMeshStruct<1>::getValidDiscretizationParameters() const;
  template<> Teuchos::RCP<const Teuchos::ParameterList> TmplSTKMeshStruct<2>::getValidDiscretizationParameters() const;
  template<> Teuchos::RCP<const Teuchos::ParameterList> TmplSTKMeshStruct<3>::getValidDiscretizationParameters() const;

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
