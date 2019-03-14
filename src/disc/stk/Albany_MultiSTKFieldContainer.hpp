//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/12/14: Epetra ifdef'ed out if ALBANY_EPETRA_EXE turned off

#ifndef ALBANY_MULTISTKFIELDCONT_HPP
#define ALBANY_MULTISTKFIELDCONT_HPP

#include "Albany_GenericSTKFieldContainer.hpp"
#include "Teuchos_Array.hpp"

namespace Albany {

template<bool Interleaved>
class MultiSTKFieldContainer : public GenericSTKFieldContainer<Interleaved>
{
public:

  MultiSTKFieldContainer(const Teuchos::RCP<Teuchos::ParameterList>& params_,
                         const Teuchos::RCP<stk::mesh::MetaData>& metaData_,
                         const Teuchos::RCP<stk::mesh::BulkData>& bulkData_,
                         const int neq_,
                         const AbstractFieldContainer::FieldContainerRequirements& req,
                         const int numDim_,
                         const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                         const Teuchos::Array<Teuchos::Array<std::string> >& solution_vector,
                         const Teuchos::Array<std::string>& residual_vector);

  ~MultiSTKFieldContainer() = default;

  bool hasResidualField           () const { return haveResidual;            }
  bool hasSphereVolumeField       () const { return buildSphereVolume;       }
  bool hasLatticeOrientationField () const { return buildLatticeOrientation; }

  void fillSolnVector(Thyra_Vector& soln, stk::mesh::Selector& sel, const Teuchos::RCP<const Thyra_VectorSpace>& node_vs);
  void fillVector(Thyra_Vector& field_vector, const std::string&  field_name, stk::mesh::Selector& field_selection,
                  const Teuchos::RCP<const Thyra_VectorSpace>& field_node_vs, const NodalDOFManager& nodalDofManager);
  void fillSolnMultiVector(Thyra_MultiVector& soln, stk::mesh::Selector& sel, const Teuchos::RCP<const Thyra_VectorSpace>& node_vs);
  void saveVector(const Thyra_Vector& field_vector, const std::string&  field_name, stk::mesh::Selector& field_selection,
                  const Teuchos::RCP<const Thyra_VectorSpace>& field_node_vs, const NodalDOFManager& nodalDofManager);
  void saveSolnVector(const Thyra_Vector& soln, stk::mesh::Selector& sel, const Teuchos::RCP<const Thyra_VectorSpace>& node_vs);
  void saveSolnVector(const Thyra_Vector& soln, const Thyra_Vector& soln_dot, stk::mesh::Selector& sel,
                      const Teuchos::RCP<const Thyra_VectorSpace>& node_vs);
  void saveSolnVector(const Thyra_Vector& soln, const Thyra_Vector& soln_dot,const Thyra_Vector& soln_dotdot, stk::mesh::Selector& sel,
                      const Teuchos::RCP<const Thyra_VectorSpace>& node_vs);
  void saveResVector(const Thyra_Vector& res, stk::mesh::Selector& sel, const Teuchos::RCP<const Thyra_VectorSpace>& node_vs);
  void saveSolnMultiVector(const Thyra_MultiVector& soln, stk::mesh::Selector& sel, const Teuchos::RCP<const Thyra_VectorSpace>& node_vs);

  void transferSolutionToCoords();

private:
  void fillVectorImpl (Thyra_Vector& field_vector,
                       const std::string& field_name,
                       stk::mesh::Selector& field_selection,
                       const Teuchos::RCP<const Thyra_VectorSpace>& field_node_vs,
                       const NodalDOFManager& nodalDofManager,
                       const int offset);
  void saveVectorImpl (const Thyra_Vector& field_vector,
                       const std::string& field_name,
                       stk::mesh::Selector& field_selection,
                       const Teuchos::RCP<const Thyra_VectorSpace>& field_node_vs,
                       const NodalDOFManager& nodalDofManager,
                       const int offset);

  void initializeSTKAdaptation();

  bool haveResidual;

  bool buildSphereVolume;
  bool buildLatticeOrientation;

  // Containers for residual and solution

  Teuchos::Array<Teuchos::Array<std::string> > sol_vector_name;
  Teuchos::Array<Teuchos::Array<int> > sol_index;

  Teuchos::Array<std::string> res_vector_name;
  Teuchos::Array<int> res_index;
};

} // namespace Albany



// Define macro for explicit template instantiation
#define MULTISTKFIELDCONTAINER_INSTANTIATE_TEMPLATE_CLASS_NONINTERLEAVED(name) \
  template class name<false>;
#define MULTISTKFIELDCONTAINER_INSTANTIATE_TEMPLATE_CLASS_INTERLEAVED(name) \
  template class name<true>;

#define MULTISTKFIELDCONTAINER_INSTANTIATE_TEMPLATE_CLASS(name) \
  MULTISTKFIELDCONTAINER_INSTANTIATE_TEMPLATE_CLASS_NONINTERLEAVED(name) \
  MULTISTKFIELDCONTAINER_INSTANTIATE_TEMPLATE_CLASS_INTERLEAVED(name)


#endif
