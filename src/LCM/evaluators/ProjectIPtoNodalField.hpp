//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_ProjectIPtoNodalField_hpp)
#define LCM_ProjectIPtoNodalField_hpp

#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_MDField.hpp>
#include <Phalanx_DataLayout.hpp>
#include <Teuchos_ParameterList.hpp>
#include "Albany_ProblemUtils.hpp"

#include "Stratimikos_DefaultLinearSolverBuilder.hpp"

//#define PROJ_INTERP_TEST

namespace LCM {
/*! 
 * \brief Evaluator to compute a nodal stress field from integration points.
 *
 * This class implements the method described in Section 3.1 of
 *     Jiao, Xiangmin, and Michael T. Heath. "Common‐refinement‐based data
 *     transfer between non‐matching meshes in multiphysics simulations."
 *     Int. J. for Num. Meth. in Eng. 61.14 (2004): 2402-2427.
 * evaluateFields() assembles (i) the consistent mass matrix or, optionally, the
 * lumped mass matrix M and (ii) the integral over each element of the projected
 * quantity b. Then postEvaluate() solves the linear equation M x = b and
 * reports x to STK's nodal database.
 *   The graph describing the mass matrix's structure is created in Albany::
 * STKDiscretization::meshToGraph().
 */
template<typename EvalT, typename Traits>
class ProjectIPtoNodalFieldBase : 
    public PHX::EvaluatorWithBaseImpl<Traits>,
    public PHX::EvaluatorDerived<EvalT, Traits>
{
public:
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  ///
  /// Constructor
  ///
  ProjectIPtoNodalFieldBase(Teuchos::ParameterList& p,
                            const Teuchos::RCP<Albany::Layouts>& dl,
                            const Albany::MeshSpecsStruct* mesh_specs);
  
  ///
  /// Phalanx method to allocate space
  ///
  void postRegistrationSetup(typename Traits::SetupData d,
                             PHX::FieldManager<Traits>& vm);

  ///
  /// These functions are defined in the specializations
  ///
  void preEvaluate(typename Traits::PreEvalData d) = 0;
  void postEvaluate(typename Traits::PostEvalData d) = 0;
  void evaluateFields(typename Traits::EvalData d) = 0;

  Teuchos::RCP<const PHX::FieldTag> getEvaluatedFieldTag() const {
    return field_tag_;
  }

  Teuchos::RCP<const PHX::FieldTag> getResponseFieldTag() const {
    return field_tag_;
  }
    
protected:
  int number_of_fields_;

  bool output_to_exodus_;
  bool output_node_data_;

  // Represent the Field Layout by an enumerated type.
  struct EFieldLayout {
    enum Enum { scalar, vector, tensor };
    static Enum fromString(const std::string& user_str)
      throw (Teuchos::Exceptions::InvalidParameterValue);
  };

  std::vector<std::string> ip_field_names_;
  std::vector<typename EFieldLayout::Enum> ip_field_layouts_;
  std::vector<std::string> nodal_field_names_;

  int num_vecs_;
  int num_pts_;
  int num_dims_;
  int num_nodes_;
  int num_vertices_;
    
  std::vector<PHX::MDField<ScalarT> > ip_fields_;
  PHX::MDField<RealType,Cell,Node,QuadPoint> BF;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint> wBF;

  bool sep_by_eb_;
  std::string eb_name_;

  Teuchos::RCP< PHX::Tag<ScalarT> > field_tag_;
  Albany::StateManager* p_state_mgr_;

  Teuchos::RCP<Tpetra_MultiVector> source_load_vector_;

  Stratimikos::DefaultLinearSolverBuilder linearSolverBuilder_;
  Teuchos::RCP<Thyra::LinearOpWithSolveFactoryBase<ST> > lowsFactory_;

#ifdef PROJ_INTERP_TEST
  PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim> coords_qp_;
  PHX::MDField<MeshScalarT,Cell,Vertex,Dim> coords_verts_;
#endif
};

template<typename EvalT, typename Traits>
class ProjectIPtoNodalField
  : public ProjectIPtoNodalFieldBase<EvalT, Traits> {
public:
  ProjectIPtoNodalField(Teuchos::ParameterList& p,
                        const Teuchos::RCP<Albany::Layouts>& dl,
                        const Albany::MeshSpecsStruct* mesh_specs)
    : ProjectIPtoNodalFieldBase<EvalT, Traits>(p, dl, mesh_specs) {}
  void preEvaluate(typename Traits::PreEvalData d) {}
  void postEvaluate(typename Traits::PostEvalData d) {}
  void evaluateFields(typename Traits::EvalData d) {}
};

template<typename Traits>
class ProjectIPtoNodalField<PHAL::AlbanyTraits::Residual,Traits>
  : public ProjectIPtoNodalFieldBase<PHAL::AlbanyTraits::Residual, Traits> {
public:
  ProjectIPtoNodalField(Teuchos::ParameterList& p,
                        const Teuchos::RCP<Albany::Layouts>& dl,
                        const Albany::MeshSpecsStruct* mesh_specs);
  void preEvaluate(typename Traits::PreEvalData d);
  void postEvaluate(typename Traits::PostEvalData d);
  void evaluateFields(typename Traits::EvalData d);

  static ProjectIPtoNodalField* create(
    Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl);

private:
  // Declare a class hierarchy of mass matrix types. mass_matrix_ has to be in
  // this specialization, at least for now, because its implementation of fill()
  // is valid only for AlbanyTraits::Residual. Later we might move it up to the
  // nonspecialized class and create separate fill() impls for each trait.
  class MassMatrix;
  class FullMassMatrix;
  class LumpedMassMatrix;
  Teuchos::RCP<MassMatrix> mass_matrix_;

  void fillRHS(const typename Traits::EvalData workset);
};

} // namespace LCM

#endif  // ProjectIPtoNodalField.hpp
