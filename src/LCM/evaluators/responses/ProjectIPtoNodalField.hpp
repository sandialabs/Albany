//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_ProjectIPtoNodalField_hpp)
#define LCM_ProjectIPtoNodalField_hpp

#include <Phalanx_DataLayout.hpp>
#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_MDField.hpp>
#include <Teuchos_ParameterList.hpp>
#include "Albany_ProblemUtils.hpp"
#include "Albany_StateManager.hpp"

#include "Stratimikos_DefaultLinearSolverBuilder.hpp"

// Running this test all the time is generally quite cheap, so let's do that.
#define PROJ_INTERP_TEST

namespace LCM {
/*!
 * \brief Evaluator to compute a nodal stress field from integration
 *        points. Only the Residual evaluation type is implemented.
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

template <typename EvalT, typename Traits>
class ProjectIPtoNodalFieldBase : public PHX::EvaluatorWithBaseImpl<Traits>,
                                  public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  ProjectIPtoNodalFieldBase(const Teuchos::RCP<Albany::Layouts>& dl)
  {
    field_tag_ = Teuchos::rcp(new PHX::Tag<typename EvalT::ScalarT>(
        "Project IP to Nodal Field", dl->dummy));
    this->addEvaluatedField(*field_tag_);
  }
  Teuchos::RCP<const PHX::FieldTag>
  getEvaluatedFieldTag() const
  {
    return field_tag_;
  }
  Teuchos::RCP<const PHX::FieldTag>
  getResponseFieldTag() const
  {
    return field_tag_;
  }

 private:
  Teuchos::RCP<PHX::Tag<typename EvalT::ScalarT>> field_tag_;
};

template <typename EvalT, typename Traits>
class ProjectIPtoNodalField : public ProjectIPtoNodalFieldBase<EvalT, Traits>
{
 public:
  ProjectIPtoNodalField(
      Teuchos::ParameterList&              p,
      const Teuchos::RCP<Albany::Layouts>& dl,
      const Albany::MeshSpecsStruct*       mesh_specs)
      : ProjectIPtoNodalFieldBase<EvalT, Traits>(dl)
  {
  }
  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm)
  {
  }
  void
  preEvaluate(typename Traits::PreEvalData d)
  {
  }
  void
  postEvaluate(typename Traits::PostEvalData d)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true, std::logic_error, "Should never be called.");
  }
  void
  evaluateFields(typename Traits::EvalData d)
  {
  }
};

class ProjectIPtoNodalFieldManager;
class ProjectIPtoNodalFieldQuadrature;

template <typename Traits>
class ProjectIPtoNodalField<PHAL::AlbanyTraits::Residual, Traits>
    : public ProjectIPtoNodalFieldBase<PHAL::AlbanyTraits::Residual, Traits>
{
 public:
  ProjectIPtoNodalField(
      Teuchos::ParameterList&              p,
      const Teuchos::RCP<Albany::Layouts>& dl,
      const Albany::MeshSpecsStruct*       mesh_specs);
  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);
  void
  preEvaluate(typename Traits::PreEvalData d);
  void
  postEvaluate(typename Traits::PostEvalData d);
  void
  evaluateFields(typename Traits::EvalData d);

 private:
  typedef PHAL::AlbanyTraits::Residual::ScalarT     ScalarT;
  typedef PHAL::AlbanyTraits::Residual::MeshScalarT MeshScalarT;

  Teuchos::RCP<ProjectIPtoNodalFieldManager> mgr_;

  bool output_to_exodus_;
  bool output_node_data_;

  // Represent the Field Layout by an enumerated type.
  struct EFieldLayout
  {
    enum Enum
    {
      scalar,
      vector,
      tensor
    };
    static Enum
    fromString(const std::string& user_str);
  };

  std::vector<std::string>                 ip_field_names_;
  std::vector<typename EFieldLayout::Enum> ip_field_layouts_;
  std::vector<std::string>                 nodal_field_names_;

  int ndb_start_, num_fields_, num_pts_, num_dims_, num_nodes_;

  std::vector<PHX::MDField<const ScalarT>>               ip_fields_;
  PHX::MDField<const RealType, Cell, Node, QuadPoint>    BF;
  PHX::MDField<const MeshScalarT, Cell, Node, QuadPoint> wBF;

#ifdef PROJ_INTERP_TEST
  PHX::MDField<ScalarT>                                 test_ip_field_;
  PHX::MDField<const MeshScalarT, Cell, QuadPoint, Dim> coords_qp_;
#endif
  typedef Intrepid2::Basis<PHX::Device, RealType, RealType> Intrepid2Basis;
  PHX::MDField<const MeshScalarT, Cell, Vertex, Dim>        coords_verts_;
  Teuchos::RCP<ProjectIPtoNodalFieldQuadrature>             quad_mgr_;

  Albany::StateManager* p_state_mgr_;

  Stratimikos::DefaultLinearSolverBuilder               linearSolverBuilder_;
  Teuchos::RCP<Thyra::LinearOpWithSolveFactoryBase<ST>> lowsFactory_;

  bool
  initManager(Teuchos::ParameterList* const pl, const std::string& key_suffix);
  void
  fillRHS(const typename Traits::EvalData workset);
};

}  // namespace LCM

#endif  // ProjectIPtoNodalField.hpp
