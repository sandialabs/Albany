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

namespace LCM
{
/// 
/// \brief Evaltuator to compute a nodal stress field
///
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
                     const Teuchos::RCP<Albany::Layouts>& dl);
  
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

  Teuchos::RCP<const Teuchos::ParameterList> getValidProjectIPtoNodalFieldParameters() const;

  int number_of_fields_;

  std::vector<std::string> ip_field_names_;
  std::vector<std::string> ip_field_layouts_;
  std::vector<std::string> nodal_field_names_;

  std::size_t num_vecs_;

  std::size_t num_pts_;
  std::size_t num_dims_;
  std::size_t num_nodes_;
  std::size_t num_vertices_;
    
  std::vector<PHX::MDField<ScalarT> > ip_fields_;
  PHX::MDField<RealType,Cell,Node,QuadPoint> BF;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint> wBF;

  bool output_to_exodus_;
  bool output_node_data_;

  Teuchos::RCP< PHX::Tag<ScalarT> > field_tag_;
  Albany::StateManager* p_state_mgr_;

  Teuchos::RCP<Tpetra_CrsMatrix> mass_matrix;
  Teuchos::RCP<Tpetra_MultiVector> source_load_vector;
  Teuchos::RCP<Tpetra_MultiVector> node_projected_ip_vector;

};

template<typename EvalT, typename Traits>
class ProjectIPtoNodalField
  : public ProjectIPtoNodalFieldBase<EvalT, Traits> {
public:
  ProjectIPtoNodalField(Teuchos::ParameterList& p,
                   const Teuchos::RCP<Albany::Layouts>& dl) :
    ProjectIPtoNodalFieldBase<EvalT, Traits>(p, dl){}
  void preEvaluate(typename Traits::PreEvalData d){}
  void postEvaluate(typename Traits::PostEvalData d){}
  void evaluateFields(typename Traits::EvalData d){}
};

  // **************************************************************
  // **************************************************************
  // * Specializations
  // **************************************************************
  // **************************************************************

// **************************************************************
// Residual 
// **************************************************************
template<typename Traits>
class ProjectIPtoNodalField<PHAL::AlbanyTraits::Residual,Traits>
  : public ProjectIPtoNodalFieldBase<PHAL::AlbanyTraits::Residual, Traits> {
public:
  ProjectIPtoNodalField(Teuchos::ParameterList& p,
                   const Teuchos::RCP<Albany::Layouts>& dl);
  void preEvaluate(typename Traits::PreEvalData d);
  void postEvaluate(typename Traits::PostEvalData d);
  void evaluateFields(typename Traits::EvalData d);
};
}

#endif  // ProjectIPtoNodalField.hpp
