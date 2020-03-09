//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"

#include "Albany_ThyraUtils.hpp"
#include "Albany_NodalDOFManager.hpp"
#include "Albany_DistributedParameterLibrary.hpp"
#include "Albany_AbstractDiscretization.hpp"
#include "Albany_GlobalLocalIndexer.hpp"

#include "PHAL_SDirichletField.hpp"

#include "Albany_TpetraThyraUtils.hpp"

// **********************************************************************
// Genereric Template Code for Constructor and PostRegistrationSetup
// **********************************************************************

namespace PHAL {

template <typename EvalT, typename Traits>
SDirichletField_Base<EvalT, Traits>::
SDirichletField_Base(Teuchos::ParameterList& p) :
  PHAL::DirichletBase<EvalT, Traits>(p) {

  // Get field type and corresponding layouts
  field_name = p.get<std::string>("Field Name");
}

// **********************************************************************
// Specialization: Residual
// **********************************************************************
template<typename Traits>
SDirichletField<PHAL::AlbanyTraits::Residual, Traits>::
SDirichletField(Teuchos::ParameterList& p) :
  SDirichletField_Base<PHAL::AlbanyTraits::Residual, Traits>(p) {
}

template<typename Traits>
void
SDirichletField<PHAL::AlbanyTraits::Residual, Traits>::preEvaluate(
    typename Traits::EvalData dirichlet_workset)
{
#ifdef DEBUG_OUTPUT
  Teuchos::RCP<Teuchos::FancyOStream> out = Teuchos::VerboseObjectBase::getDefaultOStream();
  *out << "SDirichletField preEvaluate Residual\n";
#endif
  Teuchos::RCP<const Thyra_Vector> x = dirichlet_workset.x;
  Teuchos::ArrayRCP<ST> x_view = Teuchos::arcp_const_cast<ST>(Albany::getLocalData(x));

  const Albany::NodalDOFManager& fieldDofManager = dirichlet_workset.disc->getDOFManager(this->field_name);
  //MP: If the parameter is scalar, then the parameter offset is set to zero. Otherwise the parameter offset is the same of the solution's one.
  auto fieldNodeVs = dirichlet_workset.disc->getNodeVectorSpace(this->field_name);
  auto fieldVs = dirichlet_workset.disc->getVectorSpace(this->field_name);
  bool isFieldScalar = (fieldNodeVs->dim() == fieldVs->dim());
  int fieldOffset = isFieldScalar ? 0 : this->offset;
  const std::vector<GO>& nsNodesGIDs = dirichlet_workset.disc->getNodeSetGIDs().find(this->nodeSetID)->second;

  Teuchos::RCP<const Thyra_Vector> pvec = dirichlet_workset.distParamLib->get(this->field_name)->vector();
  Teuchos::ArrayRCP<const ST> p_constView = Albany::getLocalData(pvec);

  const std::vector<std::vector<int> >& nsNodes = dirichlet_workset.nodeSets->find(this->nodeSetID)->second;
  auto field_node_indexer = Albany::createGlobalLocalIndexer(fieldNodeVs);
  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
      int lunk = nsNodes[inode][this->offset];
      GO node_gid = nsNodesGIDs[inode];
      int lfield = fieldDofManager.getLocalDOF(field_node_indexer->getLocalElement(node_gid),fieldOffset);
      x_view[lunk] = p_constView[lfield];
  }
}



// **********************************************************************
template<typename Traits>
void
SDirichletField<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData dirichlet_workset) {
  Teuchos::RCP<Thyra_Vector> f      = dirichlet_workset.f;
  Teuchos::ArrayRCP<ST>      f_view = Albany::getNonconstLocalData(f);

  // Grab the vector of node GIDs for this Node Set ID
  auto&  ns_nodes = dirichlet_workset.nodeSets->find(this->nodeSetID)->second;

  for (size_t ns_node = 0; ns_node < ns_nodes.size(); ns_node++) {
    int const dof = ns_nodes[ns_node][this->offset];
    f_view[dof]   = 0.0;
  }
}

// **********************************************************************
// Specialization: Jacobian
// **********************************************************************
template<typename Traits>
SDirichletField<PHAL::AlbanyTraits::Jacobian, Traits>::
SDirichletField(Teuchos::ParameterList& p) :
  SDirichletField_Base<PHAL::AlbanyTraits::Jacobian, Traits>(p) {
}

template<typename Traits>
void
SDirichletField<PHAL::AlbanyTraits::Jacobian, Traits>::preEvaluate(
    typename Traits::EvalData dirichlet_workset)
{
#ifdef DEBUG_OUTPUT
  Teuchos::RCP<Teuchos::FancyOStream> out = Teuchos::VerboseObjectBase::getDefaultOStream();
  *out << "SDirichletField preEvaluate Jacobian\n";
#endif
  if(Teuchos::nonnull(dirichlet_workset.f)) {
    Teuchos::RCP<const Thyra_Vector> x = dirichlet_workset.x;
    Teuchos::ArrayRCP<ST> x_view = Teuchos::arcp_const_cast<ST>(Albany::getLocalData(x));

    const Albany::NodalDOFManager& fieldDofManager = dirichlet_workset.disc->getDOFManager(this->field_name);
    //MP: If the parameter is scalar, then the parameter offset is set to zero. Otherwise the parameter offset is the same of the solution's one.
    auto fieldNodeVs = dirichlet_workset.disc->getNodeVectorSpace(this->field_name);
    auto fieldVs = dirichlet_workset.disc->getVectorSpace(this->field_name);
    bool isFieldScalar = (fieldNodeVs->dim() == fieldVs->dim());
    int fieldOffset = isFieldScalar ? 0 : this->offset;
    const std::vector<GO>& nsNodesGIDs = dirichlet_workset.disc->getNodeSetGIDs().find(this->nodeSetID)->second;

    Teuchos::RCP<const Thyra_Vector> pvec = dirichlet_workset.distParamLib->get(this->field_name)->vector();
    Teuchos::ArrayRCP<const ST> p_constView = Albany::getLocalData(pvec);

    const std::vector<std::vector<int> >& nsNodes = dirichlet_workset.nodeSets->find(this->nodeSetID)->second;
    auto field_node_indexer = Albany::createGlobalLocalIndexer(fieldNodeVs);
    for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
        int lunk = nsNodes[inode][this->offset];
        GO node_gid = nsNodesGIDs[inode];
        int lfield = fieldDofManager.getLocalDOF(field_node_indexer->getLocalElement(node_gid),fieldOffset);
        x_view[lunk] = p_constView[lfield];
    }
  }
}

template <typename Traits>
void
SDirichletField<PHAL::AlbanyTraits::Jacobian, Traits>::set_row_and_col_is_dbc(
    typename Traits::EvalData dirichlet_workset)
{
  Teuchos::RCP<const Thyra_LinearOp> J = dirichlet_workset.Jac;

  auto  range_vs  = J->range();
  auto  col_vs    = Albany::getColumnSpace(J);
  auto& ns_nodes  = dirichlet_workset.nodeSets->find(this->nodeSetID)->second;
  auto  domain_vs = range_vs;  // we are assuming this!

  row_is_dbc_ = Thyra::createMember(range_vs);
  col_is_dbc_ = Thyra::createMember(col_vs);
  row_is_dbc_->assign(0.0);
  col_is_dbc_->assign(0.0);

  auto row_is_dbc_data = Albany::getNonconstLocalData(row_is_dbc_);
  for (size_t ns_node = 0; ns_node < ns_nodes.size(); ns_node++) {
    auto dof             = ns_nodes[ns_node][this->offset];
    row_is_dbc_data[dof] = 1.0;
  }

  auto cas_manager = Albany::createCombineAndScatterManager(domain_vs, col_vs);
  cas_manager->scatter(row_is_dbc_, col_is_dbc_, Albany::CombineMode::INSERT);
}

// **********************************************************************
template<typename Traits>
void SDirichletField<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData dirichlet_workset)
{

  Teuchos::RCP<Thyra_Vector>       f = dirichlet_workset.f;
  if(Teuchos::nonnull(f)) {
    Teuchos::RCP<const Thyra_Vector> x = dirichlet_workset.x;
    Teuchos::ArrayRCP<ST> x_view = Teuchos::arcp_const_cast<ST>(Albany::getLocalData(x));

    const Albany::NodalDOFManager& fieldDofManager = dirichlet_workset.disc->getDOFManager(this->field_name);
    //MP: If the parameter is scalar, then the parameter offset is set to zero. Otherwise the parameter offset is the same of the solution's one.
    auto fieldNodeVs = dirichlet_workset.disc->getNodeVectorSpace(this->field_name);
    auto fieldVs = dirichlet_workset.disc->getVectorSpace(this->field_name);
    bool isFieldScalar = (fieldNodeVs->dim() == fieldVs->dim());
    int fieldOffset = isFieldScalar ? 0 : this->offset;
    const std::vector<GO>& nsNodesGIDs = dirichlet_workset.disc->getNodeSetGIDs().find(this->nodeSetID)->second;

    Teuchos::RCP<const Thyra_Vector> pvec = dirichlet_workset.distParamLib->get(this->field_name)->vector();
    Teuchos::ArrayRCP<const ST> p_constView = Albany::getLocalData(pvec);

    const std::vector<std::vector<int> >& nsNodes = dirichlet_workset.nodeSets->find(this->nodeSetID)->second;
    auto field_node_indexer = Albany::createGlobalLocalIndexer(fieldNodeVs);
    for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
        int lunk = nsNodes[inode][this->offset];
        GO node_gid = nsNodesGIDs[inode];
        int lfield = fieldDofManager.getLocalDOF(field_node_indexer->getLocalElement(node_gid),fieldOffset);
        x_view[lunk] = p_constView[lfield];
    }
  }

  const std::vector<std::vector<int> >& nsNodes = dirichlet_workset.nodeSets->find(this->nodeSetID)->second;

  Teuchos::RCP<const Thyra_Vector> x = dirichlet_workset.x;
  //Teuchos::RCP<Thyra_Vector>       f = dirichlet_workset.f;
  Teuchos::RCP<Thyra_LinearOp>     J = dirichlet_workset.Jac;

  bool const fill_residual = f != Teuchos::null;

  auto f_view = fill_residual ? Albany::getNonconstLocalData(f) : Teuchos::null;
  Teuchos::Array<ST> entries;
  Teuchos::Array<LO> indices;
  Teuchos::Array<ST> value(1);
  value[0] = dirichlet_workset.j_coeff;;

  this->set_row_and_col_is_dbc(dirichlet_workset);

  auto     col_is_dbc_data = Albany::getLocalData(col_is_dbc_.getConst());
  auto     range_spmd_vs   = Albany::getSpmdVectorSpace(J->range());
  const LO num_local_rows  = range_spmd_vs->localSubDim();

  for (LO local_row = 0; local_row < num_local_rows; ++local_row) {
    Albany::getLocalRowValues(J, local_row, indices, entries);
    auto row_is_dbc = col_is_dbc_data[local_row] > 0;
    if (row_is_dbc && fill_residual == true) {
      int lunk = nsNodes[local_row][this->offset];
      f_view[lunk] = 0.0;
    }

    const LO num_row_entries = entries.size();

    for (LO row_entry = 0; row_entry < num_row_entries; ++row_entry) {
      auto local_col         = indices[row_entry];
      auto is_diagonal_entry = local_col == local_row;
      if ( is_diagonal_entry) { continue; }

      auto col_is_dbc = col_is_dbc_data[local_col] > 0;
      if (row_is_dbc || col_is_dbc) { entries[row_entry] = 0.0; }
    }
    Albany::setLocalRowValues(J, local_row, indices(), entries());
  }
}

// **********************************************************************
// Specialization: Tangent
// **********************************************************************
template<typename Traits>
SDirichletField<PHAL::AlbanyTraits::Tangent, Traits>::
SDirichletField(Teuchos::ParameterList& p) :
  SDirichletField_Base<PHAL::AlbanyTraits::Tangent, Traits>(p) {
}

template<typename Traits>
void
SDirichletField<PHAL::AlbanyTraits::Tangent, Traits>::preEvaluate(
    typename Traits::EvalData dirichlet_workset)
{

  #ifdef DEBUG_OUTPUT
    Teuchos::RCP<Teuchos::FancyOStream> out = Teuchos::VerboseObjectBase::getDefaultOStream();
    *out << "SDirichletField preEvaluate Tangent\n";
  #endif
  if(Teuchos::nonnull(dirichlet_workset.f)) {
    Teuchos::RCP<const Thyra_Vector> x = dirichlet_workset.x;
    Teuchos::ArrayRCP<ST> x_view = Teuchos::arcp_const_cast<ST>(Albany::getLocalData(x));

    const Albany::NodalDOFManager& fieldDofManager = dirichlet_workset.disc->getDOFManager(this->field_name);
    //MP: If the parameter is scalar, then the parameter offset is set to zero. Otherwise the parameter offset is the same of the solution's one.
    auto fieldNodeVs = dirichlet_workset.disc->getNodeVectorSpace(this->field_name);
    auto fieldVs = dirichlet_workset.disc->getVectorSpace(this->field_name);
    bool isFieldScalar = (fieldNodeVs->dim() == fieldVs->dim());
    int fieldOffset = isFieldScalar ? 0 : this->offset;
    const std::vector<GO>& nsNodesGIDs = dirichlet_workset.disc->getNodeSetGIDs().find(this->nodeSetID)->second;

    Teuchos::RCP<const Thyra_Vector> pvec = dirichlet_workset.distParamLib->get(this->field_name)->vector();
    Teuchos::ArrayRCP<const ST> p_constView = Albany::getLocalData(pvec);

    const std::vector<std::vector<int> >& nsNodes = dirichlet_workset.nodeSets->find(this->nodeSetID)->second;
    auto field_node_indexer = Albany::createGlobalLocalIndexer(fieldNodeVs);
    for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
        int lunk = nsNodes[inode][this->offset];
        GO node_gid = nsNodesGIDs[inode];
        int lfield = fieldDofManager.getLocalDOF(field_node_indexer->getLocalElement(node_gid),fieldOffset);
        x_view[lunk] = p_constView[lfield];
    }
  }
}

// **********************************************************************
template<typename Traits>
void SDirichletField<PHAL::AlbanyTraits::Tangent, Traits>::
evaluateFields(typename Traits::EvalData dirichlet_workset) {

  Teuchos::RCP<const Thyra_Vector> x = dirichlet_workset.x;
  Teuchos::RCP<Thyra_Vector>       f = dirichlet_workset.f;

  Teuchos::RCP<const Thyra_MultiVector> Vx = dirichlet_workset.Vx;
  Teuchos::RCP<Thyra_MultiVector>       fp = dirichlet_workset.fp;
  Teuchos::RCP<Thyra_MultiVector>       JV = dirichlet_workset.JV;

  Teuchos::ArrayRCP<const ST> x_constView;
  Teuchos::ArrayRCP<ST>       f_nonconstView;

  Teuchos::ArrayRCP<Teuchos::ArrayRCP<const ST>> Vx_const2dView;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>>       JV_nonconst2dView;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>>       fp_nonconst2dView;

  if (f != Teuchos::null) {
    f_nonconstView = Albany::getNonconstLocalData(f);
  }
  if (JV != Teuchos::null) {
    JV_nonconst2dView = Albany::getNonconstLocalData(JV);
    Vx_const2dView    = Albany::getLocalData(Vx);
  }

  if (fp != Teuchos::null) {
    fp_nonconst2dView = Albany::getNonconstLocalData(fp);
  }

  const RealType j_coeff = dirichlet_workset.j_coeff;
  const std::vector<std::vector<int> >& nsNodes = dirichlet_workset.nodeSets->find(this->nodeSetID)->second;

  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
    int lunk = nsNodes[inode][this->offset];

    if (f != Teuchos::null) {
      f_nonconstView[lunk] = 0.0;
    }

    if (JV != Teuchos::null) {
      for (int i=0; i<dirichlet_workset.num_cols_x; i++) {
        JV_nonconst2dView[i][lunk] = j_coeff*Vx_const2dView[i][lunk];
      }
    }

    if (fp != Teuchos::null) {
      for (int i=0; i<dirichlet_workset.num_cols_p; i++) {
        fp_nonconst2dView[i][lunk] = 0;
      }
    }
  }
}

// **********************************************************************
// Specialization: DistParamDeriv
// **********************************************************************
template<typename Traits>
SDirichletField<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
SDirichletField(Teuchos::ParameterList& p) :
  SDirichletField_Base<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p) {
}

// **********************************************************************
template<typename Traits>
void SDirichletField<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
evaluateFields(typename Traits::EvalData dirichlet_workset) {

  bool isFieldParameter =  dirichlet_workset.dist_param_deriv_name == this->field_name;
  TEUCHOS_TEST_FOR_EXCEPTION(
      isFieldParameter,
      std::logic_error,
      "Error, SDirichletField cannot handle dirichlet parameter " <<  this->field_name << ", use DirichletField instead." << std::endl);

  bool trans = dirichlet_workset.transpose_dist_param_deriv;

  Teuchos::RCP<Thyra_MultiVector> fpV = dirichlet_workset.fpV;

  int num_cols = fpV->domain()->dim();

  const std::vector<std::vector<int> >& nsNodes = dirichlet_workset.nodeSets->find(this->nodeSetID)->second;

  if (trans) {
    // For (df/dp)^T*V we zero out corresponding entries in V
    Teuchos::RCP<Thyra_MultiVector> Vp = dirichlet_workset.Vp_bc;
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>> Vp_nonconst2dView = Albany::getNonconstLocalData(Vp);

    for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
      int lunk = nsNodes[inode][this->offset];
      for (int col=0; col<num_cols; ++col) {
        Vp_nonconst2dView[col][lunk] = 0.0;
      }
    }
  } else {
    // for (df/dp)*V we zero out corresponding entries in df/dp
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>> fpV_nonconst2dView = Albany::getNonconstLocalData(fpV);
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<const ST>> Vp_const2dView = Albany::getLocalData(dirichlet_workset.Vp);
    for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
      int lunk = nsNodes[inode][this->offset];
      for (int col=0; col<num_cols; ++col) {
        fpV_nonconst2dView[col][lunk] = 0.0;
      }
    }
  }
}

} // namespace PHAL
