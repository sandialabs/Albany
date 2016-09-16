//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout_MDALayout.hpp"
#include "Tpetra_CrsMatrix.hpp"
#include "Albany_STKDiscretization.hpp"

// **********************************************************************
// Genereric Template Code for Constructor
// **********************************************************************

namespace FELIX {

template <typename EvalT, typename Traits>
HydrologyDirichletBase<EvalT, Traits>::
HydrologyDirichletBase(Teuchos::ParameterList& p) :
  offset    (p.get<int>("Equation Offset")),
  nodeSetID (p.get<std::string>("Node Set ID")),
  H_name    (p.get<std::string>("Ice Thickness Variable Name")),
  s_name    (p.get<std::string>("Surface Height Variable Name"))
{
  Teuchos::ParameterList& physics = *p.get<Teuchos::ParameterList*> ("FELIX Physical Parameters");
  rho_w = physics.get<double>("Water Density");
  g     = physics.get<double>("Gravity Acceleration");

  std::string name = "Hydrology Dirichlet " + nodeSetID;
  Teuchos::RCP<PHX::DataLayout> dummy = Teuchos::rcp(new PHX::MDALayout<Dummy>(0));
  const PHX::Tag<ScalarT> fieldTag(name, dummy);

  this->addEvaluatedField(fieldTag);
  this->setName(name);
}

// **********************************************************************
// Specialization: Residual
// **********************************************************************
template<typename Traits>
HydrologyDirichlet<PHAL::AlbanyTraits::Residual, Traits>::
HydrologyDirichlet(Teuchos::ParameterList& p) :
  HydrologyDirichletBase<PHAL::AlbanyTraits::Residual, Traits>(p) {
}

// **********************************************************************
template<typename Traits>
void
HydrologyDirichlet<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  Teuchos::RCP<Tpetra_Vector> fT = dirichletWorkset.fT;
  Teuchos::ArrayRCP<ST> fT_nonconstView = fT->get1dViewNonConst();
  Teuchos::RCP<const Tpetra_Vector> xT = dirichletWorkset.xT;
  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();

  Teuchos::RCP<Albany::STKDiscretization> disc = Teuchos::rcp_dynamic_cast<Albany::STKDiscretization>(dirichletWorkset.disc);
  TEUCHOS_TEST_FOR_EXCEPTION (disc==Teuchos::null, std::runtime_error, "Error! The discretization must be of type STKDiscretization.\n");

  const std::vector<GO>& nsNodesGIDs = disc->getNodeSetGIDs().find(this->nodeSetID)->second;
  const std::vector<std::vector<int> >& nsNodes = disc->getNodeSets().find(this->nodeSetID)->second;

  typedef Albany::AbstractSTKFieldContainer::ScalarFieldType  SFT;

  SFT* H = disc->getSTKMetaData().get_field<SFT> (stk::topology::NODE_RANK, this->H_name);
  SFT* s = disc->getSTKMetaData().get_field<SFT> (stk::topology::NODE_RANK, this->s_name);

  const std::size_t numNsNodes = nsNodesGIDs.size();
  stk::mesh::Entity node;
  int row;
  double H_val, s_val;
  for (std::size_t i=0; i<numNsNodes; ++i)
  {
    node = disc->getSTKBulkData().get_entity(stk::topology::NODE_RANK, nsNodesGIDs[i]+1);
    H_val = (stk::mesh::field_data(*H, node))[0];
    s_val = (stk::mesh::field_data(*s, node))[0];

    row = nsNodes[i][this->offset];
    fT_nonconstView[row] = xT_constView[row] - this->rho_w*this->g*(s_val-H_val);
  }
}

// **********************************************************************
// Specialization: Jacobian
// **********************************************************************
template<typename Traits>
HydrologyDirichlet<PHAL::AlbanyTraits::Jacobian, Traits>::
HydrologyDirichlet(Teuchos::ParameterList& p) :
  HydrologyDirichletBase<PHAL::AlbanyTraits::Jacobian, Traits>(p) {
}

// **********************************************************************
template<typename Traits>
void HydrologyDirichlet<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  Teuchos::RCP<Tpetra_CrsMatrix> jacT = dirichletWorkset.JacT;

  const RealType j_coeff = dirichletWorkset.j_coeff;
  const std::vector<std::vector<int> >& nsNodes = dirichletWorkset.nodeSets->find(this->nodeSetID)->second;

  Teuchos::Array<LO> index(1);
  Teuchos::Array<ST> value(1);
  size_t numEntriesT;
  value[0] = j_coeff;
  Teuchos::Array<ST> matrixEntriesT;
  Teuchos::Array<LO> matrixIndicesT;

  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
      int lunk = nsNodes[inode][this->offset];
      index[0] = lunk;
      numEntriesT = jacT->getNumEntriesInLocalRow(lunk);
      matrixEntriesT.resize(numEntriesT);
      matrixIndicesT.resize(numEntriesT);

      jacT->getLocalRowCopy(lunk, matrixIndicesT(), matrixEntriesT(), numEntriesT);
      for (int i=0; i<numEntriesT; i++) matrixEntriesT[i]=0;
      jacT->replaceLocalValues(lunk, matrixIndicesT(), matrixEntriesT());

      jacT->replaceLocalValues(lunk, index(), value());
  }

  Teuchos::RCP<Tpetra_Vector> fT = dirichletWorkset.fT;
  if (fT != Teuchos::null)
  {
    Teuchos::ArrayRCP<ST> fT_nonconstView = fT->get1dViewNonConst();
    Teuchos::RCP<const Tpetra_Vector> xT = dirichletWorkset.xT;
    Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();

    Teuchos::RCP<Albany::STKDiscretization> disc = Teuchos::rcp_dynamic_cast<Albany::STKDiscretization>(dirichletWorkset.disc);
    TEUCHOS_TEST_FOR_EXCEPTION (disc==Teuchos::null, std::runtime_error, "Error! The discretization must be of type STKDiscretization.\n");

    const std::vector<GO>& nsNodesGIDs = disc->getNodeSetGIDs().find(this->nodeSetID)->second;
    const std::vector<std::vector<int> >& nsNodes = disc->getNodeSets().find(this->nodeSetID)->second;

    typedef Albany::AbstractSTKFieldContainer::ScalarFieldType  SFT;

    SFT* H = disc->getSTKMetaData().get_field<SFT> (stk::topology::NODE_RANK, this->H_name);
    SFT* s = disc->getSTKMetaData().get_field<SFT> (stk::topology::NODE_RANK, this->s_name);

    const std::size_t numNsNodes = nsNodesGIDs.size();
    stk::mesh::Entity node;
    int row;
    double H_val, s_val;
    for (std::size_t i=0; i<numNsNodes; ++i)
    {
      node = disc->getSTKBulkData().get_entity(stk::topology::NODE_RANK, nsNodesGIDs[i]+1);
      H_val = (stk::mesh::field_data(*H, node))[0];
      s_val = (stk::mesh::field_data(*s, node))[0];

      row = nsNodes[i][this->offset];
      fT_nonconstView[row] = xT_constView[row] - this->rho_w*this->g*(s_val-H_val);
    }
  }
}

// **********************************************************************
// Specialization: Tangent
// **********************************************************************
template<typename Traits>
HydrologyDirichlet<PHAL::AlbanyTraits::Tangent, Traits>::
HydrologyDirichlet(Teuchos::ParameterList& p) :
  HydrologyDirichletBase<PHAL::AlbanyTraits::Tangent, Traits>(p) {
}

// **********************************************************************
template<typename Traits>
void HydrologyDirichlet<PHAL::AlbanyTraits::Tangent, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  Teuchos::RCP<Tpetra_MultiVector> JVT = dirichletWorkset.JVT;
  Teuchos::RCP<const Tpetra_MultiVector> VxT = dirichletWorkset.VxT;

  Teuchos::ArrayRCP<const ST> VxT_constView;

  const RealType j_coeff = dirichletWorkset.j_coeff;
  const std::vector<std::vector<int> >& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;

  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
    int lunk = nsNodes[inode][this->offset];

    if (JVT != Teuchos::null) {
      Teuchos::ArrayRCP<ST> JVT_nonconstView;
      for (int i=0; i<dirichletWorkset.num_cols_x; i++) {
        JVT_nonconstView = JVT->getDataNonConst(i);
        VxT_constView = VxT->getData(i);
        JVT_nonconstView[lunk] = j_coeff*VxT_constView[lunk];
      }
    }

  }

  Teuchos::RCP<Tpetra_Vector> fT = dirichletWorkset.fT;
  Teuchos::RCP<Tpetra_MultiVector> fpT = dirichletWorkset.fpT;
  if (fT != Teuchos::null || fpT != Teuchos::null)
  {
    Teuchos::RCP<const Tpetra_Vector> xT = dirichletWorkset.xT;
    Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();

    Teuchos::RCP<Albany::STKDiscretization> disc = Teuchos::rcp_dynamic_cast<Albany::STKDiscretization>(dirichletWorkset.disc);
    TEUCHOS_TEST_FOR_EXCEPTION (disc==Teuchos::null, std::runtime_error, "Error! The discretization must be of type STKDiscretization.\n");

    const std::vector<GO>& nsNodesGIDs = disc->getNodeSetGIDs().find(this->nodeSetID)->second;
    const std::vector<std::vector<int> >& nsNodes = disc->getNodeSets().find(this->nodeSetID)->second;

    typedef Albany::AbstractSTKFieldContainer::ScalarFieldType  SFT;

    SFT* H = disc->getSTKMetaData().get_field<SFT> (stk::topology::NODE_RANK, this->H_name);
    SFT* s = disc->getSTKMetaData().get_field<SFT> (stk::topology::NODE_RANK, this->s_name);

    const std::size_t numNsNodes = nsNodesGIDs.size();
    stk::mesh::Entity node;
    int row;
    double H_val, s_val;
    for (std::size_t i=0; i<numNsNodes; ++i)
    {
      node = disc->getSTKBulkData().get_entity(stk::topology::NODE_RANK, nsNodesGIDs[i]+1);
      H_val = (stk::mesh::field_data(*H, node))[0];
      s_val = (stk::mesh::field_data(*s, node))[0];

      row = nsNodes[i][this->offset];
      if (fT != Teuchos::null)
        fT->get1dViewNonConst()[row] = xT_constView[row] - this->rho_w*this->g*(s_val-H_val);

      if (fpT != Teuchos::null)
      {
        for (int i=0; i<dirichletWorkset.num_cols_p; i++) {
          fpT->getDataNonConst(i)[row] = 0;
        }
      }
    }
  }
}

// **********************************************************************
// Specialization: DistParamDeriv
// **********************************************************************
template<typename Traits>
HydrologyDirichlet<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
HydrologyDirichlet(Teuchos::ParameterList& p) :
  HydrologyDirichletBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p) {
}

// **********************************************************************
template<typename Traits>
void HydrologyDirichlet<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  Teuchos::RCP<Tpetra_MultiVector> fpVT = dirichletWorkset.fpVT;
  Teuchos::ArrayRCP<ST> fpVT_nonconstView;

  bool trans = dirichletWorkset.transpose_dist_param_deriv;
  int num_cols = fpVT->getNumVectors();

  const std::vector<std::vector<int> >& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;

  // For (df/dp)^T*V we zero out corresponding entries in V
  if (trans) {
    Teuchos::RCP<Tpetra_MultiVector> VpT = dirichletWorkset.Vp_bcT;
    //non-const view of VpT
    Teuchos::ArrayRCP<ST> VpT_nonconstView;
    for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
      int lunk = nsNodes[inode][this->offset];

      for (int col=0; col<num_cols; ++col) {
        VpT_nonconstView = VpT->getDataNonConst(col);
        VpT_nonconstView[lunk] = 0.0;
       }
    }
  }

  // for (df/dp)*V we zero out corresponding entries in df/dp
  else {
    for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
      int lunk = nsNodes[inode][this->offset];

      for (int col=0; col<num_cols; ++col) {
        fpVT_nonconstView = fpVT->getDataNonConst(col);
        fpVT_nonconstView[lunk] = 0.0;
      }
    }
  }

  Teuchos::RCP<Tpetra_Vector> fT = dirichletWorkset.fT;
  if (fT != Teuchos::null)
  {
    Teuchos::ArrayRCP<ST> fT_nonconstView = fT->get1dViewNonConst();
    Teuchos::RCP<const Tpetra_Vector> xT = dirichletWorkset.xT;
    Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();

    Teuchos::RCP<Albany::STKDiscretization> disc = Teuchos::rcp_dynamic_cast<Albany::STKDiscretization>(dirichletWorkset.disc);
    TEUCHOS_TEST_FOR_EXCEPTION (disc==Teuchos::null, std::runtime_error, "Error! The discretization must be of type STKDiscretization.\n");

    const std::vector<GO>& nsNodesGIDs = disc->getNodeSetGIDs().find(this->nodeSetID)->second;
    const std::vector<std::vector<int> >& nsNodes = disc->getNodeSets().find(this->nodeSetID)->second;

    typedef Albany::AbstractSTKFieldContainer::ScalarFieldType  SFT;

    SFT* H = disc->getSTKMetaData().get_field<SFT> (stk::topology::NODE_RANK, this->H_name);
    SFT* s = disc->getSTKMetaData().get_field<SFT> (stk::topology::NODE_RANK, this->s_name);

    const std::size_t numNsNodes = nsNodesGIDs.size();
    stk::mesh::Entity node;
    int row;
    double H_val, s_val;
    for (std::size_t i=0; i<numNsNodes; ++i)
    {
      node = disc->getSTKBulkData().get_entity(stk::topology::NODE_RANK, nsNodesGIDs[i]+1);
      H_val = (stk::mesh::field_data(*H, node))[0];
      s_val = (stk::mesh::field_data(*s, node))[0];

      row = nsNodes[i][this->offset];
      fT_nonconstView[row] = xT_constView[row] - this->rho_w*this->g*(s_val-H_val);
    }
  }
}

} // Namespace FELIX
