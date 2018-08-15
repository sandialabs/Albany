//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"

#include "Albany_ThyraUtils.hpp"

// **********************************************************************
// Genereric Template Code for Constructor and PostRegistrationSetup
// **********************************************************************

namespace PHAL {

template <typename EvalT, typename Traits>
DirichletField_Base<EvalT, Traits>::
DirichletField_Base(Teuchos::ParameterList& p) :
  PHAL::DirichletBase<EvalT, Traits>(p) {

  // Get field type and corresponding layouts
  field_name = p.get<std::string>("Field Name");
}

// **********************************************************************
// Specialization: Residual
// **********************************************************************
template<typename Traits>
DirichletField<PHAL::AlbanyTraits::Residual, Traits>::
DirichletField(Teuchos::ParameterList& p) :
  DirichletField_Base<PHAL::AlbanyTraits::Residual, Traits>(p) {
}

// **********************************************************************
template<typename Traits>
void
DirichletField<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset) {


  const Albany::NodalDOFManager& fieldDofManager = dirichletWorkset.disc->getDOFManager(this->field_name);
  //MP: If the parameter is scalar, then the parameter offset is seto to zero. Otherwise the parameter offset is the same of the solution's one.
  Teuchos::RCP<const Tpetra_Map> fieldNodeMap = dirichletWorkset.disc->getNodeMapT(this->field_name);
  bool isFieldScalar = (fieldNodeMap->getNodeNumElements() == dirichletWorkset.disc->getMapT(this->field_name)->getNodeNumElements());
  int fieldOffset = isFieldScalar ? 0 : this->offset;
  const std::vector<GO>& nsNodesGIDs = dirichletWorkset.disc->getNodeSetGIDs().find(this->nodeSetID)->second;

  Teuchos::RCP<Tpetra_Vector> pvecT = dirichletWorkset.distParamLib->get(this->field_name)->vector();
  Teuchos::ArrayRCP<const ST> pT = pvecT->get1dView();

  Teuchos::RCP<const Thyra_Vector> x = dirichletWorkset.x;
  Teuchos::RCP<Thyra_Vector>       f = dirichletWorkset.f;

  Teuchos::ArrayRCP<const ST> x_constView    = Albany::getLocalData(x);
  Teuchos::ArrayRCP<ST>       f_nonconstView = Albany::getNonconstLocalData(f);

  const std::vector<std::vector<int> >& nsNodes = dirichletWorkset.nodeSets->find(this->nodeSetID)->second;
  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
      int lunk = nsNodes[inode][this->offset];
      GO node_gid = nsNodesGIDs[inode];
      int lfield = fieldDofManager.getLocalDOF(fieldNodeMap->getLocalElement(node_gid),fieldOffset);
      f_nonconstView[lunk] = x_constView[lunk] - pT[lfield];
  }
}

// **********************************************************************
// Specialization: Jacobian
// **********************************************************************
template<typename Traits>
DirichletField<PHAL::AlbanyTraits::Jacobian, Traits>::
DirichletField(Teuchos::ParameterList& p) :
  DirichletField_Base<PHAL::AlbanyTraits::Jacobian, Traits>(p) {
}

// **********************************************************************
template<typename Traits>
void DirichletField<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset) {

  const Albany::NodalDOFManager& fieldDofManager = dirichletWorkset.disc->getDOFManager(this->field_name);
  Teuchos::RCP<const Tpetra_Map> fieldNodeMap = dirichletWorkset.disc->getNodeMapT(this->field_name);
  bool isFieldScalar = (fieldNodeMap->getNodeNumElements() == dirichletWorkset.disc->getMapT(this->field_name)->getNodeNumElements());
  int fieldOffset = isFieldScalar ? 0 : this->offset;
  const std::vector<GO>& nsNodesGIDs = dirichletWorkset.disc->getNodeSetGIDs().find(this->nodeSetID)->second;
  Teuchos::RCP<Tpetra_Vector> pvecT = dirichletWorkset.distParamLib->get(this->field_name)->vector();
  Teuchos::ArrayRCP<const ST> pT = pvecT->get1dView();

  Teuchos::RCP<const Thyra_Vector> x   = dirichletWorkset.x;
  Teuchos::RCP<Thyra_Vector>       f   = dirichletWorkset.f;
  Teuchos::RCP<Thyra_LinearOp>     jac = dirichletWorkset.Jac;

  Teuchos::ArrayRCP<const ST> x_constView;
  Teuchos::ArrayRCP<ST>       f_nonconstView;

  const RealType j_coeff = dirichletWorkset.j_coeff;
  const std::vector<std::vector<int> >& nsNodes = dirichletWorkset.nodeSets->find(this->nodeSetID)->second;

  bool fillResid = (f != Teuchos::null);
  if (fillResid) { 
    x_constView = Albany::getLocalData(x);
    f_nonconstView = Albany::getNonconstLocalData(f);
  }

  Teuchos::Array<LO> index(1);
  Teuchos::Array<ST> value(1);
  value[0] = j_coeff;
  Teuchos::Array<ST> matrixEntries;
  Teuchos::Array<LO> matrixIndices;

  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
    int lunk = nsNodes[inode][this->offset];
    index[0] = lunk;

    // Extract the row, zero it out, then put j_coeff on diagonal
    Albany::getLocalRowValues(jac,lunk,matrixIndices,matrixEntries);
    for (auto& val : matrixEntries) { val = 0.0; }
    Albany::setLocalRowValues(jac, lunk, matrixIndices(), matrixEntries());
    Albany::setLocalRowValues(jac, lunk, index(), value());

    if (fillResid) {
      GO node_gid = nsNodesGIDs[inode];
      int lfield = fieldDofManager.getLocalDOF(fieldNodeMap->getLocalElement(node_gid),fieldOffset);
      f_nonconstView[lunk] = x_constView[lunk] - pT[lfield];
    }
  }
}

// **********************************************************************
// Specialization: Tangent
// **********************************************************************
template<typename Traits>
DirichletField<PHAL::AlbanyTraits::Tangent, Traits>::
DirichletField(Teuchos::ParameterList& p) :
  DirichletField_Base<PHAL::AlbanyTraits::Tangent, Traits>(p) {
}

// **********************************************************************
template<typename Traits>
void DirichletField<PHAL::AlbanyTraits::Tangent, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset) {

  const Albany::NodalDOFManager& fieldDofManager = dirichletWorkset.disc->getDOFManager(this->field_name);
  Teuchos::RCP<const Tpetra_Map> fieldNodeMap = dirichletWorkset.disc->getNodeMapT(this->field_name);
  bool isFieldScalar = (fieldNodeMap->getNodeNumElements() == dirichletWorkset.disc->getMapT(this->field_name)->getNodeNumElements());
  int fieldOffset = isFieldScalar ? 0 : this->offset;
  const std::vector<GO>& nsNodesGIDs = dirichletWorkset.disc->getNodeSetGIDs().find(this->nodeSetID)->second;

  Teuchos::RCP<Tpetra_Vector> pvecT = dirichletWorkset.distParamLib->get(this->field_name)->vector();
  Teuchos::ArrayRCP<const ST> pT = pvecT->get1dView();

  Teuchos::RCP<const Thyra_Vector> x = dirichletWorkset.x;
  Teuchos::RCP<Thyra_Vector>       f = dirichletWorkset.f;

  Teuchos::RCP<const Thyra_MultiVector> Vx = dirichletWorkset.Vx;
  Teuchos::RCP<Thyra_MultiVector>       fp = dirichletWorkset.fp;
  Teuchos::RCP<Thyra_MultiVector>       JV = dirichletWorkset.JV;

  Teuchos::ArrayRCP<const ST> x_constView;
  Teuchos::ArrayRCP<ST>       f_nonconstView;

  Teuchos::ArrayRCP<Teuchos::ArrayRCP<const ST>> Vx_const2dView;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>>       JV_nonconst2dView;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>>       fp_nonconst2dView;

  if (f != Teuchos::null) {
    x_constView = Albany::getLocalData(x);
    f_nonconstView = Albany::getNonconstLocalData(f);
  }
  if (JV != Teuchos::null) {
    JV_nonconst2dView = Albany::getNonconstLocalData(JV);
    Vx_const2dView    = Albany::getLocalData(Vx);
  }

  if (fp != Teuchos::null) {
    fp_nonconst2dView = Albany::getNonconstLocalData(fp);
  }

  const RealType j_coeff = dirichletWorkset.j_coeff;
  const std::vector<std::vector<int> >& nsNodes = dirichletWorkset.nodeSets->find(this->nodeSetID)->second;

  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
    int lunk = nsNodes[inode][this->offset];

    if (f != Teuchos::null) {
      GO node_gid = nsNodesGIDs[inode];
      int lfield = fieldDofManager.getLocalDOF(fieldNodeMap->getLocalElement(node_gid),fieldOffset);
      f_nonconstView[lunk] = x_constView[lunk] - pT[lfield];
    }

    if (JV != Teuchos::null) {
      for (int i=0; i<dirichletWorkset.num_cols_x; i++) {
        JV_nonconst2dView[i][lunk] = j_coeff*Vx_const2dView[i][lunk];
      }
    }

    if (fp != Teuchos::null) {
      for (int i=0; i<dirichletWorkset.num_cols_p; i++) {
        fp_nonconst2dView[i][lunk] = 0;
      }
    }
  }
}

// **********************************************************************
// Specialization: DistParamDeriv
// **********************************************************************
template<typename Traits>
DirichletField<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
DirichletField(Teuchos::ParameterList& p) :
  DirichletField_Base<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p) {
}

// **********************************************************************
template<typename Traits>
void DirichletField<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset) {

  bool isFieldParameter =  dirichletWorkset.dist_param_deriv_name == this->field_name;
  bool trans = dirichletWorkset.transpose_dist_param_deriv;

  Teuchos::RCP<Thyra_MultiVector> fpV = dirichletWorkset.fpV;

  int num_cols = fpV->domain()->dim();

  const std::vector<std::vector<int> >& nsNodes = dirichletWorkset.nodeSets->find(this->nodeSetID)->second;

  if (trans) {
    // For (df/dp)^T*V we zero out corresponding entries in V
    Teuchos::RCP<Thyra_MultiVector> Vp = dirichletWorkset.Vp_bc;
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>> Vp_nonconst2dView = Albany::getNonconstLocalData(Vp);

    if(isFieldParameter) {
      const Albany::NodalDOFManager& fieldDofManager = dirichletWorkset.disc->getDOFManager(this->field_name);
      Teuchos::RCP<const Tpetra_Map> fieldNodeMap = dirichletWorkset.disc->getNodeMapT(this->field_name);
      bool isFieldScalar = (fieldNodeMap->getNodeNumElements() == dirichletWorkset.disc->getMapT(this->field_name)->getNodeNumElements());
      int fieldOffset = isFieldScalar ? 0 : this->offset;
      const std::vector<GO>& nsNodesGIDs = dirichletWorkset.disc->getNodeSetGIDs().find(this->nodeSetID)->second;
      Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>> fpV_nonconst2dView = Albany::getNonconstLocalData(fpV);
      for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
        int lunk = nsNodes[inode][this->offset];
        GO node_gid = nsNodesGIDs[inode];
        int lfield = fieldDofManager.getLocalDOF(fieldNodeMap->getLocalElement(node_gid),fieldOffset);
        for (int col=0; col<num_cols; ++col) {
          fpV_nonconst2dView[col][lfield] -= Vp_nonconst2dView[col][lunk];
          Vp_nonconst2dView[col][lunk] = 0.0;
         }
      }
    } else {
      for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
        int lunk = nsNodes[inode][this->offset];

        for (int col=0; col<num_cols; ++col) {
          Vp_nonconst2dView[col][lunk] = 0.0;
         }
      }
    }
  } else {
    // for (df/dp)*V we zero out corresponding entries in df/dp
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>> fpV_nonconst2dView = Albany::getNonconstLocalData(fpV);
    if(isFieldParameter) {
      const Albany::NodalDOFManager& fieldDofManager = dirichletWorkset.disc->getDOFManager(this->field_name);
      Teuchos::RCP<const Tpetra_Map> fieldNodeMap = dirichletWorkset.disc->getNodeMapT(this->field_name);
      bool isFieldScalar = (fieldNodeMap->getNodeNumElements() == dirichletWorkset.disc->getMapT(this->field_name)->getNodeNumElements());
      int fieldOffset = isFieldScalar ? 0 : this->offset;
      const std::vector<GO>& nsNodesGIDs = dirichletWorkset.disc->getNodeSetGIDs().find(this->nodeSetID)->second;
      for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
        int lunk = nsNodes[inode][this->offset];
        GO node_gid = nsNodesGIDs[inode];
        int lfield = fieldDofManager.getLocalDOF(fieldNodeMap->getLocalElement(node_gid),fieldOffset);
        for (int col=0; col<num_cols; ++col) {
          //(*fpV)[col][lunk] = 0.0;
          fpV_nonconst2dView[col][lunk] = -double(col == lfield);
        }
      }
    } else {
      for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
        int lunk = nsNodes[inode][this->offset];

        for (int col=0; col<num_cols; ++col) {
          fpV_nonconst2dView[col][lunk] = 0.0;
        }
      }
    }
  }
}

} // namespace PHAL
