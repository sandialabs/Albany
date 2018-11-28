//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"

#include "Albany_ThyraUtils.hpp"

#include <set>

// **********************************************************************
// Genereric Template Code for Constructor and PostRegistrationSetup
// **********************************************************************

namespace PHAL
{

// **********************************************************************
// Specialization: Residual
// **********************************************************************
template<typename Traits>
DirichletOffNodeSet<PHAL::AlbanyTraits::Residual, Traits>::
DirichletOffNodeSet(Teuchos::ParameterList& p) :
  DirichletBase<PHAL::AlbanyTraits::Residual, Traits>(p),
  nodeSets (*p.get<Teuchos::RCP<std::vector<std::string> > >("Node Sets"))
{
}

// **********************************************************************
template<typename Traits>
void DirichletOffNodeSet<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  // Gather all node IDs from all the stored nodesets
  std::set<int> nodeSetsRows;
  for (int ins(0); ins<nodeSets.size(); ++ins)
  {
    const std::vector<std::vector<int> >& nsNodes = dirichletWorkset.nodeSets->find(nodeSets[ins])->second;
    for (int inode=0; inode<nsNodes.size(); ++inode)
    {
      nodeSetsRows.insert(nsNodes[inode][this->offset]);
    }
  }

  Teuchos::RCP<const Thyra_Vector> x = dirichletWorkset.x;
  Teuchos::RCP<Thyra_Vector> f = dirichletWorkset.f;

  Teuchos::ArrayRCP<const ST> x_constView    = Albany::getLocalData(x);
  Teuchos::ArrayRCP<ST>       f_nonconstView = Albany::getNonconstLocalData(f);

  // Loop on all local dofs and set the BC on those not in nodeSetsRows
  LO num_local_dofs = Albany::getSpmdVectorSpace(f->space())->localSubDim();
  for (LO row=0; row<num_local_dofs; ++row)
  {
    if (nodeSetsRows.find(row)==nodeSetsRows.end()) {
      f_nonconstView[row] = x_constView[row] - this->value;
    }
  }
}

// **********************************************************************
// Specialization: Jacobian
// **********************************************************************
template<typename Traits>
DirichletOffNodeSet<PHAL::AlbanyTraits::Jacobian, Traits>::
DirichletOffNodeSet(Teuchos::ParameterList& p) :
  DirichletBase<PHAL::AlbanyTraits::Jacobian, Traits>(p),
  nodeSets (*p.get<Teuchos::RCP<std::vector<std::string> > >("Node Sets"))
{
}

// **********************************************************************
template<typename Traits>
void DirichletOffNodeSet<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  // Gather all node IDs from all the stored nodesets
  std::set<int> nodeSetsRows;
  for (int ins(0); ins<nodeSets.size(); ++ins)
  {
    const std::vector<std::vector<int> >& nsNodes = dirichletWorkset.nodeSets->find(nodeSets[ins])->second;
    for (int inode=0; inode<nsNodes.size(); ++inode)
    {
      nodeSetsRows.insert(nsNodes[inode][this->offset]);
    }
  }

  Teuchos::RCP<const Thyra_Vector> x   = dirichletWorkset.x;
  Teuchos::RCP<Thyra_Vector>       f   = dirichletWorkset.f;
  Teuchos::RCP<Thyra_LinearOp>     jac = dirichletWorkset.Jac;

  Teuchos::ArrayRCP<const ST> x_constView;
  Teuchos::ArrayRCP<ST>       f_nonconstView;

  const RealType j_coeff = dirichletWorkset.j_coeff;
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

  // Loop on all local dofs and set the BC on those not in nodeSetsRows
  LO num_local_dofs = Albany::getSpmdVectorSpace(jac->range())->localSubDim();
  for (LO row=0; row<num_local_dofs; ++row) {
    if (nodeSetsRows.find(row)==nodeSetsRows.end()) {
      // It's a row not on the given node sets
      index[0] = row;

      // Extract the row, zero it out, then put j_coeff on diagonal
      Albany::getLocalRowValues(jac,row,matrixIndices,matrixEntries);
      for (auto& val : matrixEntries) { val = 0.0; }
      Albany::setLocalRowValues(jac, row, matrixIndices(), matrixEntries());
      Albany::setLocalRowValues(jac, row, index(), value());

      if (fillResid) {
        f_nonconstView[row] = x_constView[row] - this->value.val();
      }
    }
  }
}

// **********************************************************************
// Specialization: Tangent
// **********************************************************************
template<typename Traits>
DirichletOffNodeSet<PHAL::AlbanyTraits::Tangent, Traits>::
DirichletOffNodeSet(Teuchos::ParameterList& p) :
  DirichletBase<PHAL::AlbanyTraits::Tangent, Traits>(p),
  nodeSets (*p.get<Teuchos::RCP<std::vector<std::string> > >("Node Sets"))
{
}

// **********************************************************************
template<typename Traits>
void DirichletOffNodeSet<PHAL::AlbanyTraits::Tangent, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  // Gather all node IDs from all the stored nodesets
  std::set<int> nodeSetsRows;
  for (int ins(0); ins<nodeSets.size(); ++ins)
  {
    const std::vector<std::vector<int> >& nsNodes = dirichletWorkset.nodeSets->find(nodeSets[ins])->second;
    for (int inode=0; inode<nsNodes.size(); ++inode)
    {
      nodeSetsRows.insert(nsNodes[inode][this->offset]);
    }
  }

  Teuchos::RCP<const Thyra_Vector>       x = dirichletWorkset.x;
  Teuchos::RCP<const Thyra_MultiVector> Vx = dirichletWorkset.Vx;
  Teuchos::RCP<Thyra_Vector>             f = dirichletWorkset.f;
  Teuchos::RCP<Thyra_MultiVector>       fp = dirichletWorkset.fp;
  Teuchos::RCP<Thyra_MultiVector>       JV = dirichletWorkset.JV;

  Teuchos::ArrayRCP<const ST> x_constView;
  Teuchos::ArrayRCP<ST>       f_nonconstView;

  Teuchos::ArrayRCP<Teuchos::ArrayRCP<const ST>> Vx_const2dView;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>>       fp_nonconst2dView;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>>       JV_nonconst2dView;

  if (f != Teuchos::null) {
    f_nonconstView = Albany::getNonconstLocalData(f);
    x_constView = Albany::getLocalData(x);
  }
  if (JV != Teuchos::null) {
    JV_nonconst2dView = Albany::getNonconstLocalData(JV);
    Vx_const2dView    = Albany::getLocalData(Vx);
  }

  if (fp != Teuchos::null) {
    fp_nonconst2dView = Albany::getNonconstLocalData(fp);
  }

  const RealType j_coeff = dirichletWorkset.j_coeff;
  // Loop on all local dofs and set the BC on those not in nodeSetsRows
  LO num_local_dofs = fp!=Teuchos::null ? Albany::getSpmdVectorSpace(fp->range())->localSubDim() :
                     (JV!=Teuchos::null ? Albany::getSpmdVectorSpace(JV->range())->localSubDim() :
                     (f !=Teuchos::null ? Albany::getSpmdVectorSpace(f->space())->localSubDim() : 0));
  for (LO row=0; row<num_local_dofs; ++row) {
    if (nodeSetsRows.find(row)==nodeSetsRows.end()) {
      // It's a row not on the given node sets
      if (f != Teuchos::null) {
        f_nonconstView[row] = x_constView[row] - this->value.val();
      }

      if (JV != Teuchos::null) {
        for (int i=0; i<dirichletWorkset.num_cols_x; i++) {
          JV_nonconst2dView[i][row] = j_coeff*Vx_const2dView[i][row];
        }
      }

      if (fp != Teuchos::null) {
        for (int i=0; i<dirichletWorkset.num_cols_p; i++) {
          fp_nonconst2dView[i][row] = -this->value.dx(dirichletWorkset.param_offset+i);
        }
      }
    }
  }
}

// **********************************************************************
// Specialization: DistParamDeriv
// **********************************************************************
template<typename Traits>
DirichletOffNodeSet<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
DirichletOffNodeSet(Teuchos::ParameterList& p) :
  DirichletBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p)
{
}

// **********************************************************************
template<typename Traits>
void DirichletOffNodeSet<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  // Gather all node IDs from all the stored nodesets
  std::set<int> nodeSetsRows;
  for (int ins(0); ins<nodeSets.size(); ++ins) {
    const std::vector<std::vector<int> >& nsNodes = dirichletWorkset.nodeSets->find(nodeSets[ins])->second;
    for (int inode=0; inode<nsNodes.size(); ++inode) {
      nodeSetsRows.insert(nsNodes[inode][this->offset]);
    }
  }

  Teuchos::RCP<Thyra_MultiVector> fpV = dirichletWorkset.fpV;
  bool trans = dirichletWorkset.transpose_dist_param_deriv;
  int num_cols = fpV->domain()->dim();

  LO num_local_dofs = Albany::getSpmdVectorSpace(fpV->range())->localSubDim();
  if (trans) {
    // For (df/dp)^T*V we zero out corresponding entries in V
    Teuchos::RCP<Thyra_MultiVector> Vp = dirichletWorkset.Vp_bc;
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>> Vp_nonconst2dView = Albany::getNonconstLocalData(Vp);

    // Loop on all local dofs and set the BC on those not in nodeSetsRows
    for (LO row=0; row<num_local_dofs; ++row) {
      if (nodeSetsRows.find(row)==nodeSetsRows.end()) {
        // It's a row not on the given node sets
        for (int col=0; col<num_cols; ++col) {
          //(*Vp)[col][row] = 0.0;
          Vp_nonconst2dView[col][row] = 0.0;
        }
      }
    }
  } else {
    // for (df/dp)*V we zero out corresponding entries in df/dp
    // Loop on all local dofs and set the BC on those not in nodeSetsRows
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>> fpV_nonconst2dView = Albany::getNonconstLocalData(fpV);
    for (LO row=0; row<num_local_dofs; ++row) {
      if (nodeSetsRows.find(row)==nodeSetsRows.end()) {
        // It's a row not on the given node sets
        for (int col=0; col<num_cols; ++col) {
          //(*fpV)[col][row] = 0.0;
          fpV_nonconst2dView[col][row] = 0.0;
        }
      }
    }
  }
}

} // Namespace PHAL
