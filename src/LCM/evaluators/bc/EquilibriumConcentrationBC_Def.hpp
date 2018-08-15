//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Albany_ThyraUtils.hpp"

namespace LCM {

template <typename EvalT, typename Traits>
EquilibriumConcentrationBC_Base<EvalT, Traits>::
EquilibriumConcentrationBC_Base(Teuchos::ParameterList& p) :
  coffset_(p.get<int>("Equation Offset")),
  poffset_(p.get<int>("Pressure Offset")),
  PHAL::DirichletBase<EvalT, Traits>(p),
  applied_conc_(p.get<RealType>("Applied Concentration")),
  pressure_fac_(p.get<RealType>("Pressure Factor"))
{
}
//------------------------------------------------------------------------------
//
template<typename EvalT, typename Traits>
void
EquilibriumConcentrationBC_Base<EvalT, Traits>::
computeBCs(ScalarT& pressure, ScalarT& Cval)
{
  Cval = applied_conc_ * std::exp(pressure_fac_ * pressure);
}
//------------------------------------------------------------------------------
// Specialization: Residual
//
template<typename Traits>
EquilibriumConcentrationBC<PHAL::AlbanyTraits::Residual, Traits>::
EquilibriumConcentrationBC(Teuchos::ParameterList& p) :
  EquilibriumConcentrationBC_Base<PHAL::AlbanyTraits::Residual, Traits>(p)
{
}
//------------------------------------------------------------------------------
//
template<typename Traits>
void
EquilibriumConcentrationBC<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  Teuchos::RCP<const Thyra_Vector> x = dirichletWorkset.x;
  Teuchos::RCP<Thyra_Vector>       f = dirichletWorkset.f;

  Teuchos::ArrayRCP<const ST> x_constView    = Albany::getLocalData(x);
  Teuchos::ArrayRCP<ST>       f_nonconstView = Albany::getNonconstLocalData(f);

  // Grab the vector of node GIDs for this Node Set ID from the std::map
  const std::vector<std::vector<int>>& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;

  int cunk, punk;
  ScalarT Cval;
  ScalarT pressure;

  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
    cunk = nsNodes[inode][this->coffset_];
    punk = nsNodes[inode][this->poffset_];
    pressure = x_constView[punk];
    this->computeBCs(pressure, Cval);

    f_nonconstView[cunk] = x_constView[cunk] - Cval;
  }
}
//------------------------------------------------------------------------------
// Specialization: Jacobian
//
template<typename Traits>
EquilibriumConcentrationBC<PHAL::AlbanyTraits::Jacobian, Traits>::
EquilibriumConcentrationBC(Teuchos::ParameterList& p) :
  EquilibriumConcentrationBC_Base<PHAL::AlbanyTraits::Jacobian, Traits>(p)
{
}
//------------------------------------------------------------------------------
template<typename Traits>
void EquilibriumConcentrationBC<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  Teuchos::RCP<const Thyra_Vector> x   = dirichletWorkset.x;
  Teuchos::RCP<Thyra_Vector>       f   = dirichletWorkset.f;
  Teuchos::RCP<Thyra_LinearOp>     jac = dirichletWorkset.Jac;

  Teuchos::ArrayRCP<const ST> x_constView = Albany::getLocalData(x);
  Teuchos::ArrayRCP<ST>       f_nonconstView;

  const RealType j_coeff = dirichletWorkset.j_coeff;
  const std::vector<std::vector<int>>& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;

  bool fillResid = (f != Teuchos::null);
  if (fillResid) {
    f_nonconstView = Albany::getNonconstLocalData(f);
  }

  int cunk, punk;
  ScalarT Cval;
  ScalarT pressure;

  Teuchos::Array<LO> index(1);
  Teuchos::Array<ST> value(1);
  value[0] = j_coeff;
  Teuchos::Array<ST> matrixEntries;
  Teuchos::Array<LO> matrixIndices;


  for (unsigned int inode = 0; inode < nsNodes.size(); inode++)
  {
    cunk = nsNodes[inode][this->coffset_];
    punk = nsNodes[inode][this->poffset_];
    pressure = x_constView[punk];
    this->computeBCs(pressure, Cval);

    // replace jac values for the C dof
    Albany::getLocalRowValues(jac, cunk, matrixIndices, matrixEntries);
    for (auto& val : matrixEntries) { val = 0.0; }
    Albany::setLocalRowValues(jac, cunk, matrixIndices(), matrixEntries());
    index[0] = cunk;
    Albany::setLocalRowValues(jac, cunk, index(), value());

    if (fillResid)
    {
      f_nonconstView[cunk] = x_constView[cunk] - Cval.val();
    }
  }
}
//------------------------------------------------------------------------------
// Specialization: Tangent
//
template<typename Traits>
EquilibriumConcentrationBC<PHAL::AlbanyTraits::Tangent, Traits>::
EquilibriumConcentrationBC(Teuchos::ParameterList& p) :
  EquilibriumConcentrationBC_Base<PHAL::AlbanyTraits::Tangent, Traits>(p)
{
}
//------------------------------------------------------------------------------
template<typename Traits>
void EquilibriumConcentrationBC<PHAL::AlbanyTraits::Tangent, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  Teuchos::RCP<const Thyra_Vector>       x = dirichletWorkset.x;
  Teuchos::RCP<const Thyra_MultiVector> Vx = dirichletWorkset.Vx;
  Teuchos::RCP<Thyra_Vector>             f = dirichletWorkset.f;
  Teuchos::RCP<Thyra_MultiVector>       fp = dirichletWorkset.fp;
  Teuchos::RCP<Thyra_MultiVector>       JV = dirichletWorkset.JV;

  Teuchos::ArrayRCP<const ST> x_constView = Albany::getLocalData(x);
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

  const RealType j_coeff = dirichletWorkset.j_coeff;
  const std::vector<std::vector<int>>& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;

  int cunk, punk;
  ScalarT Cval;
  ScalarT pressure;

  for (unsigned int inode = 0; inode < nsNodes.size(); inode++)
  {
    cunk = nsNodes[inode][this->coffset_];
    punk = nsNodes[inode][this->poffset_];
    pressure = x_constView[punk];
    this->computeBCs(pressure, Cval);

    if (f != Teuchos::null)
    {
      f_nonconstView[cunk] = x_constView[cunk] - Cval.val();
    }

    if (JV != Teuchos::null) {
      for (int i=0; i<dirichletWorkset.num_cols_x; i++) {
	      JV_nonconst2dView[i][cunk] = j_coeff*Vx_const2dView[i][cunk];
      }
    }

    if (fp != Teuchos::null) {
      for (int i=0; i<dirichletWorkset.num_cols_p; i++) {
        fp_nonconst2dView[i][cunk] = -Cval.dx(dirichletWorkset.param_offset+i);
      }
    }

  }
}
//------------------------------------------------------------------------------
// Specialization: DistParamDeriv
//
template<typename Traits>
EquilibriumConcentrationBC<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
EquilibriumConcentrationBC(Teuchos::ParameterList& p) :
  EquilibriumConcentrationBC_Base<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p)
{
}
//------------------------------------------------------------------------------
template<typename Traits>
void EquilibriumConcentrationBC<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  Teuchos::RCP<Thyra_MultiVector> fpV = dirichletWorkset.fpV;
  bool trans = dirichletWorkset.transpose_dist_param_deriv;
  int num_cols = fpV->domain()->dim();

  //
  // We're currently assuming Dirichlet BC's can't be distributed parameters.
  // Thus we don't need to actually evaluate the BC's here.  The code to do
  // so is still here, just commented out for future reference.
  //

  const std::vector<std::vector<int>>& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;

  int cunk; // global and local indicies into unknown vector

  if (trans) {
    // For (df/dp)^T*V we zero out corresponding entries in V
    Teuchos::RCP<Thyra_MultiVector> Vp = dirichletWorkset.Vp_bc;
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>> Vp_nonconst2dView = Albany::getNonconstLocalData(Vp);
    for (unsigned int inode = 0; inode < nsNodes.size(); inode++)
    {
      cunk = nsNodes[inode][this->coffset_];

      for (int col=0; col<num_cols; ++col) {
        Vp_nonconst2dView[col][cunk] = 0.0;
      }
    }
  } else {
    // for (df/dp)*V we zero out corresponding entries in df/dp
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>> fpV_nonconst2dView = Albany::getNonconstLocalData(fpV);
    for (unsigned int inode = 0; inode < nsNodes.size(); inode++)
    {
      cunk = nsNodes[inode][this->coffset_];

      for (int col=0; col<num_cols; ++col) {
        fpV_nonconst2dView[col][cunk] = 0.0;
      }
    }
  }
}

//------------------------------------------------------------------------------
} // namespace LCM
