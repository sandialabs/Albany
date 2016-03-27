//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

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
  Teuchos::RCP<Tpetra_Vector> fT = dirichletWorkset.fT;
  Teuchos::RCP<const Tpetra_Vector> xT = dirichletWorkset.xT;
  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();
  Teuchos::ArrayRCP<ST> fT_nonconstView = fT->get1dViewNonConst();

  // Grab the vector of node GIDs for this Node Set ID from the std::map
  const std::vector<std::vector<int>>& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;

  int cunk, punk;
  ScalarT Cval;
  ScalarT pressure;
  
  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
    cunk = nsNodes[inode][this->coffset_];
    punk = nsNodes[inode][this->poffset_];
    pressure = xT_constView[punk];
    this->computeBCs(pressure, Cval);

    fT_nonconstView[cunk] = xT_constView[cunk] - Cval;
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
  Teuchos::RCP<Tpetra_Vector> fT = dirichletWorkset.fT;
  Teuchos::RCP<const Tpetra_Vector> xT = dirichletWorkset.xT;
  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();
  Teuchos::RCP<Tpetra_CrsMatrix> jacT = dirichletWorkset.JacT;

  const RealType j_coeff = dirichletWorkset.j_coeff;
  const std::vector<std::vector<int>>& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;

  bool fillResid = (fT != Teuchos::null);
  Teuchos::ArrayRCP<ST> fT_nonconstView;
  if (fillResid) fT_nonconstView = fT->get1dViewNonConst();

  int cunk, punk;
  ScalarT Cval;
  ScalarT pressure;

  Teuchos::Array<LO> index(1);
  Teuchos::Array<ST> value(1);
  size_t numEntriesT;
  value[0] = j_coeff;
  Teuchos::Array<ST> matrixEntriesT;
  Teuchos::Array<LO> matrixIndicesT;


  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) 
  {
    cunk = nsNodes[inode][this->coffset_];
    punk = nsNodes[inode][this->poffset_];
    pressure = xT_constView[punk];
    this->computeBCs(pressure, Cval);
    
    // replace jac values for the C dof 
    numEntriesT = jacT->getNumEntriesInLocalRow(cunk);
    matrixEntriesT.resize(numEntriesT);
    matrixIndicesT.resize(numEntriesT);
    jacT->getLocalRowCopy(cunk, matrixIndicesT(), matrixEntriesT(), numEntriesT);
    for (int i=0; i<numEntriesT; i++) matrixEntriesT[i]=0;
    jacT->replaceLocalValues(cunk, matrixIndicesT(), matrixEntriesT());

    index[0] = cunk;
    jacT->replaceLocalValues(cunk, index(), value());

    if (fillResid)
    {
      fT_nonconstView[cunk] = xT_constView[cunk] - Cval.val();
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
  Teuchos::RCP<Tpetra_Vector> fT = dirichletWorkset.fT;
  Teuchos::RCP<Tpetra_MultiVector> fpT = dirichletWorkset.fpT;
  Teuchos::RCP<Tpetra_MultiVector> JVT = dirichletWorkset.JVT;
  Teuchos::RCP<const Tpetra_Vector> xT = dirichletWorkset.xT;
  Teuchos::RCP<const Tpetra_MultiVector> VxT = dirichletWorkset.VxT;

  Teuchos::ArrayRCP<const ST> VxT_constView; 
  Teuchos::ArrayRCP<ST> fT_nonconstView;                                         
  if (fT != Teuchos::null) fT_nonconstView = fT->get1dViewNonConst();
  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();                                       
  
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
    pressure = xT_constView[punk];
    this->computeBCs(pressure, Cval);

    if (fT != Teuchos::null)
    {
      fT_nonconstView[cunk] = xT_constView[cunk] - Cval.val();
    } 

    if (JVT != Teuchos::null) {
      Teuchos::ArrayRCP<ST> JVT_nonconstView; 
      for (int i=0; i<dirichletWorkset.num_cols_x; i++)
      {
        JVT_nonconstView = JVT->getDataNonConst(i); 
        VxT_constView = VxT->getData(i); 
	JVT_nonconstView[cunk] = j_coeff*VxT_constView[cunk];
      }
    }
    
    if (fpT != Teuchos::null) {
      Teuchos::ArrayRCP<ST> fpT_nonconstView; 
      for (int i=0; i<dirichletWorkset.num_cols_p; i++)
      {
        fpT_nonconstView = fpT->getDataNonConst(i); 
	fpT_nonconstView[cunk] = -Cval.dx(dirichletWorkset.param_offset+i);
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

  Teuchos::RCP<Tpetra_MultiVector> fpVT = dirichletWorkset.fpVT;
  Teuchos::ArrayRCP<ST> fpVT_nonconstView; 
  bool trans = dirichletWorkset.transpose_dist_param_deriv;
  int num_cols = fpVT->getNumVectors();

  //
  // We're currently assuming Dirichlet BC's can't be distributed parameters.
  // Thus we don't need to actually evaluate the BC's here.  The code to do
  // so is still here, just commented out for future reference.
  //

  const std::vector<std::vector<int>>& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;

  int cunk; // global and local indicies into unknown vector

  // For (df/dp)^T*V we zero out corresponding entries in V
  if (trans) {
    for (unsigned int inode = 0; inode < nsNodes.size(); inode++)
    {
      Teuchos::RCP<Tpetra_MultiVector> VpT = dirichletWorkset.Vp_bcT;
      Teuchos::ArrayRCP<ST> VpT_nonconstView; 
      cunk = nsNodes[inode][this->coffset_];

      for (int col=0; col<num_cols; ++col) {
        VpT_nonconstView = VpT->getDataNonConst(col); 
        VpT_nonconstView[cunk] = 0.0; 
      }
    }
  }

  // for (df/dp)*V we zero out corresponding entries in df/dp
  else {
    for (unsigned int inode = 0; inode < nsNodes.size(); inode++)
    {
      cunk = nsNodes[inode][this->coffset_];

      for (int col=0; col<num_cols; ++col) {
        fpVT_nonconstView = fpVT->getDataNonConst(col);
        fpVT_nonconstView[cunk] = 0.0; 
      }
    }
  }
}
//------------------------------------------------------------------------------  
// Specialization: Stochastic Galerkin Residual
//
#ifdef ALBANY_SG
template<typename Traits>
EquilibriumConcentrationBC<PHAL::AlbanyTraits::SGResidual, Traits>::
EquilibriumConcentrationBC(Teuchos::ParameterList& p) :
  EquilibriumConcentrationBC_Base<PHAL::AlbanyTraits::SGResidual, Traits>(p)
{
}
//------------------------------------------------------------------------------  
template<typename Traits>
void EquilibriumConcentrationBC<PHAL::AlbanyTraits::SGResidual, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true,
                             std::runtime_error,
                             "Error! This BC does not support SG Types");
}
//------------------------------------------------------------------------------
// Specialization: Stochastic Galerkin Jacobian
//
template<typename Traits>
EquilibriumConcentrationBC<PHAL::AlbanyTraits::SGJacobian, Traits>::
EquilibriumConcentrationBC(Teuchos::ParameterList& p) :
  EquilibriumConcentrationBC_Base<PHAL::AlbanyTraits::SGJacobian, Traits>(p)
{
}
//------------------------------------------------------------------------------  
template<typename Traits>
void EquilibriumConcentrationBC<PHAL::AlbanyTraits::SGJacobian, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true,
                             std::runtime_error,
                             "Error! This BC does not support SG Types");
}
//------------------------------------------------------------------------------  
// Specialization: Stochastic Galerkin Tangent
//
template<typename Traits>
EquilibriumConcentrationBC<PHAL::AlbanyTraits::SGTangent, Traits>::
EquilibriumConcentrationBC(Teuchos::ParameterList& p) :
  EquilibriumConcentrationBC_Base<PHAL::AlbanyTraits::SGTangent, Traits>(p)
{
}
//------------------------------------------------------------------------------  
template<typename Traits>
void EquilibriumConcentrationBC<PHAL::AlbanyTraits::SGTangent, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true,
                             std::runtime_error,
                             "Error! This BC does not support SG Types");
}

#endif 
#ifdef ALBANY_ENSEMBLE 
//------------------------------------------------------------------------------  
// Specialization: Multi-point Residual
//
template<typename Traits>
EquilibriumConcentrationBC<PHAL::AlbanyTraits::MPResidual, Traits>::
EquilibriumConcentrationBC(Teuchos::ParameterList& p) :
  EquilibriumConcentrationBC_Base<PHAL::AlbanyTraits::MPResidual, Traits>(p)
{
}
//------------------------------------------------------------------------------  
template<typename Traits>
void EquilibriumConcentrationBC<PHAL::AlbanyTraits::MPResidual, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true,
                             std::runtime_error,
                             "Error! This BC does not support ENSEMBLES");
}
//------------------------------------------------------------------------------  
// Specialization: Multi-point Jacobian
//
template<typename Traits>
EquilibriumConcentrationBC<PHAL::AlbanyTraits::MPJacobian, Traits>::
EquilibriumConcentrationBC(Teuchos::ParameterList& p) :
  EquilibriumConcentrationBC_Base<PHAL::AlbanyTraits::MPJacobian, Traits>(p)
{
}
//------------------------------------------------------------------------------
template<typename Traits>
void EquilibriumConcentrationBC<PHAL::AlbanyTraits::MPJacobian, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true,
                             std::runtime_error,
                             "Error! This BC does not support ENSEMBLES");
}

//------------------------------------------------------------------------------
// Specialization: Multi-point Tangent
//
template<typename Traits>
EquilibriumConcentrationBC<PHAL::AlbanyTraits::MPTangent, Traits>::
EquilibriumConcentrationBC(Teuchos::ParameterList& p) :
  EquilibriumConcentrationBC_Base<PHAL::AlbanyTraits::MPTangent, Traits>(p)
{
}
//------------------------------------------------------------------------------
template<typename Traits>
void EquilibriumConcentrationBC<PHAL::AlbanyTraits::MPTangent, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true,
                             std::runtime_error,
                             "Error! This BC does not support ENSEMBLES");
}
#endif
//------------------------------------------------------------------------------  
} // namespace LCM

