//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/13/14: only Epetra is SG and MP

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "Tpetra_CrsMatrix.hpp"

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
  Teuchos::RCP<Tpetra_Vector> fT = dirichletWorkset.fT;
  Teuchos::RCP<const Tpetra_Vector> xT = dirichletWorkset.xT;
  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();
  Teuchos::ArrayRCP<ST> fT_nonconstView = fT->get1dViewNonConst();

  // Grab the vector off node GIDs for this Node Set ID from the std::map
  std::set<int> nodeSetRows;
  for (int ins(0); ins<nodeSets.size(); ++ins)
  {
    const std::vector<std::vector<int> >& nsNodes = dirichletWorkset.nodeSets->find(nodeSets[ins])->second;
    for (int inode=0; inode<nsNodes.size(); ++inode)
    {
      nodeSetRows.insert(nsNodes[inode][this->offset]);
    }
  }

  const auto& wsElNodeEqID = dirichletWorkset.disc->getWsElNodeEqID();
  for (int ws=0; ws<wsElNodeEqID.size(); ++ws)
  {
    for (int cell=0; cell<wsElNodeEqID[ws].size(); ++cell)
    {
      for (int node=0; node<wsElNodeEqID[ws][cell].size(); ++node)
      {
        LO row = wsElNodeEqID[ws][cell][node][this->offset];
        if (nodeSetRows.find(row)==nodeSetRows.end())
        {
          fT_nonconstView[row] = xT_constView[row] - this->value;
        }
      }
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
  // Grab the vector off node GIDs for this Node Set ID from the std::map
  std::set<int> nodeSetRows;
  for (int ins(0); ins<nodeSets.size(); ++ins)
  {
    const std::vector<std::vector<int> >& nsNodes = dirichletWorkset.nodeSets->find(nodeSets[ins])->second;
    for (int inode=0; inode<nsNodes.size(); ++inode)
    {
      nodeSetRows.insert(nsNodes[inode][this->offset]);
    }
  }

  Teuchos::RCP<Tpetra_Vector> fT = dirichletWorkset.fT;
  Teuchos::RCP<const Tpetra_Vector> xT = dirichletWorkset.xT;
  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();
  Teuchos::RCP<Tpetra_CrsMatrix> jacT = dirichletWorkset.JacT;

  const RealType j_coeff = dirichletWorkset.j_coeff;
  bool fillResid = (fT != Teuchos::null);
  Teuchos::ArrayRCP<ST> fT_nonconstView;

  if (fillResid)
    fT_nonconstView = fT->get1dViewNonConst();

  Teuchos::Array<LO> index(1);
  Teuchos::Array<ST> value(1);
  size_t numEntriesT;
  value[0] = j_coeff;
  Teuchos::Array<ST> matrixEntriesT;
  Teuchos::Array<LO> matrixIndicesT;

  const auto& wsElNodeEqID = dirichletWorkset.disc->getWsElNodeEqID();
  for (int ws=0; ws<wsElNodeEqID.size(); ++ws)
  {
    for (int cell=0; cell<wsElNodeEqID[ws].size(); ++cell)
    {
      for (int node=0; node<wsElNodeEqID[ws][cell].size(); ++node)
      {
        LO row = wsElNodeEqID[ws][cell][node][this->offset];
        if (nodeSetRows.find(row)==nodeSetRows.end())
        {
          // It's a row not on the given node sets

          LO row = wsElNodeEqID[ws][cell][node][this->offset];
          index[0] = row;

          numEntriesT = jacT->getNumEntriesInLocalRow(row);
          matrixEntriesT.resize(numEntriesT);
          matrixIndicesT.resize(numEntriesT);

          jacT->getLocalRowCopy(row, matrixIndicesT(), matrixEntriesT(), numEntriesT);

          for (int i=0; i<numEntriesT; i++)
            matrixEntriesT[i]=0;

          jacT->replaceLocalValues(row, matrixIndicesT(), matrixEntriesT());
          jacT->replaceLocalValues(row, index(), value());

          if (fillResid)
            fT_nonconstView[row] = xT_constView[row] - this->value.val();
        }
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
  // Grab the vector off node GIDs for this Node Set ID from the std::map
  std::set<int> nodeSetRows;
  for (int ins(0); ins<nodeSets.size(); ++ins)
  {
    const std::vector<std::vector<int> >& nsNodes = dirichletWorkset.nodeSets->find(nodeSets[ins])->second;
    for (int inode=0; inode<nsNodes.size(); ++inode)
    {
      nodeSetRows.insert(nsNodes[inode][this->offset]);
    }
  }

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
  const auto& wsElNodeEqID = dirichletWorkset.disc->getWsElNodeEqID();
  for (int ws=0; ws<wsElNodeEqID.size(); ++ws)
  {
    for (int cell=0; cell<wsElNodeEqID[ws].size(); ++cell)
    {
      for (int node=0; node<wsElNodeEqID[ws][cell].size(); ++node)
      {
        LO row = wsElNodeEqID[ws][cell][node][this->offset];
        if (nodeSetRows.find(row)==nodeSetRows.end())
        {
          // It's a row not on the given node sets

          LO row = wsElNodeEqID[ws][cell][node][this->offset];

          if (fT != Teuchos::null)
          {
            fT_nonconstView[row] = xT_constView[row] - this->value.val();
          }

          if (JVT != Teuchos::null)
          {
            Teuchos::ArrayRCP<ST> JVT_nonconstView;
            for (int i=0; i<dirichletWorkset.num_cols_x; i++)
            {
              JVT_nonconstView = JVT->getDataNonConst(i);
              VxT_constView = VxT->getData(i);
              JVT_nonconstView[row] = j_coeff*VxT_constView[row];
            }
          }

          if (fpT != Teuchos::null)
          {
            Teuchos::ArrayRCP<ST> fpT_nonconstView;
            for (int i=0; i<dirichletWorkset.num_cols_p; i++)
            {
              fpT_nonconstView = fpT->getDataNonConst(i);
              fpT_nonconstView[row] = -this->value.dx(dirichletWorkset.param_offset+i);
            }
          }
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
  // Grab the vector off node GIDs for this Node Set ID from the std::map
  std::set<int> nodeSetRows;
  for (int ins(0); ins<nodeSets.size(); ++ins)
  {
    const std::vector<std::vector<int> >& nsNodes = dirichletWorkset.nodeSets->find(nodeSets[ins])->second;
    for (int inode=0; inode<nsNodes.size(); ++inode)
    {
      nodeSetRows.insert(nsNodes[inode][this->offset]);
    }
  }

  Teuchos::RCP<Tpetra_MultiVector> fpVT = dirichletWorkset.fpVT;
  //non-const view of fpVT
  Teuchos::ArrayRCP<ST> fpVT_nonconstView;
  bool trans = dirichletWorkset.transpose_dist_param_deriv;
  int num_cols = fpVT->getNumVectors();

  // For (df/dp)^T*V we zero out corresponding entries in V
  if (trans)
  {
    Teuchos::RCP<Tpetra_MultiVector> VpT = dirichletWorkset.Vp_bcT;
    //non-const view of VpT
    Teuchos::ArrayRCP<ST> VpT_nonconstView;
    const auto& wsElNodeEqID = dirichletWorkset.disc->getWsElNodeEqID();
    for (int ws=0; ws<wsElNodeEqID.size(); ++ws)
    {
      for (int cell=0; cell<wsElNodeEqID[ws].size(); ++cell)
      {
        for (int node=0; node<wsElNodeEqID[ws][cell].size(); ++node)
        {
          LO row = wsElNodeEqID[ws][cell][node][this->offset];
          if (nodeSetRows.find(row)==nodeSetRows.end())
          {
            // It's a row not on the given node sets
            LO row = wsElNodeEqID[ws][cell][node][this->offset];

            for (int col=0; col<num_cols; ++col)
            {
              //(*Vp)[col][row] = 0.0;
              VpT_nonconstView = VpT->getDataNonConst(col);
              VpT_nonconstView[row] = 0.0;
            }
          }
        }
      }
    }
  }

  // for (df/dp)*V we zero out corresponding entries in df/dp
  else
  {
    const auto& wsElNodeEqID = dirichletWorkset.disc->getWsElNodeEqID();
    for (int ws=0; ws<wsElNodeEqID.size(); ++ws)
    {
      for (int cell=0; cell<wsElNodeEqID[ws].size(); ++cell)
      {
        for (int node=0; node<wsElNodeEqID[ws][cell].size(); ++node)
        {
          LO row = wsElNodeEqID[ws][cell][node][this->offset];
          if (nodeSetRows.find(row)==nodeSetRows.end())
          {
            // It's a row not on the given node sets
            LO row = wsElNodeEqID[ws][cell][node][this->offset];

            for (int col=0; col<num_cols; ++col)
            {
              //(*fpV)[col][row] = 0.0;
              fpVT_nonconstView = fpVT->getDataNonConst(col);
              fpVT_nonconstView[row] = 0.0;
            }
          }
        }
      }
    }
  }
}

// **********************************************************************
// Specialization: Stochastic Galerkin Residual
// **********************************************************************

#ifdef ALBANY_SG
template<typename Traits>
DirichletOffNodeSet<PHAL::AlbanyTraits::SGResidual, Traits>::
DirichletOffNodeSet(Teuchos::ParameterList& p) :
  DirichletBase<PHAL::AlbanyTraits::SGResidual, Traits>(p)
{
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Specialization not yet ipmlemented.\n");
}
// **********************************************************************
template<typename Traits>
void DirichletOffNodeSet<PHAL::AlbanyTraits::SGResidual, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Specialization not yet ipmlemented.\n");
}

// **********************************************************************
// Specialization: Stochastic Galerkin Jacobian
// **********************************************************************
template<typename Traits>
DirichletOffNodeSet<PHAL::AlbanyTraits::SGJacobian, Traits>::
DirichletOffNodeSet(Teuchos::ParameterList& p) :
  DirichletBase<PHAL::AlbanyTraits::SGJacobian, Traits>(p)
{
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Specialization not yet ipmlemented.\n");
}

// **********************************************************************
template<typename Traits>
void DirichletOffNodeSet<PHAL::AlbanyTraits::SGJacobian, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Specialization not yet ipmlemented.\n");
}

// **********************************************************************
// Specialization: Stochastic Galerkin Tangent
// **********************************************************************
template<typename Traits>
DirichletOffNodeSet<PHAL::AlbanyTraits::SGTangent, Traits>::
DirichletOffNodeSet(Teuchos::ParameterList& p) :
  DirichletBase<PHAL::AlbanyTraits::SGTangent, Traits>(p)
{
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Specialization not yet ipmlemented.\n");
}

// **********************************************************************
template<typename Traits>
void DirichletOffNodeSet<PHAL::AlbanyTraits::SGTangent, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Specialization not yet ipmlemented.\n");
}
#endif
#ifdef ALBANY_ENSEMBLE

// **********************************************************************
// Specialization: Multi-point Residual
// **********************************************************************

template<typename Traits>
DirichletOffNodeSet<PHAL::AlbanyTraits::MPResidual, Traits>::
DirichletOffNodeSet(Teuchos::ParameterList& p) :
  DirichletBase<PHAL::AlbanyTraits::MPResidual, Traits>(p)
{
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Specialization not yet ipmlemented.\n");
}
// **********************************************************************
template<typename Traits>
void DirichletOffNodeSet<PHAL::AlbanyTraits::MPResidual, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Specialization not yet ipmlemented.\n");
}

// **********************************************************************
// Specialization: Multi-point Jacobian
// **********************************************************************
template<typename Traits>
DirichletOffNodeSet<PHAL::AlbanyTraits::MPJacobian, Traits>::
DirichletOffNodeSet(Teuchos::ParameterList& p) :
  DirichletBase<PHAL::AlbanyTraits::MPJacobian, Traits>(p)
{
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Specialization not yet ipmlemented.\n");
}

// **********************************************************************
template<typename Traits>
void DirichletOffNodeSet<PHAL::AlbanyTraits::MPJacobian, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Specialization not yet ipmlemented.\n");
}

// **********************************************************************
// Specialization: Multi-point Tangent
// **********************************************************************
template<typename Traits>
DirichletOffNodeSet<PHAL::AlbanyTraits::MPTangent, Traits>::
DirichletOffNodeSet(Teuchos::ParameterList& p) :
  DirichletBase<PHAL::AlbanyTraits::MPTangent, Traits>(p)
{
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Specialization not yet ipmlemented.\n");
}

// **********************************************************************
template<typename Traits>
void DirichletOffNodeSet<PHAL::AlbanyTraits::MPTangent, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Specialization not yet ipmlemented.\n");
}
#endif

} // Namespace PHAL
