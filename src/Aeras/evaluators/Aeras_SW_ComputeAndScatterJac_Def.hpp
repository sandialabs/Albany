//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <vector>
#include <string>

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Aeras_Layouts.hpp"
#include "Albany_Utils.hpp"

namespace Aeras {

template<typename EvalT, typename Traits>
SW_ComputeAndScatterJacBase<EvalT, Traits>::
SW_ComputeAndScatterJacBase(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Aeras::Layouts>& dl) :
  BF            (p.get<std::string>  ("BF Name"),           dl->node_qp_scalar),
  wBF           (p.get<std::string>  ("Weighted BF Name"),  dl->node_qp_scalar),
  GradBF        (p.get<std::string>  ("Gradient BF Name"),  dl->node_qp_gradient),
  wGradBF       (p.get<std::string>  ("Weighted Gradient BF Name"), dl->node_qp_gradient),
  lambda_nodal  (p.get<std::string>  ("Lambda Coord Nodal Name"), dl->node_scalar),
  theta_nodal   (p.get<std::string>  ("Theta Coord Nodal Name"), dl->node_scalar),
  worksetSize(dl->node_scalar             ->dimension(0)),
  numNodes   (dl->node_scalar             ->dimension(1)),
  numDims    (dl->node_qp_gradient        ->dimension(3)),
  numFields  (0), numNodeVar(3)
{
  //std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";
  //std::cout << "IKT SW_ComputeAndScatterJacBase constructor! \n";

  numFields = numNodeVar;

  const std::string fieldName = p.get<std::string>("Scatter Field Name");
  // OG: why is this tag templated with ScalarT?
  //scatter_operation = Teuchos::rcp(new PHX::Tag<ScalarT>(fieldName, dl->dummy));
  scatter_operation = Teuchos::rcp(new PHX::Tag<MeshScalarT>(fieldName, dl->dummy));

  
  this->addDependentField(BF);
  this->addDependentField(wBF);
  //IKT - the following would only be needed if we were computing the Laplace operator here.
  //this->addDependentField(GradBF);
  //this->addDependentField(wGradBF);
  //this->addDependentField(lambda_nodal);
  //this->addDependentField(theta_nodal);

  this->addEvaluatedField(*scatter_operation);

  this->setName(fieldName);


}

// **********************************************************************
template<typename EvalT, typename Traits>
void SW_ComputeAndScatterJacBase<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm) 
{
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(wBF,fm);
  //this->utils.setFieldData(GradBF,fm);
  //this->utils.setFieldData(wGradBF,fm);
  //this->utils.setFieldData(lambda_nodal,fm);
  //this->utils.setFieldData(theta_nodal,fm);
}


// **********************************************************************
// Specialization: Jacobian
// **********************************************************************
template<typename Traits>
SW_ComputeAndScatterJac<PHAL::AlbanyTraits::Jacobian, Traits>::
SW_ComputeAndScatterJac(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Aeras::Layouts>& dl)
  : SW_ComputeAndScatterJacBase<PHAL::AlbanyTraits::Jacobian,Traits>(p,dl)
{ }

// **********************************************************************
template<typename Traits>
void SW_ComputeAndScatterJac<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

//std::cout << "IKT in evaluateFields!" << std::endl; 
//First, we need to compute the local mass and laplacian matrices 
//(checking the n_coeff flag for whether the laplacian is needed) as follows: 
//Mass:
//loop over cells, c
//  loop over levels, l
//    loop over nodes,node
//      q=n; m=node
//      diag(c,l,node) = BF(node,q)*wBF(m,q)
//
//Laplacian:
//loop over cells, c
//  loop over levels, l
//    loop over nodes,node
//      loop over nodes,m
//        loop over qp, q
//          loop over dim, d
//            laplace(c,l,node,m) += gradBF(node,q,d)*wGradBF(m,q,d)
//
//(There’s also a loop over unknowns per node.)
//
//Then the values of these matrices need to be scattered into the global Jacobian.

  Teuchos::RCP<Tpetra_Vector>      fT = workset.fT;
  Teuchos::RCP<Tpetra_CrsMatrix> JacT = workset.JacT;

  const bool loadResid = (fT != Teuchos::null);
  LO rowT; 
  Teuchos::Array<LO> colT; 

  //std::cout << "DEBUG in SW_ComputeAndScatterJac::EvaluateFields: " << __PRETTY_FUNCTION__ << "\n";
  //std::cout << "LOAD RESIDUAL? " << loadResid << "\n";

  //AMET calls for mass with (j, m, n ) = (0, -1, 0)

  RealType mc = workset.m_coeff;

  bool buildMass = true; // ( ( workset.j_coeff == 0.0 )&&( workset.m_coeff != 0.0 )&&( workset.n_coeff == 0.0 ) );

  int numcells_ = workset.numCells,
	  numnodes_ = this->numNodes;
  //for mass we do not really need even this
  /*
  Kokkos::DynRankView<ScalarT, PHX::Device> localMassMatr(numcells_,numnodes_,numnodes_);
  for(int cell = 0; cell < numcells_; cell++){
	  for(int node = 0; node < numnodes_; node++)
	     localMassMatr(cell, node, node) = this -> wBF(cell, node, node);
  }*/

  if ( buildMass ) {
    for (int cell=0; cell < workset.numCells; ++cell ) {
      const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];
      const int neq = nodeID[0].size();

      for (int node = 0; node < this->numNodes; ++node) {
        const Teuchos::ArrayRCP<int>& eqID  = nodeID[node];
        int n = 0, eq = 0;
        for (int j = eq; j < eq+this->numNodeVar; ++j, ++n) {
          rowT = eqID[n];
          RealType val2 = mc * this -> wBF(cell, node, node);
          JacT->sumIntoLocalValues(rowT, Teuchos::arrayView(&rowT,1), Teuchos::arrayView(&val2,1));
        }
        eq += this->numNodeVar;
      }
    }
  }

}

#ifdef ALBANY_ENSEMBLE
// **********************************************************************
// Specialization: Multi-point Residual
// **********************************************************************
template<typename Traits>
SW_ComputeAndScatterJac<PHAL::AlbanyTraits::MPResidual,Traits>::
SW_ComputeAndScatterJac(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Aeras::Layouts>& dl)
  : SW_ComputeAndScatterJacBase<PHAL::AlbanyTraits::MPResidual,Traits>(p,dl)
{}

template<typename Traits>
void SW_ComputeAndScatterJac<PHAL::AlbanyTraits::MPResidual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  //FIXME, IKT, 5/9/16: this function needs to be implemented! 
  std::cout << "SW_ComputeAndScatterJac evaluateFields MPResidual specialization has not been implemented yet!" << std::endl; 
}

// **********************************************************************
// Specialization: Multi-point Jacobian
// **********************************************************************

template<typename Traits>
SW_ComputeAndScatterJac<PHAL::AlbanyTraits::MPJacobian, Traits>::
SW_ComputeAndScatterJac(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Aeras::Layouts>& dl)
  : SW_ComputeAndScatterJacBase<PHAL::AlbanyTraits::MPJacobian,Traits>(p,dl)
{ }

// **********************************************************************
template<typename Traits>
void SW_ComputeAndScatterJac<PHAL::AlbanyTraits::MPJacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  //FIXME, IKT, 5/9/16: this function needs to be implemented! 
  std::cout << "SW_ComputeAndScatterJac evaluateFields MPJacobian specialization has not been implemented yet!" << std::endl; 
}
#endif

}
