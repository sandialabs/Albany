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

namespace Aeras {

template<typename EvalT, typename Traits>
ComputeAndScatterJacBase<EvalT, Traits>::
ComputeAndScatterJacBase(const Teuchos::ParameterList& p,
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
  numLevels  (dl->node_scalar_level       ->dimension(2)), 
  numFields  (0), numNodeVar(0), numVectorLevelVar(0),  numScalarLevelVar(0), numTracerVar(0)
{
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";

  const Teuchos::ArrayRCP<std::string> node_names         = p.get< Teuchos::ArrayRCP<std::string> >("Node Residual Names");
  const Teuchos::ArrayRCP<std::string> vector_level_names = p.get< Teuchos::ArrayRCP<std::string> >("Vector Level Residual Names");
  const Teuchos::ArrayRCP<std::string> scalar_level_names = p.get< Teuchos::ArrayRCP<std::string> >("Scalar Level Residual Names");
  const Teuchos::ArrayRCP<std::string> tracer_names       = p.get< Teuchos::ArrayRCP<std::string> >("Tracer Residual Names");

  numNodeVar   = node_names  .size();
  numVectorLevelVar  = vector_level_names .size();
  numScalarLevelVar  = scalar_level_names .size();
  numTracerVar = tracer_names.size();
  numFields = numNodeVar +  numVectorLevelVar + numScalarLevelVar +  numTracerVar;

  const std::string fieldName = p.get<std::string>("Scatter Field Name");
  // OG: why is this tag templated with ScalarT?
  //scatter_operation = Teuchos::rcp(new PHX::Tag<ScalarT>(fieldName, dl->dummy));
  scatter_operation = Teuchos::rcp(new PHX::Tag<MeshScalarT>(fieldName, dl->dummy));

  
  this->addDependentField(BF);
  this->addDependentField(wBF);
  this->addDependentField(GradBF);
  this->addDependentField(wGradBF);
  this->addDependentField(lambda_nodal);
  this->addDependentField(theta_nodal);

  this->addEvaluatedField(*scatter_operation);

  this->setName(fieldName);

 //read the hv coef
  double HVcoef = p.get<double>("HV coefficient");
  sqrtHVcoef = std::sqrt(HVcoef);

}

// **********************************************************************
template<typename EvalT, typename Traits>
void ComputeAndScatterJacBase<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm) 
{
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(GradBF,fm);
  this->utils.setFieldData(wGradBF,fm);
  this->utils.setFieldData(lambda_nodal,fm);
  this->utils.setFieldData(theta_nodal,fm);
}


// **********************************************************************
// Specialization: Jacobian
// **********************************************************************

template<typename Traits>
ComputeAndScatterJac<PHAL::AlbanyTraits::Jacobian, Traits>::
ComputeAndScatterJac(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Aeras::Layouts>& dl)
  : ComputeAndScatterJacBase<PHAL::AlbanyTraits::Jacobian,Traits>(p,dl)
{ }
// *********************************************************************
// Kokkos kernels
//#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT

//FIXME, IKT, 5/9/16: Kokkos functor implementations go here. 
//
//#endif
// **********************************************************************
template<typename Traits>
void ComputeAndScatterJac<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

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

//FIXME, IKT, 5/24/16: uncomment out the following once Kokkos functors have been implemented
//#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT

  Teuchos::RCP<Tpetra_Vector>      fT = workset.fT;
  Teuchos::RCP<Tpetra_CrsMatrix> JacT = workset.JacT;

  const bool loadResid = (fT != Teuchos::null);
  LO rowT; 
  Teuchos::Array<LO> colT; 

  std::cout << "DEBUG in ComputeAndScatterJac::EvaluateFields: " << __PRETTY_FUNCTION__ << "\n";
  std::cout << "LOAD RESIDUAL? " << loadResid << "\n";


//AMET calls for mass with (j, m, n ) = (0, -1, 0)
//HVDecorator calls for mass with 0, -1, 0 (for the time step as in AMET) and 0, 1, 0 for the HV operator

  ST mc = workset.m_coeff;

  std::cout <<"Workset coefficients: j = "<< workset.j_coeff << ", m = " <<workset.m_coeff << ", n = " <<workset.n_coeff << "\n";

  bool buildMass = ( ( workset.j_coeff == 0.0 )&&( workset.m_coeff != 0.0 )&&( workset.n_coeff == 0.0 ) );
  bool buildLaplace = ( workset.j_coeff == 0.0 )&&( workset.m_coeff == 0.0 )&&( workset.n_coeff == 1.0 );

  std::cout << "buildMass, buildLaplace " << buildMass << ", " << buildLaplace << "\n";

  int numcells_ = workset.numCells,
	  numnodes_ = this->numNodes;
  //for mass we do not really need even this
  /*
  Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> localMassMatr(numcells_,numnodes_,numnodes_);
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
          ST val2 = mc * this -> wBF(cell, node, node);
          JacT->sumIntoLocalValues(rowT, Teuchos::arrayView(&rowT,1), Teuchos::arrayView(&val2,1));
        }
        eq += this->numNodeVar;
        for (int level = 0; level < this->numLevels; level++) {
          for (int j = eq; j < eq+this->numVectorLevelVar; ++j) {
            for (int dim = 0; dim < this->numDims; ++dim, ++n) {
              rowT = eqID[n];
              ST val2 = mc * this -> wBF(cell, node, node);
              JacT->sumIntoLocalValues(rowT, Teuchos::arrayView(&rowT,1), Teuchos::arrayView(&val2,1));
            }
          }
          for (int j = eq+this->numVectorLevelVar; j < eq+this->numVectorLevelVar+this->numScalarLevelVar; ++j, ++n) {
            rowT = eqID[n];
            ST val2 = mc *  this -> wBF(cell, node, node);
            JacT->sumIntoLocalValues(rowT, Teuchos::arrayView(&rowT,1), Teuchos::arrayView(&val2,1));
          }
        }
        eq += this->numVectorLevelVar+this->numScalarLevelVar;
        for (int level = 0; level < this->numLevels; ++level) {
          for (int j = eq; j < eq+this->numTracerVar; ++j, ++n) {
            rowT = eqID[n];
            //Minus!
            ST val2 = mc * this -> wBF(cell, node, node);
            JacT->sumIntoLocalValues(rowT, Teuchos::arrayView(&rowT,1), Teuchos::arrayView(&val2,1));
          }
        }
        eq += this->numTracerVar;
      }
    }
  }


////////////////////////////////////////////////////////
  if ( buildLaplace ) {
    int numn = this->numNodes;
    Intrepid2::FieldContainer_Kokkos<ST, PHX::Layout, PHX::Device>  KK(numn*3, numn*2), KT(numn*2, numn*3),
   	                                                            GR(numn*3, numn*3), L(numn*2, numn*2),
								    GRKK(numn*3,numn*2), KTGRKK(numn*2,numn*2); 
    for (int cell=0; cell < workset.numCells; ++cell ) {
      const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];
      const int neq = nodeID[0].size();
      colT.resize(neq * this->numNodes);
      for (int node=0; node<this->numNodes; node++){
        for (int eq_col=0; eq_col<neq; eq_col++) {
          colT[neq * node + eq_col] =  nodeID[node][eq_col];
        }
      }

      //Build a submatrix (local matrix) for Laplace. For any particular cell:
      //The matrix acts on vector (u1 v1 u2 v2 u3 v3 ... uNumNodes vNumNodes)
      //and returns (WGrad(u1) WGrad(v1) WGrad(u2) WGrad(v2) ...), that is, the output vector is ordered
      //the same way as the input vector. After this matrix is built,
      //if we are filling Laplace matrix's row corresponding to node N for u vector, then we need row
      //2N from the local matrix. In other words, this Laplace operator takes u and v from all nodes on
      //a certain level and produces weak Laplace for them. If we fix a nodal value for u,
      //then Laplace at this nodal value for u depends on local matrix' row 2N. For v at node
      //N it should be row 2N+1.

      //The local matrix is K^T*WeakGrad*K. Here K is a transform from lon/lat velocity
      //to xyz velocity, WeakGrad calculates a weak Laplace op (grad*grad) for each (x, or y, or z)
      //component, and K^T is equivalent to K^{inverse}.

      //The biggest effort is to find how to insert values of the local matrix to the big matrix.
      KK.initialize();
      GR.initialize();
      for (int node=0; node<this->numNodes; node++) {
        ST lam = this -> lambda_nodal(cell, node),
	    th = this -> theta_nodal(cell, node);
        const ST k11 = -sin(lam),
	         k12 = -sin(th)*cos(lam),
	         k21 =  cos(lam),
	         k22 = -sin(th)*sin(lam),
	         k32 =  cos(th);
        KK(node*3, node*2) = k11;
        KK(node*3, node*2+1) = k12;
        KK(node*3+1, node*2) = k21;
        KK(node*3+1, node*2+1) = k22;
        KK(node*3+2, node*2+1) = k32;
      }

      for (int no = 0; no < this->numNodes; no++) {
        for (int mo = 0; mo< this->numNodes; mo++) {
          ST val = 0;
	  for (int qp = 0; qp < this->numNodes; qp++) {
	    val += this->GradBF(cell,no,qp,0)*this->GradBF(cell,mo,qp,0)*this->wBF(cell,qp,qp)
	        +  this->GradBF(cell,no,qp,1)*this->GradBF(cell,mo,qp,1)*this->wBF(cell,qp,qp);
	  }
	  GR(no*3,mo*3) = val;
	  GR(no*3+1,mo*3+1) = val;
	  GR(no*3+2,mo*3+2) = val;
        }
      }

      //now let's multiply everything
      GRKK.initialize();
      for (int ii = 0; ii < 3*numn; ii++)
        for (int jj = 0; jj < 2*numn; jj++)
          for(int cc = 0; cc < 3*numn; cc++)
            GRKK(ii,jj) += GR(ii,cc)*KK(cc,jj);
      KTGRKK.initialize();
      for(int ii = 0; ii < 2*numn; ii++)
        for (int jj = 0; jj < 2*numn; jj++)
          for(int cc = 0; cc < 3*numn; cc++)
            KTGRKK(ii,jj) += KK(cc,ii)*GRKK(cc,jj);

      for (int node = 0; node < this->numNodes; ++node) {
        const Teuchos::ArrayRCP<int>& eqID  = nodeID[node];
        int n = 0, eq = 0;
        //dealing with surf pressure
        for (int j = eq; j < eq+this->numNodeVar; ++j, ++n) {
          /*
	  //OG Disabling Laplace for surface pressure.
	  //rowT is LID in row map for vector x=all variables
	  //so rowT here is a row index corresponding to pressure eqn at (node, cell)
	  rowT = eqID[n];
          //loop over nodes on the same level
	  for (unsigned int m=0; m< this->numNodes; m++) {
            const int col_ = colT[m*neq];//= nodeID[m][n];
	    //at this point we know node, cell, m
            ST val = 0;
	    for (int qp = 0; qp < this->numNodes; qp++) {
	      val += this->GradBF(cell,node,qp,0)*this->GradBF(cell,m,qp,0)*this->wBF(cell,qp,qp)
	  	  + this->GradBF(cell,node,qp,1)*this->GradBF(cell,m,qp,1)*this->wBF(cell,qp,qp);
            }
	    val *= this->sqrtHVcoef;
	    JacT->sumIntoLocalValues(rowT, Teuchos::arrayView(&col_,1), Teuchos::arrayView(&val,1));
	  }
	  */
        }
        eq += this->numNodeVar;

        for (int level = 0; level < this->numLevels; level++) {
          //dealing with velocity
	  for (int j = eq; j < eq+this->numVectorLevelVar; ++j) {
	    for (int dim = 0; dim < this->numDims; ++dim, ++n) {
	      rowT = eqID[n];
	      for (unsigned int m=0; m< this->numNodes; m++) {
	        //filling u values
	        if (dim == 0) {
	          //filling dependency on u values
	          const int col1_ = nodeID[m][n];
	          ST val = this->sqrtHVcoef * KTGRKK(node*2,m*2);
	          JacT->sumIntoLocalValues(rowT, Teuchos::arrayView(&col1_,1), Teuchos::arrayView(&val,1));
   	          //filling dependency on v values, so, it is eqn n+1
	          const int col2_ = nodeID[m][n+1];
	          val = this->sqrtHVcoef * KTGRKK(node*2,m*2+1);
	          JacT->sumIntoLocalValues(rowT, Teuchos::arrayView(&col2_,1), Teuchos::arrayView(&val,1));
	        }
	        //filling v values
	        if (dim == 1) {
	          //filling dependencies on u values. The current eqn is n, so, we need to look at n-1 level IDs.
	          const int col1_ = nodeID[m][n-1];
	          ST val = this->sqrtHVcoef * KTGRKK(node*2+1,m*2);
	          JacT->sumIntoLocalValues(rowT, Teuchos::arrayView(&col1_,1), Teuchos::arrayView(&val,1));
	          //filling dependencies on v values.
	          const int col2_ = nodeID[m][n];
	          val = this->sqrtHVcoef * KTGRKK(node*2+1,m*2+1);
	          JacT->sumIntoLocalValues(rowT, Teuchos::arrayView(&col2_,1), Teuchos::arrayView(&val,1));
	        }
	      }
	    }//dim loop
	  }
	  //dealing with temperature
	  for (int j = eq+this->numVectorLevelVar; j < eq+this->numVectorLevelVar+this->numScalarLevelVar; ++j, ++n) {
	    rowT = eqID[n];//the same as nodeID[node][n]
	    //loop over nodes on the same level
	    for (unsigned int m=0; m< this->numNodes; m++) {
     	      const int col_ = nodeID[m][n];
	      //const int row_ = nodeID[node][n];
	      //at this point we know node, cell, m
    	      ST val = 0;
	      for (int qp = 0; qp < this->numNodes; qp++) {
	        val += this->GradBF(cell,node,qp,0)*this->GradBF(cell,m,qp,0)*this->wBF(cell,qp,qp)
	            + this->GradBF(cell,node,qp,1)*this->GradBF(cell,m,qp,1)*this->wBF(cell,qp,qp);
              }
	      val *= this->sqrtHVcoef;
	      JacT->sumIntoLocalValues(rowT, Teuchos::arrayView(&col_,1), Teuchos::arrayView(&val,1));
	    }
          }
        }//end of level loop for velocity and temperature

        eq += this->numVectorLevelVar+this->numScalarLevelVar;
        for (int level = 0; level < this->numLevels; ++level) {
          //dealing with tracers
          for (int j = eq; j < eq+this->numTracerVar; ++j, ++n) {
     	    rowT = eqID[n];//the same as nodeID[node][n]
	    //loop over nodes on the same level
	    for (unsigned int m=0; m< this->numNodes; m++) {
    	      const int col_ = nodeID[m][n];
	      //const int row_ = nodeID[node][n];
	      //at this point we know node, cell, m
    	      ST val = 0;
	      for (int qp = 0; qp < this->numNodes; qp++) {
	        val += this->GradBF(cell,node,qp,0)*this->GradBF(cell,m,qp,0)*this->wBF(cell,qp,qp)
	            + this->GradBF(cell,node,qp,1)*this->GradBF(cell,m,qp,1)*this->wBF(cell,qp,qp);
	      }
	      val *= this->sqrtHVcoef;
	      JacT->sumIntoLocalValues(rowT, Teuchos::arrayView(&col_,1), Teuchos::arrayView(&val,1));
	    }
          }
        }//end of level loop for tracers
        eq += this->numTracerVar;
      }//end of loop over nodes
    }//end of loop over cells
  }//end of if buildLaplace

//#else

  //FIXME, IKT, 5/9/16: this function needs to be Kokkos-ized!  Kokkos implementation goes here.
  //std::cout << "ComputeAndScatterJac evaluateFields Jacobian specialization has not been Kokkos-ized yet!" << std::endl; 
 
//#endif
}

#ifdef ALBANY_ENSEMBLE
// **********************************************************************
// Specialization: Multi-point Residual
// **********************************************************************
template<typename Traits>
ComputeAndScatterJac<PHAL::AlbanyTraits::MPResidual,Traits>::
ComputeAndScatterJac(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Aeras::Layouts>& dl)
  : ComputeAndScatterJacBase<PHAL::AlbanyTraits::MPResidual,Traits>(p,dl)
{}

template<typename Traits>
void ComputeAndScatterJac<PHAL::AlbanyTraits::MPResidual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  //FIXME, IKT, 5/9/16: this function needs to be implemented! 
  std::cout << "ComputeAndScatterJac evaluateFields MPResidual specialization has not been implemented yet!" << std::endl; 
}

// **********************************************************************
// Specialization: Multi-point Jacobian
// **********************************************************************

template<typename Traits>
ComputeAndScatterJac<PHAL::AlbanyTraits::MPJacobian, Traits>::
ComputeAndScatterJac(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Aeras::Layouts>& dl)
  : ComputeAndScatterJacBase<PHAL::AlbanyTraits::MPJacobian,Traits>(p,dl)
{ }

// **********************************************************************
template<typename Traits>
void ComputeAndScatterJac<PHAL::AlbanyTraits::MPJacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  //FIXME, IKT, 5/9/16: this function needs to be implemented! 
  std::cout << "ComputeAndScatterJac evaluateFields MPJacobian specialization has not been implemented yet!" << std::endl; 
}
#endif

}
