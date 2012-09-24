/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

namespace PHAL {

//base flow values
const double ubar = 0.0; 
const double vbar = 0.0; 
const double wbar = 0.0; 
const double zetabar = 1.0; 
const double pbar = 0.714285714285714;
//fluid parameters  
const double alpha = 1.0; 
const double gamma_gas = 1.4; //gas constant 


//**********************************************************************
template<typename EvalT, typename Traits>
LinComprNSResid<EvalT, Traits>::
LinComprNSResid(const Teuchos::ParameterList& p) :
  wBF     (p.get<std::string>                   ("Weighted BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Scalar Data Layout") ),
  wGradBF    (p.get<std::string>                   ("Weighted Gradient BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Gradient Data Layout") ),
  C          (p.get<std::string>                   ("QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  Cgrad      (p.get<std::string>                   ("Gradient QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
  CDot       (p.get<std::string>                   ("QP Time Derivative Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  Residual   (p.get<std::string>                   ("Residual Name"),
              p.get<Teuchos::RCP<PHX::DataLayout> >("Node Vector Data Layout") )
{
  this->addDependentField(C);
  this->addDependentField(Cgrad);
  this->addDependentField(CDot);
  this->addDependentField(wBF);
  this->addDependentField(wGradBF);

  this->addEvaluatedField(Residual);


  this->setName("LinComprNSResid"+PHX::TypeString<EvalT>::value);

  std::vector<PHX::DataLayout::size_type> dims;
  wGradBF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];
  numDims  = dims[3];

  C.fieldTag().dataLayout().dimensions(dims);
  vecDim  = dims[2];

cout << " vecDim = " << vecDim << endl;
cout << " numDims = " << numDims << endl;
}

//**********************************************************************
template<typename EvalT, typename Traits>
void LinComprNSResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(C,fm);
  this->utils.setFieldData(Cgrad,fm);
  this->utils.setFieldData(CDot,fm);
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(wGradBF,fm);

  this->utils.setFieldData(Residual,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void LinComprNSResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  typedef Intrepid::FunctionSpaceTools FST;


  //convective flux matrices 
  //A[0][][] = A1 (vecDim x vecDim)
  //A[1][][] = A2 (vecDim x vecDim)
  //A[2][][] = A3 (vecDim x vecDim)
  //these are allocated statically for a maximum 3D problem (numDims = 3, vecDim = 5)
  std::vector<std::vector<std::vector<double> > > A; 
  A.resize(3); 
  for (int i=0; i<3; ++i) {
     A[i].resize(5); 
     for (int j=0; j<5; ++j) 
        A[i][j].resize(5); 
  }

  //zero out convective flux matrices
  for (std::size_t dim=0; dim<numDims; dim++){
    for (std::size_t i=0; i<vecDim; i++) {
      for (std::size_t j=0; j<vecDim; j++) {
         A[dim][i][j] = 0.0; 
       }
     }
  }    


  //populate A1 and A2
  for (std::size_t i=0; i<vecDim; i++) {
     A[0][i][i] = ubar; 
     A[1][i][i] = vbar; 
  }
  A[0][0][vecDim-1] = zetabar; 
  A[0][vecDim-2][0] = -1.0*zetabar; 
  A[0][vecDim-1][0] = gamma_gas*pbar; 
  A[1][1][vecDim-1] = zetabar; 
  A[1][vecDim-2][1] = -1.0*zetabar; 
  A[1][vecDim-1][1] = gamma_gas*pbar; 

  //if 3D problem, populate A3
  if (numDims == 3) {
    for (std::size_t i=0; i<vecDim; i++) {
      A[2][i][i] = wbar; 
    }
    A[2][2][vecDim-1] = zetabar; 
    A[2][vecDim-2][2] = -1.0*zetabar; 
    A[2][vecDim-1][2] = gamma_gas*pbar; 
  }

  /*cout.precision(3); 

  cout << "A1: " << endl; 
  for (std::size_t i=0; i<vecDim; i++) {
    for (std::size_t j=0; j<vecDim; j++) {
     cout << A[0][i][j] << '\t'; 
    }
   cout << endl; 
  }

  cout << "A2: " << endl; 
  for (std::size_t i=0; i<vecDim; i++) {
    for (std::size_t j=0; j<vecDim; j++) {
     cout << A[1][i][j] << '\t'; 
    }
   cout << endl; 
  }
  if (numDims == 3) {
    cout << "A3: " << endl; 
    for (std::size_t i=0; i<vecDim; i++) {
      for (std::size_t j=0; j<vecDim; j++) {
       cout << A[2][i][j] << '\t'; 
      }
     cout << endl; 
    }
  }
  */


    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t node=0; node < numNodes; ++node) {
          for (std::size_t i=0; i<vecDim; i++)  
             Residual(cell,node,i) = 0.0; 
          for (std::size_t qp=0; qp < numQPs; ++qp) {
            for (std::size_t i=0; i<vecDim; i++) {
               Residual(cell,node,i) += CDot(cell,qp,i)*wBF(cell,node,qp); //equation is unsteady
               for (std::size_t j=0; j<vecDim; j++) {
                 for (std::size_t dim=0; dim<numDims; dim++) {
                    Residual(cell,node,i) += wBF(cell,node,qp)*(A[dim][i][j]*Cgrad(cell,qp,j,dim));
                  }
               }
              //Cgrad(cell, qp, i, dim) * wGradBF(cell, node, qp, dim);
    	     } 
            } 
          } 
        }
}

//**********************************************************************
}

