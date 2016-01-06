//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

namespace PHAL {
//#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
template<typename EvalT, typename Traits>
ComputeBasisFunctions<EvalT, Traits>::
ComputeBasisFunctions(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
  coordVec      (p.get<std::string>  ("Coordinate Vector Name"), dl->vertices_vector ),
  cubature      (p.get<Teuchos::RCP <Intrepid::Cubature<RealType> > >("Cubature")),
  intrepidBasis (p.get<Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > > ("Intrepid Basis") ),
  cellType      (p.get<Teuchos::RCP <shards::CellTopology> > ("Cell Type")),
  weighted_measure (p.get<std::string>  ("Weights Name"), dl->qp_scalar ),
  jacobian_det (p.get<std::string>  ("Jacobian Det Name"), dl->qp_scalar ),
  BF            (p.get<std::string>  ("BF Name"), dl->node_qp_scalar),
  wBF           (p.get<std::string>  ("Weighted BF Name"), dl->node_qp_scalar),
  GradBF        (p.get<std::string>  ("Gradient BF Name"), dl->node_qp_gradient),
  wGradBF       (p.get<std::string>  ("Weighted Gradient BF Name"), dl->node_qp_gradient)
{
  this->addDependentField(coordVec);
  this->addEvaluatedField(weighted_measure);
  this->addEvaluatedField(jacobian_det);
  this->addEvaluatedField(BF);
  this->addEvaluatedField(wBF);
  this->addEvaluatedField(GradBF);
  this->addEvaluatedField(wGradBF);

  // Get Dimensions
  std::vector<PHX::DataLayout::size_type> dim;
  dl->node_qp_gradient->dimensions(dim);

  int containerSize = dim[0];
  numNodes = dim[1];
  numQPs = dim[2];
  numDims = dim[3];


  std::vector<PHX::DataLayout::size_type> dims;
  dl->vertices_vector->dimensions(dims);
  numVertices = dims[1];

  // Allocate Temporary FieldContainers
  val_at_cub_points.resize(numNodes, numQPs);
  grad_at_cub_points.resize(numNodes, numQPs, numDims);
  refPoints.resize(numQPs, numDims);
  refWeights.resize(numQPs);
  jacobian.resize(containerSize, numQPs, numDims, numDims);
  jacobian_inv.resize(containerSize, numQPs, numDims, numDims);

  // Pre-Calculate reference element quantitites
  cubature->getCubature(refPoints, refWeights);
  intrepidBasis->getValues(val_at_cub_points, refPoints, Intrepid::OPERATOR_VALUE);
  intrepidBasis->getValues(grad_at_cub_points, refPoints, Intrepid::OPERATOR_GRAD);

  this->setName("ComputeBasisFunctions"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ComputeBasisFunctions<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(coordVec,fm);
  this->utils.setFieldData(weighted_measure,fm);
  this->utils.setFieldData(jacobian_det,fm);
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(GradBF,fm);
  this->utils.setFieldData(wGradBF,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ComputeBasisFunctions<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

  /** The allocated size of the Field Containers must currently 
    * match the full workset size of the allocated PHX Fields, 
    * this is the size that is used in the computation. There is
    * wasted effort computing on zeroes for the padding on the
    * final workset. Ideally, these are size numCells.
  //int containerSize = workset.numCells;
    */

  Intrepid::CellTools<MeshScalarT>::setJacobian(jacobian, refPoints, coordVec, intrepidBasis);
  Intrepid::CellTools<MeshScalarT>::setJacobianInv(jacobian_inv, jacobian);
  Intrepid::CellTools<MeshScalarT>::setJacobianDet(jacobian_det, jacobian);

  Intrepid::FunctionSpaceTools::computeCellMeasure<MeshScalarT>
    (weighted_measure, jacobian_det, refWeights);
  Intrepid::FunctionSpaceTools::HGRADtransformVALUE<RealType>
    (BF, val_at_cub_points);
  Intrepid::FunctionSpaceTools::multiplyMeasure<MeshScalarT>
    (wBF, weighted_measure, BF);
  Intrepid::FunctionSpaceTools::HGRADtransformGRAD<MeshScalarT>
    (GradBF, jacobian_inv, grad_at_cub_points);
  Intrepid::FunctionSpaceTools::multiplyMeasure<MeshScalarT>
    (wGradBF, weighted_measure, GradBF);
}
/*#else // ALBANY_KOKKOS_UNDER_DEVELOPMENT
//**********************************************************************
template<typename EvalT, typename Traits>
ComputeBasisFunctions<EvalT, Traits>::
ComputeBasisFunctions(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
  coordVec      (p.get<std::string>  ("Coordinate Vector Name"), dl->vertices_vector ),
  cubature      (p.get<Teuchos::RCP <Intrepid::Cubature<RealType> > >("Cubature")),
  intrepidBasis (p.get<Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > > ("Intrepid Basis") ),
  cellType      (p.get<Teuchos::RCP <shards::CellTopology> > ("Cell Type")),
  weighted_measure (p.get<std::string>  ("Weights Name"), dl->qp_scalar ),
  jacobian_inv  (p.get<std::string>  ("Jacobian Inv Name"), dl->qp_tensor ),
  jacobian_det  (p.get<std::string>  ("Jacobian Det Name"), dl->qp_scalar ),
  BF            (p.get<std::string>  ("BF Name"), dl->node_qp_scalar),
  wBF           (p.get<std::string>  ("Weighted BF Name"), dl->node_qp_scalar),
  GradBF        (p.get<std::string>  ("Gradient BF Name"), dl->node_qp_gradient),
  wGradBF       (p.get<std::string>  ("Weighted Gradient BF Name"), dl->node_qp_gradient),
  jacobian  (p.get<std::string>  ("Jacobian Name"), dl->qp_tensor )
{
  this->addDependentField(coordVec);
  this->addEvaluatedField(weighted_measure);
  this->addEvaluatedField(jacobian_det);
  this->addEvaluatedField(jacobian_inv);
  this->addEvaluatedField(jacobian); 
  this->addEvaluatedField(BF);
  this->addEvaluatedField(wBF);
  this->addEvaluatedField(GradBF);
  this->addEvaluatedField(wGradBF);

  // Get Dimensions
  std::vector<PHX::DataLayout::size_type> dim;
  dl->node_qp_gradient->dimensions(dim);

  int containerSize = dim[0];
  numNodes = dim[1];
  numQPs = dim[2];
  numDims = dim[3];


  std::vector<PHX::DataLayout::size_type> dims;
  dl->vertices_vector->dimensions(dims);
  numVertices = dims[1];

  // Allocate Temporary FieldContainers
  val_at_cub_points.resize(numNodes, numQPs);
  grad_at_cub_points.resize(numNodes, numQPs, numDims);  
  refPoints.resize(numQPs, numDims);
  refWeights.resize(numQPs);

 
  val_at_cub_points_CUDA=Kokkos::View <RealType**, PHX::Device>("val_at_cub_points", numNodes, numQPs);
  grad_at_cub_points_CUDA=Kokkos::View <RealType***, PHX::Device>("grad_at_cub_points", numNodes, numQPs, numDims);
  refPoints_CUDA=Kokkos::View <RealType**, PHX::Device>("refPoints", numQPs, numDims);
  refWeights_CUDA=Kokkos::View <RealType*, PHX::Device>("refWeights", numQPs);

  // Pre-Calculate reference element quantitites
  cubature->getCubature(refPoints, refWeights);
  intrepidBasis->getValues(val_at_cub_points, refPoints, Intrepid::OPERATOR_VALUE);
  intrepidBasis->getValues(grad_at_cub_points, refPoints, Intrepid::OPERATOR_GRAD);


  for (int i=0; i < numQPs; i++)
  {
   refWeights_CUDA(i)=refWeights(i);
   for (int j=0; j< numDims; j++)
     refPoints_CUDA(i,j)=refPoints(i,j);
  }

  for (int i=0; i< numNodes; i++){
     for (int j=0; j<numQPs; j++) {
        val_at_cub_points_CUDA(i,j)=val_at_cub_points(i,j);
        for (int k=0; k< numDims; k++)
            grad_at_cub_points_CUDA(i,j,k)=grad_at_cub_points(i,j,k);
     }
  }

  this->setName("ComputeBasisFunctions" );
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ComputeBasisFunctions<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(coordVec,fm);
  this->utils.setFieldData(weighted_measure,fm);
  this->utils.setFieldData(jacobian_det,fm);
  this->utils.setFieldData(jacobian_inv,fm);
  this->utils.setFieldData(jacobian,fm);
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(GradBF,fm);
  this->utils.setFieldData(wGradBF,fm);
}
// ********************************************************************
template<class Scalar, class ArrayOutData, class ArrayInDataLeft, class ArrayInDataRight>
void scalarMultiplyDataData(ArrayOutData &           outputData,
                            ArrayInDataLeft &        inputDataLeft,
                            ArrayInDataRight &       inputDataRight,
                            const bool               reciprocal) {

 // const bool reciprocal = false;
  int invalRank      = inputDataRight.rank();
  int outvalRank     = outputData.rank();
  int numCells       = outputData.dimension(0);
  int numPoints      = outputData.dimension(1);
  int numDataPoints  = inputDataLeft.dimension(1);
  int dim1Tens       = 0;
  int dim2Tens       = 0;
  if (outvalRank > 2) {
    dim1Tens = outputData.dimension(2);
    if (outvalRank > 3) {
      dim2Tens = outputData.dimension(3);
    }
  }


  if (outvalRank == invalRank) {

    if (numDataPoints != 1) { 
       switch(invalRank) {
        case 2: {
          if (reciprocal) {
            for(int cl = 0; cl < numCells; cl++) {
              for(int pt = 0; pt < numPoints; pt++) {
                  outputData(cl, pt) = inputDataRight(cl, pt)/inputDataLeft(cl, pt);
              } // P-loop
            } // C-loop
          }
        else {
            for(int cl = 0; cl < numCells; cl++) {
              for(int pt = 0; pt < numPoints; pt++) {
                  outputData(cl, pt) = inputDataRight(cl, pt)*inputDataLeft(cl, pt);
              } // P-loop
            } // C-loop
          }
        }// case 2
        break;
      }// invalRank
     }
    else { // constant left data

      switch(invalRank) {
        case 2: {
          if (reciprocal) {
            for(int cl = 0; cl < numCells; cl++) {
              for(int pt = 0; pt < numPoints; pt++) {
                  outputData(cl, pt) = inputDataRight(cl, pt)/inputDataLeft(cl, 0);
              } // P-loop
            } // C-loop
          }
          else {
            for(int cl = 0; cl < numCells; cl++) {
              for(int pt = 0; pt < numPoints; pt++) {
                  outputData(cl, pt) = inputDataRight(cl, pt)*inputDataLeft(cl, 0);
              } // P-loop
            } // C-loop
          }
        }// case 2
        break;

      } // invalRank
    } // numDataPoints
    }
  else {

   } // end if (outvalRank = invalRank)

} // scalarMultiplyDataData



template<class Scalar, class ArrayOut, class ArrayDet, class ArrayWeights>
inline void computeCellMeasure(ArrayOut             & outVals,
                               const ArrayDet       & inDet,
                               const ArrayWeights   & inWeights) {
 scalarMultiplyDataData<Scalar>(outVals, inDet, inWeights, true);
  for (int cell=0; cell<outVals.dimension(0); cell++) {
         if (inDet(cell,0) < 0.0) {
               for (int point=0; point<outVals.dimension(1); point++) {
                       outVals(cell, point) *= -1.0;
                }
          }
       }

}

// *********************************************************************
template<class Scalar, class ArrayOutFields, class ArrayInFields>
void cloneFields(ArrayOutFields &       outputFields,
                 const ArrayInFields &  inputFields) {
  int invalRank      = inputFields.rank();
  int outvalRank     = outputFields.rank();
  int numCells       = outputFields.dimension(0);
  int numFields      = outputFields.dimension(1);
  int numPoints      = outputFields.dimension(2);
  int dim1Tens       = 0;
  int dim2Tens       = 0;
  if (outvalRank > 3) {
    dim1Tens = outputFields.dimension(3);
    if (outvalRank > 4) {
      dim2Tens = outputFields.dimension(4);
    }
  }
 
 for(int cl = 0; cl < numCells; cl++) {
        for(int bf = 0; bf < numFields; bf++) {
          for(int pt = 0; pt < numPoints; pt++) {
            outputFields(cl, bf, pt) = inputFields(bf, pt);
          } // P-loop
        } // F-loop
      } // C-loop

}
// **********************************************************************
template<class Scalar, class ArrayTypeOut, class ArrayTypeIn>
void HGRADtransformVALUE(ArrayTypeOut       & outVals,
                                             const ArrayTypeIn  & inVals) {

  cloneFields<Scalar>(outVals, inVals);

}
// ***********************************************************************
template<class Scalar, class ArrayOutFields, class ArrayInData, class ArrayInFields>
void scalarMultiplyDataField(ArrayOutFields &     outputFields,
                                         const ArrayInData &  inputData,
                                         ArrayInFields &      inputFields,
                                         const bool           reciprocal) {

  int invalRank      = inputFields.rank();
  int outvalRank     = outputFields.rank();
  int numCells       = outputFields.dimension(0);
  int numFields      = outputFields.dimension(1);
  int numPoints      = outputFields.dimension(2);
  int numDataPoints  = inputData.dimension(1);
  int dim1Tens       = 0;
  int dim2Tens       = 0;
  if (outvalRank > 3) {
    dim1Tens = outputFields.dimension(3);
    if (outvalRank > 4) {
      dim2Tens = outputFields.dimension(4);
    }
  }
  
    if (outvalRank == invalRank) {

    if (numDataPoints != 1) { // nonconstant data

      switch(invalRank) {
        case 3: {
          if (reciprocal) {
            for(int cl = 0; cl < numCells; cl++) {
              for(int bf = 0; bf < numFields; bf++) {
                for(int pt = 0; pt < numPoints; pt++) {
                  outputFields(cl, bf, pt) = inputFields(cl, bf, pt)/inputData(cl, pt);
                } // P-loop
              } // F-loop
            } // C-loop
          }
          else {
            for(int cl = 0; cl < numCells; cl++) {
              for(int bf = 0; bf < numFields; bf++) {
                for(int pt = 0; pt < numPoints; pt++) {
                  outputFields(cl, bf, pt) = inputFields(cl, bf, pt)*inputData(cl, pt);
                } // P-loop
              } // F-loop
            } // C-loop
          }
        }// case 3
        break;
      } 
     }
    else { //constant data

      switch(invalRank) {
        case 3: {
          if (reciprocal) {
            for(int cl = 0; cl < numCells; cl++) {
              for(int bf = 0; bf < numFields; bf++) {
                for(int pt = 0; pt < numPoints; pt++) {
                  outputFields(cl, bf, pt) = inputFields(cl, bf, pt)/inputData(cl, 0);
                } // P-loop
              } // F-loop
            } // C-loop
          }
          else {
            for(int cl = 0; cl < numCells; cl++) {
              for(int bf = 0; bf < numFields; bf++) {
                for(int pt = 0; pt < numPoints; pt++) {
                  outputFields(cl, bf, pt) = inputFields(cl, bf, pt)*inputData(cl, 0);
                } // P-loop
              } // F-loop
            } // C-loop
          }
        }// case 3
        break;

     } // invalRank 
   } // numDataPoints
  }
}
// ***********************************************************************
template<class Scalar, class ArrayTypeOut, class ArrayTypeMeasure, class ArrayTypeIn>
void multiplyMeasure(ArrayTypeOut             & outVals,
                                         const ArrayTypeMeasure   & inMeasure,
                                         const ArrayTypeIn        & inVals) {

  scalarMultiplyDataField<Scalar>(outVals, inMeasure, inVals, false);

} // multiplyMeasure


// ************************************************************************
// Kokkos functor
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void ComputeBasisFunctions<EvalT, Traits>:: operator () (const int i) const
{
  
  int numCells_=GradBF.dimension(0);
  int numQPs_= GradBF.dimension(2);
  int numDims_=GradBF.dimension(3);
  int numNodes_= GradBF.dimension(1);

  for(int nodes = 0; nodes < numNodes_; nodes++) 
     for(int pt = 0; pt < numQPs_; pt++) 
       for(int row = 0; row < numDims_; row++)
              GradBF(i, nodes, pt, row) = 0.0;


  //Intrepid::setJacobian
//PHX::MDField <MeshScalarT,Cell,QuadPoint,Dim,Dim> 
  int dim0 = refPoints_CUDA.dimension(0);
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;

  double BasisGrads[8][8][3];
 
  for (int i0 = 0; i0 < dim0; i0++) {
        x = refPoints_CUDA(i0, 0);
        y = refPoints_CUDA(i0, 1);
        z = refPoints_CUDA(i0, 2);
        
        BasisGrads[0][i0][0] = -(1.0 - y)*(1.0 - z)/8.0;
        BasisGrads[0][i0][1] = -(1.0 - x)*(1.0 - z)/8.0;
        BasisGrads[0][i0][2] = -(1.0 - x)*(1.0 - y)/8.0;

        BasisGrads[1][i0][0] =  (1.0 - y)*(1.0 - z)/8.0;
        BasisGrads[1][i0][1] = -(1.0 + x)*(1.0 - z)/8.0;
        BasisGrads[1][i0][2] = -(1.0 + x)*(1.0 - y)/8.0;

        BasisGrads[2][i0][0] =  (1.0 + y)*(1.0 - z)/8.0;
        BasisGrads[2][i0][1] =  (1.0 + x)*(1.0 - z)/8.0;
        BasisGrads[2][i0][2] = -(1.0 + x)*(1.0 + y)/8.0;

        BasisGrads[3][i0][0] = -(1.0 + y)*(1.0 - z)/8.0;
        BasisGrads[3][i0][1] =  (1.0 - x)*(1.0 - z)/8.0;
        BasisGrads[3][i0][2] = -(1.0 - x)*(1.0 + y)/8.0;

        BasisGrads[4][i0][0] = -(1.0 - y)*(1.0 + z)/8.0;
        BasisGrads[4][i0][1] = -(1.0 - x)*(1.0 + z)/8.0;
        BasisGrads[4][i0][2] =  (1.0 - x)*(1.0 - y)/8.0;

        BasisGrads[5][i0][0] =  (1.0 - y)*(1.0 + z)/8.0;
        BasisGrads[5][i0][1] = -(1.0 + x)*(1.0 + z)/8.0;
        BasisGrads[5][i0][2] =  (1.0 + x)*(1.0 - y)/8.0;

        BasisGrads[6][i0][0] =  (1.0 + y)*(1.0 + z)/8.0;
        BasisGrads[6][i0][1] =  (1.0 + x)*(1.0 + z)/8.0;
        BasisGrads[6][i0][2] =  (1.0 + x)*(1.0 + y)/8.0;

        BasisGrads[7][i0][0] = -(1.0 + y)*(1.0 + z)/8.0;
        BasisGrads[7][i0][1] =  (1.0 - x)*(1.0 + z)/8.0;
        BasisGrads[7][i0][2] =  (1.0 - x)*(1.0 + y)/8.0;

 }

 
  for(int qp = 0; qp < numQPs_; qp++) 
        for(int row = 0; row < numDims_; row++)
           for(int col = 0; col < numDims_; col++) 
            jacobian(i, qp,row,col)=0.0;
 

  for(int qp = 0; qp < numQPs_; qp++) {
        for(int row = 0; row < numDims_; row++){
           for(int col = 0; col < numDims_; col++){
              for(int node = 0; node < dim0; node++){
                jacobian(i, qp, row, col)+= coordVec(i, node, row)*BasisGrads[node][qp][col];
               } // node
            } // col
        } // row
     } // qp

   // Intrepid::setJacobianInv & setJacobianDet
   
  for (int i1=0; i1<numQPs_; i1++) {
    int k, j, rowID = 0, colID = 0;
    int rowperm[3]={0,1,2};
    int colperm[3]={0,1,2};
    const int i0=i;
    MeshScalarT determinant;
    MeshScalarT emax(0);

    for(k=0; k < 3; ++k){
         for(j=0; j < 3; ++j){
              if( std::abs( jacobian(i0,i1,k,j) ) >  emax){
                rowID = k;  colID = j; emax = std::abs( jacobian(i0,i1,k,j) );
              }
          }
        }
     if( rowID ){
        rowperm[0] = rowID;
        rowperm[rowID] = 0;
      }
     if( colID ){
        colperm[0] = colID;
        colperm[colID] = 0;
      }
     MeshScalarT B[3][3], S[2][2], Bi[3][3]; // B=rowperm inMat colperm, S=Schur complement(Boo)
      for(k=0; k < 3; ++k){
        for(j=0; j < 3; ++j){
          B[k][j] = jacobian(i0, i1, rowperm[k],colperm[j]);
         }
      }
     B[1][0] /= B[0][0]; B[2][0] /= B[0][0];// B(:,0)/=pivot
     for(k=0; k < 2; ++k){
       for(j=0; j < 2; ++j){
         S[k][j] = B[k+1][j+1] - B[k+1][0] * B[0][j+1]; // S = B -z*y'
       }
     }
     MeshScalarT detS = S[0][0]*S[1][1]- S[0][1]*S[1][0], Si[2][2];

     Si[0][0] =  S[1][1]/detS;                  Si[0][1] = -S[0][1]/detS;
     Si[1][0] = -S[1][0]/detS;                  Si[1][1] =  S[0][0]/detS;

     for(j=0; j<2;j++)
       Bi[0][j+1] = -( B[0][1]*Si[0][j] + B[0][2]* Si[1][j])/B[0][0];
     for(k=0; k<2;k++)
       Bi[k+1][0] = -(Si[k][0]*B[1][0] + Si[k][1]*B[2][0]);

     Bi[0][0] =  ((MeshScalarT)1/B[0][0])-Bi[0][1]*B[1][0]-Bi[0][2]*B[2][0];
     Bi[1][1] =  Si[0][0];
     Bi[1][2] =  Si[0][1];
     Bi[2][1] =  Si[1][0];
     Bi[2][2] =  Si[1][1];
     for(k=0; k < 3; ++k){
       for(j=0; j < 3; ++j){
         jacobian_inv(i0, i1, k,j) = Bi[colperm[k]][rowperm[j]]; // set inverse
       }
     }
  //determinant:
      determinant = B[0][0] * (S[0][0] * S[1][1] - S[0][1] * S[1][0]);
      if( rowID ) determinant = -determinant;
      if( colID ) determinant = -determinant;
      jacobian_det(i0,i1) = determinant;

  }

  //computeCellMeasure

  if (jacobian_det(i,0) < 0.0) {
    for(int pt = 0; pt < numQPs_; pt++) {
       weighted_measure(i, pt) = -1* refWeights_CUDA(pt)*jacobian_det(i, pt);
     } // P-loop
   }
   else {
   for(int pt = 0; pt < numQPs_; pt++) {
       weighted_measure(i, pt) = refWeights_CUDA(pt)*jacobian_det(i, pt);
     }
   }
   
  //HGRADtransformVALUE
  for(int nodes = 0; nodes < numNodes_; nodes++) {
     for(int pt = 0; pt < numQPs_; pt++) {
        BF(i, nodes, pt) = val_at_cub_points_CUDA(nodes, pt);
     } // pt-loop
   } // nodes-loop

  //compute wBF
  for(int nodes = 0; nodes < numNodes_; nodes++) {
     for(int pt = 0; pt < numQPs_; pt++) {
         wBF(i, nodes, pt) = BF(i, nodes, pt)*weighted_measure(i, pt);
     } // P-loop
   } //

  //compute GradBF

  for(int nodes = 0; nodes < numNodes_; nodes++) {
     for(int pt = 0; pt < numQPs_; pt++) {
       for(int row = 0; row < numDims_; row++){
              GradBF(i, nodes, pt, row) = 0.0;
              for(int col = 0; col < numDims_; col++){
                  GradBF(i, nodes, pt, row) +=jacobian_inv(i, pt, col, row)*grad_at_cub_points_CUDA(nodes, pt, col);
               }// col
            } //row
     } // pt-loop
   } //


  //computewGradBF
  for(int nodes = 0; nodes < numNodes_; nodes++) {
        for(int pt = 0; pt < numQPs_; pt++) {
           for( int dim = 0; dim < numDims_; dim++) {
              wGradBF(i, nodes, pt, dim) = GradBF(i, nodes, pt, dim)*weighted_measure(i, pt);
           } // D1-loop
         } // P-loop
      } 


}


// **********************************************************************
template<typename EvalT, typename Traits>
void ComputeBasisFunctions<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

  // The allocated size of the Field Containers must currently 
    // match the full workset size of the allocated PHX Fields, 
    // this is the size that is used in the computation. There is
    // wasted effort computing on zeroes for the padding on the
    // final workset. Ideally, these are size numCells.
  //int containerSize = workset.numCells;
    //
  Kokkos::parallel_for (GradBF.dimension(0), *this);
//std::cout << "ComputeBasisFunction" <<std::endl;
//std::cout << wGradBF(1, 1, 1, 1) <<"  " <<jacobian_inv(1,1,1,1) <<"   "<<grad_at_cub_points_CUDA(1,1,1) <<"   "<< weighted_measure(1,1)<<std::endl; 

//  Intrepid::CellTools<RealType>::setJacobian(jacobian, refPoints, coordVec, *cellType);
//  Intrepid::CellTools<MeshScalarT>::setJacobianInv(jacobian_inv, jacobian);
//  Intrepid::CellTools<MeshScalarT>::setJacobianDetTemp(jacobian_det, jacobian);
  
//  Intrepid::FunctionSpaceTools::computeCellMeasureTemp(weighted_measure, jacobian_det, refWeights);
//  computeCellMeasure<MeshScalarT>(weighted_measure, jacobian_det, refWeights);
//  Intrepid::FunctionSpaceTools::HGRADtransformVALUETemp<RealType> (BF, val_at_cub_points); 
//  Intrepid::FunctionSpaceTools::multiplyMeasureTemp<MeshScalarT>(wBF, weighted_measure, BF);
   

 // Intrepid::FunctionSpaceTools::HGRADtransformGRADTemp<MeshScalarT>
//    (GradBF, jacobian_inv, grad_at_cub_points);
//  Intrepid::FunctionSpaceTools::multiplyMeasureTemp<MeshScalarT>
//    (wGradBF, weighted_measure, GradBF);

}
#endif // ALBANY_KOKKOS_UNDER_DEVELOPMENT
*/
//**********************************************************************
}
