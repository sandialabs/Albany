//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "PHAL_Utilities.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"
#include "Aeras_ShallowWaterConstants.hpp"

namespace Aeras {

//**********************************************************************
template<typename EvalT, typename Traits>
ComputeBasisFunctions<EvalT, Traits>::
ComputeBasisFunctions(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Aeras::Layouts>& dl) :
                              spatialDimension( p.get<std::size_t>("spatialDim") ),
  coordVec      (p.get<std::string>  ("Coordinate Vector Name"),
      spatialDimension == 3 ? dl->node_3vector : dl->node_vector ),
  cubature      (p.get<Teuchos::RCP <Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > > >("Cubature")),
  intrepidBasis (p.get<Teuchos::RCP<Intrepid2::Basis<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > > > ("Intrepid2 Basis") ),
  cellType      (p.get<Teuchos::RCP <shards::CellTopology> > ("Cell Type")),
  weighted_measure (p.get<std::string>  ("Weights Name"),   dl->qp_scalar ),
  sphere_coord  (p.get<std::string>  ("Spherical Coord Name"), dl->qp_gradient ),
  lambda_nodal  (p.get<std::string>  ("Lambda Coord Nodal Name"), dl->node_scalar), 
  theta_nodal   (p.get<std::string>  ("Theta Coord Nodal Name"), dl->node_scalar), 
  jacobian_inv  (p.get<std::string>  ("Jacobian Inv Name"), dl->qp_tensor ),
  jacobian_det  (p.get<std::string>  ("Jacobian Det Name"), dl->qp_scalar ),
  BF            (p.get<std::string>  ("BF Name"),           dl->node_qp_scalar),
  wBF           (p.get<std::string>  ("Weighted BF Name"),  dl->node_qp_scalar),
  GradBF        (p.get<std::string>  ("Gradient BF Name"),  dl->node_qp_gradient),
  wGradBF       (p.get<std::string>  ("Weighted Gradient BF Name"), dl->node_qp_gradient),
  GradGradBF    (p.get<std::string>  ("Gradient Gradient BF Name"), dl->node_qp_tensor),
  wGradGradBF   (p.get<std::string>  ("Weighted Gradient Gradient BF Name"), dl->node_qp_tensor),
  jacobian  (p.get<std::string>  ("Jacobian Name"), dl->qp_tensor ),
  earthRadius(ShallowWaterConstants::self().earthRadius)
{
  this->addDependentField(coordVec);
  this->addEvaluatedField(weighted_measure);
  this->addEvaluatedField(sphere_coord);
  this->addEvaluatedField(lambda_nodal);
  this->addEvaluatedField(theta_nodal);
  this->addEvaluatedField(jacobian_det);
  this->addEvaluatedField(jacobian_inv);
  this->addEvaluatedField(jacobian);
  this->addEvaluatedField(BF);
  this->addEvaluatedField(wBF);
  this->addEvaluatedField(GradBF);
  this->addEvaluatedField(wGradBF);
  this->addEvaluatedField(GradGradBF);
  this->addEvaluatedField(wGradGradBF);

  
  // Get Dimensions
  std::vector<PHX::DataLayout::size_type> dim;
  dl->node_qp_gradient->dimensions(dim);

  const int containerSize   = dim[0];
  numNodes                  = dim[1];
  numQPs                    = dim[2];
  const int basisDims       =      2;


  // Allocate Temporary FieldContainers
  val_at_cub_points .resize     (numNodes, numQPs);
  grad_at_cub_points.resize     (numNodes, numQPs, basisDims);
  D2_at_cub_points  .resize     (numNodes, numQPs, Intrepid2::getDkCardinality(Intrepid2::OPERATOR_D2, basisDims));
  refPoints         .resize               (numQPs, basisDims);
  refWeights        .resize               (numQPs);

  // Pre-Calculate reference element quantitites
  cubature->getCubature(refPoints, refWeights);
  
  intrepidBasis->getValues(val_at_cub_points,  refPoints, Intrepid2::OPERATOR_VALUE);
  intrepidBasis->getValues(grad_at_cub_points, refPoints, Intrepid2::OPERATOR_GRAD);
  intrepidBasis->getValues(D2_at_cub_points,   refPoints, Intrepid2::OPERATOR_D2);

  this->setName("Aeras::ComputeBasisFunctions"+PHX::typeAsString<EvalT>());
/*
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  refWeights_CUDA=Kokkos::View<RealType*, PHX::Device>("refWeights_CUDA", numQPs);
  val_at_cub_points_CUDA=Kokkos::View<RealType**, PHX::Device>("val_at_cub_points_CUDA", numNodes, numQPs);
  grad_at_cub_points_CUDA=Kokkos::View<RealType***, PHX::Device>("grad_at_cub_points_CUDA", numNodes, numQPs, basisDims);

  for (int i =0; i< numQPs; i++)
    refWeights_CUDA(i)=refWeights(i);

  for (int i =0; i < numNodes; i++){
    for (int j=0; j < numQPs; j++){
      val_at_cub_points_CUDA(i,j)=val_at_cub_points(i,j);
      for (int k=0; k < basisDims; k++)
        grad_at_cub_points_CUDA(i,j,k)=grad_at_cub_points(i,j,k);
    }
  }
#endif*/
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ComputeBasisFunctions<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(coordVec,fm);
  this->utils.setFieldData(weighted_measure,fm);
  this->utils.setFieldData(sphere_coord,fm);
  this->utils.setFieldData(lambda_nodal,fm);
  this->utils.setFieldData(theta_nodal,fm);
  this->utils.setFieldData(jacobian_det,fm);
  this->utils.setFieldData(jacobian_inv,fm);
  this->utils.setFieldData(jacobian,fm);
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(GradBF,fm);
  this->utils.setFieldData(wGradBF,fm);
  this->utils.setFieldData(GradGradBF,fm);
  this->utils.setFieldData(wGradGradBF,fm);
  
}

//**********************************************************************
//Kokkos functors:
/*#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void ComputeBasisFunctions<EvalT, Traits>::
operator() (const ComputeBasisFunctions_Tag& tag, const int& cell) const
{
  for(int nodes = 0; nodes < numNodes; nodes++) 
        for(int pt = 0; pt < numQPs; pt++) 
           for( int dim = 0; dim < numDims; dim++) 
              GradBF(cell, nodes,pt,dim) =0.0;
 
   compute_jacobian(cell);
   compute_jacobian_inv(cell);
   compute_jacobian_det(cell);
   computeCellMeasure(cell);
   compute_BF(cell);
   compute_wBF(cell);
   compute_GradBF(cell);
   compute_wGradBF(cell);
}

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void ComputeBasisFunctions<EvalT, Traits>::
operator() (const ComputeBasisFunctions_basisDim_Tag& tag, const int& cell) const
{

  for(int nodes = 0; nodes < numNodes; nodes++)
        for(int pt = 0; pt < numQPs; pt++)
           for( int dim = 0; dim < numDims; dim++)
              GradBF(cell, nodes,pt,dim) =0.0;

 
   compute_jacobian(cell); 
   compute_jacobian_inv(cell);
   compute_jacobian_det(cell);
   computeCellMeasure(cell);
   compute_BF(cell);
   compute_wBF(cell);
}

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void ComputeBasisFunctions<EvalT, Traits>::
operator() (const ComputeBasisFunctions_no_Jacobian_Tag& tag, const int& cell) const
{

  for(int nodes = 0; nodes < numNodes; nodes++)
        for(int pt = 0; pt < numQPs; pt++)
           for( int dim = 0; dim < numDims; dim++)
              GradBF(cell, nodes,pt,dim) =0.0;


   compute_jacobian_inv(cell);
   compute_jacobian_det(cell);
   computeCellMeasure(cell);
   compute_BF(cell);
   compute_wBF(cell);
   compute_GradBF(cell);
   compute_wGradBF(cell);
}

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void ComputeBasisFunctions<EvalT, Traits>::
operator() (const ComputeBasisFunctions_no_Jacobian_basisDim_Tag& tag, const int& cell) const
{

  for(int nodes = 0; nodes < numNodes; nodes++)
        for(int pt = 0; pt < numQPs; pt++)
           for( int dim = 0; dim < numDims; dim++)
              GradBF(cell, nodes,pt,dim) =0.0;


   compute_jacobian_inv(cell);
   compute_jacobian_det(cell);
   computeCellMeasure(cell);
   compute_BF(cell);
   compute_wBF(cell);
}


template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void ComputeBasisFunctions<EvalT, Traits>::
compute_jacobian (const int e) const
{
  for (int q = 0; q<numQPs;          ++q)
       for (int b1= 0; b1<basisDim;     ++b1)
          for (int b2= 0; b2<basisDim;   ++b2)
            for (int d = 0; d<spatialDim;++d)
              jacobian(e,q,b1,b2) = 0;


      for (int q = 0; q<numQPs;         ++q)
        for (int d = 0; d<spatialDim;   ++d)
          for (int v = 0; v<numNodes;  ++v)
            phi(q,d) += coordVec(e,v,d) * val_at_cub_points_CUDA(v,q);

      for (int v = 0; v<numNodes;      ++v)
        for (int q = 0; q<numQPs;       ++q)
          for (int d = 0; d<spatialDim; ++d)
            for (int b = 0; b<basisDim; ++b)
              dphi(q,d,b) += coordVec(e,v,d) * grad_at_cub_points_CUDA(v,q,b);

      for (int q = 0; q<numQPs;         ++q)
        for (int d = 0; d<spatialDim;   ++d)
          norm(q) += phi(q,d)*phi(q,d);

      for (int q = 0; q<numQPs;         ++q) {
         norm(q) = std::sqrt(norm(q));
      }

      for (int q = 0; q<numQPs;         ++q)
        for (int d = 0; d<spatialDim;   ++d)
          phi(q,d) /= norm(q);

      for (int q = 0; q<numQPs;         ++q) {
      const MeshScalarT latitude  = std::asin(phi(q,2));  //theta

        MeshScalarT longitude = std::atan2(phi(q,1),phi(q,0));  //lambda
        if (std::abs(std::abs(latitude)-pi/2) < DIST_THRESHOLD) longitude = 0;
        else if (longitude < 0) longitude += 2*pi;


        sphere_coord(e,q,0) = longitude;
        sphere_coord(e,q,1) = latitude;

        sinT(q) = std::sin(latitude);
        cosT(q) = std::cos(latitude);
        sinL(q) = std::sin(longitude);
        cosL(q) = std::cos(longitude);
      }

      for (int q = 0; q<numQPs;         ++q) {
        D1(q,0,0) = -sinL(q);
        D1(q,0,1) =  cosL(q);
        D1(q,1,2) =        1;
      }

      for (int q = 0; q<numQPs;         ++q) {
        D2(q,0,0) =  sinL(q)*sinL(q)*cosT(q)*cosT(q) + sinT(q)*sinT(q);
        D2(q,0,1) = -sinL(q)*cosL(q)*cosT(q)*cosT(q);
        D2(q,0,2) = -cosL(q)*sinT(q)*cosT(q);

        D2(q,1,0) = -sinL(q)*cosL(q)*cosT(q)*cosT(q);
        D2(q,1,1) =  cosL(q)*cosL(q)*cosT(q)*cosT(q) + sinT(q)*sinT(q);
        D2(q,1,2) = -sinL(q)*sinT(q)*cosT(q);

        D2(q,2,0) = -cosL(q)*sinT(q);
        D2(q,2,1) = -sinL(q)*sinT(q);
        D2(q,2,2) =  cosT(q);
      }

      for (int q = 0; q<numQPs;          ++q)
        for (int b = 0; b<basisDim;      ++b)
          for (int d = 0; d<spatialDim;  ++d)
            for (int j = 0; j<spatialDim;++j)
              D3(q,b,d) += D1(q,b,j)*D2(q,j,d);

      for (int q = 0; q<numQPs;          ++q)
        for (int b1= 0; b1<basisDim;     ++b1)
          for (int b2= 0; b2<basisDim;   ++b2)
            for (int d = 0; d<spatialDim;++d)
              jacobian(e,q,b1,b2) += D3(q,b1,d) *  dphi(q,d,b2);

      for (int q = 0; q<numQPs;          ++q)
        for (int b1= 0; b1<basisDim;     ++b1)
          for (int b2= 0; b2<basisDim;   ++b2)
            jacobian(e,q,b1,b2) *= earthRadius/norm(q);


}

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void ComputeBasisFunctions<EvalT, Traits>:: 
compute_jacobian_inv (const int i) const
 {
   MeshScalarT determinant;
   for (int i1=0; i1<numQPs; i1++) {
   determinant = jacobian(i, i1, 0, 0)*jacobian(i, i1, 1, 1)-jacobian(i, i1, 0, 1)*jacobian(i, i1, 1, 0);
    jacobian_inv(i, i1, 0, 0) =jacobian (i,i1, 1, 1)/ determinant;
    jacobian_inv(i, i1, 0, 1) =jacobian (i,i1, 0, 1)/ determinant;
    jacobian_inv(i, i1, 1, 0) =jacobian (i,i1, 1, 0)/ determinant;
    jacobian_inv(i, i1, 1, 1) =jacobian (i,i1, 0, 0)/ determinant;
    }
 }

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void ComputeBasisFunctions<EvalT, Traits>::
compute_jacobian_det(const int i) const
{
   for (int i1=0; i1<numQPs; i1++) {
    jacobian_det(i, i1)=jacobian(i, i1, 0, 0)*jacobian(i, i1, 1, 1)-jacobian(i, i1, 0, 1)*jacobian(i, i1, 1, 0);
   }
}

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void ComputeBasisFunctions<EvalT, Traits>::
computeCellMeasure(const int i) const 
{
   if (jacobian_det(i,0) < 0.0) {
    for(int pt = 0; pt < numQPs; pt++) {
       weighted_measure(i, pt) = -1* refWeights_CUDA(pt)*jacobian_det(i, pt);
     } // P-loop
   }
   else {
   for(int pt = 0; pt < numQPs; pt++) {
       weighted_measure(i, pt) = refWeights_CUDA(pt)*jacobian_det(i, pt);
     }
   }
}

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void ComputeBasisFunctions<EvalT, Traits>::
compute_BF(const int i) const 
{
  for(int nodes = 0; nodes < numNodes; nodes++) {
     for(int pt = 0; pt < numQPs; pt++) {
        BF(i, nodes, pt) = val_at_cub_points_CUDA(nodes, pt);
     } // pt-loop
   } // nodes-loop
}

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void ComputeBasisFunctions<EvalT, Traits>::
compute_wBF(const int i) const 
{
  for(int nodes = 0; nodes < numNodes; nodes++) {
     for(int pt = 0; pt < numQPs; pt++) {
         wBF(i, nodes, pt) = BF(i, nodes, pt)*weighted_measure(i, pt);
     } // P-loop
   } //
}

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void ComputeBasisFunctions<EvalT, Traits>::
compute_GradBF(const int i) const
{
  for(int nodes = 0; nodes < numNodes; nodes++) {
     for(int pt = 0; pt < numQPs; pt++) {
       for(int row = 0; row < numDims; row++){
              GradBF(i, nodes, pt, row) = 0.0;
              for(int col = 0; col < numDims; col++){
                  GradBF(i, nodes, pt, row) +=jacobian_inv(i, pt, col, row)*grad_at_cub_points_CUDA(nodes, pt, col);
               }// col
            } //row
     } // P-loop
   } //
}

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void ComputeBasisFunctions<EvalT, Traits>::
compute_wGradBF(const int i) const
{
  for(int nodes = 0; nodes < numNodes; nodes++) {
        for(int pt = 0; pt < numQPs; pt++) {
           for( int dim = 0; dim < numDims; dim++) {
              wGradBF(i, nodes, pt, dim) = GradBF(i, nodes, pt, dim)*weighted_measure(i, pt);
           } // D1-loop
         } // P-loop
      } // F-loop
}

#endif
*/
//**********************************************************************
template<typename EvalT, typename Traits>
void ComputeBasisFunctions<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  if (memoizer_.haveStoredData(workset)) return;

  /** The allocated size of the Field Containers must currently 
    * match the full workset size of the allocated PHX Fields, 
    * this is the size that is used in the computation. There is
    * wasted effort computing on zeroes for the padding on the
    * final workset. Ideally, these are size numCells.
  //int containerSize = workset.numCells;
    */


  // setJacobian only needs to be RealType since the data type is only
  //  used internally for Basis Fns on reference elements, which are
  //  not functions of coordinates. This save 18min of compile time!!!
//#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT

  const double pi = ShallowWaterConstants::self().pi;
  const double DIST_THRESHOLD = ShallowWaterConstants::self().distanceThreshold;

  const int numelements = coordVec.dimension(0);
  const int spatialDim  = coordVec.dimension(2);
  const int basisDim    =                    2;


  if (spatialDim==basisDim) {
    //Check that we don't have a higher order spectral element.  The node_count is based on 
    //2D quad/shellquad elements.  This logic will only get hit if spatialDim = 2.
    //Only a quad or shellquad element can be enriched according to the logic in Aeras::SpectralDiscretization.
    TEUCHOS_TEST_FOR_EXCEPTION(cellType->getNodeCount() > 9, 
                               Teuchos::Exceptions::InvalidParameter,
                               std::endl << "Error!  Intrepid2::CellTools<RealType>::setJacobian " <<
                               "is only implemented for bilinear and biquadratic elements!  Attempting " <<
                               "to call this function for a higher order element. \n"); 
    Intrepid2::CellTools<RealType>::setJacobian(jacobian, refPoints, coordVec, *cellType);
  } else {
    Intrepid2::FieldContainer<MeshScalarT>  phi(numQPs,spatialDim);
    Intrepid2::FieldContainer<MeshScalarT> dphi(numQPs,spatialDim,basisDim);
    Intrepid2::FieldContainer<MeshScalarT> norm(numQPs);
    Intrepid2::FieldContainer<MeshScalarT> sinL(numQPs);
    Intrepid2::FieldContainer<MeshScalarT> cosL(numQPs);
    Intrepid2::FieldContainer<MeshScalarT> sinT(numQPs);
    Intrepid2::FieldContainer<MeshScalarT> cosT(numQPs);
    Intrepid2::FieldContainer<MeshScalarT>   D1(numQPs,basisDim,spatialDim);
    Intrepid2::FieldContainer<MeshScalarT>   D2(numQPs,spatialDim,spatialDim);
    Intrepid2::FieldContainer<MeshScalarT>   D3(numQPs,basisDim,spatialDim);
    
    for (int e = 0; e<numelements;      ++e) {
      for (int v = 0; v<numNodes;      ++v) {
          //  phi(q,d) += coordVec(e,v,d) * val_at_cub_points(v,q);
          //const MeshScalarT latitude  = std::asin(phi(q,2));  //theta
          const MeshScalarT latitude  = std::asin(coordVec(e,v,2));  //theta

          //MeshScalarT longitude = std::atan2(phi(q,1),phi(q,0));  //lambda
          MeshScalarT longitude = std::atan2(coordVec(e,v,1),coordVec(e,v,0));  //lambda
          if (std::abs(std::abs(latitude)-pi/2) < DIST_THRESHOLD) longitude = 0;
          else if (longitude < 0) longitude += 2*pi;

          lambda_nodal(e,v) = longitude;
          theta_nodal(e,v) = latitude;

      }
    }
    
    for (int e = 0; e<numelements;      ++e) {
      phi.initialize(); 
      dphi.initialize(); 
      norm.initialize(); 
      sinL.initialize(); 
      cosL.initialize(); 
      sinT.initialize(); 
      cosT.initialize(); 
      D1.initialize(); 
      D2.initialize(); 
      D3.initialize();

      for (int q = 0; q<numQPs;          ++q)
        for (int b1= 0; b1<basisDim;     ++b1)
          for (int b2= 0; b2<basisDim;   ++b2)
            for (int d = 0; d<spatialDim;++d)
              jacobian(e,q,b1,b2) = 0;
      

      for (int q = 0; q<numQPs;         ++q) 
        for (int d = 0; d<spatialDim;   ++d) 
          for (int v = 0; v<numNodes;  ++v)
            phi(q,d) += coordVec(e,v,d) * val_at_cub_points(v,q);

      for (int v = 0; v<numNodes;      ++v)
        for (int q = 0; q<numQPs;       ++q) 
          for (int d = 0; d<spatialDim; ++d) 
            for (int b = 0; b<basisDim; ++b) 
              dphi(q,d,b) += coordVec(e,v,d) * grad_at_cub_points(v,q,b);
  
      for (int q = 0; q<numQPs;         ++q) 
        for (int d = 0; d<spatialDim;   ++d) 
          norm(q) += phi(q,d)*phi(q,d);

      for (int q = 0; q<numQPs;         ++q) {
         norm(q) = std::sqrt(norm(q));
      }

      for (int q = 0; q<numQPs;         ++q) 
        for (int d = 0; d<spatialDim;   ++d) 
          phi(q,d) /= norm(q);
     
      for (int q = 0; q<numQPs;         ++q) {

        // ==========================================================
        // enforce three facts:
        //
        // 1) lon at poles is defined to be zero
        //
        // 2) Grid points must be separated by about .01 Meter (on earth)
        //   from pole to be considered "not the pole".
        //
        // 3) range of lon is { 0<= lon < 2*PI }
        //
        // ==========================================================

        const MeshScalarT latitude  = std::asin(phi(q,2));  //theta

        MeshScalarT longitude = std::atan2(phi(q,1),phi(q,0));  //lambda
        if (std::abs(std::abs(latitude)-pi/2) < DIST_THRESHOLD) longitude = 0;
        else if (longitude < 0) longitude += 2*pi;


        sphere_coord(e,q,0) = longitude;
        sphere_coord(e,q,1) = latitude;

        sinT(q) = std::sin(latitude);
        cosT(q) = std::cos(latitude);
        sinL(q) = std::sin(longitude);
        cosL(q) = std::cos(longitude);
      }

      for (int q = 0; q<numQPs;         ++q) {
        D1(q,0,0) = -sinL(q);
        D1(q,0,1) =  cosL(q);
        D1(q,1,2) =        1;
      }

      for (int q = 0; q<numQPs;         ++q) {
        D2(q,0,0) =  sinL(q)*sinL(q)*cosT(q)*cosT(q) + sinT(q)*sinT(q);
        D2(q,0,1) = -sinL(q)*cosL(q)*cosT(q)*cosT(q);
        D2(q,0,2) = -cosL(q)*sinT(q)*cosT(q);

        D2(q,1,0) = -sinL(q)*cosL(q)*cosT(q)*cosT(q); 
        D2(q,1,1) =  cosL(q)*cosL(q)*cosT(q)*cosT(q) + sinT(q)*sinT(q); 
        D2(q,1,2) = -sinL(q)*sinT(q)*cosT(q);

        D2(q,2,0) = -cosL(q)*sinT(q);   
        D2(q,2,1) = -sinL(q)*sinT(q);   
        D2(q,2,2) =  cosT(q);
      }

      for (int q = 0; q<numQPs;          ++q) 
        for (int b = 0; b<basisDim;      ++b) 
          for (int d = 0; d<spatialDim;  ++d) 
            for (int j = 0; j<spatialDim;++j) 
              D3(q,b,d) += D1(q,b,j)*D2(q,j,d);

      for (int q = 0; q<numQPs;          ++q) 
        for (int b1= 0; b1<basisDim;     ++b1) 
          for (int b2= 0; b2<basisDim;   ++b2) 
            for (int d = 0; d<spatialDim;++d) 
              jacobian(e,q,b1,b2) += D3(q,b1,d) *  dphi(q,d,b2);

      for (int q = 0; q<numQPs;          ++q) 
        for (int b1= 0; b1<basisDim;     ++b1) 
          for (int b2= 0; b2<basisDim;   ++b2)
            jacobian(e,q,b1,b2) *= earthRadius/norm(q);

    }
    
  }//end else
  
  
  
  /////////////implementing the map and Jacobian exactly like homme's,
  /////////////no generality.
  if(0){
  
  Intrepid2::FieldContainer<MeshScalarT>   Q(4);
  Intrepid2::FieldContainer<MeshScalarT>   C(3,4);
  Intrepid2::FieldContainer<MeshScalarT>   xx(3);
  Intrepid2::FieldContainer<MeshScalarT>   CartC(3);
  
  Intrepid2::FieldContainer<MeshScalarT>   dd(4,2);
  Intrepid2::FieldContainer<MeshScalarT>   D1(2,3);
  Intrepid2::FieldContainer<MeshScalarT>   D2(3,3);
  Intrepid2::FieldContainer<MeshScalarT>   D3(3,2);
  Intrepid2::FieldContainer<MeshScalarT>   D4(3,2);
  
  for (int e = 0; e<numelements; ++e) {
    
    for (int q = 0; q<numQPs;          ++q)
      for (int b1= 0; b1<basisDim;     ++b1)
        for (int b2= 0; b2<basisDim;   ++b2){
          jacobian(e,q,b1,b2) = 0.;
        }
    
    for (int q = 0; q<numQPs; ++q){
      
      //reference coords are refPoints(q,0) and refPoints(q,1) because BasisDim = 2
      //in this case
      
      MeshScalarT a = refPoints(q,0);
      MeshScalarT b = refPoints(q,1);
      
      Q.initialize();
      C.initialize();
      xx.initialize();
      CartC.initialize();
      
      Q(0) = (1-a)*(1-b)/4.; Q(1) = (1+a)*(1-b)/4.;
      Q(2) = (1+a)*(1+b)/4.; Q(3) = (1-a)*(1+b)/4.;
      
      
      //corner 1,2,3,4 = Vertex 0,1,2,3
      //can be pulled out from q loop
      for (int v = 0; v<4; v++) {
        for (int d = 0; d<3; d++)
          C(d,v) = coordVec(e,v,d);
      }
      
      
      for (int i = 0; i<3; i++)
        for(int j = 0; j<4; j++)
          xx(i) += C(i,j)*Q(j);
      
      MeshScalarT rr = std::sqrt( xx(0)*xx(0) + xx(1)*xx(1) + xx(2)*xx(2) );
      
      for (int i = 0; i<3; i++)
        CartC(i) = xx(i)/rr;
      
      const MeshScalarT latitude  = std::asin(CartC(2));  //theta
      MeshScalarT longitude = std::atan2(CartC(1),CartC(0));  //lambda
      if (std::abs(std::abs(latitude)-pi/2) < DIST_THRESHOLD) longitude = 0;
      else if (longitude < 0) longitude += 2*pi;
      
      
      //where is this used?
      sphere_coord(e,q,0) = longitude;
      sphere_coord(e,q,1) = latitude;
      
      MeshScalarT sinT, cosT, sinL, cosL;
      
      sinT = std::sin(latitude);
      cosT = std::cos(latitude);
      sinL = std::sin(longitude);
      cosL = std::cos(longitude);
      
      dd.initialize();
      D1.initialize();
      D2.initialize();
      D3.initialize();
      D4.initialize();
      
      D1(0,0) = -sinL;
      D1(0,1) =  cosL;
      D1(1,2) =  1.;
      
      
      D2(0,0) =  sinL*sinL*cosT*cosT + sinT*sinT;
      D2(0,1) = -sinL*cosL*cosT*cosT;
      D2(0,2) = -cosL*sinT*cosT;
      
      D2(1,0) = -sinL*cosL*cosT*cosT;
      D2(1,1) =  cosL*cosL*cosT*cosT + sinT*sinT;
      D2(1,2) = -sinL*sinT*cosT;
      
      D2(2,0) = -cosL*sinT;
      D2(2,1) = -sinL*sinT;
      D2(2,2) =  cosT;
      
      dd(0,0)=(-1+b)/4.; dd(0,1)=(-1+a)/4.;
      dd(1,0)=(1-b)/4.;  dd(1,1)=(-1-a)/4.;
      dd(2,0)=(1+b)/4.;  dd(2,1)=(1+a)/4.;
      dd(3,0)=(-1-b)/4.; dd(3,1)=(1-a)/4.;
      
      
      for (int i=0; i<3; i++)
        for (int j=0; j<2; j++)
          for (int k=0; k<4; k++)
            D3(i,j) += C(i,k)*dd(k,j);
      
      
      for (int i=0; i<3; i++)
        for (int j=0; j<2; j++)
          for (int k=0; k<3; k++)
            D4(i,j) += D2(i,k)*D3(k,j);
      
      for (int i=0; i<2; i++)
        for (int j=0; j<2; j++)
          for (int k=0; k<3; k++)
            jacobian(e,q,i,j) += D1(i,k)*D4(k,j)/rr*earthRadius;
      
    }//end q loop for quad points
    
  }//end e loop for elements
  }//end of if-statement which turns on/off homme;s map
  //////////////////////////////////////////////////////////////////////




  Intrepid2::CellTools<MeshScalarT>::setJacobianInv(jacobian_inv, jacobian);
  Intrepid2::CellTools<MeshScalarT>::setJacobianDet(jacobian_det, jacobian);

  for (int e = 0; e<numelements;      ++e) {
    for (int q = 0; q<numQPs;          ++q) {
      TEUCHOS_TEST_FOR_EXCEPTION(std::abs(jacobian_det(e,q))<.1e-8,
                  std::logic_error,"Bad Jacobian Found.");
    }
  }

  Intrepid2::FunctionSpaceTools::computeCellMeasure<MeshScalarT>
    (weighted_measure, jacobian_det, refWeights);

  Intrepid2::FunctionSpaceTools::HGRADtransformVALUE<RealType>
    (BF, val_at_cub_points);
  Intrepid2::FunctionSpaceTools::multiplyMeasure<MeshScalarT>
    (wBF, weighted_measure, BF);
  Intrepid2::FunctionSpaceTools::HGRADtransformGRAD<MeshScalarT>
    (GradBF, jacobian_inv, grad_at_cub_points);
  Intrepid2::FunctionSpaceTools::multiplyMeasure<MeshScalarT>
    (wGradBF, weighted_measure, GradBF);

  PHAL::set(GradGradBF, 0.0);
  if (spatialDim!=basisDim)
    for (int e=0; e<numelements; ++e)
      for (int v=0; v<numNodes; ++v)
        for (int q=0; q<numQPs; ++q)
          for (int i=0; i<basisDim; i++)
            for (int j=0; j<basisDim; j++)
              for (int k=0; k<basisDim; k++)
                GradGradBF(e,v,q,i,j) += jacobian_inv(e,q,i,k)*D2_at_cub_points(v,q,i+j);

  if (spatialDim!=basisDim)
    for (int e=0; e<numelements; ++e)
      for (int v=0; v<numNodes; ++v)
        for (int q=0; q<numQPs; ++q)
          for (int i=0; i<basisDim; i++)
            for (int j=0; j<basisDim; j++)
              wGradGradBF(e,v,q,i,j) = weighted_measure(e,q) * GradGradBF(e,v,q,i,j);



/*#else // ALBANY_KOKKOS_UNDER_DEVELOPMENT

 pi = ShallowWaterConstants::self().pi;
 DIST_THRESHOLD = ShallowWaterConstants::self().distanceThreshold;

 numelements = coordVec.dimension(0);
 spatialDim  = coordVec.dimension(2);
 basisDim    =                    2;


 phi  =  Kokkos::View<MeshScalarT**,  PHX::Device> ("phi", numQPs,spatialDim);
 dphi =  Kokkos::View<MeshScalarT***, PHX::Device> ("dphi", numQPs,spatialDim, basisDim);
 norm =  Kokkos::View<MeshScalarT*,   PHX::Device> ("norm", numQPs);
 sinL =  Kokkos::View<MeshScalarT*,   PHX::Device> ("sinL", numQPs);
 cosL =  Kokkos::View<MeshScalarT*,   PHX::Device> ("cosL", numQPs);
 sinT =  Kokkos::View<MeshScalarT*,   PHX::Device> ("sinT", numQPs);
 cosT =  Kokkos::View<MeshScalarT*,   PHX::Device> ("cosT", numQPs);
 D1   =  Kokkos::View<MeshScalarT***, PHX::Device> ("D1", numQPs,basisDim,spatialDim);
 D2   =  Kokkos::View<MeshScalarT***, PHX::Device> ("D2", numQPs,spatialDim,spatialDim);
 D3   =  Kokkos::View<MeshScalarT***, PHX::Device> ("D3", numQPs,basisDim,spatialDim);

// PHAL::set(GradGradBF, 0.0);

 if (spatialDim==basisDim) {
 //Check that we don't have a higher order spectral element.  The node_count is based on 
 //2D quad/shellquad elements.  This logic will only get hit if spatialDim = 2.
 //Only a quad or shellquad element can be enriched according to the logic in Aeras::SpectralDiscretization
   TEUCHOS_TEST_FOR_EXCEPTION(cellType->getNodeCount() > 9,
                       Teuchos::Exceptions::InvalidParameter,
                       std::endl << "Error!  Intrepid2::CellTools<RealType>::setJacobian " <<
                       "is only implemented for bilinear and biquadratic elements!  Attempting " <<
                  "to call this function for a higher order element. \n");
  Intrepid2::CellTools<RealType>::setJacobian(jacobian, refPoints, coordVec, *cellType);
   if (spatialDim!=basisDim)
      Kokkos::parallel_for(ComputeBasisFunctions_no_Jacobian_Policy(0,workset.numCells),*this);
   else
      Kokkos::parallel_for(ComputeBasisFunctions_no_Jacobian_basisDim_Policy(0,workset.numCells),*this);
 }
 else {
     if (spatialDim!=basisDim)
        Kokkos::parallel_for(ComputeBasisFunctions_Policy(0,workset.numCells),*this);
     else
        Kokkos::parallel_for(ComputeBasisFunctions_basisDim_Policy(0,workset.numCells),*this);
 }

 for (int e = 0; e<numelements;      ++e) {
    for (int q = 0; q<numQPs;          ++q) {
      TEUCHOS_TEST_FOR_EXCEPTION(std::abs(jacobian_det(e,q))<.1e-8,
                  std::logic_error,"Bad Jacobian Found.");
    }
  }

   

#endif // ALBANY_KOKKOS_UNDER_DEVELOPMENT
*/
  //div_check(spatialDim, numelements);
}



template<typename EvalT, typename Traits>
void ComputeBasisFunctions<EvalT, Traits>::
initialize_grad(Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device > &grad_at_quadrature_points) const
{
  const unsigned N = static_cast<unsigned>(std::floor(std::sqrt(numQPs)+.1));
  Intrepid2::FieldContainer<RealType> dLdx(N,N); dLdx.initialize(); 

  for (unsigned m=0; m<N; ++m) {
    for (unsigned n=0; n<N; ++n) {
      for (unsigned i=m?0:1; i<N; (++i==m)?++i:i) {
        double prod = 1;
        for (unsigned j=0; j<N; ++j) {
          if (j!=m && j!=i) prod *= (refPoints(n,0)-refPoints(j,0)) / 
                                    (refPoints(m,0)-refPoints(j,0));
        }
        dLdx(m,n) += prod/(refPoints(m,0)-refPoints(i,0));
      }
    }
  }

  for (unsigned m=0; m<numQPs; ++m) {
    for (unsigned n=0; n<numQPs; ++n) {
      if (m/N == n/N) grad_at_quadrature_points(m,n,0) = dLdx(m%N,n%N);
      if (m%N == n%N) grad_at_quadrature_points(m,n,1) = dLdx(m/N,n/N);
    }
  }
}


template<typename EvalT, typename Traits>
void ComputeBasisFunctions<EvalT, Traits>::
spherical_divergence (Intrepid2::FieldContainer<MeshScalarT> &div_v,
                      const Intrepid2::FieldContainer<MeshScalarT> &v_lambda_theta,
                      const int e,
                      const double rrearth) const
{
  Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> grad_at_quadrature_points(numQPs,numQPs,2);
  static bool init_grad = true;
  if (init_grad) initialize_grad(grad_at_quadrature_points);
  init_grad = false;

  std::vector<MeshScalarT> jac_weighted_contravarient_x(numQPs);
  std::vector<MeshScalarT> jac_weighted_contravarient_y(numQPs);
  for (int q = 0; q<numQPs;         ++q) {
    // push to reference space
    const MeshScalarT contravarient_x = jacobian_inv(e,q,0,0)*v_lambda_theta(q,0) + jacobian_inv(e,q,0,1)*v_lambda_theta(q,1);
    const MeshScalarT contravarient_y = jacobian_inv(e,q,1,0)*v_lambda_theta(q,0) + jacobian_inv(e,q,1,1)*v_lambda_theta(q,1);
    jac_weighted_contravarient_x[q] = jacobian_det(e,q) * contravarient_x;
    jac_weighted_contravarient_y[q] = jacobian_det(e,q) * contravarient_y;
  }

  MeshScalarT dv_xi_dxi;
  MeshScalarT dv_eta_deta;

  for (int q = 0; q<numQPs;          ++q) {
    for (int v = 0; v<numQPs;  ++v) {
      dv_xi_dxi   += grad_at_quadrature_points(v,q,0)*jac_weighted_contravarient_x[v];  
      dv_eta_deta += grad_at_quadrature_points(v,q,1)*jac_weighted_contravarient_y[v];  
    }

    const MeshScalarT rjacobian_det = 1/jacobian_det(e,q);
    const MeshScalarT div_unit = (dv_xi_dxi + dv_eta_deta)*rjacobian_det;
    div_v(q)    = div_unit*rrearth;
  }
}


template<typename EvalT, typename Traits>
void ComputeBasisFunctions<EvalT, Traits>::
div_check(const int spatialDim, const int numelements) const
{
  
  const static double  rearth = 6.376e06;
  const static double rrearth = 1; // 1/rearth;
   
  for (int c = 0; c<2; ++c) {
    if (!c && numQPs!=16) continue;
    for (int e = 0; e<numelements;      ++e) {
      static const MeshScalarT DIST_THRESHOLD = 1.0e-6;

      Intrepid2::FieldContainer<MeshScalarT>  phi(numQPs,spatialDim);
      phi.initialize(); 
      for (int q = 0; q<numQPs;         ++q) 
        for (int d = 0; d<spatialDim;   ++d) 
          for (int v = 0; v<numNodes;  ++v)
            phi(q,d) += earthRadius*coordVec(e,v,d) * val_at_cub_points(v,q);

      std::vector<MeshScalarT> divergence_v(numQPs);
      Intrepid2::FieldContainer<MeshScalarT> v_lambda_theta(numQPs,2);
      switch (c) {
        case 0: {
          //  Example copied from homme, the climate code, from function divergence_sphere()
          //  in derivative_mod.F90. The solution is given as a fixed input/output.
          //  Note: This requires that the mesh read in be a projected cube-to-sphere with 600
          //  total elements, 10 along each cube edge, and the first element be a certain one.

          const static double v[16] = 
                   {7473.7470865518399, 7665.3190370095717, 7918.441032072672,  8041.3357610390449,
                    7665.3190370095717, 7854.2794792071199, 8093.0041509498678, 8201.3952993149032,
                    7918.4410320726756, 8093.0041509498706, 8289.0810252631491, 8360.5202251186929,
                    8041.3357610390449, 8201.3952993149032, 8360.5202251186929, 8401.7866189014148} ;
          const static double s[16] = 
                   {0.000000000000000, -5.2458958075096795, -3.672098804806848, -12.19657502178409,
                    5.245895807589972,  0.0000000000000000,  5.468563750710966,  -2.68353065400852,
                    3.672098804671908, -5.4685637506901852,  0.000000000000000, -12.88362080627209,
                   12.196575021808101,  2.6835306540696267, 12.883620806137678,   0.00000000000000};

          const static double d[16] = 
                  {234.2042288813096, 226.01008098131695, 215.8066813838328, 211.20330567403138, 
                   226.0100809813168, 218.35802236416407, 208.9244466731406, 204.74540616901601, 
                   215.8066813838327, 208.92444667314072, 200.6069171296133, 197.05760538656338, 
                   211.2033056740313, 204.74540616901612, 197.0576053865633, 193.87282576804154};

          
          bool null_test = false;
          for (int q = 0; q<numQPs; ++q) if (DIST_THRESHOLD<std::abs(jacobian_det(e,q)-d[q])) null_test=true;
          if (null_test) {
            for (int q = 0; q<numQPs; ++q) {
              v_lambda_theta(q,0) = 0; 
              divergence_v[q]     = 0;
              v_lambda_theta(q,1) = 0;
            }
          } else {
            for (int q = 0; q<numQPs; ++q) {
              v_lambda_theta(q,0) = v[q]; 
              divergence_v[q]     = s[q];
              v_lambda_theta(q,1) = 0;
            }
          }
        }
        break;
        case 1: {
          //  Spherical Divergence:  (d/d_theta) (sin(theta) v_theta) + d/d_lambda v_lambda
          //
          //     First case v_theta = 1/sin(theta)    
          //               v_lambda = 1
          //                div v = 0
          for (int q = 0; q<numQPs; ++q) {
            v_lambda_theta(q,0) = 
              (val_at_cub_points(0,q)*jacobian(e,q,0,0)+val_at_cub_points(1,q)*jacobian(e,q,0,1))/jacobian_det(e,q);
            v_lambda_theta(q,1) = 
              (val_at_cub_points(2,q)*jacobian(e,q,1,0)+val_at_cub_points(3,q)*jacobian(e,q,1,1))/jacobian_det(e,q);
            divergence_v[q] = 0;
          }
        }
        break;
        case 2: {
          //
          //     Second case v_theta = lambda * cos(theta)
          //                v_lambda = -lambda^2 * (cos^2(theta) - sin^2(theta))/2
          //                div v = 0
          for (int q = 0; q<numQPs; ++q) {
            const MeshScalarT R = std::sqrt(phi(q,0)*phi(q,0)+phi(q,1)*phi(q,1)+phi(q,2)*phi(q,2));
            const MeshScalarT X = phi(q,0)/R;
            const MeshScalarT Y = phi(q,1)/R;
            const MeshScalarT Z = phi(q,2)/R;
            const MeshScalarT cos_theta = Z;
            const MeshScalarT sin_theta = 1-Z*Z;
            const MeshScalarT     theta = std::acos (Z);
            const MeshScalarT    lambda = std::atan2(Y,X);
            v_lambda_theta(q,0) = -lambda*lambda*(cos_theta*cos_theta - sin_theta*sin_theta)/2;
            v_lambda_theta(q,1) = lambda * cos_theta;                                               
            divergence_v[q] = 0;
          }
        }
        break;
        case 3: {
          //
          //     third  case v_theta = theta^2                 
          //                v_lambda = lambda^2 * cos(theta)
          //                div v    = 2*theta*sin(theta) + theta^2*cos(theta) + 2*lambda*cos(theta)
          for (int q = 0; q<numQPs; ++q) {
            const MeshScalarT R = std::sqrt(phi(q,0)*phi(q,0)+phi(q,1)*phi(q,1)+phi(q,2)*phi(q,2));
            const MeshScalarT X = phi(q,0)/R;
            const MeshScalarT Y = phi(q,1)/R;
            const MeshScalarT Z = phi(q,2)/R;
            const MeshScalarT cos_theta = Z;
            const MeshScalarT sin_theta = 1-Z*Z;
            const MeshScalarT     theta = std::acos (Z);
            const MeshScalarT    lambda = std::atan2(Y,X);
            v_lambda_theta(q,0) = lambda*lambda*cos_theta;
            v_lambda_theta(q,1) = theta * theta;                                                    
            divergence_v[q] = 2*theta*sin_theta + theta*theta*cos_theta + 2*lambda*cos_theta;
          }
        }
        break;
      }
      

      Intrepid2::FieldContainer<MeshScalarT> div_v(numQPs);
      spherical_divergence(div_v, v_lambda_theta, e, rrearth);
      for (int q = 0; q<numQPs;          ++q) {
        if (DIST_THRESHOLD<std::abs(div_v(q)/rrearth - divergence_v[q])) 
          std::cout <<"Aeras_ComputeBasisFunctions_Def"<<":"<<__LINE__
                    <<" case, element, quadpoint:("<<c<<","<<e<<","<<q
                    <<") divergence_v, div_v:("<<divergence_v[q] <<","<<div_v(q)/rrearth
                    <<")" <<std::endl;
      }
      if (!c) e = numelements;
    }
  }
}

}
