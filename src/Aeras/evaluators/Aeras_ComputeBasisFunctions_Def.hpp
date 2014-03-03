//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

namespace Aeras {

//**********************************************************************
template<typename EvalT, typename Traits>
ComputeBasisFunctions<EvalT, Traits>::
ComputeBasisFunctions(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
  coordVec      (p.get<std::string>  ("Coordinate Vector Name"), dl->node_3vector ),
  cubature      (p.get<Teuchos::RCP <Intrepid::Cubature<RealType> > >("Cubature")),
  intrepidBasis (p.get<Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > > ("Intrepid Basis") ),
  cellType      (p.get<Teuchos::RCP <shards::CellTopology> > ("Cell Type")),
  weighted_measure (p.get<std::string>  ("Weights Name"),   dl->qp_scalar ),
  sphere_coord  (p.get<std::string>  ("Spherical Coord Name"), dl->qp_gradient ),
  jacobian_inv  (p.get<std::string>  ("Jacobian Inv Name"), dl->qp_tensor ),
  jacobian_det  (p.get<std::string>  ("Jacobian Det Name"), dl->qp_scalar ),
  BF            (p.get<std::string>  ("BF Name"),           dl->node_qp_scalar),
  wBF           (p.get<std::string>  ("Weighted BF Name"),  dl->node_qp_scalar),
  GradBF        (p.get<std::string>  ("Gradient BF Name"),  dl->node_qp_gradient),
  wGradBF       (p.get<std::string>  ("Weighted Gradient BF Name"), dl->node_qp_gradient),
  jacobian  (p.get<std::string>  ("Jacobian Name"), dl->qp_tensor )
{
  this->addDependentField(coordVec);
  this->addEvaluatedField(weighted_measure);
  this->addEvaluatedField(sphere_coord);
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

  const int containerSize   = dim[0];
  numNodes                  = dim[1];
  numQPs                    = dim[2];
  const int basisDims       =      2;


  // Allocate Temporary FieldContainers
  val_at_cub_points .resize     (numNodes, numQPs);
  grad_at_cub_points.resize     (numNodes, numQPs, basisDims);
  refPoints         .resize               (numQPs, basisDims);
  refWeights        .resize               (numQPs);

  // Pre-Calculate reference element quantitites
  cubature->getCubature(refPoints, refWeights);
  intrepidBasis->getValues(val_at_cub_points,  refPoints, Intrepid::OPERATOR_VALUE);
  intrepidBasis->getValues(grad_at_cub_points, refPoints, Intrepid::OPERATOR_GRAD);

  this->setName("Aeras::ComputeBasisFunctions"+PHX::TypeString<EvalT>::value);
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
  this->utils.setFieldData(jacobian_det,fm);
  this->utils.setFieldData(jacobian_inv,fm);
  this->utils.setFieldData(jacobian,fm);
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
  
  const int numelements = coordVec.dimension(0);
  const int spatialDim  = coordVec.dimension(2);
  const int basisDim    =                    2;

  // setJacobian only needs to be RealType since the data type is only
  //  used internally for Basis Fns on reference elements, which are
  //  not functions of coordinates. This save 18min of compile time!!!
  if (spatialDim==basisDim) {
    Intrepid::CellTools<RealType>::setJacobian(jacobian, refPoints, coordVec, *cellType);

  } else {
    Intrepid::FieldContainer<MeshScalarT>  phi(numQPs,spatialDim);
    Intrepid::FieldContainer<MeshScalarT> dphi(numQPs,spatialDim,basisDim);
    Intrepid::FieldContainer<MeshScalarT> norm(numQPs);
    Intrepid::FieldContainer<MeshScalarT> sinL(numQPs);
    Intrepid::FieldContainer<MeshScalarT> cosL(numQPs);
    Intrepid::FieldContainer<MeshScalarT> sinT(numQPs);
    Intrepid::FieldContainer<MeshScalarT> cosT(numQPs);
    Intrepid::FieldContainer<MeshScalarT>   D1(numQPs,basisDim,spatialDim);
    Intrepid::FieldContainer<MeshScalarT>   D2(numQPs,spatialDim,spatialDim);
    Intrepid::FieldContainer<MeshScalarT>   D3(numQPs,basisDim,spatialDim);

 // print out coords of vertices -- looks good
//for (int e = 0; e<numelements;      ++e)
//  for (int v = 0; v<numNodes;  ++v)
//  std::cout << "XXX: coord vec: " << e << " v: " << v << " =  "  << coordVec(e,v,0) << " " << coordVec(e,v,1) << " " << coordVec(e,v,2) << " " <<std::endl;


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

      for (int q = 0; q<numQPs;         ++q) 
         norm(q) = std::sqrt(norm(q));

      for (int q = 0; q<numQPs;         ++q) 
        for (int d = 0; d<spatialDim;   ++d) 
          phi(q,d) /= norm(q);

      for (int q = 0; q<numQPs;         ++q) {
        sinT(q) = phi(q,2);  
        cosT(q) = std::sqrt(1-sinT(q)*sinT(q));
        sinL(q) = cosT(q)>.0001 ? phi(q,1)/cosT(q) : MeshScalarT(0);
        cosL(q) = cosT(q)>.0001 ? phi(q,0)/cosT(q) : MeshScalarT(1);

//        sinL(q) = cosT(q) != 0 ? phi(q,1)/cosT(q) : MeshScalarT(0);
//        cosL(q) = cosT(q) != 0 ? phi(q,0)/cosT(q) : MeshScalarT(1);
  
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
        static const double pi = 3.1415926535897932385;
        static const double DIST_THRESHOLD = 1.0e-9;
        const MeshScalarT latitude  = std::asin(phi(q,2));

        MeshScalarT longitude = std::atan2(phi(q,1),phi(q,0));
        if (std::abs(std::abs(latitude)-pi/2) < DIST_THRESHOLD) longitude = 0;
        else if (longitude < 0) longitude += 2*pi;


        sphere_coord(e,q,0) = longitude;
        sphere_coord(e,q,1) = latitude;
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
            jacobian(e,q,b1,b2) /= norm(q);

    }
  }
  
  Intrepid::CellTools<MeshScalarT>::setJacobianInv(jacobian_inv, jacobian);

  Intrepid::CellTools<MeshScalarT>::setJacobianDet(jacobian_det, jacobian);

  for (int e = 0; e<numelements;      ++e) {
    for (int q = 0; q<numQPs;          ++q) {
      TEUCHOS_TEST_FOR_EXCEPTION(abs(jacobian_det(e,q))<.0001,
                  std::logic_error,"Bad Jacobian Found.");
    }
  }

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

//**********************************************************************
}
