//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Intrepid2_HGRAD_TRI_Cn_FEM.hpp"
#include "Intrepid2_HGRAD_SIA_LINE_FEM.hpp"
#include "Intrepid2_DerivedBasis_HGRAD_WEDGE.hpp"
//#include "Intrepid2_RealSpaceTools.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"

namespace PHAL {

  template<class T, class MDField>
  constexpr typename std::enable_if< std::is_pod< T >::value,  T >::type
  getADValue(const MDField& field) {return Albany::ADValue(field(0));}

  template<class T, class MDField>
  constexpr typename std::enable_if< !std::is_pod< T >::value, T >::type
  getADValue(const MDField& field) {return field(0);}

/*
    template<class T>
  constexpr typename std::enable_if< std::is_pod< T >::value,  double>::type
  getDevNorm2(const T& val) {return 0.0;}

  template<class T>
  constexpr typename std::enable_if< !std::is_pod< T >::value, typename T::value_type >::type
  getDevNorm2(const T& val) {typename T::value_type dx, norm=0; for(int i=0; i<val.size(); ++i) {dx = val.dx(i); norm += std::pow(dx,2);} return norm;}
*/

// We need the following two functions because the parameters (inView) are ScalarT whereas the outViews are MeshScalarT.
// When in Jacobian evaluation mode, MeshScalarT is RealType whereas ScalarT is a Fad type.

  template<class outViewT, class inViewT>
  constexpr typename std::enable_if< !std::is_pod< typename outViewT::value_type >::value, void >::type
  initView(outViewT& outView, const inViewT inView, std::string outViewName, int extent0, int extent1, int extent2=0) {
    if(extent2>0)
      outView = Kokkos::createDynRankView(inView, outViewName, extent0, extent1, extent2);
    else
      outView = Kokkos::createDynRankView(inView, outViewName, extent0, extent1);
  }

  template<class outViewT, class inViewT>
  constexpr typename std::enable_if< std::is_pod< typename outViewT::value_type >::value, void >::type
  initView(outViewT& outView, const inViewT /*inView*/, std::string outViewName, int extent0, int extent1, int extent2=0) {
    if(extent2>0)
      outView = outViewT(outViewName, extent0, extent1, extent2);
    else
      outView = outViewT(outViewName, extent0, extent1);
  }


template<typename EvalT, typename Traits>
ComputeBasisFunctions<EvalT, Traits>::
ComputeBasisFunctions(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
  coordVec      (p.get<std::string>  ("Coordinate Vector Name"), dl->vertices_vector ),
  cellType      (p.get<Teuchos::RCP <shards::CellTopology> > ("Cell Type")),
  cubature      (p.get<Teuchos::RCP <Intrepid2::Cubature<PHX::Device> > >("Cubature")),
  intrepidBasis (p.get<Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> > > ("Intrepid2 Basis") ),
  BF            (p.get<std::string>  ("BF Name"), dl->node_qp_scalar),
  GradBF        (p.get<std::string>  ("Gradient BF Name"), dl->node_qp_gradient)
{
  depthIntegrated = (intrepidBasis->getBaseCellTopology().getKey() == shards::Wedge<6>::key) && 
    (p.isParameter("Depth-integrated Model") ? p.get<bool>("Depth-integrated Model") : false);

  this->addDependentField(coordVec.fieldTag());
  this->addEvaluatedField(BF);  
  this->addEvaluatedField(GradBF);
  
  computeWeights = p.isParameter("Weights Name");

  if(computeWeights) {
    weighted_measure = decltype(weighted_measure)(p.get<std::string>  ("Weights Name"), dl->qp_scalar );
    jacobian_det  = decltype(jacobian_det)(p.get<std::string>  ("Jacobian Det Name"), dl->qp_scalar );
    wBF = decltype(wBF)(p.get<std::string>  ("Weighted BF Name"), dl->node_qp_scalar);
    wGradBF  = decltype(wGradBF)(p.get<std::string>  ("Weighted Gradient BF Name"), dl->node_qp_gradient);
    this->addEvaluatedField(weighted_measure);
    this->addEvaluatedField(jacobian_det);
    this->addEvaluatedField(wBF);
    this->addEvaluatedField(wGradBF);
  }

  if(depthIntegrated) {
    c0_ = decltype(c0_)(p.get<std::string>("C_0 Parameter Name"), dl->shared_param);
    c1_ = decltype(c1_)(p.get<std::string>("C_1 Parameter Name"), dl->shared_param);
    c2_ = decltype(c2_)(p.get<std::string>("C_2 Parameter Name"), dl->shared_param);
    this->addDependentField(c0_);
    this->addDependentField(c1_);
    this->addDependentField(c2_);
  }

  // Get Dimensions
  std::vector<PHX::DataLayout::size_type> dim;
  dl->node_qp_gradient->dimensions(dim);

  numCells = dim[0];
  numNodes = dim[1];
  numQPs = dim[2];
  numDims = dim[3];

  std::vector<PHX::DataLayout::size_type> dims;
  dl->vertices_vector->dimensions(dims);
  numVertices = dims[1];

  this->setName("ComputeBasisFunctions"+PHX::print<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ComputeBasisFunctions<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(coordVec,fm);
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(GradBF,fm);
  if(computeWeights) {
    this->utils.setFieldData(jacobian_det,fm);
    this->utils.setFieldData(weighted_measure,fm);
    this->utils.setFieldData(wBF,fm);
    this->utils.setFieldData(wGradBF,fm);
  }

  jacobian = Kokkos::createDynRankView(coordVec.get_view(), "jacobian", numCells, numQPs, numDims, numDims);
  jacobian_inv = Kokkos::createDynRankView(coordVec.get_view(), "jacobian_inv", numCells, numQPs, numDims, numDims);
  

  // Allocate Temporary Kokkos Views
  if(depthIntegrated) {
    initView(val_at_cub_points, c0_.get_view(), "val_at_cub_points", numNodes, numQPs);
    initView(grad_at_cub_points, c0_.get_view(), "val_at_cub_points", numNodes, numQPs, numDims);
    c3_ = Kokkos::createDynRankView(c0_.get_view(), "c3", c0_.extent(0));
  } else {
    val_at_cub_points_RT = Kokkos::DynRankView<RealType, PHX::Device>("val_at_cub_points_RT", numNodes, numQPs);
    grad_at_cub_points_RT = Kokkos::DynRankView<RealType, PHX::Device>("grad_at_cub_points_RT", numNodes, numQPs, numDims);
  }

  refPoints = Kokkos::DynRankView<RealType, PHX::Device>("refPoints", numQPs, numDims);
  refWeights = Kokkos::DynRankView<RealType, PHX::Device>("refWeights", numQPs);

  // Pre-Calculate reference element quantities
  cubature->getCubature(refPoints, refWeights);
  if(depthIntegrated) {
    depthIntegratedBasis = Teuchos::rcp(new Intrepid2::Basis_Derived_HGRAD_WEDGE<Intrepid2::Basis_HGRAD_TRI_Cn_FEM<PHX::Device, MeshScalarT>, Intrepid2::Basis_HGRAD_SIA_LINE_FEM<PHX::Device, MeshScalarT> >(1));
  } else {
    intrepidBasis->getValues(val_at_cub_points_RT, refPoints, Intrepid2::OPERATOR_VALUE);
    intrepidBasis->getValues(grad_at_cub_points_RT, refPoints, Intrepid2::OPERATOR_GRAD);
    depthIntegratedBasis = Teuchos::null;
  }

  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
  if (d.memoizer_active()) memoizer.enable_memoizer();
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ComputeBasisFunctions<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  if (memoizer.have_saved_data(workset,this->evaluatedFields())) return;

  /** The allocated size of the Field Containers must currently
    * match the full workset size of the allocated PHX Fields,
    * this is the size that is used in the computation. There is
    * wasted effort computing on zeroes for the padding on the
    * final workset. Ideally, these are size numCells.
  //int containerSize = workset.numCells;
    */

  typedef typename Intrepid2::CellTools<PHX::Device>   ICT;
  typedef Intrepid2::FunctionSpaceTools<PHX::Device>   IFST;

  ICT::setJacobian(jacobian, refPoints, coordVec.get_view(), intrepidBasis);
  ICT::setJacobianInv (jacobian_inv, jacobian);
  

  bool isJacobianDetNegative(false);
  if(Teuchos::nonnull(depthIntegratedBasis)) {
    auto wedgebasis = Teuchos::rcp_dynamic_cast<Intrepid2::Basis_Derived_HGRAD_WEDGE<Intrepid2::Basis_HGRAD_TRI_Cn_FEM<PHX::Device, MeshScalarT>, Intrepid2::Basis_HGRAD_SIA_LINE_FEM<PHX::Device, MeshScalarT> > >(depthIntegratedBasis);
    auto lineBasisSIA = Teuchos::rcp_dynamic_cast<Intrepid2::Basis_HGRAD_SIA_LINE_FEM<PHX::Device, MeshScalarT>>(wedgebasis->getLineBasis());

    c3_(0) = 1.0 - c0_(0) - c1_(0) - c2_(0);
    lineBasisSIA->setCoefficients(getADValue<MeshScalarT>(c0_),getADValue<MeshScalarT>(c1_),getADValue<MeshScalarT>(c2_),getADValue<MeshScalarT>(c3_));
    //lineBasisSIA->setCoefficients(getADValue<RealType>(c0_),getADValue<RealType>(c1_),getADValue<RealType>(c2_));
    depthIntegratedBasis->getValues(val_at_cub_points, refPoints, Intrepid2::OPERATOR_VALUE);
    depthIntegratedBasis->getValues(grad_at_cub_points, refPoints, Intrepid2::OPERATOR_GRAD);
    IFST::HGRADtransformVALUE(BF.get_view(), val_at_cub_points);
    IFST::HGRADtransformGRAD (GradBF.get_view(), jacobian_inv, grad_at_cub_points);

    if(std::is_pod<ScalarT>::value) {
      std::cout << "coefficients: " << c0_(0) << " " << c1_(0) << " " << c2_(0) <<std::endl;
    }

/*
    if(!std::is_pod<MeshScalarT>::value) {
       std::cout << "\ncoord: ";
      for(int i=0; i<coordVec.extent_int(0); i++) {       
        for(int j=0; j<coordVec.extent_int(1); j++) {
          for(int k=0; k<coordVec.extent_int(2); k++) {
            if(getDevNorm2(coordVec(i,j,k)) > 0.0)
              std::cout << coordVec(i,j,k) << " ";
          }
        }
      }

      std::cout << "\nval_at_cub_points: ";
      for(int i=0; i<val_at_cub_points.extent_int(0); i++) {      
        for(int j=0; j<val_at_cub_points.extent_int(1); j++) {
          if(getDevNorm2(val_at_cub_points(i,j)) > 0.0)
            std::cout << val_at_cub_points(i,j) << " ";
        }
      }

      std::cout << "\ngrad_at_cub_points: ";
      for(int i=0; i<grad_at_cub_points.extent_int(0); i++) {       
        for(int j=0; j<grad_at_cub_points.extent_int(1); j++) {
          for(int k=0; k<grad_at_cub_points.extent_int(2); k++) {
            if(getDevNorm2(grad_at_cub_points(i,j,k)) > 0.0)
              std::cout << grad_at_cub_points(i,j,k) << " ";
          }
        }
      }
      
      bool issue = false;
      
      std::cout << "\nBF: ";
      for(int i=0; i<BF.extent_int(0); i++) {
        
        for(int j=0; j<BF.extent_int(1); j++) {
          for(int k=0; k<BF.extent_int(2); k++)
            if(getDevNorm2(BF(i,j,k)) > 0.0) {
              issue = true; std::cout << "("<<i << "," << j << "," <<  k << ") " <<  BF(i,j,k) << " " << val_at_cub_points(j,k) << " bool: ";
              break;
            }
          if(issue) break;
        }
        if(issue) break;
      }
      std::cout << issue;

      std::cout << "\njacobian: ";
      issue = false;
      for(int i=0; i<jacobian.extent_int(0); i++) {
        for(int j=0; j<jacobian.extent_int(1); j++) {
          for(int k=0; k<jacobian.extent_int(2); k++)
            for(int d=0; d<jacobian.extent_int(3); d++)
              if(getDevNorm2(jacobian(i,j,k,d)) > 0.0)
                issue = true; //std::cout << jacobian(i,j,k,d) << " ";
        }
      }
      std::cout << issue;

      std::cout << "\njacobian_inv: ";
      issue = false;
      for(int i=0; i<jacobian_inv.extent_int(0); i++) {
        for(int j=0; j<jacobian_inv.extent_int(1); j++) {
          for(int k=0; k<jacobian_inv.extent_int(2); k++)
            for(int d=0; d<jacobian_inv.extent_int(3); d++)
              if(getDevNorm2(jacobian_inv(i,j,k,d)) > 0.0)
                issue = true;//std::cout << jacobian_inv(i,j,k,d) << " ";
        }
      }
      std::cout << issue;

      std::cout << "\njacobian_det: ";
      issue = false;
      for(int i=0; i<jacobian_det.extent_int(0); i++) {
        for(int j=0; j<jacobian_det.extent_int(1); j++) {
          if(getDevNorm2(jacobian_det(i,j)) > 0.0)            
            issue = true;//std::cout << jacobian_det(i,j) << " ";
        }
      }
      std::cout << issue;

      std::cout << "\nGBF: ";
      issue = false;
      for(int i=0; i<GradBF.extent_int(0); i++) {
        for(int j=0; j<GradBF.extent_int(1); j++) {
          for(int k=0; k<GradBF.extent_int(2); k++)
            for(int d=0; d<GradBF.extent_int(3); d++)
              if(getDevNorm2(GradBF(i,j,k,d)) > 0.0)
                issue = true;//std::cout << GradBF(i,j,k,d) << " ";
        }
      }
      std::cout << issue;
      
    }

*/
  } else {
    IFST::HGRADtransformVALUE(BF.get_view(), val_at_cub_points_RT);
    IFST::HGRADtransformGRAD (GradBF.get_view(), jacobian_inv, grad_at_cub_points_RT);
  }

  if(computeWeights) {
    ICT::setJacobianDet (jacobian_det.get_view(), jacobian);
    isJacobianDetNegative = IFST::computeCellMeasure (weighted_measure.get_view(), jacobian_det.get_view(), refWeights);
    IFST::multiplyMeasure    (wBF.get_view(), weighted_measure.get_view(), BF.get_view());  
    IFST::multiplyMeasure    (wGradBF.get_view(), weighted_measure.get_view(), GradBF.get_view());
  }





  (void)isJacobianDetNegative;
}

//**********************************************************************
} // namespace PHAL
