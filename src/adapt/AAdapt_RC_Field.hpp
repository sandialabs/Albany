//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef AADAPT_RC_FIELD
#define AADAPT_RC_FIELD

#include "AAdapt_RC_DataTypes.hpp"

namespace AAdapt {
namespace rc {

/*! \brief Encapsulate an MDField that holds accumulated data associated with
 *         another MDField containing incremental data.
 *
 * Equations in evaluators are written relative to the current RC. Certain
 * quantities are therefore incremental. These must be combined with accumulated
 * data from before the current RC. This class encapsulates accumulated data and
 * provides methods to do the combination.
 *
 * Typical calculations are of the following sort. Define
 *     u[n,n-1] = x[n] - x[n-1]
 *     U = u[n,0]
 *     F[n,k] = dx[n]/dx[k].
 *   First, we need to compute the deformation gradient
 *     F[n,0] = dx[n]/dx[0] = du[n,0]/dx[0] + dx[0]/dx[0] = dU/dx + I.
 * We typically have grad_u_, which is u times the basis functions:
 *     du[n,n-1]/dx[n-1] = dx[n]/dx[n-1] - dx[n-1]/dx[n-1] = F[n,n-1] - I.
 * So compute
 *     F[n,n-1] = I + du[n,n-1]/dx[n-1]
 *     F[n,0] = F[n,n-1] F[n-1,0].
 * This uses multiplyInto.
 *   Second, we need to compute something like strain w.r.t. x[0]:
 *     dU/dx[0] = dx[n]/dx[0] - dx[0]/dx[0] = F[n,0] - I.
 *     strain = 1/2 (dU/dx[0] + dU/dx[0]^T).
 *   At this point, we have stress with respect to x[0]. Now we need the
 * weighted basis function gradient to also be w.r.t. x[0]. w_grad_bf_ is
 *     f det(dx[n-1]/dr) dw/dx[n-1],
 * where f is the quadrature weight, r is the element reference coordinates, and
 * the det gives the volume change relative to the reference element. We need
 *     f det(dx[0]/dr) dw/dx[0].
 * One factor is given by
 *     dw/dx[0] = dw/dx[n-1] F[n-1,0],
 * which is w_grad_bf_ times def_grad_rc_. The other is
 *     det(dx[0]/dr) = det(dx[0]/dx[n-1] dx[n-1]/dr)
 *                   = det(dx[0]/dx[n-1]) det(dx[n-1]/dr)
 *                   = det(dx[n-1]/dr) / det(F[n-1,0]).
 * This third calculation is performed by transformWeightedGradientBF.
 */
template<int rank> class Field {
public:
  Field();

  bool init(const Teuchos::ParameterList& p, const std::string& name);

  //! \c init has been called.
  operator bool() const;

  typename RTensor<rank>::type& operator() () { return f_; }
  const typename RTensor<rank>::type& operator() () const { return f_; }

  //! f_incr = f_incr * f_accum. Call as \code f_rc.multiplyInto<typename
  //  EvalT::ScalarT>(f, cell, qp); \endcode inside loops over workset.numCells
  //  and number of quadrature points.
  template<typename ad_type>
  void multiplyInto(typename Tensor<ad_type, 2>::type& f_incr) const;
  //! f_incr = f_incr * f_accum.
  template<typename ad_type>
  void multiplyInto(typename Tensor<ad_type, 2>::type& f_incr,
                    const std::size_t cell, const std::size_t qp) const;

  //! f_incr += f_accum.
  template<typename ad_type>
  void addTo(typename Tensor<ad_type, rank>::type& f_incr) const;
  //! f_incr += f_accum. 
  template<typename ad_type>
  void addTo(typename Tensor<ad_type, rank>::type& f_incr,
             const std::size_t cell, const std::size_t qp) const;

private:
  typename RTensor<rank>::type f_;
  bool valid_;
};

//! Transform \c w_grad_bf using F[n-1,0].
void transformWeightedGradientBF(
  const Field<2>& F, const RealType& det_F,
  const PHX::MDField<RealType, Cell, Node, QuadPoint, Dim>& w_grad_bf,
  const int cell, const int pt, const int node, RealType w[3]);

} // namespace rc
} // namespace AAdapt

#endif // AADAPT_RC_FIELD
