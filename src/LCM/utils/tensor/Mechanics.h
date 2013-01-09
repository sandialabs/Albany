//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(tensor_Mechanics_h)
#define tensor_Mechanics_h

#include "Vector.h"

namespace LCM {

  ///
  /// Volumetric part of 2nd-order tensor
  /// \param A tensor
  /// \return \f$ \frac{1}{3} \mathrm{tr}\:A I \f$
  ///
  template<typename T>
  Tensor<T>
  vol(Tensor<T> const & A);

  ///
  /// Deviatoric part of 2nd-order tensor
  /// \param A tensor
  /// \return \f$ A - vol(A) \f$
  ///
  template<typename T>
  Tensor<T>
  dev(Tensor<T> const & A);

  ///
  /// Push forward covariant vector
  /// \param \f$ F, u \f$
  /// \return \f$ F^{-T} u \f$
  ///
  template<typename T>
  Vector<T>
  push_forward_covariant(Tensor<T> const & F, Vector<T> const & u);

  ///
  /// Pull back covariant vector
  /// \param \f$ F, u \f$
  /// \return \f$ F^T u \f$
  ///
  template<typename T>
  Vector<T>
  pull_back_covariant(Tensor<T> const & F, Vector<T> const & u);

  ///
  /// Push forward contravariant vector
  /// \param \f$ F, u \f$
  /// \return \f$ F u \f$
  ///
  template<typename T>
  Vector<T>
  push_forward_contravariant(Tensor<T> const & F, Vector<T> const & u);

  ///
  /// Pull back contravariant vector
  /// \param \f$ F, u \f$
  /// \return \f$ F^{-1} u \f$
  ///
  template<typename T>
  Vector<T>
  pull_back_contravariant(Tensor<T> const & F, Vector<T> const & u);

  ///
  /// Push forward covariant tensor
  /// \param \f$ F, A \f$
  /// \return \f$ F^{-T} A F^{-1} \f$
  ///
  template<typename T>
  Tensor<T>
  push_forward_covariant(Tensor<T> const & F, Tensor<T> const & A);

  ///
  /// Pull back covariant tensor
  /// \param \f$ F, A \f$
  /// \return \f$ F^T A F\f$
  ///
  template<typename T>
  Tensor<T>
  pull_back_covariant(Tensor<T> const & F, Tensor<T> const & A);

  ///
  /// Push forward contravariant tensor
  /// \param \f$ F, A \f$
  /// \return \f$ F A F^T \f$
  ///
  template<typename T>
  Tensor<T>
  push_forward_contravariant(Tensor<T> const & F, Tensor<T> const & A);

  ///
  /// Pull back contravariant tensor
  /// \param \f$ F, A \f$
  /// \return \f$ F^{-1} A F^{-T} \f$
  ///
  template<typename T>
  Tensor<T>
  pull_back_contravariant(Tensor<T> const & F, Tensor<T> const & A);

  ///
  /// Piola transformation for vector
  /// \param \f$ F, u \f$
  /// \return \f$ \det F F^{-1} u \f$
  ///
  template<typename T>
  Vector<T>
  piola(Tensor<T> const & F, Vector<T> const & u);

  ///
  /// Inverse Piola transformation for vector
  /// \param \f$ F, u \f$
  /// \return \f$ (\det F)^{-1} F u \f$
  ///
  template<typename T>
  Vector<T>
  piola_inverse(Tensor<T> const & F, Vector<T> const & u);

  ///
  /// Piola transformation for tensor, applied on second
  /// index. Useful for transforming Cauchy stress to 1PK stress.
  /// \param \f$ F, sigma \f$
  /// \return \f$ \det F sigma F^{-T} \f$
  ///
  template<typename T>
  Tensor<T>
  piola(Tensor<T> const & F, Tensor<T> const & sigma);

  ///
  /// Inverse Piola transformation for tensor, applied on second
  /// index. Useful for transforming 1PK stress to Cauchy stress.
  /// \param \f$ F, P \f$
  /// \return \f$ (\det F)^{-1} P F^T \f$
  ///
  template<typename T>
  Tensor<T>
  piola_inverse(Tensor<T> const & F, Tensor<T> const & P);

} // namespace LCM

#include "Mechanics.i.cc"
#include "Mechanics.t.cc"

#endif // tensor_Mechanics_h
