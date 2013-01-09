//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(tensor_Mechanics_i_cc)
#define tensor_Mechanics_i_cc

namespace LCM {

  //
  // R^N volumetric part of 2nd-order tensor
  // \return \f$ \frac{1}{N} \mathrm{tr}\:(A) I \f$
  //
  template<typename T>
  inline
  Tensor<T>
  vol(Tensor<T> const & A)
  {
    const Index
    N = A.get_dimension();

    const T theta = (1.0/T(N)) * trace(A);

    return theta * eye<T>(N);
  }

  //
  // R^N deviatoric part of 2nd-order tensor
  // \return \f$ A - vol(A) \f$
  //
  template<typename T>
  inline
  Tensor<T>
  dev(Tensor<T> const & A)
  {
    return A - vol(A);
  }

} // namespace LCM

#endif // tensor_Mechanics_i_cc
