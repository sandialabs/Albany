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

/*! Encapsulate an MDField that holds accumulated data associated with another
 *! MDField containing incremental data.
 *
 * Equations in evaluators are written relative to the current RC. Certain
 * quantities are therefore incremental. These must be combined with accumulated
 * data from before the current RC. This class encapsulates accumulated data and
 * provides methods to do the combination.
 */
template<int rank>
class Field {
public:
  Field();

  bool init(const Teuchos::ParameterList& p, const std::string& name);

  //! init has been called.
  operator bool() const;

  typename RTensor<rank>::type& operator() () { return f_; }

  //! f_incr += f_accum. Call as \code f_rc.addTo<typename EvalT::ScalarT>(f);
  //! \endcode
  template<typename ad_type>
  void addTo(typename Tensor<ad_type, rank>::type& f_incr) const;

  //! f_incr = f_incr * f_accum.
  template<typename ad_type>
  void multiplyInto(typename Tensor<ad_type, 2>::type& f_incr) const;

private:
  typename RTensor<rank>::type f_;
  bool valid_;
};

} // namespace rc
} // namespace AAdapt

#endif // AADAPT_RC_FIELD
