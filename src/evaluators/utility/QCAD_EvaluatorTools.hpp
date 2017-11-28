//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef QCAD_EVALUATORTOOLS_HPP
#define QCAD_EVALUATORTOOLS_HPP

/**
 * \brief Provides general-purpose template-specialized functions
 *  for use in other evaluator classes.
 */
namespace QCAD
{
  template<typename EvalT, typename Traits> class EvaluatorTools;

  //! Specializations

  // Residual
  template<typename Traits>
  class EvaluatorTools<PHAL::AlbanyTraits::Residual, Traits>
  {
  public:
    typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
    typedef typename PHAL::AlbanyTraits::Residual::MeshScalarT MeshScalarT;

    EvaluatorTools();
    double getDoubleValue(const ScalarT& t) const;
    double getMeshDoubleValue(const MeshScalarT& t) const;
    std::string getEvalType() const;
  };


  // Jacobian
  template<typename Traits>
  class EvaluatorTools<PHAL::AlbanyTraits::Jacobian, Traits>
  {
  public:
    typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
    typedef typename PHAL::AlbanyTraits::Jacobian::MeshScalarT MeshScalarT;

    EvaluatorTools();
    double getDoubleValue(const ScalarT& t) const;
    double getMeshDoubleValue(const MeshScalarT& t) const;
    std::string getEvalType() const;
  };


  // Tangent
  template<typename Traits>
  class EvaluatorTools<PHAL::AlbanyTraits::Tangent, Traits>
  {
  public:
    typedef typename PHAL::AlbanyTraits::Tangent::ScalarT ScalarT;
    typedef typename PHAL::AlbanyTraits::Tangent::MeshScalarT MeshScalarT;

    EvaluatorTools();
    double getDoubleValue(const ScalarT& t) const;
    double getMeshDoubleValue(const MeshScalarT& t) const;
    std::string getEvalType() const;
  };

  // DistParamDeriv
  template<typename Traits>
  class EvaluatorTools<PHAL::AlbanyTraits::DistParamDeriv, Traits>
  {
  public:
    typedef typename PHAL::AlbanyTraits::DistParamDeriv::ScalarT ScalarT;
    typedef typename PHAL::AlbanyTraits::DistParamDeriv::MeshScalarT MeshScalarT;

    EvaluatorTools();
    double getDoubleValue(const ScalarT& t) const;
    double getMeshDoubleValue(const MeshScalarT& t) const;
    std::string getEvalType() const;
  };

}

#endif
