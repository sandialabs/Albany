//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
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


#ifdef ALBANY_SG_MP
  // Stochastic Galerkin Residual
  template<typename Traits>
  class EvaluatorTools<PHAL::AlbanyTraits::SGResidual, Traits>
  {
  public:
    typedef typename PHAL::AlbanyTraits::SGResidual::ScalarT ScalarT;
    typedef typename PHAL::AlbanyTraits::SGResidual::MeshScalarT MeshScalarT;

    EvaluatorTools();
    double getDoubleValue(const ScalarT& t) const;
    double getMeshDoubleValue(const MeshScalarT& t) const;
    std::string getEvalType() const;
  };


  // Stochastic Galerkin Jacobian
  template<typename Traits>
  class EvaluatorTools<PHAL::AlbanyTraits::SGJacobian, Traits>
  {
  public:
    typedef typename PHAL::AlbanyTraits::SGJacobian::ScalarT ScalarT;
    typedef typename PHAL::AlbanyTraits::SGJacobian::MeshScalarT MeshScalarT;

    EvaluatorTools();
    double getDoubleValue(const ScalarT& t) const;
    double getMeshDoubleValue(const MeshScalarT& t) const;
    std::string getEvalType() const;
  };

  // Stochastic Galerkin Tangent
  template<typename Traits>
  class EvaluatorTools<PHAL::AlbanyTraits::SGTangent, Traits>
  {
  public:
    typedef typename PHAL::AlbanyTraits::SGTangent::ScalarT ScalarT;
    typedef typename PHAL::AlbanyTraits::SGTangent::MeshScalarT MeshScalarT;

    EvaluatorTools();
    double getDoubleValue(const ScalarT& t) const;
    double getMeshDoubleValue(const MeshScalarT& t) const;
    std::string getEvalType() const;
  };


  // Multi-point residual
  template<typename Traits>
  class EvaluatorTools<PHAL::AlbanyTraits::MPResidual, Traits>
  {
  public:
    typedef typename PHAL::AlbanyTraits::MPResidual::ScalarT ScalarT;
    typedef typename PHAL::AlbanyTraits::MPResidual::MeshScalarT MeshScalarT;

    EvaluatorTools();
    double getDoubleValue(const ScalarT& t) const;
    double getMeshDoubleValue(const MeshScalarT& t) const;
    std::string getEvalType() const;
  };


  // Multi-point Jacobian
  template<typename Traits>
  class EvaluatorTools<PHAL::AlbanyTraits::MPJacobian, Traits>
  {
  public:
    typedef typename PHAL::AlbanyTraits::MPJacobian::ScalarT ScalarT;
    typedef typename PHAL::AlbanyTraits::MPJacobian::MeshScalarT MeshScalarT;

    EvaluatorTools();
    double getDoubleValue(const ScalarT& t) const;
    double getMeshDoubleValue(const MeshScalarT& t) const;
    std::string getEvalType() const;
  };

  // Multi-point Tangent
  template<typename Traits>
  class EvaluatorTools<PHAL::AlbanyTraits::MPTangent, Traits>
  {
  public:
    typedef typename PHAL::AlbanyTraits::MPTangent::ScalarT ScalarT;
    typedef typename PHAL::AlbanyTraits::MPTangent::MeshScalarT MeshScalarT;

    EvaluatorTools();
    double getDoubleValue(const ScalarT& t) const;
    double getMeshDoubleValue(const MeshScalarT& t) const;
    std::string getEvalType() const;
  };
#endif //ALBANY_SG_MP

}

#endif
