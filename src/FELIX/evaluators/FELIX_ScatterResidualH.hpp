//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_SCATTER_RESIDUALH_HPP
#define FELIX_SCATTER_RESIDUALH_HPP

#include "PHAL_ScatterResidual.hpp"



namespace PHAL {
/** \brief Scatters result from the residual fields into the
    global (epetra) data structurs.  This includes the
    post-processing of the AD data type for all evaluation
    types besides Residual.

*/
// **************************************************************
// Base Class for code that is independent of evaluation type
// **************************************************************



template<typename EvalT, typename Traits> class ScatterResidualH;
template<typename EvalT, typename Traits> class ScatterResidualH3D;

// **************************************************************
// **************************************************************
// * Specializations
// **************************************************************
// **************************************************************


// **************************************************************
// Residual
// **************************************************************
template<typename Traits>
class ScatterResidualH<PHAL::AlbanyTraits::Residual,Traits>
  : public ScatterResidual<PHAL::AlbanyTraits::Residual, Traits>  {
public:
  ScatterResidualH(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
};

// **************************************************************
// Jacobian
// **************************************************************
template<typename Traits>
class ScatterResidualH<PHAL::AlbanyTraits::Jacobian,Traits>
  : public ScatterResidual<PHAL::AlbanyTraits::Jacobian, Traits>  {
public:
  ScatterResidualH(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
  void evaluateFields(typename Traits::EvalData d);
private:
  const std::size_t numFields;
  int HLevel;
  std::string meshPart;
  Teuchos::RCP<const CellTopologyData> cell_topo;
  typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
};

// **************************************************************
// Tangent
// **************************************************************
template<typename Traits>
class ScatterResidualH<PHAL::AlbanyTraits::Tangent,Traits>
  : public ScatterResidual<PHAL::AlbanyTraits::Tangent, Traits>  {
public:
  ScatterResidualH(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
//  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Distributed parameter derivative
// **************************************************************
template<typename Traits>
class ScatterResidualH<PHAL::AlbanyTraits::DistParamDeriv,Traits>
  : public ScatterResidual<PHAL::AlbanyTraits::DistParamDeriv, Traits>  {
public:
  ScatterResidualH(const Teuchos::ParameterList& p,
                  const Teuchos::RCP<Albany::Layouts>& dl);
//  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Stochastic Galerkin Residual
// **************************************************************
#ifdef ALBANY_SG
template<typename Traits>
class ScatterResidualH<PHAL::AlbanyTraits::SGResidual,Traits>
  : public ScatterResidual<PHAL::AlbanyTraits::SGResidual, Traits>  {
public:
  ScatterResidualH(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
};

// **************************************************************
// Stochastic Galerkin Jacobian
// **************************************************************
template<typename Traits>
class ScatterResidualH<PHAL::AlbanyTraits::SGJacobian,Traits>
  : public ScatterResidual<PHAL::AlbanyTraits::SGJacobian, Traits>  {
public:
  ScatterResidualH(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
};

// **************************************************************
// Stochastic Galerkin Tangent
// **************************************************************
template<typename Traits>
class ScatterResidualH<PHAL::AlbanyTraits::SGTangent,Traits>
  : public ScatterResidual<PHAL::AlbanyTraits::SGTangent, Traits>  {
public:
  ScatterResidualH(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
};

#endif

#ifdef ALBANY_ENSEMBLE
// **************************************************************
// Multi-point Residual
// **************************************************************
template<typename Traits>
class ScatterResidualH<PHAL::AlbanyTraits::MPResidual,Traits>
  : public ScatterResidual<PHAL::AlbanyTraits::MPResidual, Traits>  {
public:
  ScatterResidualH(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
};

// **************************************************************
// Multi-point Jacobian
// **************************************************************
template<typename Traits>
class ScatterResidualH<PHAL::AlbanyTraits::MPJacobian,Traits>
  : public ScatterResidual<PHAL::AlbanyTraits::MPJacobian, Traits>  {
public:
  ScatterResidualH(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
};

// **************************************************************
// Multi-point Tangent
// **************************************************************
template<typename Traits>
class ScatterResidualH<PHAL::AlbanyTraits::MPTangent,Traits>
  : public ScatterResidual<PHAL::AlbanyTraits::MPTangent, Traits>  {
public:
  ScatterResidualH(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
};
#endif

// **************************************************************


// **************************************************************
// Residual
// **************************************************************
template<typename Traits>
class ScatterResidualH3D<PHAL::AlbanyTraits::Residual,Traits>
  : public ScatterResidual<PHAL::AlbanyTraits::Residual, Traits>  {
public:
  ScatterResidualH3D(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
};

// **************************************************************
// Jacobian
// **************************************************************
template<typename Traits>
class ScatterResidualH3D<PHAL::AlbanyTraits::Jacobian,Traits>
  : public ScatterResidual<PHAL::AlbanyTraits::Jacobian, Traits>  {
public:
  ScatterResidualH3D(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
  void evaluateFields(typename Traits::EvalData d);
private:
  const std::size_t numFields;
  int HOffset;
  int HLevel;
  typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
};

// **************************************************************
// Tangent
// **************************************************************
template<typename Traits>
class ScatterResidualH3D<PHAL::AlbanyTraits::Tangent,Traits>
  : public ScatterResidual<PHAL::AlbanyTraits::Tangent, Traits>  {
public:
  ScatterResidualH3D(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
//  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Distributed parameter derivative
// **************************************************************
template<typename Traits>
class ScatterResidualH3D<PHAL::AlbanyTraits::DistParamDeriv,Traits>
  : public ScatterResidual<PHAL::AlbanyTraits::DistParamDeriv, Traits>  {
public:
  ScatterResidualH3D(const Teuchos::ParameterList& p,
                  const Teuchos::RCP<Albany::Layouts>& dl);
//  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Stochastic Galerkin Residual
// **************************************************************
#ifdef ALBANY_SG
template<typename Traits>
class ScatterResidualH3D<PHAL::AlbanyTraits::SGResidual,Traits>
  : public ScatterResidual<PHAL::AlbanyTraits::SGResidual, Traits>  {
public:
  ScatterResidualH3D(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
};

// **************************************************************
// Stochastic Galerkin Jacobian
// **************************************************************
template<typename Traits>
class ScatterResidualH3D<PHAL::AlbanyTraits::SGJacobian,Traits>
  : public ScatterResidual<PHAL::AlbanyTraits::SGJacobian, Traits>  {
public:
  ScatterResidualH3D(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
};

// **************************************************************
// Stochastic Galerkin Tangent
// **************************************************************
template<typename Traits>
class ScatterResidualH3D<PHAL::AlbanyTraits::SGTangent,Traits>
  : public ScatterResidual<PHAL::AlbanyTraits::SGTangent, Traits>  {
public:
  ScatterResidualH3D(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
};
#endif

#ifdef ALBANY_ENSEMBLE
// **************************************************************
// Multi-point Residual
// **************************************************************
template<typename Traits>
class ScatterResidualH3D<PHAL::AlbanyTraits::MPResidual,Traits>
  : public ScatterResidual<PHAL::AlbanyTraits::MPResidual, Traits>  {
public:
  ScatterResidualH3D(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
};

// **************************************************************
// Multi-point Jacobian
// **************************************************************
template<typename Traits>
class ScatterResidualH3D<PHAL::AlbanyTraits::MPJacobian,Traits>
  : public ScatterResidual<PHAL::AlbanyTraits::MPJacobian, Traits>  {
public:
  ScatterResidualH3D(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
};

// **************************************************************
// Multi-point Tangent
// **************************************************************
template<typename Traits>
class ScatterResidualH3D<PHAL::AlbanyTraits::MPTangent,Traits>
  : public ScatterResidual<PHAL::AlbanyTraits::MPTangent, Traits>  {
public:
  ScatterResidualH3D(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
};
#endif 

}

#endif
