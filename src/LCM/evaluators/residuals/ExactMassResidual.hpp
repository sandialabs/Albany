//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LCM_ExactMass_RESIDUAL_HPP
#define LCM_ExactMass_RESIDUAL_HPP

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_MDField.hpp>
#include <Phalanx_config.hpp>
#include <Sacado_ParameterAccessor.hpp>
#include "Albany_Layouts.hpp"


namespace LCM {
/** \brief This evaluator computes the residual and Jacobian contributions
 * coming from the analytic mass matrix for various elements.  The main
 * element of interest is the composite tet element, where the exact mass comes from 
 * eqn. (C.4) in (Ostien et al, 2016).  One may be interested in the exact 
 * mass for other elements, like the isoparameteric tet10, to avoid having to use
 * very high cubature degrees to numericall integrate the mass to a sufficient accuracy. 

*/
// **************************************************************
// Base Class for code that is independent of evaluation type
// **************************************************************

template<typename EvalT, typename Traits>
class ExactMassResidualBase
  : public PHX::EvaluatorWithBaseImpl<Traits>,
    public PHX::EvaluatorDerived<EvalT, Traits>  {

public:
  ExactMassResidualBase(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  virtual void evaluateFields(typename Traits::EvalData d)=0;

protected:
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  /// Local function: return row of exact composite tet local mass 
  std::vector<RealType> compositeTet10LocalMassRow(const int cell, const int row) const;
  /// Local function: return row of exact lumped composite tet local mass 
  std::vector<RealType> compositeTet10LocalMassRowLumped(const int cell, const int row) const;
  /// Local function: return row of exact 8-node hexahedron local mass 
  std::vector<RealType> hex8LocalMassRow(const int cell, const int row) const;
  /// Local function: return row of exact 4-node tetrahedron local mass 
  std::vector<RealType> tet4LocalMassRow(const int cell, const int row) const;
  /// Local function: return row of exact 10-node (isoparametric) tetrahedron 
  //local mass
  std::vector<RealType> tet10LocalMassRow(const int cell, const int row) const;
  /// Local function: return row of exact lumped 10-node (isoparametric) tetrahedron 
  //local mass
  std::vector<RealType> tet10LocalMassRowLumped(const int cell, const int row) const;
  /// Local function: returns \int w_bf d\Omega for a given cell as a given node, 
  //  needed to compute the volume of each element to multiply local mass by.
  RealType computeElementVolScaling(const int cell, const int node) const; 
  /// Local function: returns elt volume = \int d\Omega for a given cell
  RealType computeElementVolume(const int cell) const; 
  /// Local function: helper function for computing value of residual to 
  //  minimize code duplication b/w Residual and Jacobian specializations.
  void computeResidualValue(typename Traits::EvalData workset) const; 

  /// Input: Weighted Basis Functions
  PHX::MDField<const MeshScalarT, Cell, Node, QuadPoint> w_bf_;
  /// Input: acceleration at quad points 
  PHX::MDField<const ScalarT, Cell, QuadPoint, Dim> accel_qps_;
  /// Input: acceleration at nodes 
  PHX::MDField<const ScalarT, Cell, Node, Dim> accel_nodes_;
  /// Input: integration weights
  PHX::MDField<const MeshScalarT> weights_;
  /// Output: mass contribution to residual/Jacobian 
  PHX::MDField<ScalarT, Cell, Node, Dim> mass_;
  /// Number of element nodes
  int num_nodes_;
  /// Number of integration points
  int num_pts_;
  /// Number of spatial dimensions
  int num_dims_;
  /// Number of cells
  int num_cells_;
  /// Density
  RealType density_{1.0};
  /// Dynamics flag
  bool enable_dynamics_;
  /// FOS for debug output
  Teuchos::RCP<Teuchos::FancyOStream> out_;
  /// Boolean telling code whether to use cubature to compute residual
  /// (if false, mass matrix will be used) 
  bool resid_using_cub_; 
  /// Flag to mark if using composite tet
  bool use_composite_tet_; 
  /// Flag to mark if using composite tet exact mass
  bool use_exact_mass_;

  enum ELT_TYPE {TET4, HEX8, TET10, CT10, UNSUPPORTED};
  ELT_TYPE elt_type;
 
};

template<typename EvalT, typename Traits> class ExactMassResidual;


// **************************************************************
// **************************************************************
// * Specializations
// **************************************************************
// **************************************************************


// **************************************************************
// Residual
// **************************************************************
template<typename Traits>
class ExactMassResidual<PHAL::AlbanyTraits::Residual,Traits>
  : public ExactMassResidualBase<PHAL::AlbanyTraits::Residual, Traits>  {
public:
  ExactMassResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
  void evaluateFields(typename Traits::EvalData d);
protected:

private:
  typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;

};

// **************************************************************
// Jacobian
// **************************************************************
template<typename Traits>
class ExactMassResidual<PHAL::AlbanyTraits::Jacobian,Traits>
  : public ExactMassResidualBase<PHAL::AlbanyTraits::Jacobian, Traits>  {
public:
  ExactMassResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
  void evaluateFields(typename Traits::EvalData d);
protected:

private:
  typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;

};

// **************************************************************
// Tangent
// **************************************************************
template<typename Traits>
class ExactMassResidual<PHAL::AlbanyTraits::Tangent,Traits>
  : public ExactMassResidualBase<PHAL::AlbanyTraits::Tangent, Traits>  {
public:
  ExactMassResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
  void evaluateFields(typename Traits::EvalData d);
protected:
  std::vector<RealType> compositeTet10LocalMassRow(const int row) const;  
private:
  typedef typename PHAL::AlbanyTraits::Tangent::ScalarT ScalarT;
};

// **************************************************************
// Distributed parameter derivative
// **************************************************************
template<typename Traits>
class ExactMassResidual<PHAL::AlbanyTraits::DistParamDeriv,Traits>
  : public ExactMassResidualBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>  {
public:
  ExactMassResidual(const Teuchos::ParameterList& p,
                  const Teuchos::RCP<Albany::Layouts>& dl);
  void evaluateFields(typename Traits::EvalData d);
protected:

private:
  typedef typename PHAL::AlbanyTraits::DistParamDeriv::ScalarT ScalarT;
};


// **************************************************************
}

#endif
