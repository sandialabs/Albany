//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AERAS_SHALLOWWATERRESID_HPP
#define AERAS_SHALLOWWATERRESID_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"
#include "Sacado_ParameterAccessor.hpp"

#include <Shards_CellTopology.hpp>
#include <Intrepid2_Basis.hpp>
#include <Intrepid2_Cubature.hpp>


//#define ALBANY_KOKKOS_UNDER_DEVELOPMENT


namespace Aeras {
/** \brief ShallowWater equation Residual for atmospheric modeling

    This evaluator computes the residual of the ShallowWater equation for
    atmospheric dynamics.

 */

template<typename EvalT, typename Traits>
class ShallowWaterResid : public PHX::EvaluatorWithBaseImpl<Traits>,
public PHX::EvaluatorDerived<EvalT, Traits>,
public Sacado::ParameterAccessor<EvalT, SPL_Traits>  {

public:
	typedef typename EvalT::ScalarT ScalarT;
	typedef typename EvalT::MeshScalarT MeshScalarT;

	ShallowWaterResid(const Teuchos::ParameterList& p,
			const Teuchos::RCP<Albany::Layouts>& dl);

	void postRegistrationSetup(typename Traits::SetupData d,
			PHX::FieldManager<Traits>& vm);

	void evaluateFields(typename Traits::EvalData d);

	ScalarT& getValue(const std::string &n);

private:

	// Input:
	PHX::MDField<MeshScalarT,Cell,Node,QuadPoint> wBF;
	PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> wGradBF;
	PHX::MDField<ScalarT,Cell,Node,VecDim> U;  //vecDim works but its really Dim+1?
	PHX::MDField<ScalarT,Cell,Node,VecDim> UNodal;
	PHX::MDField<ScalarT,Cell,Node,VecDim> UDotDotNodal;
	PHX::MDField<ScalarT,Cell,QuadPoint,VecDim,Dim> Ugrad;
	PHX::MDField<ScalarT,Cell,Node,VecDim> UDot;
	PHX::MDField<ScalarT,Cell,Node,VecDim> UDotDot;
	Teuchos::RCP<shards::CellTopology> cellType;

	PHX::MDField<ScalarT,Cell,QuadPoint> mountainHeight;

	PHX::MDField<MeshScalarT,Cell,QuadPoint> weighted_measure;

	PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim,Dim> jacobian;
	PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim,Dim> jacobian_inv;
	PHX::MDField<MeshScalarT,Cell,QuadPoint> jacobian_det;
	Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device>    grad_at_cub_points;
	PHX::MDField<ScalarT,Cell,Node,VecDim> hyperviscosity;

	// Output:
	PHX::MDField<ScalarT,Cell,Node,VecDim> Residual;


	bool usePrescribedVelocity;
	bool useExplHyperviscosity;
	bool useImplHyperviscosity;
	bool plotVorticity;

	Teuchos::RCP<Intrepid2::Basis<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > > intrepidBasis;
	Teuchos::RCP<Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout,PHX::Device> > > cubature;
	Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device>    refPoints;
	Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device>    refWeights;
#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
	Intrepid2::FieldContainer_Kokkos<MeshScalarT, PHX::Layout, PHX::Device>  nodal_jacobian;
	Intrepid2::FieldContainer_Kokkos<MeshScalarT, PHX::Layout, PHX::Device>  nodal_inv_jacobian;
	Intrepid2::FieldContainer_Kokkos<MeshScalarT, PHX::Layout, PHX::Device>  nodal_det_j;
	Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> wrk_;
#endif

	PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>   sphere_coord;
	PHX::MDField<MeshScalarT,Cell,Node> lambda_nodal;
	PHX::MDField<MeshScalarT,Cell,Node> theta_nodal;
	PHX::MDField<ScalarT,Cell,QuadPoint,VecDim> source;

	ScalarT gravity; // gravity parameter -- Sacado-ized for sensitivities
	ScalarT Omega;   //rotation of earth  -- Sacado-ized for sensitivities

	double RRadius;   // 1/radius_of_earth
	double AlphaAngle;
	bool doNotDampRotation;
	bool obtainLaplaceOp;

	int numCells;
	int numNodes;
	int numQPs; //the same as vecDims
	int numDims;
	int vecDim; //the same as numDims, why doubled?
	int spatialDim;

	//OG: this is temporary
	double sHvTau;


#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
	void divergence(const Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device>  & fieldAtNodes,
			std::size_t cell, Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device>  & div);

	//gradient returns vector in physical basis
	void gradient(const Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device>  & fieldAtNodes,
			std::size_t cell, Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device>  & gradField);

	// curl only returns the component in the radial direction
	void curl(const Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device>  & fieldAtNodes,
			std::size_t cell, Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device>  & curl);

	void fill_nodal_metrics(std::size_t cell);

	void get_coriolis(std::size_t cell, Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device>  & coriolis);

	std::vector<LO> qpToNodeMap;
	std::vector<LO> nodeToQPMap;

#else
public:

	//OG why is everything here public?


	//this will stay
	Kokkos::View<MeshScalarT*, PHX::Device> refWeights_Kokkos;
	Kokkos::View<MeshScalarT***, PHX::Device> grad_at_cub_points_Kokkos;
	Kokkos::View<MeshScalarT**, PHX::Device> refPoints_kokkos;

	typedef PHX::KokkosViewFactory<ScalarT,PHX::Device> ViewFactory;

	PHX::MDField<ScalarT, Cell, Node> csurf;
	PHX::MDField<ScalarT, Cell, Node> csurftilde;
	PHX::MDField<ScalarT, Cell, QuadPoint, Dim> cgradsurf;
	PHX::MDField<ScalarT, Cell, QuadPoint, Dim> cgradsurftilde;
	PHX::MDField<ScalarT, Cell, Node> cUX, cUY, cUZ, cUTX, cUTY, cUTZ;
	PHX::MDField<ScalarT, Cell, QuadPoint, Dim> cgradUX, cgradUY, cgradUZ;
	PHX::MDField<ScalarT, Cell, QuadPoint, Dim> cgradUTX, cgradUTY, cgradUTZ;
	PHX::MDField<ScalarT, Cell, Node, Dim> tempnodalvec1, tempnodalvec2;
	PHX::MDField<ScalarT, Cell, Node, Dim> chuv;
	PHX::MDField<ScalarT, Cell, QuadPoint> cdiv;
	PHX::MDField<ScalarT, Cell, QuadPoint> ccor;
	PHX::MDField<ScalarT, Cell, QuadPoint> cvort;
	PHX::MDField<ScalarT, Cell, Node> ckineticEnergy, cpotentialEnergy;
	PHX::MDField<ScalarT, Cell, Node, Dim> cvelocityVec;
	PHX::MDField<ScalarT, Cell, QuadPoint, Dim> cgradKineticEnergy;
	PHX::MDField<ScalarT, Cell, QuadPoint, Dim> cgradPotentialEnergy;


	std::vector<LO> qpToNodeMap;
	std::vector<LO> nodeToQPMap;
	Kokkos::View<int*, PHX::Device> nodeToQPMap_Kokkos;

	double a, myPi;


//	ScalarT k11, k12, k21, k22, k32;

	void divergence4(const PHX::MDField<ScalarT, Cell, Node, Dim>  & field,
			const PHX::MDField<ScalarT, Cell, QuadPoint>  & div_,
			const int & cell) const;

	KOKKOS_INLINE_FUNCTION
	void gradient4(const PHX::MDField<ScalarT, Cell, Node>  & field,
			const PHX::MDField<ScalarT, Cell, QuadPoint, Dim>  & gradient_,
			const int & cell) const;

	KOKKOS_INLINE_FUNCTION
	void curl4(const PHX::MDField<ScalarT, Cell, Node, Dim>  & field,
			const PHX::MDField<ScalarT, Cell, QuadPoint>  & curl_,
			const int &cell) const;

	KOKKOS_INLINE_FUNCTION
	void get_coriolis4(const PHX::MDField<ScalarT, Cell, QuadPoint>  & cor_,
			const int &cell) const;

	typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;

	struct ShallowWaterResid_VecDim3_usePrescribedVelocity_Tag{};
	struct ShallowWaterResid_VecDim3_no_usePrescribedVelocity_Tag{};
	//The following are for hyperviscosity
	struct ShallowWaterResid_VecDim4_Tag{};
	struct ShallowWaterResid_VecDim6_Tag{};

	struct ShallowWaterResid_BuildLaplace_for_h_Tag{};
	struct ShallowWaterResid_BuildLaplace_for_huv_Tag{};

	typedef Kokkos::RangePolicy<ExecutionSpace, ShallowWaterResid_VecDim3_usePrescribedVelocity_Tag> ShallowWaterResid_VecDim3_usePrescribedVelocity_Policy;
	typedef Kokkos::RangePolicy<ExecutionSpace, ShallowWaterResid_VecDim3_no_usePrescribedVelocity_Tag> ShallowWaterResid_VecDim3_no_usePrescribedVelocity_Policy;
	typedef Kokkos::RangePolicy<ExecutionSpace, ShallowWaterResid_VecDim4_Tag> ShallowWaterResid_VecDim4_Policy;
	typedef Kokkos::RangePolicy<ExecutionSpace, ShallowWaterResid_VecDim6_Tag> ShallowWaterResid_VecDim6_Policy;

	//building Laplace op
	typedef Kokkos::RangePolicy<ExecutionSpace, ShallowWaterResid_BuildLaplace_for_h_Tag>  ShallowWaterResid_BuildLaplace_for_h_Policy;
	typedef Kokkos::RangePolicy<ExecutionSpace, ShallowWaterResid_BuildLaplace_for_huv_Tag>  ShallowWaterResid_BuildLaplace_for_huv_Policy;


	KOKKOS_INLINE_FUNCTION
	void operator() (const ShallowWaterResid_VecDim3_usePrescribedVelocity_Tag& tag, const int& cell) const;
	KOKKOS_INLINE_FUNCTION
	void operator() (const ShallowWaterResid_VecDim3_no_usePrescribedVelocity_Tag& tag, const int& cell) const;
	KOKKOS_INLINE_FUNCTION
	void operator() (const ShallowWaterResid_VecDim4_Tag& tag, const int& cell) const;
	KOKKOS_INLINE_FUNCTION
	void operator() (const ShallowWaterResid_VecDim6_Tag& tag, const int& cell) const;

	KOKKOS_INLINE_FUNCTION
	void operator() (const ShallowWaterResid_BuildLaplace_for_h_Tag& tag, const int& cell) const;
	KOKKOS_INLINE_FUNCTION
	void operator() (const ShallowWaterResid_BuildLaplace_for_huv_Tag& tag, const int& cell) const;

	KOKKOS_INLINE_FUNCTION
	void compute_Residual0(const int& cell) const;

	KOKKOS_INLINE_FUNCTION
	void compute_h_ImplHV(const int& cell) const;

	KOKKOS_INLINE_FUNCTION
	void compute_uv_ImplHV(const int& cell) const;

	KOKKOS_INLINE_FUNCTION
	void BuildLaplace_for_h (const int& cell) const;

	KOKKOS_INLINE_FUNCTION
	void BuildLaplace_for_uv (const int& cell) const;

	KOKKOS_INLINE_FUNCTION
	void compute_Residuals12_prescribed (const int& cell) const;

	KOKKOS_INLINE_FUNCTION
	void compute_Residuals12_notprescribed (const int& cell) const;

	KOKKOS_INLINE_FUNCTION
	void zeroing_Residual (const int& cell) const;


	// KOKKOS_INLINE_FUNCTION
	//void compute_coefficients_K(const MeshScalarT lam, const MeshScalarT th   );

	void compute_3Dvelocity4(std::size_t node, const ScalarT lam, const ScalarT th, const ScalarT ulambda, const ScalarT utheta,
			const PHX::MDField<ScalarT, Cell, Node>  & ux, const PHX::MDField<ScalarT, Cell, Node>  & uy,
			const PHX::MDField<ScalarT, Cell, Node>  & uz, const int& cell) const;

#endif
};

// Warning: these maps are a temporary fix, introduced by Steve Bova,
// to use the correct node ordering for node-point quadrature.  This
// should go away when spectral elements are fully implemented for
// Aeras.
//const int qpToNodeMap[9] = {0, 4, 1, 7, 8, 5, 3, 6, 2};
//const int nodeToQPMap[9] = {0, 2, 8, 6, 1, 5, 7, 3, 4};
// const int qpToNodeMap[4] = {0, 1, 3, 2};
// const int nodeToQPMap[4] = {0, 1, 3, 2};
// const int qpToNodeMap[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
// const int nodeToQPMap[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
}

#endif
