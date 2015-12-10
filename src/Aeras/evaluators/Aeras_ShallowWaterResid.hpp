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
#include <Intrepid_Basis.hpp>
#include <Intrepid_Cubature.hpp>


#define ALBANY_KOKKOS_UNDER_DEVELOPMENT


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
	PHX::MDField<ScalarT,Cell,Node,VecDim> U;  //vecDim works but its really Dim+1
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
	Intrepid::FieldContainer<RealType>    grad_at_cub_points;
	PHX::MDField<ScalarT,Cell,Node,VecDim> hyperviscosity;

	// Output:
	PHX::MDField<ScalarT,Cell,Node,VecDim> Residual;


	bool usePrescribedVelocity;
	bool useExplHyperviscosity;
	bool useImplHyperviscosity;
	bool plotVorticity;

	Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > intrepidBasis;
	Teuchos::RCP<Intrepid::Cubature<RealType> > cubature;
	Intrepid::FieldContainer<RealType>    refPoints;
	Intrepid::FieldContainer<RealType>    refWeights;
#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
	Intrepid::FieldContainer<MeshScalarT>  nodal_jacobian;
	Intrepid::FieldContainer<MeshScalarT>  nodal_inv_jacobian;
	Intrepid::FieldContainer<MeshScalarT>  nodal_det_j;
	Intrepid::FieldContainer<ScalarT> wrk_;
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
	int numQPs;
	int numDims;
	int vecDim;
	int spatialDim;

	//OG: this is temporary
	double sHvTau;


	PHX::MDField<ScalarT,QuadPoint> wrk1qp_scalar_scope1_;
	PHX::MDField<ScalarT,QuadPoint> wrk2qp_scalar_scope1_;
	PHX::MDField<ScalarT,QuadPoint> wrk3qp_scalar_scope1_;
	PHX::MDField<ScalarT,QuadPoint> wrk4qp_scalar_scope1_;

//	PHX::MDField<ScalarT,Node> wrk1node_scalar_scope1_;
//	PHX::MDField<ScalarT,Node> wrk2node_scalar_scope1_;
//	PHX::MDField<ScalarT,QuadPoint> wrk1qp_scalar_scope1_;
//	PHX::MDField<ScalarT,QuadPoint> wrk2qp_scalar_scope1_;

	PHX::MDField<ScalarT,QuadPoint, Dim> wrk1qp_vector_scope1_;
	PHX::MDField<ScalarT,QuadPoint, Dim> wrk2qp_vector_scope1_;
	PHX::MDField<ScalarT,QuadPoint, Dim> wrk3qp_vector_scope1_;

	PHX::MDField<ScalarT,Node> wrk1node_scalar_scope1_;
	PHX::MDField<ScalarT,Node> wrk2node_scalar_scope1_;
	PHX::MDField<ScalarT,Node> wrk3node_scalar_scope1_;

	PHX::MDField<ScalarT,Node, Dim> wrk1node_vector_scope1_;
	PHX::MDField<ScalarT,Node, Dim> wrk2node_vector_scope1_;
//	PHX::MDField<ScalarT,Node, Dim> wrk3_vector_scope1_;

	//this is a vec of dim 3
	//OG: There may be a confusion about what vecDim is. It is only Dim+1
	//and I should have probably set dimension here to 3.
	//PHX::MDField<ScalarT,Node,VecDim> wrk3_vector_scope1_;

	PHX::MDField<ScalarT,Node, Dim> wrk1node_vector_scope2_;



#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
	void divergence(const Intrepid::FieldContainer<ScalarT>  & fieldAtNodes,
			std::size_t cell, Intrepid::FieldContainer<ScalarT>  & div);

	//gradient returns vector in physical basis
	void gradient(const Intrepid::FieldContainer<ScalarT>  & fieldAtNodes,
			std::size_t cell, Intrepid::FieldContainer<ScalarT>  & gradField);

	// curl only returns the component in the radial direction
	void curl(const Intrepid::FieldContainer<ScalarT>  & fieldAtNodes,
			std::size_t cell, Intrepid::FieldContainer<ScalarT>  & curl);

	void fill_nodal_metrics(std::size_t cell);

	void get_coriolis(std::size_t cell, Intrepid::FieldContainer<ScalarT>  & coriolis);

	std::vector<LO> qpToNodeMap;
	std::vector<LO> nodeToQPMap;

#else
public:

	//OG why is everything here public?

	//these three lines will go away
	Kokkos::View<MeshScalarT***, PHX::Device> nodal_jacobian;
	Kokkos::View<MeshScalarT***, PHX::Device> nodal_inv_jacobian;
	Kokkos::View<MeshScalarT*, PHX::Device> nodal_det_j;

	//this will stay
	Kokkos::View<MeshScalarT*, PHX::Device> refWeights_Kokkos;
	Kokkos::View<MeshScalarT***, PHX::Device> grad_at_cub_points_Kokkos;
	Kokkos::View<MeshScalarT**, PHX::Device> refPoints_kokkos;

	typedef PHX::KokkosViewFactory<ScalarT,PHX::Device> ViewFactory;

	//this needs cell dimensions
	PHX::MDField<ScalarT,Node> surf;
	PHX::MDField<ScalarT,Node> surftilde;
	PHX::MDField<ScalarT,QuadPoint, Dim> hgradNodes;
	PHX::MDField<ScalarT,QuadPoint, Dim> htildegradNodes;

	PHX::MDField<ScalarT,Node> uX, uY, uZ, utX, utY,utZ;
	PHX::MDField<ScalarT,QuadPoint, Dim> uXgradNodes, uYgradNodes, uZgradNodes;
	PHX::MDField<ScalarT,QuadPoint, Dim> utXgradNodes, utYgradNodes, utZgradNodes;
//end of the block that needs cell dim
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

	KOKKOS_INLINE_FUNCTION
	void divergence3(const PHX::MDField<ScalarT, Node, Dim>  & field,
			const PHX::MDField<ScalarT, QuadPoint>  & div_,
			const int & cell) const;

	void divergence4(const PHX::MDField<ScalarT, Cell, Node, Dim>  & field,
			const PHX::MDField<ScalarT, Cell, QuadPoint>  & div_,
			const int & cell) const;

	KOKKOS_INLINE_FUNCTION
	void gradient3(const PHX::MDField<ScalarT, Node>  & field,
			const PHX::MDField<ScalarT, QuadPoint, Dim>  & gradient_,
			const int & cell) const;

	KOKKOS_INLINE_FUNCTION
	void gradient4(const PHX::MDField<ScalarT, Cell, Node>  & field,
			const PHX::MDField<ScalarT, Cell, QuadPoint, Dim>  & gradient_,
			const int & cell) const;

	KOKKOS_INLINE_FUNCTION
	void curl3(const PHX::MDField<ScalarT, Node, Dim>  & field,
			const PHX::MDField<ScalarT, QuadPoint>  & curl_,
			const int &cell) const;

	KOKKOS_INLINE_FUNCTION
	void curl4(const PHX::MDField<ScalarT, Cell, Node, Dim>  & field,
			const PHX::MDField<ScalarT, Cell, QuadPoint>  & curl_,
			const int &cell) const;

	KOKKOS_INLINE_FUNCTION
	void get_coriolis3(const PHX::MDField<ScalarT,QuadPoint>  & cor_,
			const int &cell) const;

	KOKKOS_INLINE_FUNCTION
	void get_coriolis4(const PHX::MDField<ScalarT, Cell, QuadPoint>  & cor_,
			const int &cell) const;

	//This function puts (Residual(0)*Residual(1), Residual(0)*residual(2)) into huv_ .
	//KOKKOS_INLINE_FUNCTION
	//void product_h_uv(const PHX::MDField<ScalarT,Node, Dim>  & huv_,
	//		const int & cell) const;

	KOKKOS_INLINE_FUNCTION
	void fill_nodal_metrics (const int &cell) const;



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


//	KOKKOS_INLINE_FUNCTION
//	void compute_product_h_vel(const int& cell) const;

	KOKKOS_INLINE_FUNCTION
	void compute_Residual0(const int& cell) const;
	KOKKOS_INLINE_FUNCTION
	void compute_h_ImplHV(const int& cell) const;
	KOKKOS_INLINE_FUNCTION
	void compute_Residual3(const int& cell) const;
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

	//KOKKOS_INLINE_FUNCTION
	//void compute_coefficients_K(const typename PHAL::Ref<const MeshScalarT>::type lam,
	//		                     const typename PHAL::Ref<const MeshScalarT>::type th   );
	// KOKKOS_INLINE_FUNCTION
	void compute_coefficients_K(const MeshScalarT lam, const MeshScalarT th   );


	// KOKKOS_INLINE_FUNCTION
//	void compute_3Dvelocity(std::size_t node, const ScalarT lam, const ScalarT th, const ScalarT ulambda, const ScalarT utheta,
//			const PHX::MDField<ScalarT, Node, VecDim>  & uxyz) const;
	void compute_3Dvelocity(std::size_t node, const ScalarT lam, const ScalarT th, const ScalarT ulambda, const ScalarT utheta,
			const PHX::MDField<ScalarT, Node>  & ux, const PHX::MDField<ScalarT, Node>  & uy, const PHX::MDField<ScalarT, Node>  & uz) const;

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
