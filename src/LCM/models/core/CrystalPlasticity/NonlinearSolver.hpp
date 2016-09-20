//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(Core_NonlinearSolver_hpp)
#define Core_NonlinearSolver_hpp

#include <MiniNonlinearSolver.h>
#include "CrystalPlasticityCore.hpp"

namespace CP
{
	/**
	 *	Nonlinear Solver (NLS) class for the CrystalPlasticity model; slip 
	 *	increments as unknowns.
	 */
	template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename EvalT>
	class ResidualSlipNLS:
			public Intrepid2::Function_Base<
			ResidualSlipNLS<NumDimT, NumSlipT, EvalT>,
			typename EvalT::ScalarT, NumSlipT>
	{
		using ScalarT = typename EvalT::ScalarT;

	public:

		//! Constructor.
		ResidualSlipNLS(
				Intrepid2::Tensor4<ScalarT, NumDimT> const & C,
				std::vector<SlipSystem<NumDimT>> const & slip_systems,
				std::vector<SlipFamily<NumDimT, NumSlipT>> const & slip_families,
				Intrepid2::Tensor<RealType, NumDimT> const & Fp_n,
				Intrepid2::Vector<RealType, NumSlipT> const & state_hardening_n,
				Intrepid2::Vector<RealType, NumSlipT> const & slip_n,
				Intrepid2::Tensor<ScalarT, NumDimT> const & F_np1,
				RealType dt);

		static constexpr char const * const
		NAME{"Crystal Plasticity Nonlinear System"};

	  using Base = Intrepid2::Function_Base<
	      ResidualSlipNLS<NumDimT, NumSlipT, EvalT>,
	      typename EvalT::ScalarT, NumSlipT>;

		//! Default implementation of value.
		template<typename T, Intrepid2::Index N = Intrepid2::DYNAMIC>
		T
		value(Intrepid2::Vector<T, N> const & x);

		//! Gradient function; returns the residual vector as a function of the slip 
		// at step N+1.
		template<typename T, Intrepid2::Index N = Intrepid2::DYNAMIC>
		Intrepid2::Vector<T, N>
		gradient(Intrepid2::Vector<T, N> const & x);


		//! Default implementation of hessian.
		template<typename T, Intrepid2::Index N = Intrepid2::DYNAMIC>
		Intrepid2::Tensor<T, N>
		hessian(Intrepid2::Vector<T, N> const & x);

	private:

		RealType
		num_dim_;

		RealType
		num_slip_;

		Intrepid2::Tensor4<ScalarT, NumDimT> const &
		C_;

		std::vector<SlipSystem<NumDimT>> const &
		slip_systems_;

		std::vector<SlipFamily<NumDimT, NumSlipT>> const &
		slip_families_;

		Intrepid2::Tensor<RealType, NumDimT> const &
		Fp_n_;

		Intrepid2::Vector<RealType, NumSlipT> const &
		state_hardening_n_;

		Intrepid2::Vector<RealType, NumSlipT> const &
		slip_n_;

		Intrepid2::Tensor<ScalarT, NumDimT> const &
		F_np1_;

		RealType
		dt_;
	};


	//
	//! Nonlinear Solver (NLS) class for the CrystalPlasticity model; slip 
	//  increments and hardnesses as unknowns.
	//
	template<Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT, typename EvalT>
	class ResidualSlipHardnessNLS:
			public Intrepid2::Function_Base<
			ResidualSlipHardnessNLS<NumDimT, NumSlipT, EvalT>,
			typename EvalT::ScalarT, NumSlipT>
	{
		using ScalarT = typename EvalT::ScalarT;

	public:

		//! Constructor.
		ResidualSlipHardnessNLS(
				Intrepid2::Tensor4<ScalarT, NumDimT> const & C,
				std::vector<SlipSystem<NumDimT>> const & slip_systems,
				std::vector<SlipFamily<NumDimT, NumSlipT>> const & slip_families,
				Intrepid2::Tensor<RealType, NumDimT> const & Fp_n,
				Intrepid2::Vector<RealType, NumSlipT> const & state_hardening_n,
				Intrepid2::Vector<RealType, NumSlipT> const & slip_n,
				Intrepid2::Tensor<ScalarT, NumDimT> const & F_np1,
				RealType dt);

		static constexpr char const * const
		NAME{"Slip and Hardness Residual Nonlinear System"};

		using Base = Intrepid2::Function_Base<
	      ResidualSlipHardnessNLS<NumDimT, NumSlipT, EvalT>,
	      typename EvalT::ScalarT, NumSlipT>;

		//! Default implementation of value.
		template<typename T, Intrepid2::Index N = Intrepid2::DYNAMIC>
		T
		value(Intrepid2::Vector<T, N> const & x);

		//! Gradient function; returns the residual vector as a function of the slip 
		// and hardness at step N+1.
		template<typename T, Intrepid2::Index N = Intrepid2::DYNAMIC>
		Intrepid2::Vector<T, N>
		gradient(Intrepid2::Vector<T, N> const & x);


		//! Default implementation of hessian.
		template<typename T, Intrepid2::Index N = Intrepid2::DYNAMIC>
		Intrepid2::Tensor<T, N>
		hessian(Intrepid2::Vector<T, N> const & x);

	private:

		RealType
		num_dim_;

		RealType
		num_slip_;

		Intrepid2::Tensor4<ScalarT, NumDimT> const &
		C_;

		std::vector<SlipSystem<NumDimT>> const &
		slip_systems_;

		std::vector<SlipFamily<NumDimT, NumSlipT>> const &
		slip_families_;

		Intrepid2::Tensor<RealType, NumDimT> const &
		Fp_n_;

		Intrepid2::Vector<RealType, NumSlipT> const &
		state_hardening_n_;

		Intrepid2::Vector<RealType, NumSlipT> const &
		slip_n_;

		Intrepid2::Tensor<ScalarT, NumDimT> const &
		F_np1_;

		RealType
		dt_;
	};
}

#include "NonlinearSolver_Def.hpp"

#endif

