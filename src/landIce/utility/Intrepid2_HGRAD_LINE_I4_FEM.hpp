// @HEADER
// ************************************************************************
//
//                           Intrepid2 Package
//                 Copyright (2007) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Kyungjoo Kim  (kyukim@sandia.gov), or
//                    Mauro Perego  (mperego@sandia.gov)
//
// ************************************************************************
// @HEADER

/** \file   Intrepid2_HGRAD_LINE_I4_FEM.hpp
    \brief  Header file for the Intrepid2::Basis_HGRAD_LINE_I4_FEM class.
    \author Created by M. Perego
*/

#ifndef __INTREPID2_HGRAD_LINE_I4_FEM_HPP__
#define __INTREPID2_HGRAD_LINE_I4_FEM_HPP__

#include "Intrepid2_Basis.hpp"

namespace Intrepid2 {

  /** \class  Intrepid2::Basis_HGRAD_LINE_I4_FEM   
      \brief  Implementation of the H(grad)-compatible Lagrangian polynomial FEM basis on Line cell.

      Implements Lagrangian polynomial basis on the reference Line cell. The basis has
      cardinality 2 and reproduces constants.
      Following Intrepid2 notations, the I4 in the name, stands for the fact that the basis funcitons are 
      polynomials of dregree 4 but they do not span the complete polynomial space P^4 (I = Incomplete).
      The defualt basis functions are
      \phi_0(x) = 1-f(1/2-x/2), \phi_1(x) = f(1/2-x/2), with f(z)= 1-z^4,  x \in [-1,1]
      The basis can be changed setting coefficients c_0, c_1, c_2 via the setCoefficients function, in which case,
      f(z) = (1-z)(1+c_0 z+c_1 z^2+c_2 z^3)
      Basis functions are dual to a unisolvent set of degrees-of-freedom (DoF) defined and enumerated as follows:

      \verbatim
      =================================================================================================
      |         |           degree-of-freedom-tag table                    |                           |
      |   DoF   |----------------------------------------------------------|      DoF definition       |
      | ordinal |  subc dim    | subc ordinal | subc DoF ord |subc num DoF |                           |
      |=========|==============|==============|==============|=============|===========================|
      |    0    |       0      |       0      |       0      |      1      |   L_0(u) = u(-1)          |
      |---------|--------------|--------------|--------------|-------------|---------------------------|
      |    1    |       0      |       1      |       0      |      1      |   L_1(u) = u(1)           |
      |=========|==============|==============|==============|=============|===========================|
      |   MAX   |  maxScDim=0  |  maxScOrd=1  |  maxDfOrd=0  |      -      |                           |
      |=========|==============|==============|==============|=============|===========================|
      \endverbatim
  */

  namespace Impl {

    /**
      \brief See Intrepid2::Basis_HGRAD_LINE_I4_FEM
    */
    class Basis_HGRAD_LINE_I4_FEM {
    public:
      typedef struct Line<2> cell_topology_type;
      /**
        \brief See Intrepid2::Basis_HGRAD_LINE_I4_FEM
      */
      template<EOperator opType>
      struct Serial {
        template<typename OutputViewType,
                 typename inputViewType,
                 typename coeffType>
        KOKKOS_INLINE_FUNCTION
        static void
        getValues(       OutputViewType output,
                   const inputViewType input,
                   const coeffType c0,
                   const coeffType c1,
                   const coeffType c2);

      };

      template<typename DeviceType,
               typename outputValueValueType, class ...outputValueProperties,
               typename inputPointValueType,  class ...inputPointProperties,
               typename coeffType>
      static void
      getValues(       Kokkos::DynRankView<outputValueValueType,outputValueProperties...> outputValues,
                 const Kokkos::DynRankView<inputPointValueType, inputPointProperties...>  inputPoints,
                 const coeffType c0,
                 const coeffType c1,
                 const coeffType c2,
                 const EOperator operatorType);

      /**
        \brief See Intrepid2::Basis_HGRAD_LINE_I4_FEM
      */
      template<typename outputValueViewType,
               typename inputPointViewType,
               typename coeffType,
               EOperator opType>
      struct Functor {
             outputValueViewType _outputValues;
        const inputPointViewType  _inputPoints;
        const coeffType  _c0;
        const coeffType  _c1;
        const coeffType  _c2;

        KOKKOS_INLINE_FUNCTION
        Functor(       outputValueViewType outputValues_,
                       inputPointViewType  inputPoints_,
                       coeffType c0_,
                       coeffType c1_,
                       coeffType c2_)
          : _outputValues(outputValues_), _inputPoints(inputPoints_), _c0(c0_), _c1(c1_), _c2(c2_) {}

        KOKKOS_INLINE_FUNCTION
        void operator()(const ordinal_type pt) const {
         // std::cout << " eccof: " <<std::endl; 
          switch (opType) {
          case OPERATOR_VALUE : {
            auto       output = Kokkos::subview( _outputValues, Kokkos::ALL(), pt );
            const auto input  = Kokkos::subview( _inputPoints,                 pt, Kokkos::ALL() );
            Serial<opType>::getValues( output, input, _c0, _c1, _c2 );
            break;
          }
          case OPERATOR_GRAD :
          case OPERATOR_MAX : {
            auto       output = Kokkos::subview( _outputValues, Kokkos::ALL(), pt, Kokkos::ALL() );
            const auto input  = Kokkos::subview( _inputPoints,                 pt, Kokkos::ALL() );
            Serial<opType>::getValues( output, input, _c0, _c1, _c2 );
            break;
          }
          default: {
            INTREPID2_TEST_FOR_ABORT( opType != OPERATOR_VALUE &&
                                      opType != OPERATOR_GRAD &&
                                      opType != OPERATOR_MAX,
                                      ">>> ERROR: (Intrepid2::Basis_HGRAD_LINE_I4_FEM::Serial::getValues) operator is not supported");

          }
          }
        }
      };
    };
  }

  template<typename DeviceType = void,
           typename outputValueType = double,
           typename pointValueType = double>
  class Basis_HGRAD_LINE_I4_FEM
    : public Basis<DeviceType,outputValueType,pointValueType> {
  public:
    using BasisBase = Basis<DeviceType,outputValueType,pointValueType>;
    using HostBasis = Basis_HGRAD_LINE_I4_FEM<typename Kokkos::HostSpace::device_type,outputValueType,pointValueType>;
    
    using OrdinalTypeArray1DHost = typename BasisBase::OrdinalTypeArray1DHost;
    using OrdinalTypeArray2DHost = typename BasisBase::OrdinalTypeArray2DHost;
    using OrdinalTypeArray3DHost = typename BasisBase::OrdinalTypeArray3DHost;
    
    using OutputViewType = typename BasisBase::OutputViewType;
    using PointViewType  = typename BasisBase::PointViewType ;
    using ScalarViewType = typename BasisBase::ScalarViewType;
      
    /** \brief  Constructor.
     */
    Basis_HGRAD_LINE_I4_FEM(const ordinal_type order = 0,
                            const EPointType   pointType = POINTTYPE_EQUISPACED);

    using BasisBase::getValues;

    outputValueType c0_, c1_, c2_;

    void setCoefficients (const outputValueType c0, const outputValueType c1, const outputValueType c2) {
      c0_=c0; c1_= c1; c2_= c2;
    }

    virtual
    void
    getValues(       OutputViewType outputValues,
               const PointViewType  inputPoints,
               const EOperator operatorType = OPERATOR_VALUE) const override {
          //      std::cout << " ecco0: "  <<std::endl; 
#ifdef HAVE_INTREPID2_DEBUG
      // Verify arguments
      Intrepid2::getValues_HGRAD_Args(outputValues,
                                      inputPoints,
                                      operatorType,
                                      this->getBaseCellTopology(),
                                      this->getCardinality() );
#endif
      Impl::Basis_HGRAD_LINE_I4_FEM::
        getValues<DeviceType>( outputValues,
                                  inputPoints,
                                  c0_,
                                  c1_,
                                  c2_,
                                  operatorType );
    }

    virtual
    void
    getDofCoords( ScalarViewType dofCoords ) const override {
#ifdef HAVE_INTREPID2_DEBUG
      // Verify rank of output array.
      INTREPID2_TEST_FOR_EXCEPTION( dofCoords.rank() != 2, std::invalid_argument,
                                    ">>> ERROR: (Intrepid2::Basis_HGRAD_LINE_I4_FEM::getDofCoords) rank = 2 required for dofCoords array");
      // Verify 0th dimension of output array.
      INTREPID2_TEST_FOR_EXCEPTION( static_cast<ordinal_type>(dofCoords.extent(0)) != this->basisCardinality_, std::invalid_argument,
                                    ">>> ERROR: (Intrepid2::Basis_HGRAD_LINE_I4_FEM::getDofCoords) mismatch in number of dof and 0th dimension of dofCoords array");
      // Verify 1st dimension of output array.
      INTREPID2_TEST_FOR_EXCEPTION( dofCoords.extent(1) != this->getBaseCellTopology().getDimension(), std::invalid_argument,
                                    ">>> ERROR: (Intrepid2::Basis_HGRAD_LINE_I4_FEM::getDofCoords) incorrect reference cell (1st) dimension in dofCoords array");
#endif
      Kokkos::deep_copy(dofCoords, this->dofCoords_);
    }

    virtual
    void
    getDofCoeffs( ScalarViewType dofCoeffs ) const override {
#ifdef HAVE_INTREPID2_DEBUG
      // Verify rank of output array.
      INTREPID2_TEST_FOR_EXCEPTION( dofCoeffs.rank() != 1, std::invalid_argument,
                                    ">>> ERROR: (Intrepid2::Basis_HGRAD_LINE_I4_FEM::getdofCoeffs) rank = 1 required for dofCoeffs array");
      // Verify 0th dimension of output array.
      INTREPID2_TEST_FOR_EXCEPTION( static_cast<ordinal_type>(dofCoeffs.extent(0)) != this->getCardinality(), std::invalid_argument,
                                    ">>> ERROR: (Intrepid2::Basis_HGRAD_LINE_I4_FEM::getdofCoeffs) mismatch in number of dof and 0th dimension of dofCoeffs array");
#endif
      Kokkos::deep_copy(dofCoeffs, 1.0);
    }

    virtual
    const char*
    getName() const override {
      return "Intrepid2_HGRAD_LINE_I4_FEM";
    }

    BasisPtr<typename Kokkos::HostSpace::device_type,outputValueType,pointValueType>
    getHostBasis() const override{
      auto hostBasis = Teuchos::rcp(new HostBasis(this->basisDegree_));
      return hostBasis;
    }

  };

}// namespace Intrepid2

#include "Intrepid2_HGRAD_LINE_I4_FEMDef.hpp"

#endif
