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

/** \file   Intrepid2_HGRAD_LINE_I4_FEMDef.hpp
    \brief  Definition file for FEM basis functions for H(grad) functions on a Line.
    \author Created by M. Perego
 */

#ifndef __INTREPID2_HGRAD_LINE_I4_FEM_DEF_HPP__
#define __INTREPID2_HGRAD_LINE_I4_FEM_DEF_HPP__

#include "KokkosExp_View_Fad.hpp"

namespace Intrepid2 {

  // -------------------------------------------------------------------------------------

  namespace Impl {

    template<EOperator opType>
    template<typename OutputViewType,
             typename inputViewType,
             typename coeffType>
    KOKKOS_INLINE_FUNCTION
    void
    Basis_HGRAD_LINE_I4_FEM::Serial<opType>::
    getValues(       OutputViewType output,
               const inputViewType input,
               const coeffType c0,
               const coeffType c1,
               const coeffType c2) {
                
      switch (opType) {        
      case OPERATOR_VALUE : {
        const auto z = (1.0 - input(0))/2.0;  //input(0) \in [-1,1];
        const typename OutputViewType::value_type f = (1-z)*(1.0+z*(c0+z*(c1+z*c2)));
        output.access(0) = 1.0-f; 
        output.access(1) = f;       
        break;
      }
      case OPERATOR_GRAD : {  
        const auto z = (1.0 - input(0))/2.0;  //input(0) \in [-1,1];
        const typename OutputViewType::value_type dfdz = c0-1.0+z*(2.0*(c1-c0)+z*(3.0*(c2-c1)-z*4.0*c2));
        output.access(0, 0) = dfdz/2.0;
        output.access(1, 0) = -dfdz/2.0;
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

    template<typename DT,
             typename outputValueValueType, class ...outputValueProperties,
             typename inputPointValueType,  class ...inputPointProperties,
             typename coeffType>
    void
    Basis_HGRAD_LINE_I4_FEM::
    getValues(       Kokkos::DynRankView<outputValueValueType,outputValueProperties...> outputValues,
               const Kokkos::DynRankView<inputPointValueType, inputPointProperties...>  inputPoints,
               const coeffType c0,
               const coeffType c1,
               const coeffType c2,
               const EOperator operatorType ) {
                //std::cout << " ecco1: "  <<std::endl; 
      typedef          Kokkos::DynRankView<outputValueValueType,outputValueProperties...>         outputValueViewType;
      typedef          Kokkos::DynRankView<inputPointValueType, inputPointProperties...>          inputPointViewType;
      typedef typename ExecSpace<typename inputPointViewType::execution_space,typename DT::execution_space>::ExecSpaceType ExecSpaceType;

      // Number of evaluation points = dim 0 of inputPoints
      const auto loopSize = inputPoints.extent(0);
      Kokkos::RangePolicy<ExecSpaceType,Kokkos::Schedule<Kokkos::Static> > policy(0, loopSize);

      switch (operatorType) {

      case OPERATOR_VALUE: {
        typedef Functor<outputValueViewType,inputPointViewType,coeffType,OPERATOR_VALUE> FunctorType;
        Kokkos::parallel_for( policy, FunctorType(outputValues, inputPoints, c0, c1, c2) );
        break;
      }
      case OPERATOR_GRAD:
      case OPERATOR_DIV:
      case OPERATOR_CURL:
      case OPERATOR_D1: {
        typedef Functor<outputValueViewType,inputPointViewType,coeffType,OPERATOR_GRAD> FunctorType;
        Kokkos::parallel_for( policy, FunctorType(outputValues, inputPoints, c0, c1, c2) );
        break;
      }
      default: {
        INTREPID2_TEST_FOR_EXCEPTION( !Intrepid2::isValidOperator(operatorType), std::invalid_argument,
                                      ">>> ERROR (Basis_HGRAD_SIA_LINE_I4_FEM): Invalid operator type");
      }
      }
    }



  }

  // -------------------------------------------------------------------------------------

  template<typename DT, typename OT, typename PT>
  Basis_HGRAD_LINE_I4_FEM<DT,OT,PT>::
  Basis_HGRAD_LINE_I4_FEM(const ordinal_type /*order*/,
                            const EPointType   pointType) {
    const ordinal_type spaceDim = 1;
    this->basisCardinality_     = 2;
    this->basisDegree_          = 4;
    this->basisCellTopologyKey_ = shards::Line<2>::key;
    this->basisType_            = BASIS_FEM_LAGRANGIAN;
    this->basisCoordinates_     = COORDINATES_CARTESIAN;
    this->functionSpace_        = FUNCTION_SPACE_HGRAD;

    c0_ = c1_ = c2_ = 1.0;

    // initialize tags
    {
      // Basis-dependent intializations
      const ordinal_type tagSize  = 4;        // size of DoF tag, i.e., number of fields in the tag
      const ordinal_type posScDim = 0;        // position in the tag, counting from 0, of the subcell dim
      const ordinal_type posScOrd = 1;        // position in the tag, counting from 0, of the subcell ordinal
      const ordinal_type posDfOrd = 2;        // position in the tag, counting from 0, of DoF ordinal relative to the subcell

      // An array with local DoF tags assigned to basis functions, in the order of their local enumeration
      ordinal_type tags[8]  = { 0, 0, 0, 1,
                                0, 1, 0, 1 };



      // host tags
      OrdinalTypeArray1DHost tagView(&tags[0], 8);

      // Basis-independent function sets tag and enum data in tagToOrdinal_ and ordinalToTag_ arrays:
      // tags are constructed on host and sent to devices
      //OrdinalTypeArray2DHost ordinalToTag;
      //OrdinalTypeArray3DHost tagToOrdinal;
      this->setOrdinalTagData(this->tagToOrdinal_,
                              this->ordinalToTag_,
                              tagView,
                              this->basisCardinality_,
                              tagSize,
                              posScDim,
                              posScOrd,
                              posDfOrd);
    }

    // dofCoords on host and create its mirror view to device
    Kokkos::DynRankView<typename ScalarViewType::value_type,typename DT::execution_space::array_layout,Kokkos::HostSpace>
      dofCoords("dofCoordsHost", this->basisCardinality_, spaceDim);

    dofCoords(0,0) = -1.0;
    dofCoords(1,0) =  1.0;

    this->dofCoords_ = Kokkos::create_mirror_view(typename DT::memory_space(), dofCoords);
    Kokkos::deep_copy(this->dofCoords_, dofCoords);
  }

}

#endif
