/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


#ifndef PHAL_ALBANYTRAITS_HPP
#define PHAL_ALBANYTRAITS_HPP

// mpl (Meta Programming Library) templates
//#include "Sacado.hpp"
#include "Sacado_mpl_vector.hpp"
#include "Sacado_mpl_find.hpp"
#include "boost/mpl/map.hpp"
#include "boost/mpl/find.hpp"

// traits Base Class
#include "Phalanx_Traits_Base.hpp"

// Include User Data Types
#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Allocator_Contiguous.hpp"
#include "Phalanx_Allocator_New.hpp"
#include "Phalanx_TypeStrings.hpp"


#include "Albany_DataTypes.hpp"
#include "PHAL_Dimension.hpp"
#include "PHAL_Workset.hpp"

#include "Albany_DiscretizationFactory.hpp"
namespace PHAL {

  struct AlbanyTraits : public PHX::TraitsBase {

    // ******************************************************************
    // *** Evaluation Types
    //   * ScalarT is for quantities that depend on solution/params
    //   * MeshScalarT  is for quantities that depend on mesh coords only
    // ******************************************************************
    struct Residual   { typedef RealType  ScalarT; typedef RealType MeshScalarT; };
    struct Jacobian   { typedef FadType   ScalarT; typedef RealType MeshScalarT; };
    struct Tangent    { typedef FadType   ScalarT;
                        typedef FadType   MeshScalarT; };  // Use this for shape opt
                       // typedef RealType MeshScalarT; }; // Uncomment for no shape opt
    struct SGResidual { typedef SGType    ScalarT; typedef RealType MeshScalarT; };
    struct SGJacobian { typedef SGFadType ScalarT; typedef RealType MeshScalarT; };
    typedef Sacado::mpl::vector<Residual, Jacobian, Tangent, 
				SGResidual, SGJacobian> EvalTypes;

    // ******************************************************************
    // *** Data Types
    // ******************************************************************
    
    // Create the data types for each evaluation type
    //AGS RPP 3/2010: Added RealType as acceptible Field
    //   type for all EvalT so that coordVec(double) for
    //   all EvalT
    
    // Residual (default scalar type is RealType)
    typedef Sacado::mpl::vector<RealType> ResidualDataTypes;
  
    // Jacobian (default scalar type is Fad<double, double>)
    typedef Sacado::mpl::vector<FadType,RealType> JacobianDataTypes;

    // Tangent (default scalar type is Fad<double>)
    typedef Sacado::mpl::vector<FadType,RealType> TangentDataTypes;

    // SG Residual (default scalar type is SGType)
    typedef Sacado::mpl::vector<SGType,RealType> SGResidualDataTypes;
  
    // SG Jacobian (default scalar type is Fad<SGType>)
    typedef Sacado::mpl::vector<SGFadType,RealType> SGJacobianDataTypes;

    // Maps the key EvalType a vector of DataTypes
    typedef boost::mpl::map<
      boost::mpl::pair<Residual, ResidualDataTypes>,
      boost::mpl::pair<Jacobian, JacobianDataTypes>,
      boost::mpl::pair<Tangent,  TangentDataTypes>,
      boost::mpl::pair<SGResidual, SGResidualDataTypes>,
      boost::mpl::pair<SGJacobian, SGJacobianDataTypes>
    >::type EvalToDataMap;

    // ******************************************************************
    // *** Allocator Type
    // ******************************************************************
    typedef PHX::NewAllocator Allocator;
    //typedef PHX::ContiguousAllocator<RealType> Allocator;

    // ******************************************************************
    // *** User Defined Object Passed in for Evaluation Method
    // ******************************************************************
    typedef const Albany::AbstractDiscretization& SetupData;
    typedef const Workset& EvalData;
    typedef void* PreEvalData;
    typedef void* PostEvalData;

  };
 
  // ******************************************************************
  // ******************************************************************
  // Debug strings.  Specialize the Evaluation and Data types for the
  // TypeString object in phalanx/src/Phalanx_TypeStrings.hpp.
  // ******************************************************************
  // ******************************************************************

}

namespace PHX {
  // Evaluation Types
  template<> struct TypeString<PHAL::AlbanyTraits::Residual> 
  { static const std::string value; };

  template<> struct TypeString<PHAL::AlbanyTraits::Jacobian> 
  { static const std::string value; };

  template<> struct TypeString<PHAL::AlbanyTraits::Tangent> 
  { static const std::string value; };

  template<> struct TypeString<PHAL::AlbanyTraits::SGResidual> 
  { static const std::string value; };

  template<> struct TypeString<PHAL::AlbanyTraits::SGJacobian> 
  { static const std::string value; };

  // Data Types
  template<> struct TypeString<RealType> 
  { static const std::string value; };

  template<> struct TypeString<FadType > 
  { static const std::string value; };

  template<> struct TypeString<SGType> 
  { static const std::string value; };

  template<> struct TypeString<SGFadType> 
  { static const std::string value; };

}

// Define macro for explicit template instantiation
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_RESIDUAL(name) \
  template class name<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits>;
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_JACOBIAN(name) \
  template class name<PHAL::AlbanyTraits::Jacobian, PHAL::AlbanyTraits>;
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_TANGENT(name) \
  template class name<PHAL::AlbanyTraits::Tangent, PHAL::AlbanyTraits>;
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_SGRESIDUAL(name) \
  template class name<PHAL::AlbanyTraits::SGResidual, PHAL::AlbanyTraits>;
#define PHAL_INSTANTIATE_TEMPLATE_CLASS_SGJACOBIAN(name) \
  template class name<PHAL::AlbanyTraits::SGJacobian, PHAL::AlbanyTraits>;

#define PHAL_INSTANTIATE_TEMPLATE_CLASS(name) \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_RESIDUAL(name)	 \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_JACOBIAN(name)      \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_TANGENT(name)	 \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_SGRESIDUAL(name)	 \
  PHAL_INSTANTIATE_TEMPLATE_CLASS_SGJACOBIAN(name)

#endif
