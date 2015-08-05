//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ATO_HOMOGENIZEDCONSTANTSRESPONSE_HPP
#define ATO_HOMOGENIZEDCONSTANTSRESPONSE_HPP

#include "PHAL_SeparableScatterScalarResponse.hpp"
#include "ATO_TopoTools.hpp"


/** 
 * \brief Response Description
 */
namespace ATO 
{
  template<typename EvalT, typename Traits>
  class HomogenizedConstantsResponseSpec : 
    public PHAL::SeparableScatterScalarResponse<EvalT,Traits>
  {
  public:
    typedef typename EvalT::ScalarT ScalarT;

    void postEvaluate(typename Traits::PostEvalData d);
	  
  protected:
    RealType global_measure;
  };

  /******************************************************************************/
  // Specialization: Jacobian
  /******************************************************************************/
  template<typename Traits>
  class HomogenizedConstantsResponseSpec<PHAL::AlbanyTraits::Jacobian,Traits> : 
    public PHAL::SeparableScatterScalarResponse<PHAL::AlbanyTraits::Jacobian,Traits>
  {
  public:
    typedef PHAL::AlbanyTraits::Jacobian EvalT;
    typedef typename EvalT::ScalarT ScalarT;
    
    void postEvaluate(typename Traits::PostEvalData d);
	  
  protected:
    RealType global_measure;
  };
  /******************************************************************************/
  // Specialization: DistParamDeriv
  /******************************************************************************/
  template<typename Traits>
  class HomogenizedConstantsResponseSpec<PHAL::AlbanyTraits::DistParamDeriv,Traits> : 
    public PHAL::SeparableScatterScalarResponse<PHAL::AlbanyTraits::DistParamDeriv,Traits>
  {
  public:
    typedef PHAL::AlbanyTraits::DistParamDeriv EvalT;
    typedef typename EvalT::ScalarT ScalarT;
    
    void postEvaluate(typename Traits::PostEvalData d);
	  
  protected:
    RealType global_measure;
  };
  /******************************************************************************/
  // Specialization: SGJacobian
  /******************************************************************************/
#ifdef ALBANY_SG
  template<typename Traits>
  class HomogenizedConstantsResponseSpec<PHAL::AlbanyTraits::SGJacobian,Traits> : 
    public PHAL::SeparableScatterScalarResponse<PHAL::AlbanyTraits::SGJacobian,Traits>
  {
  public:
    typedef PHAL::AlbanyTraits::SGJacobian EvalT;
    typedef typename EvalT::ScalarT ScalarT;
    
    void postEvaluate(typename Traits::PostEvalData d);
	  
  protected:
    RealType global_measure;
  };

#endif 
#ifdef ALBANY_ENSEMBLE 

  /******************************************************************************/
  // Specialization: MPJacobian
  /******************************************************************************/
  template<typename Traits>
  class HomogenizedConstantsResponseSpec<PHAL::AlbanyTraits::MPJacobian,Traits> : 
    public PHAL::SeparableScatterScalarResponse<PHAL::AlbanyTraits::MPJacobian,Traits>
  {
  public:
    typedef PHAL::AlbanyTraits::MPJacobian EvalT;
    typedef typename EvalT::ScalarT ScalarT;
    
    void postEvaluate(typename Traits::PostEvalData d);
	  
  protected:
    RealType global_measure;
  };
#endif

	

  template<typename EvalT, typename Traits>
  class HomogenizedConstantsResponse : public HomogenizedConstantsResponseSpec<EvalT,Traits>
  {
  public:
    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;
    
    HomogenizedConstantsResponse(Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl);
  
    void postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& vm);
    void preEvaluate(typename Traits::PreEvalData d);
    void evaluateFields(typename Traits::EvalData d);
    void postEvaluate(typename Traits::PostEvalData d);

  private:
  
    using HomogenizedConstantsResponseSpec<EvalT,Traits>::global_measure;

    std::string FName;
    static const std::string className;
    PHX::MDField<ScalarT> field;
    PHX::MDField<MeshScalarT,Cell,QuadPoint> weights;
    Teuchos::RCP< PHX::Tag<ScalarT> > objective_tag;

    Intrepid::FieldContainer<int> components;
    int tensorRank;

    RealType local_measure;

  };
}

#endif
