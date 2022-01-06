//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_LOG_GAUSSIAN_COMBINATION_PARAMETER_HPP
#define PHAL_LOG_GAUSSIAN_COMBINATION_PARAMETER_HPP

#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"
#include "Teuchos_ParameterList.hpp"

#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_SharedParameter.hpp"
#include "Albany_UnivariateDistribution.hpp"

namespace PHAL {
///
/// LogGaussianDistributedParameterBase
///
template<typename EvalT, typename Traits>
class LogGaussianDistributedParameterBase : 
  public PHX::EvaluatorWithBaseImpl<Traits>,
  public PHX::EvaluatorDerived<EvalT, Traits>
{
 private:
  typedef typename EvalT::ParamScalarT ParamScalarT;

 public:
  typedef typename EvalT::ScalarT   ScalarT;
  //typedef ParamNameEnum             EnumType;

  LogGaussianDistributedParameterBase (const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl)
  {
    std::string log_gaussian_field_name   = p.get<std::string>("Log Gaussian Name");
    std::string gaussian_field_name       = p.get<std::string>("Gaussian Name");
    std::string mean_field_name;

    mean_is_field = false;
    if (p.isParameter("Mean Name")) {
      mean_field_name = p.get<std::string>("Mean Name");
      mean_is_field = true;
    }

    eval_on_side = false;
    if (p.isParameter("Side Set Name")) {
      sideSetName = p.get<std::string>("Side Set Name");
      eval_on_side = true;
    }

    logGaussian = decltype(logGaussian)(log_gaussian_field_name,dl->node_scalar);
    numNodes = 0;

    gaussian = decltype(gaussian)(gaussian_field_name,dl->node_scalar);

    if (mean_is_field) {
      mean = decltype(mean)(mean_field_name,dl->node_scalar);
      this->addDependentField(mean);
    }
    else {
      RealType mean = p.get<RealType>("mean");
      RealType deviation = p.get<RealType>("deviation");
      a = log(mean/sqrt(1.+deviation*deviation));
      b = sqrt(log(1+deviation*deviation));
    }  
    this->addEvaluatedField(logGaussian);
    this->addDependentField(gaussian);
  
    this->setName("Log Gaussian " + log_gaussian_field_name + PHX::print<EvalT>());
  }

  void postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(logGaussian,fm);
    numNodes = logGaussian.extent(1);
    d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields(),false);
  }

protected:
  std::size_t numNodes;
  PHX::MDField<ScalarT,Cell,Node> logGaussian;
  PHX::MDField<const ScalarT,Cell,Node> gaussian;
  PHX::MDField<const RealType,Cell,Node> mean;
  RealType a, b;

  bool eval_on_side, mean_is_field;
  std::string sideSetName;
};

template<typename EvalT, typename Traits> class LogGaussianDistributedParameter;

// **************************************************************
// **************************************************************
// * Specializations
// **************************************************************
// **************************************************************


// **************************************************************
// Residual
// **************************************************************
template<typename Traits>
class LogGaussianDistributedParameter<PHAL::AlbanyTraits::Residual,Traits>
  : public LogGaussianDistributedParameterBase<PHAL::AlbanyTraits::Residual, Traits>  {

  public:
    typedef typename PHAL::AlbanyTraits::Residual::ScalarT   ScalarT;

    LogGaussianDistributedParameter (const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl) :
      LogGaussianDistributedParameterBase<PHAL::AlbanyTraits::Residual, Traits>(p, dl)
     {
     }

    void postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
    {
      LogGaussianDistributedParameterBase<PHAL::AlbanyTraits::Residual, Traits>::
        postRegistrationSetup(d, fm);
    }

    void evaluateFields(typename Traits::EvalData workset)
    {
      std::size_t n_cells;
      
      if (this->eval_on_side) {
        if (workset.sideSetViews->find(this->sideSetName)==workset.sideSetViews->end()) return;
        n_cells = workset.sideSetViews->at(this->sideSetName).size;
      } else {
        n_cells = workset.numCells;
      }

      for (std::size_t cell = 0; cell < n_cells; ++cell) {
        for (std::size_t node = 0; node < this->numNodes; ++node) {
          if (this->mean_is_field)
            (this->logGaussian)(cell, node) = exp(log((this->mean)(cell, node)) + (this->gaussian)(cell, node));
          else
            (this->logGaussian)(cell, node) = exp(this->a + this->b * (this->gaussian)(cell, node));
          //std::cout << "Residual logGaussian coeff = " << (this->logGaussian)(cell, node) << " mean = " << (this->mean)(cell, node) << " w = " <<  (this->gaussian)(cell, node) << std::endl;
        }
      }
    }
};

// **************************************************************
// Jacobian
// **************************************************************
template<typename Traits>
class LogGaussianDistributedParameter<PHAL::AlbanyTraits::Jacobian,Traits>
  : public LogGaussianDistributedParameterBase<PHAL::AlbanyTraits::Jacobian, Traits>  {

  public:
    typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT   ScalarT;

    LogGaussianDistributedParameter (const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl) :
      LogGaussianDistributedParameterBase<PHAL::AlbanyTraits::Jacobian, Traits>(p, dl)
     {
     }

    void postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
    {
      LogGaussianDistributedParameterBase<PHAL::AlbanyTraits::Jacobian, Traits>::
        postRegistrationSetup(d, fm);
    }

    void evaluateFields(typename Traits::EvalData workset)
    {
      std::size_t n_cells;
      
      if (this->eval_on_side) {
        if (workset.sideSetViews->find(this->sideSetName)==workset.sideSetViews->end()) return;
        n_cells = workset.sideSetViews->at(this->sideSetName).size;
      } else {
        n_cells = workset.numCells;
      }

      for (std::size_t cell = 0; cell < n_cells; ++cell) {
        for (std::size_t node = 0; node < this->numNodes; ++node) {
          if (this->mean_is_field)
            (this->logGaussian)(cell, node) = exp(log((this->mean)(cell, node)) + (this->gaussian)(cell, node));
          else
            (this->logGaussian)(cell, node) = exp(this->a + this->b * (this->gaussian)(cell, node));
          //std::cout << "Jacobian logGaussian coeff " << (this->logGaussian)(cell, node) << "mean = " << (this->mean)(cell, node) << " w = " <<  (this->gaussian)(cell, node) << " this->mean_is_field = " << this->mean_is_field << std::endl;
        }
      }
    }
};

// **************************************************************
// Tangent
// **************************************************************
template<typename Traits>
class LogGaussianDistributedParameter<PHAL::AlbanyTraits::Tangent,Traits>
  : public LogGaussianDistributedParameterBase<PHAL::AlbanyTraits::Tangent, Traits>  {

  public:
    typedef typename PHAL::AlbanyTraits::Tangent::ScalarT   ScalarT;

    LogGaussianDistributedParameter (const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl) :
      LogGaussianDistributedParameterBase<PHAL::AlbanyTraits::Tangent, Traits>(p, dl)
     {
     }

    void postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
    {
      LogGaussianDistributedParameterBase<PHAL::AlbanyTraits::Tangent, Traits>::
        postRegistrationSetup(d, fm);
    }

    void evaluateFields(typename Traits::EvalData workset)
    {
      std::size_t n_cells;
      
      if (this->eval_on_side) {
        if (workset.sideSetViews->find(this->sideSetName)==workset.sideSetViews->end()) return;
        n_cells = workset.sideSetViews->at(this->sideSetName).size;
      } else {
        n_cells = workset.numCells;
      }

      for (std::size_t cell = 0; cell < n_cells; ++cell) {
        for (std::size_t node = 0; node < this->numNodes; ++node) {
          if (this->mean_is_field)
            (this->logGaussian)(cell, node) = exp(log((this->mean)(cell, node)) + (this->gaussian)(cell, node));
          else
            (this->logGaussian)(cell, node) = exp(this->a + this->b * (this->gaussian)(cell, node));
          //std::cout << "Tangent logGaussian coeff " << (this->logGaussian)(cell, node) << " " <<  (this->gaussian)(cell, node) << std::endl;
        }
      }
    }
};

// **************************************************************
// DistParamDeriv
// **************************************************************
template<typename Traits>
class LogGaussianDistributedParameter<PHAL::AlbanyTraits::DistParamDeriv,Traits>
  : public LogGaussianDistributedParameterBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>  {

  public:
    typedef typename PHAL::AlbanyTraits::DistParamDeriv::ScalarT   ScalarT;

    LogGaussianDistributedParameter (const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl) :
      LogGaussianDistributedParameterBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p, dl)
     {
     }

    void postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
    {
      LogGaussianDistributedParameterBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
        postRegistrationSetup(d, fm);
    }

    void evaluateFields(typename Traits::EvalData workset)
    {
      std::size_t n_cells;
      
      if (this->eval_on_side) {
        if (workset.sideSetViews->find(this->sideSetName)==workset.sideSetViews->end()) return;
        n_cells = workset.sideSetViews->at(this->sideSetName).size;
      } else {
        n_cells = workset.numCells;
      }

      for (std::size_t cell = 0; cell < n_cells; ++cell) {
        for (std::size_t node = 0; node < this->numNodes; ++node) {
          if (this->mean_is_field)
            (this->logGaussian)(cell, node) = exp(log((this->mean)(cell, node)) + (this->gaussian)(cell, node));
          else
            (this->logGaussian)(cell, node) = exp(this->a + this->b * (this->gaussian)(cell, node));
          //std::cout << "DistParamDeriv logGaussian coeff " << (this->logGaussian)(cell, node) << " " <<  (this->gaussian)(cell, node) << std::endl;
        }
      }
    }
};

// **************************************************************
// HessianVec
// **************************************************************
template<typename Traits>
class LogGaussianDistributedParameter<PHAL::AlbanyTraits::HessianVec,Traits>
  : public LogGaussianDistributedParameterBase<PHAL::AlbanyTraits::HessianVec, Traits>  {

  public:
    typedef typename PHAL::AlbanyTraits::HessianVec::ScalarT   ScalarT;
 
    LogGaussianDistributedParameter (const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl) :
      LogGaussianDistributedParameterBase<PHAL::AlbanyTraits::HessianVec, Traits>(p, dl)
     {
     }

    void postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
    {
      LogGaussianDistributedParameterBase<PHAL::AlbanyTraits::HessianVec, Traits>::
        postRegistrationSetup(d, fm);
    }

    void evaluateFields(typename Traits::EvalData workset)
    {
      std::size_t n_cells;
      
      if (this->eval_on_side) {
        if (workset.sideSetViews->find(this->sideSetName)==workset.sideSetViews->end()) return;
        n_cells = workset.sideSetViews->at(this->sideSetName).size;
      } else {
        n_cells = workset.numCells;
      }

      for (std::size_t cell = 0; cell < n_cells; ++cell) {
        for (std::size_t node = 0; node < this->numNodes; ++node) {
          if (this->mean_is_field)
            (this->logGaussian)(cell, node) = exp(log((this->mean)(cell, node)) + (this->gaussian)(cell, node));
          else
            (this->logGaussian)(cell, node) = exp(this->a + this->b * (this->gaussian)(cell, node));
          //std::cout << "HessianVec logGaussian coeff " << (this->logGaussian)(cell, node) << " " <<  (this->gaussian)(cell, node) << std::endl;
        }
      }
    }
};

}  // Namespace PHAL

#endif  // PHAL_LOG_GAUSSIAN_COMBINATION_PARAMETER_HPP
