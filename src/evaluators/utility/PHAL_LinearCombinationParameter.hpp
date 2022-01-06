//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_LINEAR_COMBINATION_PARAMETER_HPP
#define PHAL_LINEAR_COMBINATION_PARAMETER_HPP

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
/// LinearCombinationParameterBase
///
template<typename EvalT, typename Traits>
class LinearCombinationParameterBase : 
  public PHX::EvaluatorWithBaseImpl<Traits>,
  public PHX::EvaluatorDerived<EvalT, Traits>
{
 private:
  typedef typename EvalT::ParamScalarT ParamScalarT;

 public:
  typedef typename EvalT::ScalarT   ScalarT;
  //typedef ParamNameEnum             EnumType;

  LinearCombinationParameterBase (const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl)
  {
    std::string field_name   = p.get<std::string>("Parameter Name");
    numModes = p.get<std::size_t>("Number of modes");

    eval_on_side = false;
    if (p.isParameter("Side Set Name")) {
      sideSetName = p.get<std::string>("Side Set Name");
      eval_on_side = true;
    }

    scale = false;
    if (p.isParameter("Scalar scale")) {
      scalar_scale = p.get<Teuchos::Array<double> >("Scalar scale");
      scale = true;
    }

    val = decltype(val)(field_name,dl->node_scalar);
    numNodes = 0;

    for (std::size_t i = 0; i < numModes; ++i) {
      std::string coefficient_name   = p.sublist(util::strint("Mode",i)).get<std::string>("Coefficient Name");
      std::string mode_name          = p.sublist(util::strint("Mode",i)).get<std::string>("Mode Name");

      coefficients_as_field.push_back(PHX::MDField<const ScalarT,Dim>(coefficient_name,dl->shared_param));
      modes_val.push_back(PHX::MDField<const RealType,Cell,Node>(mode_name,dl->node_scalar));
    }

    this->addEvaluatedField(val);
    for (std::size_t i = 0; i < numModes; ++i) {
      this->addDependentField(coefficients_as_field[i]);
      this->addDependentField(modes_val[i]);
    }
    this->setName("Linear Combination " + field_name + PHX::print<EvalT>());
  }

  void postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(val,fm);
    numNodes = val.extent(1);
    d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields(),false);
  }

protected:
  std::size_t numModes, numNodes;
  PHX::MDField<ScalarT,Cell,Node> val;
  std::vector<PHX::MDField<const ScalarT,Dim>>   coefficients_as_field; // or ParamScalarT
  std::vector<PHX::MDField<const RealType,Cell,Node>>   modes_val;

  bool eval_on_side;
  bool scale;
  std::string sideSetName;
  Teuchos::Array<double> scalar_scale;
};

template<typename EvalT, typename Traits> class LinearCombinationParameter;

// **************************************************************
// **************************************************************
// * Specializations
// **************************************************************
// **************************************************************


// **************************************************************
// Residual
// **************************************************************
template<typename Traits>
class LinearCombinationParameter<PHAL::AlbanyTraits::Residual,Traits>
  : public LinearCombinationParameterBase<PHAL::AlbanyTraits::Residual, Traits>  {

  public:
    typedef typename PHAL::AlbanyTraits::Residual::ScalarT   ScalarT;

    LinearCombinationParameter (const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl) :
      LinearCombinationParameterBase<PHAL::AlbanyTraits::Residual, Traits>(p, dl)
     {
     }

    void postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
    {
      LinearCombinationParameterBase<PHAL::AlbanyTraits::Residual, Traits>::
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

      // reset to zero first:

      for (std::size_t cell = 0; cell < n_cells; ++cell) {
        for (std::size_t node = 0; node < this->numNodes; ++node) {
          (this->val)(cell, node) = 0.;
        }
      }


      for (std::size_t i = 0; i < this->numModes; ++i) {
        for (std::size_t cell = 0; cell < n_cells; ++cell) {
          for (std::size_t node = 0; node < this->numNodes; ++node) {
            if (this->scale) {
              (this->val)(cell, node) +=
                this->scalar_scale[i] * (this->coefficients_as_field[i])(0) * (this->modes_val[i])(cell, node);

              //std::cout << "Residual coeff " << i << " = " << this->scalar_scale[i] << " * " << (this->coefficients_as_field[i])(0) << " * " << (this->modes_val[i])(cell, node) << std::endl;             
            }
            else {
              (this->val)(cell, node) +=
                (this->coefficients_as_field[i])(0) * (this->modes_val[i])(cell, node);
              
              //std::cout << "Unscaled Residual coeff " << i << " = " <<  (this->coefficients_as_field[i])(0) << " * " << (this->modes_val[i])(cell, node) << std::endl;
            }
          }
        }
      }
    }
};

// **************************************************************
// Jacobian
// **************************************************************
template<typename Traits>
class LinearCombinationParameter<PHAL::AlbanyTraits::Jacobian,Traits>
  : public LinearCombinationParameterBase<PHAL::AlbanyTraits::Jacobian, Traits>  {

  public:
    typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT   ScalarT;

    LinearCombinationParameter (const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl) :
      LinearCombinationParameterBase<PHAL::AlbanyTraits::Jacobian, Traits>(p, dl)
     {
     }

    void postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
    {
      LinearCombinationParameterBase<PHAL::AlbanyTraits::Jacobian, Traits>::
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

      // reset to zero first:

      for (std::size_t cell = 0; cell < n_cells; ++cell) {
        for (std::size_t node = 0; node < this->numNodes; ++node) {
          (this->val)(cell, node) = 0.;
        }
      }


      for (std::size_t i = 0; i < this->numModes; ++i) {
        for (std::size_t cell = 0; cell < n_cells; ++cell) {
          for (std::size_t node = 0; node < this->numNodes; ++node) {
            if (this->scale) {
              (this->val)(cell, node) +=
                this->scalar_scale[i] * (this->coefficients_as_field[i])(0) * (this->modes_val[i])(cell, node);              
            }
            else {
              (this->val)(cell, node) +=
                (this->coefficients_as_field[i])(0) * (this->modes_val[i])(cell, node);
              
              //std::cout << "Jacobian coeff " << i << " = " <<  (this->coefficients_as_field[i])(0) << " * " << (this->modes_val[i])(cell, node) << std::endl;
            }
          }
        }
      }
    }
};

// **************************************************************
// Tangent
// **************************************************************
template<typename Traits>
class LinearCombinationParameter<PHAL::AlbanyTraits::Tangent,Traits>
  : public LinearCombinationParameterBase<PHAL::AlbanyTraits::Tangent, Traits>  {

  public:
    typedef typename PHAL::AlbanyTraits::Tangent::ScalarT   ScalarT;

    LinearCombinationParameter (const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl) :
      LinearCombinationParameterBase<PHAL::AlbanyTraits::Tangent, Traits>(p, dl)
     {
     }

    void postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
    {
      LinearCombinationParameterBase<PHAL::AlbanyTraits::Tangent, Traits>::
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

      // reset to zero first:

      for (std::size_t cell = 0; cell < n_cells; ++cell) {
        for (std::size_t node = 0; node < this->numNodes; ++node) {
          (this->val)(cell, node) = 0.;
        }
      }


      for (std::size_t i = 0; i < this->numModes; ++i) {
        for (std::size_t cell = 0; cell < n_cells; ++cell) {
          for (std::size_t node = 0; node < this->numNodes; ++node) {
            if (this->scale) {
              (this->val)(cell, node) +=
                this->scalar_scale[i] * (this->coefficients_as_field[i])(0) * (this->modes_val[i])(cell, node);              
            }
            else {
              (this->val)(cell, node) +=
                (this->coefficients_as_field[i])(0) * (this->modes_val[i])(cell, node);
              
              //std::cout << "Tangent coeff " << i << " = " <<  (this->coefficients_as_field[i])(0) << " * " << (this->modes_val[i])(cell, node) << std::endl;
            }
          }
        }
      }
    }
};

// **************************************************************
// DistParamDeriv
// **************************************************************
template<typename Traits>
class LinearCombinationParameter<PHAL::AlbanyTraits::DistParamDeriv,Traits>
  : public LinearCombinationParameterBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>  {

  public:
    typedef typename PHAL::AlbanyTraits::DistParamDeriv::ScalarT   ScalarT;

    LinearCombinationParameter (const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl) :
      LinearCombinationParameterBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p, dl)
     {
     }

    void postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
    {
      LinearCombinationParameterBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
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

      // reset to zero first:

      for (std::size_t cell = 0; cell < n_cells; ++cell) {
        for (std::size_t node = 0; node < this->numNodes; ++node) {
          (this->val)(cell, node) = 0.;
        }
      }


      for (std::size_t i = 0; i < this->numModes; ++i) {
        for (std::size_t cell = 0; cell < n_cells; ++cell) {
          for (std::size_t node = 0; node < this->numNodes; ++node) {
            if (this->scale) {
              (this->val)(cell, node) +=
                this->scalar_scale[i] * (this->coefficients_as_field[i])(0) * (this->modes_val[i])(cell, node);              
            }
            else {
              (this->val)(cell, node) +=
                (this->coefficients_as_field[i])(0) * (this->modes_val[i])(cell, node);
              
              //std::cout << "DistParamDeriv coeff " << i << " = " <<  (this->coefficients_as_field[i])(0) << " * " << (this->modes_val[i])(cell, node) << std::endl;
            }
          }
        }
      }
    }
};

// **************************************************************
// HessianVec
// **************************************************************
template<typename Traits>
class LinearCombinationParameter<PHAL::AlbanyTraits::HessianVec,Traits>
  : public LinearCombinationParameterBase<PHAL::AlbanyTraits::HessianVec, Traits>  {

  public:
    typedef typename PHAL::AlbanyTraits::HessianVec::ScalarT   ScalarT;
 
    LinearCombinationParameter (const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl) :
      LinearCombinationParameterBase<PHAL::AlbanyTraits::HessianVec, Traits>(p, dl)
     {
     }

    void postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
    {
      LinearCombinationParameterBase<PHAL::AlbanyTraits::HessianVec, Traits>::
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

      // reset to zero first:

      for (std::size_t cell = 0; cell < n_cells; ++cell) {
        for (std::size_t node = 0; node < this->numNodes; ++node) {
          (this->val)(cell, node) = 0.;
        }
      }


      for (std::size_t i = 0; i < this->numModes; ++i) {
        for (std::size_t cell = 0; cell < n_cells; ++cell) {
          for (std::size_t node = 0; node < this->numNodes; ++node) {
            if (this->scale) {
              (this->val)(cell, node) +=
                this->scalar_scale[i] * (this->coefficients_as_field[i])(0) * (this->modes_val[i])(cell, node);              
            }
            else {
              (this->val)(cell, node) +=
                (this->coefficients_as_field[i])(0) * (this->modes_val[i])(cell, node);
              
              //std::cout << "HessianVec coeff " << i << " = " <<  (this->coefficients_as_field[i])(0) << " * " << (this->modes_val[i])(cell, node) << std::endl;
            }
          }
        }
      }
    }
};

}  // Namespace PHAL

#endif  // PHAL_LINEAR_COMBINATION_PARAMETER_HPP
