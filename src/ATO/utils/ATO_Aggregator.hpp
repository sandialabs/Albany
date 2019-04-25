//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ATO_AGGREGATOR_HPP
#define ATO_AGGREGATOR_HPP

// Albany includes
#include "Albany_ThyraTypes.hpp"
#include "Albany_CommTypes.hpp"

// Trilinos includes
#include "Teuchos_ParameterList.hpp"

// System includes
#include <string>
#include <vector>

// Albany forward declarations
namespace Albany {
class Application;
} // namespace Albany

namespace ATO {

class SolverSubSolver;

/** \brief Combines values.

    This class reads values from response functions and combines them into
  a single value for optimization.

*/
class Aggregator
{

public:
  Aggregator () = default;
  Aggregator (const Teuchos::ParameterList& aggregatorParams, int nTopos);
  virtual ~Aggregator () = default;

  const std::string& getOutputValueName ()      const { return outputValueName;      }
  const std::string& getOutputDerivativeName () const { return outputDerivativeName; }

  virtual void Evaluate () = 0;

  virtual void SetInputVariables (const std::vector<SolverSubSolver>& /* subProblems */) {
    // Do nothing by default
  }
  virtual void SetInputVariables (const std::vector<SolverSubSolver>& /* subProblems */,
                                  const std::map<std::string, std::vector<Teuchos::RCP<const Thyra_Vector>>>& /* valueMap */,
                                  const std::map<std::string, std::vector<Teuchos::RCP<Thyra_MultiVector>>>& /* derivMap */) {
    // Do nothing by default
  }

  void SetCommunicator (const Teuchos::RCP<const Teuchos_Comm>& _comm) {
    comm = _comm;
  }

  void SetOutputVariables (Teuchos::RCP<double> g,
                           Teuchos::Array<Teuchos::RCP<Thyra_Vector> > deriv) {
    valueAggregated = g;
    derivAggregated = deriv;
  }

protected:

  void parse (const Teuchos::ParameterList& aggregatorParams);

  Teuchos::Array<std::string> aggregatedValuesNames;
  Teuchos::Array<std::string> aggregatedDerivativesNames;
  std::string outputValueName;
  std::string outputDerivativeName;

  Teuchos::RCP<double> valueAggregated;
  Teuchos::Array<Teuchos::RCP<Thyra_Vector> > derivAggregated;

  Teuchos::RCP<Albany::Application> outApp;
  Teuchos::RCP<const Teuchos_Comm> comm;

  Teuchos::Array<double> normalize;
  double shiftValueAggregated;
  double scaleValueAggregated;

  int numTopologies;

  std::string normalizeMethod;
  double maxScale;
  int iteration;
  int rampInterval;
};

// ===================== Derived types ======================= //

/******************************************************************************/
class Aggregator_StateVarBased : public virtual Aggregator {
public:
  Aggregator_StateVarBased () = default;
  void SetInputVariables (const std::vector<SolverSubSolver>& subProblems) override;

protected:
  struct SubValueType {
    std::string name;
    Teuchos::RCP<Albany::Application> app;
  };
  struct SubDerivativeType {
    Teuchos::Array<std::string> name;
    Teuchos::RCP<Albany::Application> app;
  };

  std::vector<SubValueType>       values;
  std::vector<SubDerivativeType>  derivatives;
};
/******************************************************************************/


/******************************************************************************/
class Aggregator_DistParamBased : public virtual Aggregator {
public:
  Aggregator_DistParamBased () = default;
  Aggregator_DistParamBased (const Teuchos::ParameterList& aggregatorParams, int nTopos);
  void SetInputVariables (const std::vector<SolverSubSolver>& subProblems,
                          const std::map<std::string, std::vector<Teuchos::RCP<const Thyra_Vector>>>& valueMap,
                          const std::map<std::string, std::vector<Teuchos::RCP<Thyra_MultiVector>>>&  derivMap) override;
protected:
  struct SubValueType {
    std::string name;
    std::vector<Teuchos::RCP<const Thyra_Vector>> value;
  };
  struct SubDerivativeType {
    std::string name;
    std::vector<Teuchos::RCP<Thyra_MultiVector>> value;
  };

  std::vector<SubValueType>       values;
  std::vector<SubDerivativeType>  derivatives;

  double sum(std::vector<Teuchos::RCP<const Thyra_Vector>> valVector, int index);
  std::vector<double> sum(std::vector<Teuchos::RCP<const Thyra_Vector>> valVector);
};
/******************************************************************************/


/******************************************************************************/
class Aggregator_Scaled : public virtual Aggregator_StateVarBased {
public:
  Aggregator_Scaled () = default;
  Aggregator_Scaled (const Teuchos::ParameterList& aggregatorParams, int nTopos);
  void Evaluate () override;
protected:
  Teuchos::Array<double> weights;
};
/******************************************************************************/

/******************************************************************************/
template <typename CompareType>
class Aggregator_Extremum : public virtual Aggregator_StateVarBased {
public:
  Aggregator_Extremum () = default;
  Aggregator_Extremum (const Teuchos::ParameterList& aggregatorParams, int nTopos);
  void Evaluate () override;
protected:
  CompareType compare;
};
/******************************************************************************/

/******************************************************************************/
class Aggregator_Uniform : public virtual Aggregator,
                           public virtual Aggregator_Scaled
{
public:
  Aggregator_Uniform () = default;
  Aggregator_Uniform (const Teuchos::ParameterList& aggregatorParams, int nTopos);
};
/******************************************************************************/

/******************************************************************************/
class Aggregator_DistScaled : public virtual Aggregator_DistParamBased {
public:
  Aggregator_DistScaled () = default;
  Aggregator_DistScaled (const Teuchos::ParameterList& aggregatorParams, int nTopos);
  void Evaluate () override;
protected:
  Teuchos::Array<double> weights;
};
/******************************************************************************/

/******************************************************************************/
class Aggregator_Homogenized : public virtual Aggregator_DistParamBased {
public:
  Aggregator_Homogenized () = default;
  Aggregator_Homogenized (const Teuchos::ParameterList& aggregatorParams, int nTopos);
  void Evaluate () override;
protected:
  Teuchos::Array<double> m_assumedState;
  bool m_reciprocate;
  double m_initialValue;
};
/******************************************************************************/

/******************************************************************************/
template <typename CompareType>
class Aggregator_DistExtremum : public virtual Aggregator_DistParamBased {
 public:
  Aggregator_DistExtremum () = default;
  Aggregator_DistExtremum (const Teuchos::ParameterList& aggregatorParams, int nTopos);
  void Evaluate () override;
 protected:
  CompareType compare;
};
/******************************************************************************/

/******************************************************************************/
class Aggregator_DistUniform : public Aggregator_DistScaled {
 public:
  Aggregator_DistUniform (const Teuchos::ParameterList& aggregatorParams, int nTopos);
};
/******************************************************************************/


/******************************************************************************/
class AggregatorFactory {
public:
  AggregatorFactory() = delete;
  static Teuchos::RCP<Aggregator> create (const Teuchos::ParameterList& aggregatorParams,
                                          const std::string& entityType, int nTopos);
};
/******************************************************************************/

} // namespace ATO

#endif // ATO_AGGREGATOR_HPP
