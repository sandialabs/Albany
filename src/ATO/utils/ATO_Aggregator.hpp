//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ATO_Aggregator_HPP
#define ATO_Aggregator_HPP

#include "Albany_Application.hpp"

#include <string>
#include <vector>

#include "Teuchos_ParameterList.hpp"


namespace ATO {

class SolverSubSolver;

class Aggregator 
/** \brief Combines values.

    This class reads values from response functions and combines them into
  a single value for optimization.

*/
{

public:

  Aggregator(){}
  Aggregator(const Teuchos::ParameterList& aggregatorParams);
  virtual ~Aggregator(){};

  virtual void Evaluate()=0;

  virtual std::string getOutputValueName(){return outputValueName;}
  virtual std::string getOutputDerivativeName(){return outputDerivativeName;}

  virtual void SetInputVariables(const std::vector<SolverSubSolver>& subProblems){};
  virtual void SetInputVariables(const std::vector<SolverSubSolver>& subProblems,
                                 const std::map<std::string, Teuchos::RCP<const Epetra_Vector> > valueMap,
                                 const std::map<std::string, Teuchos::RCP<Epetra_MultiVector> > derivMap){};
  void SetCommunicator(const Teuchos::RCP<const Epetra_Comm>& _comm){comm = _comm;}
  void SetOutputVariables(Teuchos::RCP<double> g, Teuchos::RCP<Epetra_Vector> deriv)
         {valueAggregated = g; derivAggregated = deriv;}

protected:

  void parse(const Teuchos::ParameterList& aggregatorParams);

  Teuchos::Array<std::string> aggregatedValuesNames;
  Teuchos::Array<std::string> aggregatedDerivativesNames;
  std::string outputValueName;
  std::string outputDerivativeName;

  Teuchos::RCP<double> valueAggregated;
  Teuchos::RCP<Epetra_Vector> derivAggregated;

  Teuchos::RCP<Albany::Application> outApp;
  Teuchos::RCP<const Epetra_Comm> comm;

  Teuchos::Array<double> normalize;
  double shiftValueAggregated;
};

/******************************************************************************/
class Aggregator_StateVarBased : public virtual Aggregator {
 public:
  Aggregator_StateVarBased(){}
  void SetInputVariables(const std::vector<SolverSubSolver>& subProblems);
 protected:
  typedef struct {
    std::string name;
    Teuchos::RCP<Albany::Application> app;
  } SubVariable;

  std::vector<SubVariable> values;
  std::vector<SubVariable> derivatives;
};
/******************************************************************************/


/******************************************************************************/
class Aggregator_DistParamBased : public virtual Aggregator {
 public:
  Aggregator_DistParamBased(){}
  void SetInputVariables(const std::vector<SolverSubSolver>& subProblems,
                         const std::map<std::string, Teuchos::RCP<const Epetra_Vector> > valueMap,
                         const std::map<std::string, Teuchos::RCP<Epetra_MultiVector> > derivMap);
 protected:
  typedef struct { std::string name; Teuchos::RCP<const Epetra_Vector> value; } SubValue;
  typedef struct { std::string name; Teuchos::RCP<Epetra_MultiVector> value; } SubDerivative;

  std::vector<SubValue> values;
  std::vector<SubDerivative> derivatives;
};
/******************************************************************************/


/******************************************************************************/
class Aggregator_Scaled : public virtual Aggregator,
                          public virtual Aggregator_StateVarBased {
 public:
  Aggregator_Scaled(){}
  Aggregator_Scaled(const Teuchos::ParameterList& aggregatorParams);
  virtual void Evaluate();
 protected:
  Teuchos::Array<double> weights;
};
/******************************************************************************/

/******************************************************************************/
template <typename C>
class Aggregator_Extremum : public virtual Aggregator,
                            public virtual Aggregator_StateVarBased {
 public:
  Aggregator_Extremum(){}
  Aggregator_Extremum(const Teuchos::ParameterList& aggregatorParams);
  virtual void Evaluate();
 protected:
  C compare;
};
/******************************************************************************/

/******************************************************************************/
class Aggregator_Uniform : public Aggregator_Scaled {
 public:
  Aggregator_Uniform(const Teuchos::ParameterList& aggregatorParams);
};
/******************************************************************************/

/******************************************************************************/
class Aggregator_DistScaled : public virtual Aggregator,
                              public virtual Aggregator_DistParamBased {
 public:
  Aggregator_DistScaled(){}
  Aggregator_DistScaled(const Teuchos::ParameterList& aggregatorParams);
  void Evaluate();
 protected:
  Teuchos::Array<double> weights;
};
/******************************************************************************/

/******************************************************************************/
template <typename C>
class Aggregator_DistExtremum : public virtual Aggregator,
                                public virtual Aggregator_DistParamBased {
 public:
  Aggregator_DistExtremum(){}
  Aggregator_DistExtremum(const Teuchos::ParameterList& aggregatorParams);
  void Evaluate();
 protected:
  C compare;
};
/******************************************************************************/

/******************************************************************************/
class Aggregator_DistUniform : public Aggregator_DistScaled {
 public:
  Aggregator_DistUniform(const Teuchos::ParameterList& aggregatorParams);
};
/******************************************************************************/


/******************************************************************************/
class AggregatorFactory {
public:
  Teuchos::RCP<Aggregator> create(const Teuchos::ParameterList& aggregatorParams,
                                  std::string entityType);
};
/******************************************************************************/


}
#endif
