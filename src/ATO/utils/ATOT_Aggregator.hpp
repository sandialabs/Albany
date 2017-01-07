//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ATOT_Aggregator_HPP
#define ATOT_Aggregator_HPP

#include "Albany_Application.hpp"

#include <string>
#include <vector>

#include "Teuchos_ParameterList.hpp"


namespace ATOT {

class SolverSubSolver;

class Aggregator 
/** \brief Combines values.

    This class reads values from response functions and combines them into
  a single value for optimization.

*/
{

public:

  Aggregator(){}
  Aggregator(const Teuchos::ParameterList& aggregatorParams, int nTopos);
  virtual ~Aggregator(){};

  virtual void Evaluate()=0;
  virtual void EvaluateT()=0;

  virtual std::string getOutputValueName(){return outputValueName;}
  virtual std::string getOutputDerivativeName(){return outputDerivativeName;}

  virtual void SetInputVariablesT(const std::vector<SolverSubSolver>& subProblems){};
  virtual void SetInputVariablesT(const std::vector<SolverSubSolver>& subProblems,
                                 const std::map<std::string, Teuchos::RCP<const Tpetra_Vector> > valueMap,
                                 const std::map<std::string, Teuchos::RCP<Tpetra_MultiVector> > derivMap){};
  void SetCommunicator(const Teuchos::RCP<const Teuchos_Comm>& _comm){comm = _comm;}
  void SetOutputVariables(Teuchos::RCP<double> g, Teuchos::Array<Teuchos::RCP<Epetra_Vector> > deriv)
         {valueAggregated = g; derivAggregated = deriv;}
  void SetOutputVariablesT(Teuchos::RCP<double> g, Teuchos::Array<Teuchos::RCP<Tpetra_Vector> > derivT)
         {valueAggregated = g; derivAggregatedT = derivT;}

protected:

  void parse(const Teuchos::ParameterList& aggregatorParams);

  Teuchos::Array<std::string> aggregatedValuesNames;
  Teuchos::Array<std::string> aggregatedDerivativesNames;
  std::string outputValueName;
  std::string outputDerivativeName;

  Teuchos::RCP<double> valueAggregated;
  Teuchos::Array<Teuchos::RCP<Epetra_Vector> > derivAggregated;
  Teuchos::Array<Teuchos::RCP<Tpetra_Vector> > derivAggregatedT;

  Teuchos::RCP<Albany::Application> outApp;
  Teuchos::RCP<const Teuchos_Comm> comm;

  Teuchos::Array<double> normalize;
  double shiftValueAggregated;
  double scaleValueAggregated;

  int numTopologies;

};

/******************************************************************************/
class Aggregator_StateVarBased : public virtual Aggregator {
 public:
  Aggregator_StateVarBased(){}
  void SetInputVariablesT(const std::vector<SolverSubSolver>& subProblems);
 protected:
  typedef struct { std::string name; Teuchos::RCP<Albany::Application> app; } SubValue;
  typedef struct { Teuchos::Array<std::string> name; Teuchos::RCP<Albany::Application> app; } SubDerivative;
  typedef struct { std::string name; Teuchos::RCP<Albany::Application> app; } SubValueT;
  typedef struct { Teuchos::Array<std::string> name; Teuchos::RCP<Albany::Application> app; } SubDerivativeT;

  std::vector<SubValue> values;
  std::vector<SubDerivative> derivatives;
  std::vector<SubValueT> valuesT;
  std::vector<SubDerivativeT> derivativesT;
};
/******************************************************************************/


/******************************************************************************/
class Aggregator_DistParamBased : public virtual Aggregator {
 public:
  Aggregator_DistParamBased(){}
  void SetInputVariablesT(const std::vector<SolverSubSolver>& subProblems,
                          const std::map<std::string, Teuchos::RCP<const Tpetra_Vector> > valueMap,
                          const std::map<std::string, Teuchos::RCP<Tpetra_MultiVector> > derivMap);
 protected:
  typedef struct { std::string name; Teuchos::RCP<const Epetra_Vector> value; } SubValue;
  typedef struct { std::string name; Teuchos::RCP<const Tpetra_Vector> value; } SubValueT;
  typedef struct { std::string name; Teuchos::RCP<Epetra_MultiVector> value; } SubDerivative;
  typedef struct { std::string name; Teuchos::RCP<Tpetra_MultiVector> value; } SubDerivativeT;

  std::vector<SubValue> values;
  std::vector<SubDerivative> derivatives;
  std::vector<SubValueT> valuesT;
  std::vector<SubDerivativeT> derivativesT;
};
/******************************************************************************/


/******************************************************************************/
class Aggregator_Scaled : public virtual Aggregator,
                          public virtual Aggregator_StateVarBased {
 public:
  Aggregator_Scaled(){}
  Aggregator_Scaled(const Teuchos::ParameterList& aggregatorParams, int nTopos);
  virtual void Evaluate();
  virtual void EvaluateT();
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
  Aggregator_Extremum(const Teuchos::ParameterList& aggregatorParams, int nTopos);
  virtual void Evaluate();
  virtual void EvaluateT();
 protected:
  C compare;
};
/******************************************************************************/

/******************************************************************************/
class Aggregator_Uniform : public Aggregator_Scaled {
 public:
  Aggregator_Uniform(const Teuchos::ParameterList& aggregatorParams, int nTopos);
};
/******************************************************************************/

/******************************************************************************/
class Aggregator_DistScaled : public virtual Aggregator,
                              public virtual Aggregator_DistParamBased {
 public:
  Aggregator_DistScaled(){}
  Aggregator_DistScaled(const Teuchos::ParameterList& aggregatorParams, int nTopos);
  void Evaluate();
  void EvaluateT();
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
  Aggregator_DistExtremum(const Teuchos::ParameterList& aggregatorParams, int nTopos);
  void Evaluate();
  void EvaluateT();
 protected:
  C compare;
};
/******************************************************************************/

/******************************************************************************/
class Aggregator_DistUniform : public Aggregator_DistScaled {
 public:
  Aggregator_DistUniform(const Teuchos::ParameterList& aggregatorParams, int nTopos);
};
/******************************************************************************/


/******************************************************************************/
class AggregatorFactory {
public:
  Teuchos::RCP<Aggregator> create(const Teuchos::ParameterList& aggregatorParams,
                                  std::string entityType, int nTopos);
};
/******************************************************************************/


}
#endif
