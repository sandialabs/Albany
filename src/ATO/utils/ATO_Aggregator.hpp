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
/** \brief Combines objectives.

    This class reads objectives from response functions and combines them into
  a single objective for optimization.

*/
{

public:

  Aggregator(const Teuchos::ParameterList& aggregatorParams);
  virtual ~Aggregator(){};

  virtual void Evaluate()=0;

  virtual std::string getOutputObjectiveName(){return outputObjectiveName;}
  virtual std::string getOutputDerivativeName(){return outputDerivativeName;}

  void SetInputVariables(const std::vector<SolverSubSolver>& subProblems);
  void SetCommunicator(const Teuchos::RCP<const Epetra_Comm>& _comm){comm = _comm;}

  typedef struct {
    std::string name;
    Teuchos::RCP<Albany::Application> app;
  } SubVariable;

protected:

  void parse(const Teuchos::ParameterList& aggregatorParams);

  Teuchos::Array<std::string> aggregatedObjectivesNames;
  Teuchos::Array<std::string> aggregatedDerivativesNames;
  std::string outputObjectiveName;
  std::string outputDerivativeName;

  std::vector<SubVariable> objectives;
  std::vector<SubVariable> derivatives;

  Teuchos::RCP<Albany::Application> outApp;
  Teuchos::RCP<const Epetra_Comm> comm;

};

class Aggregator_Scaled : public Aggregator {
 public:
  Aggregator_Scaled(const Teuchos::ParameterList& aggregatorParams);
  virtual void Evaluate();
 private:
  Teuchos::Array<double> weights;

  bool shiftToZero;
};


class Aggregator_Uniform : public Aggregator {
 public:
  Aggregator_Uniform(const Teuchos::ParameterList& aggregatorParams);
  virtual void Evaluate();
};


class Aggregator_PassThru : public Aggregator {
 public:
  Aggregator_PassThru(const Teuchos::ParameterList& aggregatorParams);
  void Evaluate(){}
};


class AggregatorFactory {
public:
  Teuchos::RCP<Aggregator> create(const Teuchos::ParameterList& aggregatorParams);
};


}
#endif
