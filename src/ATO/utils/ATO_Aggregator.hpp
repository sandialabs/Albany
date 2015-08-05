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

  Aggregator(){}
  Aggregator(const Teuchos::ParameterList& aggregatorParams);
  virtual ~Aggregator(){};

  virtual void Evaluate()=0;

  virtual std::string getOutputObjectiveName(){return outputObjectiveName;}
  virtual std::string getOutputDerivativeName(){return outputDerivativeName;}

  virtual void SetInputVariables(const std::vector<SolverSubSolver>& subProblems){};
  virtual void SetInputVariables(const std::vector<SolverSubSolver>& subProblems,
                                 const std::map<std::string, Teuchos::RCP<const Epetra_Vector> > gMap,
                                 const std::map<std::string, Teuchos::RCP<Epetra_MultiVector> > dgdpMap){};
  void SetCommunicator(const Teuchos::RCP<const Epetra_Comm>& _comm){comm = _comm;}
  void SetOutputVariables(Teuchos::RCP<double> g, Teuchos::RCP<Epetra_Vector> dgdp)
         {gAggregated = g; dgdpAggregated = dgdp;}

protected:

  void parse(const Teuchos::ParameterList& aggregatorParams);

  Teuchos::Array<std::string> aggregatedObjectivesNames;
  Teuchos::Array<std::string> aggregatedDerivativesNames;
  std::string outputObjectiveName;
  std::string outputDerivativeName;

  Teuchos::RCP<double> gAggregated;
  Teuchos::RCP<Epetra_Vector> dgdpAggregated;

  Teuchos::RCP<Albany::Application> outApp;
  Teuchos::RCP<const Epetra_Comm> comm;

  Teuchos::Array<double> normalize;
};

class Aggregator_StateVarBased : public virtual Aggregator {
 public:
  Aggregator_StateVarBased(){}
  void SetInputVariables(const std::vector<SolverSubSolver>& subProblems);
 protected:
  typedef struct {
    std::string name;
    Teuchos::RCP<Albany::Application> app;
  } SubVariable;

  std::vector<SubVariable> objectives;
  std::vector<SubVariable> derivatives;


};

class Aggregator_DistParamBased : public virtual Aggregator {
 public:
  Aggregator_DistParamBased(){}
  void SetInputVariables(const std::vector<SolverSubSolver>& subProblems,
                         const std::map<std::string, Teuchos::RCP<const Epetra_Vector> > gMap,
                         const std::map<std::string, Teuchos::RCP<Epetra_MultiVector> > dgdpMap);
 protected:
  typedef struct { std::string name; Teuchos::RCP<const Epetra_Vector> value; } SubObjective;
  typedef struct { std::string name; Teuchos::RCP<Epetra_MultiVector> value; } SubDerivative;

  std::vector<SubObjective> objectives;
  std::vector<SubDerivative> derivatives;

};

class Aggregator_Scaled : public virtual Aggregator,
                          public virtual Aggregator_StateVarBased {
 public:
  Aggregator_Scaled(){}
  Aggregator_Scaled(const Teuchos::ParameterList& aggregatorParams);
  virtual void Evaluate();
 protected:
  Teuchos::Array<double> weights;
};


class Aggregator_Uniform : public Aggregator_Scaled {
 public:
  Aggregator_Uniform(const Teuchos::ParameterList& aggregatorParams);
};

class Aggregator_DistScaled : public virtual Aggregator,
                              public virtual Aggregator_DistParamBased {
 public:
  Aggregator_DistScaled(){}
  Aggregator_DistScaled(const Teuchos::ParameterList& aggregatorParams);
  void Evaluate();
 protected:
  Teuchos::Array<double> weights;
};
class Aggregator_DistUniform : public Aggregator_DistScaled {
 public:
  Aggregator_DistUniform(const Teuchos::ParameterList& aggregatorParams);
};


class AggregatorFactory {
public:
  Teuchos::RCP<Aggregator> create(const Teuchos::ParameterList& aggregatorParams,
                                  std::string entityType);
};


}
#endif
