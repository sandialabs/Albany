//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_PROBLEM_FACTORY_HPP
#define LANDICE_PROBLEM_FACTORY_HPP

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"

#include "Albany_AbstractProblem.hpp"


namespace LandIce
{

  /*!
   * \brief A factory class to instantiate LandIce objects
   */

class ProblemFactory {
public:

  //! Default constructor
  ProblemFactory (const Teuchos::RCP<Teuchos::ParameterList>& problemParams,
                  const Teuchos::RCP<Teuchos::ParameterList>& discretizaitonParams,
                  const Teuchos::RCP<ParamLib>& paramLib_);

  //! Destructor
  virtual ~ProblemFactory() {}

  virtual Teuchos::RCP<Albany::AbstractProblem> create() const;

  static bool hasProblem (const std::string& problemName);
private:

  //! Private to prohibit copying
  ProblemFactory(const ProblemFactory&) = delete;

  //! Private to prohibit copying
  ProblemFactory& operator=(const ProblemFactory&) = delete;

protected:

  //! Parameter list specifying what problem to create
  Teuchos::RCP<Teuchos::ParameterList> problemParams;

  //! Parameter list specifying what discretization to use.
  Teuchos::RCP<Teuchos::ParameterList> discretizationParams;

  //! Parameter library
  Teuchos::RCP<ParamLib> paramLib;
};

} // Namespace LandIce

#endif // LANDICE_PROBLEM_FACTORY_HPP

