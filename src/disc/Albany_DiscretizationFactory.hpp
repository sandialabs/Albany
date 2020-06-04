//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef ALBANY_DISCRETIZATIONFACTORY_HPP
#define ALBANY_DISCRETIZATIONFACTORY_HPP

#include <vector>

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"

#include "Albany_AbstractDiscretization.hpp"
#include "Albany_AbstractMeshStruct.hpp"
#include "Albany_AbstractFieldContainer.hpp"

#include "Albany_NullSpaceUtils.hpp"

namespace Albany {

/*!
 * \brief A factory class to instantiate AbstractDiscretization objects
 */
class DiscretizationFactory {
  public:

    //! Default constructor
    DiscretizationFactory(
      const Teuchos::RCP<Teuchos::ParameterList>& topLevelParams,
      const Teuchos::RCP<const Teuchos_Comm>& commT,
      const bool explicit_scheme_ = false);

    //! Destructor
    ~DiscretizationFactory() {}

    static Teuchos::RCP<Albany::AbstractMeshStruct>
    createMeshStruct (Teuchos::RCP<Teuchos::ParameterList> disc_params,
                      Teuchos::RCP<Teuchos::ParameterList> adapt_params,
                      Teuchos::RCP<const Teuchos_Comm> comm,
		      const int numParams);

    Teuchos::RCP<Albany::AbstractMeshStruct> getMeshStruct() {
      return meshStruct;
    }

    Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> > createMeshSpecs();

    Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> > createMeshSpecs(Teuchos::RCP<Albany::AbstractMeshStruct> mesh);

    Teuchos::RCP<Albany::AbstractDiscretization>
    createDiscretization(unsigned int num_equations,
                         const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                         const AbstractFieldContainer::FieldContainerRequirements& req,
                         const Teuchos::RCP<Albany::RigidBodyModes>& rigidBodyModes = Teuchos::null);


    Teuchos::RCP<Albany::AbstractDiscretization>
    createDiscretization(unsigned int num_equations,
                         const std::map<int,std::vector<std::string> >& sideSetEquations,
                         const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                         const std::map<std::string,Teuchos::RCP<Albany::StateInfoStruct> >& side_set_sis,
                         const AbstractFieldContainer::FieldContainerRequirements& req,
                         const std::map<std::string,AbstractFieldContainer::FieldContainerRequirements>& side_set_req, 
                         const Teuchos::RCP<Albany::RigidBodyModes>& rigidBodyModes = Teuchos::null);

    void
    setupInternalMeshStruct(
      unsigned int neq,
      const Teuchos::RCP<Albany::StateInfoStruct>& sis,
      const AbstractFieldContainer::FieldContainerRequirements& req); 

   void
    setupInternalMeshStruct(
      unsigned int neq,
      const Teuchos::RCP<Albany::StateInfoStruct>& sis,
      const std::map<std::string,Teuchos::RCP<Albany::StateInfoStruct> >& side_set_sis,
      const AbstractFieldContainer::FieldContainerRequirements& req,
      const std::map<std::string,AbstractFieldContainer::FieldContainerRequirements>& side_set_req);

    Teuchos::RCP<Albany::AbstractDiscretization> createDiscretizationFromInternalMeshStruct(
      const Teuchos::RCP<Albany::RigidBodyModes>& rigidBodyModes);

    Teuchos::RCP<Albany::AbstractDiscretization> createDiscretizationFromInternalMeshStruct(
      const std::map<int,std::vector<std::string> >& sideSetEquations,
      const Teuchos::RCP<Albany::RigidBodyModes>& rigidBodyModes);
    
    /* This function overwrite previous discretization parameter list */
    void
    setDiscretizationParameters(Teuchos::RCP<Teuchos::ParameterList> disc_params);
    
    private:

    //! Private to prohibit copying
    DiscretizationFactory(const DiscretizationFactory&);

    //! Private to prohibit copying
    DiscretizationFactory& operator=(const DiscretizationFactory&);

    const std::map<int,std::vector<std::string> > empty_side_set_equations;
    const std::map<std::string,Teuchos::RCP<Albany::StateInfoStruct> > empty_side_set_sis;
    const std::map<std::string,AbstractFieldContainer::FieldContainerRequirements> empty_side_set_req;

  protected:

    //! Parameter list specifying what element to create
    Teuchos::RCP<Teuchos::ParameterList> discParams;

    //! Parameter list specifying adaptation parameters (null if problem isn't adaptive)
    Teuchos::RCP<Teuchos::ParameterList> adaptParams;

    //! Parameter list specifying solver parameters
    Teuchos::RCP<Teuchos::ParameterList> piroParams;

    Teuchos::RCP<const Teuchos_Comm> commT;

    //The following are for Aeras hydrostatic problems
    int numLevels;
    int numTracers;
    
    //Flag for explicit time-integration scheme, used in Aeras
    bool explicit_scheme;

    Teuchos::RCP<Albany::AbstractMeshStruct> meshStruct;

    //Number of parameters
    int num_params{0};
};

}

#endif // ALBANY_DISCRETIZATIONFACTORY_HPP
