//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_CONTACT_MANAGER_HPP
#define ALBANY_CONTACT_MANAGER_HPP

#include "Teuchos_RCP.hpp"
#include "Albany_DataTypes.hpp"
#include "Phalanx_DataLayout.hpp"

// Moertel-specific 
#include "mrtr_interface.H"

#include <iostream>
#include <fstream>

#include "Albany_DiscretizationUtils.hpp"
#include "Albany_StateInfoStruct.hpp"


/** \brief This class implements the Mortar contact algorithm. Here is the overall sketch of how things work:

   General context: A workset of elements are processed to assemble local finite element residual contributions that
   the opposite contacting surface will impose on the current workset of elements.

   1. Fill in the master and slave contact surfaces from the discretization into the Moertel Interface object.

   2. Work with Moertel to perform the nonlinear inequality constrained optimization problem:

      a) Do a global search to find all the slave segments that can potentially intersect the 
      master segments that this processor owns. The active set that contribute to the mortar finite elements
      will change as the gap function G(x) >= 0 changes each Newton iteration.

      b) For the elements in the mortar space, find the element surfaces that are master surface segments. Do a local 
         search to find the slave segments that potentially intersect each master segment. Note that the master and slave
         elements that contribute to each mortar element will change each Newton iteration.

      c) Form the mortar integration space and assemble all the slave constraint contributions into the master side
         locations residual and gap vectors. These contributions are the M and D matrices that change each iteration
         of the solve process.

    3. The feed a nonlinear solver like a Newton method by adding the M and D to the rest of the nonlinear system. 
       Need to employ something like pseudotransient continuation - feasible direction, conditional gradient, gradient 
       projection or some such as G(x) >= 0 and x_k + d_k is not always feasible.

    4. Go back to 2 until convergence of the nonlinear inequality constrained problem is achieved.
         

*/


namespace Albany {

/*!
 * \brief This is a container class that implements mortar contact using Moertel
 *
 */
class ContactManager {

  public:

    ContactManager(const Teuchos::RCP<Teuchos::ParameterList>& params);

    //! Destructor
    virtual ~ContactManager() {}

    void initializeContactSurfaces(const std::vector<Albany::SideSetList>& ssListVec,
				const Teuchos::ArrayRCP<double>& coordArray,
				const Teuchos::RCP<const Tpetra_Map>& overlap_node_mapT,
				const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> > >::type& wsElNodeID,
				const Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >& meshSpecs);

  private:

    ContactManager();

    Teuchos::RCP<Teuchos::ParameterList> params;

    // Is this a contact problem?
    bool have_contact;

    Teuchos::Array<std::string> masterSideNames;   // master (non-mortar) side names
    Teuchos::Array<std::string> slaveSideNames;    // slave (mortar) side names
    Teuchos::Array<std::string> sideSetIDs;        // sideset ids
    Teuchos::Array<std::string> constrainedFields; // names of fields to be constrained
    Albany::MeshSpecsStruct* meshSpecs;
    Teuchos::Array<int> offset;


    // Moertel-specific library data
    Teuchos::RCP<MOERTEL::Interface> moertelInterface;

    std::ofstream sfile, mfile;


};

}

#endif // ALBANY_CONTACT_MANAGER_HPP
