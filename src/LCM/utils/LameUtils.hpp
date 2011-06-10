/********************************************************************\
*                Copyright (2011) Sandia Corporation                 *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to David Littlewood, djlittl@sandia.gov               *
\********************************************************************/

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

// LAME includes
#include <models/Material.h>
#include <models/Elastic.h>

namespace LameUtils {

Teuchos::RCP<lame::Material> constructLameMaterialModel(const Teuchos::ParameterList& lameMaterialParameters);

}
