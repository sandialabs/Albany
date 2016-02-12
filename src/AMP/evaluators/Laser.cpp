//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Laser.hpp"


namespace AMP
{
  //constructor
  Laser::Laser()
  {
    std::ifstream is("LaserCenter.txt", std::ifstream::in);
    TEUCHOS_TEST_FOR_EXCEPTION(!is, Teuchos::Exceptions::InvalidParameter,
			     std::endl << "Laser Database Error: Laser filename required: LaserCenter.txt" << std::endl);

    std::cout << "Reading file ..." << std::endl;

    RealType t,x,z;
    int power;
    while (is >> t >> x >> z >> power)
      {
	//data
	LaserCenter Data;
	//
	Data.t = t;
	Data.x = x;
	Data.z = z;
	Data.power = power;
	//
	LaserData_.push_back(Data);
      }

    is.close();

  }
  // copy constructor
  Laser::Laser(const Laser &A)
  {
    LaserData_ = A.LaserData_;
  }

  // destructor
  Laser::~Laser()
  {
    LaserData_.clear();
  }

  // get LaserData_
  const Teuchos::Array<LaserCenter> &Laser::getLaserData()
  {
    return (LaserData_);
  }

  // interpolate
  void Laser::getLaserPosition(RealType t, LaserCenter val, RealType &x, RealType &z, int &power)
  {
    Teuchos::Array<LaserCenter>::iterator low;
    // this line below works because Teuchos::Array<T> is a lighweight implementation of
    // std::vector<T>
    low = std::lower_bound(LaserData_.begin(),LaserData_.end(),val,compLaserCenter); 

    TEUCHOS_TEST_FOR_EXCEPTION(low == LaserData_.end(), Teuchos::Exceptions::InvalidParameter,
			     std::endl << "Time out of bound" << std::endl);

    
    // point 1
    RealType t1 = low->t;
    RealType x1 = low->x;	
    RealType z1 = low->z;
    int power1 = low->power;
    
    if ( low != LaserData_.begin() )
      {
	// decrement pointer
	low--;
      }
    else
      {
	// increment pointer
	low++;
      }
    
    // point 2
    RealType t2 = low->t;
    RealType x2 = low->x;	
    RealType z2 = low->z;
    int power2 = low->power;
   
    // now interpolate data between point 1 and 2:
    // P = (1-q)*A + q*B
    // P is the interpolated position
    // A is the position at time t1
    // B is the position at time t2
    // q = (t - t1) / (t2 - t1)

    // be careful by division by zero if t2 is closer to t1 (machine precision)
    RealType q = (t - t1) / (t2 - t1);

    // x position
    x = (1.0 - q)*x1 + q*x2;
    // z position
    z = (1.0 - q)*z1 + q*z2;

    // check if laser is on or off at time t
    if (  (power1 == 1 ) && (power2 == 1) )
      {
	power = 1; // on
      }
    else
      {
	power = 0; // off
      }
  }

  // function used in some STL (standard template library) containers
  bool compLaserCenter(LaserCenter A, LaserCenter B)
  {
    return (A.t < B.t ? true : false);
  }

}
