//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
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

    RealType t,x,y,power_fraction;
    int power;
    while (is >> t >> x >> y >> power >> power_fraction)
      {
	//data
	LaserCenter Data;
	//
	Data.t = t;
	Data.x = x;
	Data.y = y;
	Data.power = power;
        Data.power_fraction = power_fraction;
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
  void Laser::getLaserPosition(RealType t, LaserCenter val, RealType &x, RealType &y, int &power, RealType &power_fraction)
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
    RealType y1 = low->y;
    int power1 = low->power;
    RealType power_fraction1 = low->power_fraction; 
    
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
    RealType y2 = low->y;
    int power2 = low->power;
    RealType power_fraction2 = low->power_fraction;
   
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
    y = (1.0 - q)*y1 + q*y2;
    // power fraction
    power_fraction = (1.0 - q)*power_fraction1 + q*power_fraction2;

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
