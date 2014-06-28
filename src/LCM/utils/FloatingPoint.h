//
// $Id: FloatingPoint.h,v 1.1 2008/07/14 17:50:46 lxmota Exp $
//
// $Log: FloatingPoint.h,v $
// Revision 1.1  2008/07/14 17:50:46  lxmota
// Initial sources.
//
//

//
// 2001/10/02 22:45:15 by Jaroslaw Knap
// Imported sources.
//

#if !defined(LCM_FloatingPoint_h)
#define LCM_FloatingPoint_h

namespace LCM {

const unsigned emptyMask_     = 0x00;
const unsigned inexactMask_   = 0x01;
const unsigned divbyzeroMask_ = 0x02;
const unsigned underflowMask_ = 0x04;
const unsigned overflowMask_  = 0x08;
const unsigned invalidMask_   = 0x10;

class FloatingPoint {

public:

  FloatingPoint();
  virtual ~FloatingPoint();

  void trapInexact();
  void trapDivbyzero();
  void trapUnderflow();
  void trapOverflow();
  void trapInvalid();


private:

  static bool active_;
  static unsigned mask_;
  static unsigned oldMask_;

  // obtain and store current masks status;
  unsigned getCurrentMask();

  // set mask
  void setMask(unsigned mask);

};

} // namespace LCM

#endif // LCM_FloatingPoint_h
