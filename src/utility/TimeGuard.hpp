// @HEADER

#ifndef TIMEGUARD_HPP_
#define TIMEGUARD_HPP_

#include <Teuchos_RCPDecl.hpp>
#include <Teuchos_Time.hpp>

/**
 *  \file TimeGuard.hpp
 *  
 *  \brief 
 */

namespace util {

class TimeGuard {
public:

  TimeGuard (Teuchos::RCP<Teuchos::Time> timer, bool reset = false)
      : timer_(timer) {
    timer_->start(reset);
  }

  ~TimeGuard () {
    timer_->stop();
  }

private:

  Teuchos::RCP<Teuchos::Time> timer_;
};
}

#endif  // TIMEGUARD_HPP_
