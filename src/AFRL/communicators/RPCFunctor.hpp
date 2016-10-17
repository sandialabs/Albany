#ifndef __AFRL_RPCFunctor_hpp
#define __AFRL_RPCFunctor_hpp

#include <string>

class RPCFunctor
{
public:
  RPCFunctor();
  RPCFunctor( std::string hostname,
              int port,
              std::string exchange,
              std::string routingKey );

  virtual ~RPCFunctor();

  std::string operator() (const std::string& input);

protected:
  class Internals;
  Internals* Internal;
};

#endif
