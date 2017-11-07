#include "RPCFunctor.hpp"

#ifdef USE_RABBITMQ

#include <stdint.h>
#include <assert.h>

#include <amqp_tcp_socket.h>
#include <amqp.h>
#include <amqp_framing.h>

class RPCFunctor::Internals
{
public:
  Internals(std::string hostname,
            int port,
            std::string exchange,
            std::string routingKey);

  ~Internals();

  std::string operator() (const std::string& input);

private:
  std::string Exchange;
  std::string RoutingKey;
  amqp_socket_t* Socket;
  amqp_connection_state_t Conn;
  amqp_bytes_t ReplyToQueue;
};

RPCFunctor::Internals::Internals( std::string hostname,
                                  int port,
                                  std::string exchange,
                                  std::string routingKey ) :
  Exchange( exchange ),
  RoutingKey( routingKey ),
  Socket( NULL )
{
  // establish a channel that is used to connect RabbitMQ server

  this->Conn = amqp_new_connection();

  this->Socket = amqp_tcp_socket_new(this->Conn);
  assert("creating TCP socket" && this->Socket);

  int status = amqp_socket_open( this->Socket,
                                 hostname.c_str(),
                                 port);

  assert("opening TCP socket" && !status);

  amqp_login(this->Conn, "/", 0, 131072, 0,
             AMQP_SASL_METHOD_PLAIN, "guest", "guest");
  amqp_channel_open(this->Conn, 1);
  amqp_get_rpc_reply(this->Conn);

  // create private reply_to queue

  {
    amqp_queue_declare_ok_t *r = amqp_queue_declare(
      this->Conn, 1, amqp_empty_bytes, 0, 0, 0, 1, amqp_empty_table);
    amqp_get_rpc_reply(this->Conn);
    this->ReplyToQueue = amqp_bytes_malloc_dup(r->queue);
    if (this->ReplyToQueue.bytes == NULL)
    {
      fprintf(stderr, "Out of memory while copying queue name");
      return;
    }
  }
}

RPCFunctor::Internals::~Internals()
{
  // closing

  amqp_bytes_free(this->ReplyToQueue);
  amqp_channel_close(this->Conn, 1, AMQP_REPLY_SUCCESS);
  amqp_connection_close(this->Conn, AMQP_REPLY_SUCCESS);
  amqp_destroy_connection(this->Conn);
}

static int rows_eq(int *a, int *b)
{
  int i;

  for (i=0; i<16; i++)
    if (a[i] != b[i]) {
      return 0;
    }

  return 1;
}

std::string RPCFunctor::Internals::operator() (const std::string& input)
{
  std::string output("");

  // send the message

  {
    // set properties
    amqp_basic_properties_t props;
    props._flags = AMQP_BASIC_CONTENT_TYPE_FLAG |
                   AMQP_BASIC_DELIVERY_MODE_FLAG |
                   AMQP_BASIC_REPLY_TO_FLAG |
                   AMQP_BASIC_CORRELATION_ID_FLAG;
    props.content_type = amqp_cstring_bytes("text/plain");
    props.delivery_mode = 2; /* persistent delivery mode */
    props.reply_to = amqp_bytes_malloc_dup(this->ReplyToQueue);
    if (props.reply_to.bytes == NULL)
    {
      fprintf(stderr, "Out of memory while copying queue name");
      return output;
    }
    props.correlation_id = amqp_cstring_bytes("1");

    // publish
    amqp_basic_publish(this->Conn,
                       1,
                       amqp_cstring_bytes(this->Exchange.c_str()),
                       amqp_cstring_bytes(this->RoutingKey.c_str()),
                       0,
                       0,
                       &props,
                       amqp_cstring_bytes(input.c_str()));

    amqp_bytes_free(props.reply_to);
  }

  // wait for an answer

  {
    amqp_basic_consume(this->Conn, 1, this->ReplyToQueue, amqp_empty_bytes,
                       0, 1, 0, amqp_empty_table);
    amqp_get_rpc_reply(this->Conn);

    {
      amqp_frame_t frame;
      int result;

      amqp_basic_deliver_t *d;
      amqp_basic_properties_t *p;
      size_t body_target;
      size_t body_received;

      while (true)
      {
        amqp_maybe_release_buffers(this->Conn);
        result = amqp_simple_wait_frame(this->Conn, &frame);
        if (result < 0)
        {
          break;
        }

        if (frame.frame_type != AMQP_FRAME_METHOD)
        {
          continue;
        }

        if (frame.payload.method.id != AMQP_BASIC_DELIVER_METHOD)
        {
          continue;
        }

        d = (amqp_basic_deliver_t *) frame.payload.method.decoded;

        result = amqp_simple_wait_frame(this->Conn, &frame);
        if (result < 0)
        {
          break;
        }

        if (frame.frame_type != AMQP_FRAME_HEADER)
        {
          fprintf(stderr, "Expected header!");
          abort();
        }

        body_target = (size_t)frame.payload.properties.body_size;
        body_received = 0;

        while (body_received < body_target)
        {
          result = amqp_simple_wait_frame(this->Conn, &frame);
          if (result < 0)
          {
            break;
          }

          if (frame.frame_type != AMQP_FRAME_BODY)
          {
            fprintf(stderr, "Expected body!");
            abort();
          }

          body_received += frame.payload.body_fragment.len;
          assert(body_received <= body_target);

          output.assign((const char *) frame.payload.body_fragment.bytes,
                        frame.payload.body_fragment.len);
        }

        if (body_received != body_target)
        {
          // Can only happen when amqp_simple_wait_frame returns <= 0
          // We break here to close the connection
          break;
        }

        // everything was fine, we can quit now because we received the reply
        break;
      }

    }
  }

  return output;
}

#else /* USE_ZEROMQ */

#include <sstream>

#include <zmq.h>

class RPCFunctor::Internals
{
public:
  Internals(std::string hostname,
            int port,
            std::string,
            std::string);

  ~Internals();

  std::string operator() (const std::string& input);

private:
  void* Context;
  void* Socket;
};

RPCFunctor::Internals::Internals(std::string hostname,
                                 int port,
                                 std::string,
                                 std::string)
{
  std::stringstream s;
  s << "tcp://" << hostname << ":" << port;
  this->Context = zmq_ctx_new();
  this->Socket = zmq_socket(this->Context, ZMQ_REQ);
  int rc = zmq_connect(this->Socket, s.str().c_str());
}

RPCFunctor::Internals::~Internals()
{
  if (this->Socket) zmq_close(this->Socket);
  if (this->Context) zmq_ctx_destroy(this->Context);
}

std::string RPCFunctor::Internals::operator() (const std::string& input)
{
  zmq_send(this->Socket, input.c_str(), input.size(), 0);

  char buffer[100];
  zmq_recv(this->Socket, buffer, 100, 0);
  return std::string(buffer);
}

#endif

RPCFunctor::RPCFunctor()
{
  this->Internal = new Internals( "localhost", 5672, "", "rpc_queue" );
}

RPCFunctor::RPCFunctor( std::string hostname,
                        int port,
                        std::string exchange,
                        std::string routingKey )
{
  this->Internal = new Internals( hostname, port, exchange, routingKey );
}

RPCFunctor::~RPCFunctor()
{
  delete this->Internal;
}

std::string RPCFunctor::operator() ( const std::string& input )
{
  return this->Internal->operator()( input );
}
