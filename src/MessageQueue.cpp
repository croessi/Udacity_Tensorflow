#ifndef MESSAGEQUEUE_CPP
#define MESSAGEQUEUE_CPP

#include "MessageQueue.h"

template<class T> 
T MessageQueue<T>::receive()
  {
    // lock
    std::unique_lock<std::mutex> uLock(_mutex);
    _cond.wait(uLock, [this]
               { return !_messages.empty(); });

    // remove last element
    T message = std::move(_messages.back());
    _messages.pop_back();

    return message;
  }

template<class T> 
   void MessageQueue<T>::send(T &&message)
  {
    // lock
    std::lock_guard<std::mutex> uLock(_mutex);

    // add vector to queue
    _messages.emplace_back(std::move(message));
    _cond.notify_one();
  }

template<class T> 
  int MessageQueue<T>::GetSize()
  {
    // lock
    std::lock_guard<std::mutex> uLock(_mutex);
    return _messages.size();
  }

#endif