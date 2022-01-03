#ifndef MESSAGEQUEUE_H
#define MESSAGEQUEUE_H

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <thread>

template <class T>
class MessageQueue
{
public:
  T receive();
  void send(T &&message);
  void sendAndClear(T &&message);
  int GetSize();

private:
  std::mutex _mutex;
  std::condition_variable _cond;
  std::deque<T> _messages;
};

#endif