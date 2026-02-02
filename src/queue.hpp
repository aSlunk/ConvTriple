#ifndef QUEUE_HPP_
#define QUEUE_HPP_

#include <condition_variable>
#include <mutex>
#include <optional>
#include <queue>
#include <seal/ciphertext.h>
#include <sstream>
#include <thread>

namespace Thread {

/**
 * Thread safe Queue
 */
template <class Layer>
class Queue {
  public:
    explicit Queue() = default;

    template <class T>
    void push(std::vector<T> ciphers);
    void push(const Layer& l);

    std::optional<Layer> pop();

  private:
    std::queue<Layer> _queue;
    std::mutex _mutex;
    std::condition_variable _cv;
};

template <class Layer, class Send>
std::thread start_send(Queue<Layer>& queue, Send send);

template <class Layer, class Recv>
std::thread start_recv(Queue<Layer>& queue, Recv send);

template <class Layer, class Send>
std::thread start_send(const size_t& rounds, Queue<Layer>& queue, Send send) {
    std::thread thread([&]() {
        for (size_t i = 0; i < rounds; ++i) {
            if (auto l = queue.pop())
                send(l.value());
            else
                break;
        }
    });

    return thread;
}

template <class Layer, class Recv>
std::thread start_recv(const size_t rounds, Queue<Layer>& queue, Recv recv) {
    std::thread thread([&]() {
        for (size_t i = 0; i < rounds; ++i) {
            Layer l;
            recv(l);
            queue.push(l);
        }
    });

    return thread;
}

template <class Layer>
void Queue<Layer>::push(const Layer& l) {
    {
        std::unique_lock lock(_mutex);
        _queue.push(l);
    }
    _cv.notify_one();
}

template <class Layer>
template <class T>
void Queue<Layer>::push(std::vector<T> ciphers) {
    std::stringstream l;
    for (auto& cipher : ciphers) cipher.save(l);

    {
        std::unique_lock lock(_mutex);
        _queue.push({std::move(l), ciphers.size()});
    }
    _cv.notify_one();
}

template <class Layer>
std::optional<Layer> Queue<Layer>::pop() {
    std::unique_lock lock(_mutex);
    _cv.wait(lock, [&]() -> bool { return !_queue.empty(); });

    if (_queue.empty())
        return std::nullopt;

    Layer l = std::move(_queue.front());
    _queue.pop();
    return l;
}

} // namespace Thread

#endif
