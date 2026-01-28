#ifndef QUEUE_HPP_
#define QUEUE_HPP_

#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>

namespace Thread {

/**
 * Thread safe Queue
 */
template <class Layer>
class Queue {
    public:
    Queue() = default;

    void push(const Layer& l);
    Layer pop();

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
std::thread start_send(Queue<Layer>& queue, Send send) {
    std::thread thread([&]() {
        Layer l = queue.pop();
        send(l);
    });

    return thread;
}

template <class Layer, class Recv>
std::thread start_recv(Queue<Layer>& queue, Recv recv) {
    std::thread thread([&]() {
        Layer l;
        recv(l);
        queue.push(l);
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
Layer Queue<Layer>::pop() {
    std::unique_lock lock(_mutex);
    _cv.wait(lock, [&]() -> bool { return !_queue.empty(); });
    Layer l = _queue.front();
    _queue.pop();
    return l;
}

} // namespace Thread

#endif
