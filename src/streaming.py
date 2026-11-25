"""In-memory event channels used to stream background job progress over SSE."""

import json
from collections import OrderedDict
from queue import Empty, Queue
from threading import Lock
from time import monotonic

HEARTBEAT_SECONDS = 10
CHANNEL_TTL_SECONDS = 60
MAX_CHANNELS = 100

END = object()


def format_sse(event: str, data) -> str:
    """Serialize one Server-Sent Event frame."""
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n"


HEARTBEAT_FRAME = ": ping\n\n"


class StreamSubscription:
    def __init__(self, backlog):
        self.queue = Queue()
        for event in backlog:
            self.queue.put(event)

    def get_nowait(self):
        try:
            return self.queue.get_nowait()
        except Empty:
            return None


class StreamChannel:
    def __init__(self):
        self.backlog = []
        self.subscribers = []
        self.closed = False
        self.closed_at = None
        self.lock = Lock()

    def publish(self, event):
        with self.lock:
            if self.closed:
                return
            self.backlog.append(event)
            for subscriber in self.subscribers:
                subscriber.queue.put(event)

    def subscribe(self):
        with self.lock:
            subscription = StreamSubscription(self.backlog)
            self.subscribers.append(subscription)
            if self.closed:
                subscription.queue.put(END)
            return subscription

    def unsubscribe(self, subscription):
        with self.lock:
            if subscription in self.subscribers:
                self.subscribers.remove(subscription)

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            self.closed_at = monotonic()
            for subscriber in self.subscribers:
                subscriber.queue.put(END)

    def expired(self, now):
        with self.lock:
            return (
                self.closed
                and self.closed_at is not None
                and now - self.closed_at >= CHANNEL_TTL_SECONDS
            )


class StreamHub:
    """Track one channel per job so late subscribers can replay its events."""

    def __init__(self):
        self.channels = OrderedDict()
        self.lock = Lock()

    def open(self, job_id):
        with self.lock:
            self._prune_locked()
            channel = StreamChannel()
            self.channels[job_id] = channel
            return channel

    def get(self, job_id):
        with self.lock:
            self._prune_locked()
            return self.channels.get(job_id)

    def publish(self, job_id, event):
        channel = self.get(job_id)
        if channel is not None:
            channel.publish(event)

    def close(self, job_id):
        channel = self.get(job_id)
        if channel is not None:
            channel.close()

    def remove(self, job_id):
        with self.lock:
            self.channels.pop(job_id, None)

    def close_all(self):
        with self.lock:
            channels = list(self.channels.values())
        for channel in channels:
            channel.close()

    def _prune_locked(self):
        now = monotonic()
        expired = [
            job_id
            for job_id, channel in self.channels.items()
            if channel.expired(now)
        ]
        for job_id in expired:
            self.channels.pop(job_id, None)
        while len(self.channels) > MAX_CHANNELS:
            self.channels.popitem(last=False)
