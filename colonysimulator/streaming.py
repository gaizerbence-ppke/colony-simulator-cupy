"""Host-resident arrays streamed through the GPU one chunk of layers at a time.

The bracket-resolved cell distribution is too large for VRAM at realistic plate sizes, so it
lives in pinned host memory and is processed in chunks along its first (layer) axis. Two
device buffers on two CUDA streams let the upload of chunk i+1 and the download of chunk i-1
overlap with the computation on chunk i.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import cupy as cp
import numpy as np
from numpy.typing import DTypeLike

# The function `LayerStreamer.process` calls on every chunk: it receives the device chunk and
# the slice of layers it covers, and may return anything (collected into a list).
ChunkFunction = Callable[[cp.ndarray, slice], Any]


def pinned_zeros(shape: tuple[int, ...], dtype: DTypeLike = np.float32,
                 pinned: bool = True) -> np.ndarray:
    """Allocate a zero-filled host array, in page-locked memory when `pinned` is True.

    Page-locked memory is what makes host<->device copies asynchronous. If the allocation
    fails (pinned memory is a limited resource on some systems) a pageable NumPy array is
    returned instead and transfers silently become synchronous.
    """
    shape = tuple(int(s) for s in shape)
    count = int(np.prod(shape)) if shape else 1
    dtype = np.dtype(dtype)
    if pinned:
        try:
            memory = cp.cuda.alloc_pinned_memory(count * dtype.itemsize)
            array = np.frombuffer(memory, dtype=dtype, count=count).reshape(shape)
            array.fill(0)
            return array
        except cp.cuda.runtime.CUDARuntimeError:
            pass
    return np.zeros(shape, dtype=dtype)


def is_pinned(array: np.ndarray) -> bool:
    """True when `array` is backed by memory obtained from cupy.cuda.alloc_pinned_memory."""
    base: object = array
    while isinstance(base, np.ndarray):
        base = base.base
    if isinstance(base, memoryview):
        base = base.obj
    return isinstance(base, cp.cuda.PinnedMemoryPointer)


class LayerStreamer:
    """Streams a C-contiguous host array through the GPU along its first axis.

    `process(fn)` calls `fn(device_chunk, layer_slice)` for consecutive chunks of layers.
    `device_chunk` is a view of a preallocated device buffer holding
    `host[layer_slice]`; changes made in place are copied back to the host when
    `write_back` is set. Everything `fn` does runs on the stream assigned to that chunk, so
    other device arrays it reads or writes are automatically ordered with the copies.
    Reductions `fn` writes into GPU-resident arrays are complete once `process` returns.
    """

    host: np.ndarray                # the streamed host array, layer axis first
    n_layers: int
    chunk_layers: int               # layers per chunk
    n_buffers: int
    buffers: list[cp.ndarray]       # one preallocated device buffer per stream
    streams: list[cp.cuda.Stream]

    def __init__(self, host_array: np.ndarray, chunk_layers: int, n_buffers: int = 2) -> None:
        if not host_array.flags.c_contiguous:
            raise ValueError("host array must be C-contiguous so layer chunks are contiguous")
        if n_buffers < 1:
            raise ValueError("n_buffers must be at least 1")
        self.host = host_array
        self.n_layers = host_array.shape[0]
        self.chunk_layers = max(1, min(int(chunk_layers), self.n_layers))
        self.n_buffers = int(n_buffers)
        chunk_shape = (self.chunk_layers,) + host_array.shape[1:]
        self.buffers = [cp.empty(chunk_shape, dtype=host_array.dtype) for _ in range(self.n_buffers)]
        # Blocking streams (the default) synchronise with the null stream, so work the
        # caller issued on the default stream before `process` is visible to `fn`.
        self.streams = [cp.cuda.Stream() for _ in range(self.n_buffers)]

    @property
    def device_bytes(self) -> int:
        return sum(buffer.nbytes for buffer in self.buffers)

    @property
    def pinned(self) -> bool:
        return is_pinned(self.host)

    def chunks(self, n_layers: int | None = None) -> Iterator[tuple[int, int]]:
        n_layers = self.n_layers if n_layers is None else min(int(n_layers), self.n_layers)
        for start in range(0, n_layers, self.chunk_layers):
            yield start, min(start + self.chunk_layers, n_layers)

    def process(self, fn: ChunkFunction, n_layers: int | None = None,
                write_back: bool = True) -> list[Any]:
        """Run `fn` over the first `n_layers` layers (all by default) chunk by chunk.

        Returns the list of values `fn` returned, one per chunk. All streams are
        synchronised before returning, so host data and GPU-side results are final.
        """
        results: list[Any] = []
        for i, (start, stop) in enumerate(self.chunks(n_layers)):
            buffer_index = i % self.n_buffers
            stream = self.streams[buffer_index]
            device_chunk = self.buffers[buffer_index][:stop - start]
            host_chunk = self.host[start:stop]
            # Chunk i + n_buffers reuses this buffer and this stream; stream order
            # guarantees its upload starts only after this chunk's download finished.
            with stream:
                device_chunk.set(host_chunk, stream=stream)
                results.append(fn(device_chunk, slice(start, stop)))
                if write_back:
                    device_chunk.get(stream=stream, out=host_chunk, blocking=False)
        for stream in self.streams:
            stream.synchronize()
        return results


def chunk_layers_for_budget(layer_shape: tuple[int, ...], dtype: DTypeLike, budget_bytes: int,
                            n_buffers: int = 2, n_layers: int | None = None) -> int:
    """Largest number of layers per chunk whose `n_buffers` device buffers fit the budget."""
    bytes_per_layer = int(np.prod(layer_shape)) * np.dtype(dtype).itemsize
    chunk = max(1, int(budget_bytes) // (bytes_per_layer * n_buffers))
    if n_layers is not None:
        chunk = min(chunk, int(n_layers))
    return chunk
