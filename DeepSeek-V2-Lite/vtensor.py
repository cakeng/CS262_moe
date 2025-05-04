import torch
from collections import OrderedDict
import time
import threading
from typing import Union, List


class VTensor:
    """
    A virtual tensor class for coarse-grained tensor virtualization.
    Manages one or multiple full tensors on CPU and corresponding caches on GPU.
    If multiple tensors are provided, they must share the first dimension, and
    corresponding slices are always moved and accessed together.
    """

    def __init__(
        self,
        full_tensors_cpu: Union[torch.Tensor, List[torch.Tensor]],
        cache_budget: int,
        device: torch.device = torch.device("cuda:0"),
    ):
        """
        Initializes the VTensor.

        Args:
            full_tensors_cpu: A single complete tensor or a list of complete tensors
                              residing on the CPU. The first dimension is the
                              virtualized dimension and must be the same for all tensors
                              if a list is provided.
            cache_budget: The maximum number of slices (along the first dimension)
                          to cache on the GPU.
            device: The target GPU device.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for VTensor GPU caching.")

        self.is_multi_tensor = not isinstance(full_tensors_cpu, torch.Tensor)

        if self.is_multi_tensor:
            if not full_tensors_cpu: # Check if list is empty
                raise ValueError("Input tensor list cannot be empty.")
            self.full_tensors_cpu = full_tensors_cpu
            # Validations for list input
            first_tensor = self.full_tensors_cpu[0]
            if not isinstance(first_tensor, torch.Tensor):
                 raise TypeError("All elements in full_tensors_cpu list must be torch.Tensor")
            if first_tensor.is_cuda:
                raise ValueError("All input tensors must reside on the CPU")

            self.num_slices = first_tensor.shape[0]
            self.dtype = first_tensor.dtype
            self.slice_shapes = [t.shape[1:] for t in self.full_tensors_cpu]

            for i, t in enumerate(self.full_tensors_cpu[1:], 1):
                 if not isinstance(t, torch.Tensor):
                     raise TypeError(f"Element {i} in full_tensors_cpu list is not a torch.Tensor")
                 if t.is_cuda:
                     raise ValueError(f"Tensor {i} in the list must reside on the CPU")
                 if t.shape[0] != self.num_slices:
                     raise ValueError(f"Tensor {i} has mismatching first dimension: expected {self.num_slices}, got {t.shape[0]}")
                 if t.dtype != self.dtype:
                     raise ValueError(f"Tensor {i} has mismatching dtype: expected {self.dtype}, got {t.dtype}")

        else: # Single tensor input
            if not isinstance(full_tensors_cpu, torch.Tensor):
                 raise TypeError("full_tensors_cpu must be a torch.Tensor or list of torch.Tensor")
            if full_tensors_cpu.is_cuda:
                raise ValueError("Input tensor must reside on the CPU")
            self.full_tensors_cpu = [full_tensors_cpu] # Store as list internally for consistency
            self.num_slices = self.full_tensors_cpu[0].shape[0]
            self.dtype = self.full_tensors_cpu[0].dtype
            self.slice_shapes = [self.full_tensors_cpu[0].shape[1:]]


        self.cache_budget = min(
            cache_budget, self.num_slices
        )  # Cannot cache more than available
        self.device = device

        # Set CUDA device
        if isinstance(self.device, torch.device):
            torch.cuda.set_device(self.device)
        elif isinstance(self.device, int):
            torch.cuda.set_device(self.device)
        else:
            torch.cuda.set_device(0)
            self.device = torch.device("cuda:0")

        # Allocate GPU caches (as a list)
        self.on_gpu_tensors = []
        for shape in self.slice_shapes:
            gpu_cache = torch.empty(
                self.cache_budget, *shape,
                dtype=self.dtype,
                device=self.device
            )
            self.on_gpu_tensors.append(gpu_cache)

        # Mappings and tracking (remain the same, track slice indices)
        self.mapping = {}  # original_id -> cache_index
        self.inverse_mapping = {}  # cache_index -> original_id
        self.free_gpu_indices = list(
            range(self.cache_budget)
        )  # Indices available in on_gpu_tensor slots
        self.lru_order = (
            OrderedDict()
        )  # original_id -> None, ordered by access (least recent first)

        # Dedicated stream for asynchronous prefetching
        self.stream = torch.cuda.Stream(device=self.device)
        self._prefetch_lock = (
            threading.Lock()
        )  # Protect concurrent prefetch calls if needed

        print(
            f"VTensor initialized: {'Multi-tensor' if self.is_multi_tensor else 'Single-tensor'}, "
            f"Num Tensors={len(self.full_tensors_cpu)}, Total Slices={self.num_slices}, "
            f"Cache Budget={self.cache_budget}, Slice Shapes={[list(s) for s in self.slice_shapes]}, "
            f"Device={self.device}"
        )
        self._reset_stats()  # Initialize stats
        
    def _reset_stats(self):
        self.stats = {
            "num_slices": str(self.num_slices),
            "cache_budget": str(self.cache_budget),
            "slice_shapes": str(self.slice_shapes),
            "device": str(self.device),
            "prefetch_requested": "None",
            "mapping": "None",
            "lru_order": "None",
            "free_gpu_indices": "None",
            "get_requested": "None",
            "cache_hit": "None",
        }
        
    def _evict_one(self):
        """Evicts the least recently used slice (frees one cache index across all tensors)."""
        if not self.lru_order:
            raise RuntimeError(
                "Attempting to evict from an empty cache (this shouldn't happen if cache is full)."
            )

        lru_original_id, _ = self.lru_order.popitem(
            last=False
        )  # Pop the least recently used (first item)
        cache_index_to_free = self.mapping.pop(lru_original_id)
        del self.inverse_mapping[cache_index_to_free]
        self.free_gpu_indices.append(cache_index_to_free)
        # print(f"Evicted original_id {lru_original_id} from cache_index {cache_index_to_free}")
        return cache_index_to_free

    def async_prefetch(self, original_ids: list[int]):
        """
        Asynchronously prefetches specified slices from CPU to GPU for all managed tensors.
        Evicts least recently used slices if the cache is full.

        Args:
            original_ids: A list of original slice indices to prefetch.
        """
        if self.stats.get("prefetch_requested") == "None":
            self.stats["prefetch_requested"] = []
        self.stats["prefetch_requested"].append(original_ids)
        with self._prefetch_lock:  # Ensure atomicity for this operation
            ids_to_fetch = []
            for original_id in original_ids:
                if not (0 <= original_id < self.num_slices):
                    print(
                        f"Warning: Prefetch ID {original_id} is out of bounds. Skipping."
                    )
                    continue
                if (
                    original_id not in self.mapping
                    and original_id not in ids_to_fetch
                ):  # Avoid redundant fetches within the same call
                    ids_to_fetch.append(original_id)

            if not ids_to_fetch:
                # print("Prefetch: All requested IDs already in cache or invalid.")
                return  # Nothing to do

            num_to_fetch = len(ids_to_fetch)
            num_free = len(self.free_gpu_indices)
            num_occupied = self.cache_budget - num_free

            # Calculate evictions needed
            evictions_needed = max(0, num_to_fetch - num_free)
            # print(f"Prefetching {num_to_fetch} IDs. Need to evict {evictions_needed}.")

            # Identify candidates for eviction (those not being prefetched currently)
            eviction_candidates = [
                oid for oid in self.lru_order if oid not in original_ids
            ]  # Prioritize evicting non-targetted IDs

            # Perform evictions based on LRU among candidates
            freed_indices_from_eviction = []
            for _ in range(evictions_needed):
                if not eviction_candidates:
                    # This might happen if all cached items are in original_ids and we still need space.
                    # Evict the LRU item *among the entire cache* that IS in original_ids
                    # This is less ideal, but necessary if cache_budget is small relative to prefetch request
                    if not self.lru_order:
                        break  # Should not happen if evictions_needed > 0
                    lru_to_evict_anyway = next(
                        iter(self.lru_order)
                    )  # Get the absolute LRU item
                    if (
                        lru_to_evict_anyway in self.mapping
                    ):  # Ensure it's actually mapped
                        print(
                            f"Warning: Evicting {lru_to_evict_anyway} which was also targeted by prefetch due to cache pressure."
                        )
                        cache_idx = (
                            self._evict_one()
                        )  # Use the standard eviction logic which updates lru_order
                        freed_indices_from_eviction.append(cache_idx)
                    else:
                        # If the LRU item isn't mapped (consistency issue?), try the next
                        # This case indicates a potential bug, but we try to recover.
                        print(
                            f"Error: LRU item {lru_to_evict_anyway} not found in mapping during forced eviction."
                        )
                        # As a fallback, just break eviction loop; prefetch might fail partially.
                        break
                else:
                    # Evict the LRU item among the candidates *not* in the current prefetch list
                    lru_candidate_to_evict = None
                    for oid in self.lru_order:  # Iterate LRU order
                        if oid in eviction_candidates:
                            lru_candidate_to_evict = oid
                            break
                    if lru_candidate_to_evict is not None:
                        # print(f"Evicting candidate: {lru_candidate_to_evict}")
                        cache_idx = self.mapping.pop(lru_candidate_to_evict)
                        del self.inverse_mapping[cache_idx]
                        self.lru_order.pop(lru_candidate_to_evict)
                        self.free_gpu_indices.append(cache_idx)
                        freed_indices_from_eviction.append(cache_idx)
                        eviction_candidates.remove(
                            lru_candidate_to_evict
                        )  # Remove from candidates
                    else:
                        # This case should not happen if eviction_candidates is not empty
                        print(
                            "Error: Could not find an LRU candidate to evict."
                        )
                        break

            # Fetch data using the dedicated stream
            with torch.cuda.stream(self.stream):
                for original_id in ids_to_fetch:
                    if not self.free_gpu_indices:
                        print(
                            f"Warning: Ran out of GPU indices during prefetch for ID {original_id}, likely due to eviction issues. Skipping."
                        )
                        continue  # Should have enough space after evictions, but safety check

                    target_cache_index = self.free_gpu_indices.pop(0)

                    # Perform the non-blocking copy for *each* tensor
                    for i in range(len(self.full_tensors_cpu)):
                        self.on_gpu_tensors[i][target_cache_index].copy_(
                            self.full_tensors_cpu[i][original_id], non_blocking=True
                        )

                    # Update mappings *after* copy is issued
                    self.mapping[original_id] = target_cache_index
                    self.inverse_mapping[target_cache_index] = original_id
                    # Add to LRU order as most recently used (will be updated on actual 'get')
                    # For prefetch, we add it but don't move_to_end yet. 'get' marks true usage.
                    if original_id in self.lru_order:
                        self.lru_order.move_to_end(original_id)
                    else:
                        self.lru_order[original_id] = None

            # print(f"Prefetch issued for {len(ids_to_fetch)} items. Cache size: {len(self.mapping)}/{self.cache_budget}")

    def get(self, original_id: int) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Retrieves a slice (or list of slices if multi-tensor) by its original index.
        If the slice is not on the GPU, it's fetched from the CPU for all tensors.

        Args:
            original_id: The original index of the slice to retrieve.

        Returns:
            The requested tensor slice (or list of slices) residing on the GPU.
        """
        if self.stats.get("get_requested") == "None":
            self.stats["get_requested"] = []
            self.stats["cache_hit"] = []
        self.stats["get_requested"].append(original_id)
        if not (0 <= original_id < self.num_slices):
            raise IndexError(
                f"Index {original_id} out of bounds for tensor with {self.num_slices} slices."
            )

        # If prefetched data is still transferring, wait for it.
        # This synchronization is important if 'get' is called shortly after 'async_prefetch'
        # for the *same* ID. For different IDs or general computation overlap,
        # this sync point allows the prefetch stream to finish its work for the requested item.
        # For items *already* in cache (cache hit), this sync is usually very fast or no-op.
        # self.stream.synchronize() # Option 1: Synchronize entire stream - simpler but might reduce overlap
        # Option 2: More fine-grained sync (e.g., using events per transfer) - more complex

        cache_index = self.mapping.get(original_id)

        if cache_index is not None:
            self.stats["cache_hit"].append("True")
            # Cache Hit
            # print(f"Cache Hit: ID {original_id} at cache_index {cache_index}")
            self.lru_order.move_to_end(
                original_id
            )  # Mark as most recently used
            # Return list or single tensor based on initialization
            result = [gpu_cache[cache_index] for gpu_cache in self.on_gpu_tensors]
            return result if self.is_multi_tensor else result[0]
        else:
            self.stats["cache_hit"].append("False")
            # Cache Miss
            # print(f"Cache Miss: ID {original_id}")

            # Ensure any pending prefetches (which might affect free list/mappings) are done
            # before we potentially evict and modify state synchronously.
            # This prevents race conditions between async_prefetch and synchronous get.
            self.stream.synchronize()

            if len(self.mapping) >= self.cache_budget:
                # Cache is full, need to evict LRU
                self._evict_one()

            # Find a free spot (either newly freed or already available)
            if not self.free_gpu_indices:
                # This indicates a logic error if eviction didn't free a spot when needed
                raise RuntimeError(
                    "Cache miss occurred but no free GPU index available after potential eviction."
                )
            target_cache_index = self.free_gpu_indices.pop(0)

            # Fetch from CPU to GPU (synchronously for 'get') for *all* tensors
            # Use default stream implicitly
            for i in range(len(self.full_tensors_cpu)):
                slice_to_copy = self.full_tensors_cpu[i][original_id]
                self.on_gpu_tensors[i][target_cache_index].copy_(slice_to_copy)

            # Update mappings and LRU
            self.mapping[original_id] = target_cache_index
            self.inverse_mapping[target_cache_index] = original_id
            self.lru_order[original_id] = None  # Add as most recently used
            self.lru_order.move_to_end(original_id)
            # print(f"Fetched ID {original_id} to cache_index {target_cache_index}. Cache size: {len(self.mapping)}/{self.cache_budget}")
            hit_rate = len(self.mapping) / self.num_slices
            # Return list or single tensor based on initialization
            result = [gpu_cache[target_cache_index] for gpu_cache in self.on_gpu_tensors]
            return result if self.is_multi_tensor else result[0]

    def synchronize_prefetch(self):
        """Synchronizes the dedicated prefetch stream."""
        self.stream.synchronize()

    def get_stats(self):
        """Updates the statistics of the VTensor."""
        self.stats["mapping"] = str(self.mapping)
        self.stats["lru_order"] = str(self.lru_order)
        self.stats["free_gpu_indices"] = str(self.free_gpu_indices)
        out_stats = self.stats.copy()
        self._reset_stats()  # Reset stats after retrieval
        return out_stats

# Unit Test Section
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, skipping VTensor unit test.")
        exit()

    print("Running VTensor Unit Test...")

    # --- Configuration ---
    num_total_slices = 20
    # Increase slice dimensions to make data transfer more significant
    slice_dim1 = 1024 
    slice_dim2 = 2048 
    cache_sz = 5
    device = torch.device("cuda:0")

    # --- Setup ---
    print("\n--- Setup ---")
    # Create a dummy full tensor on CPU
    full_cpu_tensor = torch.randn(
        num_total_slices, slice_dim1, slice_dim2, dtype=torch.float32
    )
    print(
        f"Created full CPU tensor: shape={full_cpu_tensor.shape}, dtype={full_cpu_tensor.dtype}"
    )

    # Instantiate VTensor
    v_tensor = VTensor(full_cpu_tensor, cache_budget=cache_sz, device=device)

    # --- Basic Get Test (Cache Misses and Hits) ---
    print("\n--- Basic Get Test ---")
    # 1. Miss, fetch 0
    print("Getting slice 0 (miss)...")
    slice_0 = v_tensor.get(0)
    print(f"Got slice 0: shape={slice_0.shape}, device={slice_0.device}")
    assert slice_0.device == device
    assert torch.equal(slice_0.cpu(), full_cpu_tensor[0])
    print("Mapping:", v_tensor.mapping)
    print("LRU:", list(v_tensor.lru_order.keys()))

    # 2. Miss, fetch 1
    print("\nGetting slice 1 (miss)...")
    slice_1 = v_tensor.get(1)
    assert torch.equal(slice_1.cpu(), full_cpu_tensor[1])
    print("Mapping:", v_tensor.mapping)
    print("LRU:", list(v_tensor.lru_order.keys()))

    # 3. Hit, access 0 again
    print("\nGetting slice 0 (hit)...")
    slice_0_again = v_tensor.get(0)
    assert (
        slice_0_again.data_ptr() == slice_0.data_ptr()
    )  # Should be the same memory
    assert torch.equal(slice_0_again.cpu(), full_cpu_tensor[0])
    print("Mapping:", v_tensor.mapping)
    print(
        "LRU:", list(v_tensor.lru_order.keys())
    )  # 0 should be most recent now

    # --- Fill Cache & Test Eviction ---
    print("\n--- Cache Eviction Test ---")
    print(f"Filling cache (size {cache_sz})...")
    for i in range(
        2, cache_sz + 1
    ):  # Fetch 2, 3, 4, 5 (assuming cache_sz=5 initially)
        print(f"Getting slice {i} (miss)...")
        _ = v_tensor.get(i)
    print("Cache filled.")
    print("Mapping:", v_tensor.mapping)
    print(
        "LRU:", list(v_tensor.lru_order.keys())
    )  # Should be [1, 2, 3, 4, 0] if sz=5

    print(
        f"\nGetting slice {cache_sz + 1} (miss, should evict LRU)..."
    )  # e.g., get 6, evict 1
    lru_before_evict = next(iter(v_tensor.lru_order))
    slice_evict = v_tensor.get(cache_sz + 1)
    assert torch.equal(slice_evict.cpu(), full_cpu_tensor[cache_sz + 1])
    print("Mapping:", v_tensor.mapping)
    print("LRU:", list(v_tensor.lru_order.keys()))
    assert (
        lru_before_evict not in v_tensor.mapping
    )  # Check that the LRU item was evicted

    # --- Async Prefetch Test ---
    print("\n--- Async Prefetch Test ---")
    # Prefetch some items not in cache
    prefetch_ids = list(
        range(num_total_slices // 2, num_total_slices // 2 + cache_sz)
    )  # e.g., [10, 11, 12, 13, 14] if num_total=20, cache=5
    print(f"Current cache IDs: {list(v_tensor.mapping.keys())}")
    print(f"Prefetching IDs: {prefetch_ids}...")

    # Clear cache for a cleaner prefetch test (optional)
    # v_tensor = VTensor(full_cpu_tensor, cache_budget=cache_sz, device=device)
    # print("Cache cleared for prefetch test.")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Issue prefetch on the dedicated stream
    start_event.record(v_tensor.stream)
    v_tensor.async_prefetch(prefetch_ids)
    end_event.record(v_tensor.stream)

    # Wait for prefetch to complete
    v_tensor.synchronize_prefetch()  # or end_event.synchronize()
    prefetch_time = start_event.elapsed_time(end_event)
    print(
        f"Prefetch command issued and synchronized. Time: {prefetch_time:.4f} ms"
    )
    print("Mapping after prefetch:", v_tensor.mapping)
    print("LRU after prefetch:", list(v_tensor.lru_order.keys()))

    # Verify prefetched items can be retrieved (should be hits)
    print("Getting prefetched items (should be hits)...")
    for pid in prefetch_ids:
        if pid < v_tensor.num_slices:  # Ensure ID is valid
            slice_p = v_tensor.get(pid)
            assert pid in v_tensor.mapping
            assert torch.equal(slice_p.cpu(), full_cpu_tensor[pid])
    print("Prefetched items retrieved successfully.")

    # --- Overlap Test: Prefetch + Computation ---
    print("\n--- Overlap Test (Prefetch + MatMul) ---")

    # Reset VTensor state using a SINGLE tensor for this specific test
    # To avoid complexity in handling list returns within the timing loop
    v_tensor_single = VTensor(full_cpu_tensor, cache_budget=cache_sz, device=device)
    print("VTensor (single) reset for overlap test.")

    # Setup: Some initial items in cache
    initial_ids = [0, 1]
    for i_id in initial_ids:
        _ = v_tensor_single.get(i_id) # Use the single-tensor instance
    print(f"Initial cache state: {v_tensor_single.mapping.keys()}")

    # Prepare data for computation (on GPU, independent of VTensor cache)
    mat_A = torch.randn(slice_dim1, 512, device=device)
    mat_B = torch.randn(512, slice_dim2, device=device)
    print(f"Prepared matrices for MatMul: A={mat_A.shape}, B={mat_B.shape}")

    # IDs to prefetch (different from initial cache)
    overlap_prefetch_ids = list(
        range(cache_sz, cache_sz + cache_sz)
    )  # e.g., [5, 6, 7, 8, 9]
    print(f"IDs to prefetch during computation: {overlap_prefetch_ids}")

    # --- Timing without overlap ---
    print("\nTiming: Computation then Prefetch")
    # 1. Computation
    comp_start = torch.cuda.Event(enable_timing=True)
    comp_end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()  # Ensure GPU is idle
    comp_start.record()
    result_comp = torch.matmul(mat_A, mat_B)
    comp_end.record()
    torch.cuda.synchronize()  # Wait for computation
    comp_time = comp_start.elapsed_time(comp_end)
    print(f"Computation finished. Time: {comp_time:.4f} ms")

    # 2. Prefetch (after computation) - Use the single-tensor instance
    pref_start = torch.cuda.Event(enable_timing=True)
    pref_end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    pref_start.record(v_tensor_single.stream)
    v_tensor_single.async_prefetch(overlap_prefetch_ids)
    pref_end.record(v_tensor_single.stream)
    v_tensor_single.synchronize_prefetch() # Wait for prefetch
    pref_time = pref_start.elapsed_time(pref_end)
    print(f"Prefetch finished. Time: {pref_time:.4f} ms")
    total_sequential_time = comp_time + pref_time
    print(f"Total sequential time: {total_sequential_time:.4f} ms")


    # --- Timing with overlap --- Use the single-tensor instance
    print("\nTiming: Computation and Prefetch Concurrently")
    v_tensor_single = VTensor(full_cpu_tensor, cache_budget=cache_sz, device=device)
    for i_id in initial_ids:
        _ = v_tensor_single.get(i_id)
    print(f"VTensor (single) reset. Initial cache state: {v_tensor_single.mapping.keys()}")

    overall_start = torch.cuda.Event(enable_timing=True)
    prefetch_end_event = torch.cuda.Event(enable_timing=True)
    compute_end_event = torch.cuda.Event(enable_timing=True)
    overall_end = torch.cuda.Event(enable_timing=True)
    default_stream = torch.cuda.current_stream(device=device)

    torch.cuda.synchronize() 
    overall_start.record(default_stream)

    # Start prefetch on its stream (using single-tensor instance)
    with torch.cuda.stream(v_tensor_single.stream):
        v_tensor_single.async_prefetch(overlap_prefetch_ids)
        prefetch_end_event.record(v_tensor_single.stream)

    # Start computation on the default stream
    with torch.cuda.stream(default_stream):
        result_overlap = torch.matmul(mat_A, mat_B)
        compute_end_event.record(default_stream)
        default_stream.wait_event(prefetch_end_event)
        overall_end.record(default_stream)

    overall_end.synchronize()
    total_overlap_time = overall_start.elapsed_time(overall_end)

    print(f"Prefetch and Computation issued concurrently.")
    print(f"Waiting for default stream to complete (after waiting for prefetch)...)")
    print(f"Both finished. Total concurrent time: {total_overlap_time:.4f} ms")

    print(f"\nSequential Time: {total_sequential_time:.4f} ms")
    print(f"Concurrent Time: {total_overlap_time:.4f} ms") # Print concurrent time here for comparison

    if (
        total_overlap_time < total_sequential_time * 0.95
    ):  # Allow for some overhead, check for significant overlap
        print("SUCCESS: Overlap achieved! Concurrent execution was faster.")
    else:
        print(
            "WARNING: Overlap might not be significant or measurement inaccurate. Concurrent time >= Sequential time."
        )
        print(
            "         This can happen on some GPUs or if tasks are too short."
        )

    assert torch.equal(result_comp, result_overlap)
    print("Computation result consistent.")
    print("Verifying prefetched data after overlap (using single-tensor instance)...")
    for pid in overlap_prefetch_ids:
        if pid < v_tensor_single.num_slices:
             slice_o = v_tensor_single.get(pid) # Get from single-tensor instance
             assert pid in v_tensor_single.mapping
             assert torch.equal(slice_o.cpu(), v_tensor_single.full_tensors_cpu[0][pid])
    print("Prefetched data verified successfully after overlap.")

    # --- Multi-Tensor Test --- 
    print("\n--- Multi-Tensor Test ---")
    # Create multiple tensors for testing
    num_multi = 3
    slice_shapes_multi = [
        (slice_dim1 // 2, slice_dim2), 
        (slice_dim1, slice_dim2 // 2),
        (slice_dim1, slice_dim2)
    ]
    full_cpu_tensors_multi = [
        torch.randn(num_total_slices, *s_shape, dtype=torch.float32)
        for s_shape in slice_shapes_multi
    ]
    print(f"Created {num_multi} tensors for multi-tensor test.")

    # Instantiate with list
    v_tensor_multi = VTensor(full_cpu_tensors_multi, cache_budget=cache_sz, device=device)

    # Test get (miss)
    print("\nGetting slice 0 (miss) - multi-tensor...")
    slices_0_multi = v_tensor_multi.get(0)
    assert isinstance(slices_0_multi, list)
    assert len(slices_0_multi) == num_multi
    print(f"Got {len(slices_0_multi)} slices.")
    for i in range(num_multi):
        assert slices_0_multi[i].device == device
        assert slices_0_multi[i].shape == v_tensor_multi.on_gpu_tensors[i].shape[1:]
        assert torch.equal(slices_0_multi[i].cpu(), full_cpu_tensors_multi[i][0])
    print("Multi-tensor get (miss) successful.")
    print("Mapping:", v_tensor_multi.mapping)
    print("LRU:", list(v_tensor_multi.lru_order.keys()))

    # Test get (hit)
    print("\nGetting slice 0 (hit) - multi-tensor...")
    slices_0_multi_again = v_tensor_multi.get(0)
    assert isinstance(slices_0_multi_again, list)
    assert len(slices_0_multi_again) == num_multi
    print(f"Got {len(slices_0_multi_again)} slices.")
    for i in range(num_multi):
        assert slices_0_multi_again[i].data_ptr() == slices_0_multi[i].data_ptr()
        assert torch.equal(slices_0_multi_again[i].cpu(), full_cpu_tensors_multi[i][0])
    print("Multi-tensor get (hit) successful.")
    print("Mapping:", v_tensor_multi.mapping)
    print("LRU:", list(v_tensor_multi.lru_order.keys())) # Should be [0]

    # Test prefetch
    prefetch_ids_multi = [1, 2, 3]
    print(f"\nPrefetching IDs {prefetch_ids_multi} (multi-tensor)...")
    v_tensor_multi.async_prefetch(prefetch_ids_multi)
    v_tensor_multi.synchronize_prefetch()
    print("Multi-tensor prefetch synchronized.")
    print("Mapping:", v_tensor_multi.mapping)
    print("LRU:", list(v_tensor_multi.lru_order.keys())) # Prefetch adds but doesn't reorder yet

    # Test get after prefetch (should be hits)
    for pid in prefetch_ids_multi:
        print(f"\nGetting prefetched slice {pid} (hit) - multi-tensor...")
        slices_p_multi = v_tensor_multi.get(pid)
        assert isinstance(slices_p_multi, list)
        assert len(slices_p_multi) == num_multi
        assert pid in v_tensor_multi.mapping
        print(f"Got {len(slices_p_multi)} slices for ID {pid}.")
        for i in range(num_multi):
            assert torch.equal(slices_p_multi[i].cpu(), full_cpu_tensors_multi[i][pid])
    print("Multi-tensor get after prefetch successful.")
    print("Mapping:", v_tensor_multi.mapping)
    print("LRU:", list(v_tensor_multi.lru_order.keys())) # Now order should reflect gets

    print("\nVTensor Unit Test Completed.")
