"""
Latency, Throughput, and Resource Profiling Evaluator
"""
import time
import numpy as np
import psutil
import os

def profile_latency_and_throughput(inference_fn, sample_inputs, warmup=5, repetitions=30):
    """
    Measures mean latency, P95 latency, FPS throughput, and memory consumption.
    """
    # Warmup
    for _ in range(warmup):
        inference_fn(sample_inputs[0])
        
    latencies = []
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 ** 2) # MB
    
    for _ in range(repetitions):
        for inp in sample_inputs:
            t0 = time.perf_counter()
            inference_fn(inp)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0) # ms
            
    mem_after = process.memory_info().rss / (1024 ** 2) # MB
    latencies = np.array(latencies)
    
    mean_lat = float(np.mean(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    fps = float(1000.0 / mean_lat) if mean_lat > 0 else 0.0
    
    return {
        'mean_latency_ms': mean_lat,
        'p95_latency_ms': p95_lat,
        'min_latency_ms': float(np.min(latencies)),
        'max_latency_ms': float(np.max(latencies)),
        'throughput_fps': fps,
        'memory_rss_mb': mem_after,
        'memory_delta_mb': mem_after - mem_before,
        'total_evaluations': len(latencies)
    }
