"""
GPU Acceleration Benchmark for Codenames Bot

Compares performance of embedding agents with different batch sizes
and hardware backends (CPU, MPS, CUDA).
"""

import time
import torch
import numpy as np
from envs.word_batch_env import WordBatchEnv
from agents.spymaster import EmbeddingSpymaster, SpymasterParams
from agents.guesser import EmbeddingGuesser, GuesserParams
from utils.device import get_device_name


def benchmark_agents(batch_size: int, n_iterations: int = 10):
    """Benchmark agents at given batch size."""
    env = WordBatchEnv(batch_size=batch_size, seed=42)

    red_spy = EmbeddingSpymaster(
        team='red',
        params=SpymasterParams(n_candidate_clues=50, seed=42)
    )
    red_guess = EmbeddingGuesser(
        team='red',
        params=GuesserParams(seed=43)
    )

    # Warm up
    obs = env.reset(seed=42)
    _ = red_spy.get_clue(obs['red_spy'])
    _ = red_guess.get_guess(obs['red_guess'])

    # Benchmark spymaster
    start = time.time()
    for _ in range(n_iterations):
        obs = env.reset(seed=42)
        _ = red_spy.get_clue(obs['red_spy'])
    spy_time = (time.time() - start) / n_iterations

    # Benchmark guesser
    start = time.time()
    for _ in range(n_iterations):
        obs = env.reset(seed=42)
        _ = red_guess.get_guess(obs['red_guess'])
    guess_time = (time.time() - start) / n_iterations

    return spy_time, guess_time


def main():
    print("=" * 60)
    print("GPU ACCELERATION BENCHMARK - Codenames Bot")
    print("=" * 60)
    print()

    # Get device from centralized utility
    print(f"Device: {get_device_name()}")
    print(f"PyTorch version: {torch.__version__}")
    print()

    # Benchmark different batch sizes
    batch_sizes = [1, 4, 8, 16, 32]

    print(f"{'Batch':>6} | {'Spy (ms/game)':>14} | {'Guess (ms/game)':>16} | {'Total (ms/game)':>16} | {'Throughput':>12}")
    print("-" * 90)

    results = []
    for batch_size in batch_sizes:
        spy_time, guess_time = benchmark_agents(batch_size, n_iterations=10)

        spy_per_game = (spy_time * 1000) / batch_size
        guess_per_game = (guess_time * 1000) / batch_size
        total_per_game = spy_per_game + guess_per_game
        throughput = batch_size / (spy_time + guess_time)

        print(f"{batch_size:6d} | {spy_per_game:14.2f} | {guess_per_game:16.2f} | {total_per_game:16.2f} | {throughput:9.1f} g/s")

        results.append({
            'batch_size': batch_size,
            'spy_time': spy_time,
            'guess_time': guess_time,
            'spy_per_game': spy_per_game,
            'guess_per_game': guess_per_game,
            'throughput': throughput
        })

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Compare batch=1 vs batch=32
    batch_1 = results[0]
    batch_32 = results[-1]

    spy_speedup = batch_1['spy_per_game'] / batch_32['spy_per_game']
    guess_speedup = batch_1['guess_per_game'] / batch_32['guess_per_game']
    total_speedup = (batch_1['spy_per_game'] + batch_1['guess_per_game']) / \
                    (batch_32['spy_per_game'] + batch_32['guess_per_game'])

    print(f"Spymaster speedup (batch=1 → batch=32): {spy_speedup:.2f}x")
    print(f"Guesser speedup (batch=1 → batch=32):   {guess_speedup:.2f}x")
    print(f"Overall speedup:                        {total_speedup:.2f}x")
    print()
    print(f"Peak throughput: {batch_32['throughput']:.1f} games/sec @ batch_size=32")
    print()


if __name__ == "__main__":
    main()
