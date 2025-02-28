import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define target primes and intermediate primes
TARGET_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
INTERMEDIATE_PRIMES = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

# Primality test function
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

# Spiral wave equation
def spiral_wave(n, freq, amplitude, angle, phase, direction):
    return n * freq * (1 + amplitude * math.sin(2 * math.pi * (angle / 360 + phase))) * direction

# TSSM prediction function (12 waves, duplicates allowed)
def tssm_predict(layers=13):
    predictions = {'Boundary': [], 'Helix 1': []}
    wave_peaks = []
    
    # 12 angles for 12 waves, evenly spaced with key harmonics
    angles = [0, 30, 60, 90, 120, 137.5, 180, 222.5, 275, 300, 330, 360]
    dr_positive_adj = 0.2  # Boost Helix 1
    
    for n in range(1, layers + 1):
        # Boundary: Fixed, allow duplicates if waves hit them (rare)
        if n == 1:
            predictions['Boundary'].append(2)
            wave_peaks.append((2, 'Boundary', 0.0))
        elif n == 13:
            predictions['Boundary'].append(41)
            wave_peaks.append((41, 'Boundary', 0.0))
        
        # Helix 1: 12 waves to hit intermediates, duplicates expected
        for angle in angles:
            val = spiral_wave(n, freq=1.7, amplitude=1.1, angle=angle, phase=0.08, direction=1)
            val += dr_positive_adj
            prime_candidate = round(val)
            if is_prime(prime_candidate) and 2 < prime_candidate <= 41:
                wave_peaks.append((prime_candidate, 'Helix 1', abs(val - prime_candidate)))
    
    # Keep all peaks, including duplicates
    for prime, category, _ in wave_peaks:
        if prime in TARGET_PRIMES:
            predictions[category].append(prime)
    
    # No capping in raw output; duplicates reflect wave intersections
    return predictions

# Plotting function (shows duplicates)
def plot_tssm(predictions):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for category, primes in predictions.items():
        if category == 'Boundary':
            x = [0] * len(primes)
            y = [0] * len(primes)
            z = primes
            color = 'red'
        elif category == 'Helix 1':
            x = [math.cos(p * math.pi / 180) * 10 for p in primes]
            y = [math.sin(p * math.pi / 180) * 10 for p in primes]
            z = primes
            color = 'blue'
        
        ax.scatter(x, y, z, c=color, label=f"{category} ({len(primes)} points)", s=50)
    
    ax.set_xlabel('X (Toroidal Radius)')
    ax.set_ylabel('Y (Toroidal Angle)')
    ax.set_zlabel('Prime Value (Z)')
    ax.set_title('TSSM Prime Predictions in 13D Toroidal Lattice (12 Waves, Duplicates Allowed)')
    ax.legend()
    plt.savefig('tssm_plot.png')
    plt.close()

# Evaluate unique primes and success rate
def evaluate_predictions(predictions):
    unique_boundary = sorted(list(set(predictions['Boundary'])))
    unique_helix1 = sorted(list(set(predictions['Helix 1'])))
    unique_intermediates = [p for p in unique_helix1 if p in INTERMEDIATE_PRIMES]
    
    total_unique = len(unique_boundary) + len(unique_helix1)
    success_rate = len(unique_intermediates) / len(INTERMEDIATE_PRIMES) * 100
    dist_boundary = len(unique_boundary) / total_unique * 100 if total_unique > 0 else 0
    dist_helix1 = len(unique_helix1) / total_unique * 100 if total_unique > 0 else 0
    
    return {
        'Unique Boundary': unique_boundary,
        'Unique Helix 1': unique_helix1,
        'Success Rate (%)': success_rate,
        'Distribution (%)': {'Boundary': dist_boundary, 'Helix 1': dist_helix1}
    }

# Main execution
if __name__ == "__main__":
    result = tssm_predict(13)
    with open("tssm_results.txt", "w") as f:
        for category, primes in result.items():
            f.write(f"{category}: {primes}\n")
    plot_tssm(result)
    
    print("TSSM Predictions (Raw with Duplicates):")
    for category, primes in result.items():
        print(f"{category}: {primes}")
    
    eval_result = evaluate_predictions(result)
    print("\nEvaluation (Unique Primes):")
    for key, value in eval_result.items():
        print(f"{key}: {value}")
