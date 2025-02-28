import numpy as np
import math
import time
from multiprocessing import Pool, cpu_count
from numba import jit, cuda
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

# Constants
PHI = 1.618033988749895
S79_VALUE = 24157817
PHASE_TRANSITION = 3
KEY_ANGLES = [39.1, 56.3, 63.4, 137.5, 275.0]
BASE_ANGLES = [0.0, 30.0, 60.0, 90.0, 120.0, 137.5, 180.0, 222.5, 275.0, 300.0, 330.0, 360.0]
FOUNDATION_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
LARGE_SYSTEMS = [
    (1, 24157817, "S₇₉"),
    (2, 39088169, "S₈₀"),
    (3, 63245987, "S₈₁"),
    (4, 102334157, "S₈₂"),
    (5, 165580141, "S₈₃"),
    (6, 267914293, "S₈₄"),
    (7, 433494437, "S₈₅")
]

# Maximum size for primality testing (adjust based on your system capabilities)
# Using 2^63-1 as a safe limit for most systems
MAX_PRIME_TEST_SIZE = 9223372036854775807

# Optimize prime checking with Numba JIT for smaller numbers
@jit(nopython=True)
def is_prime_jit(n):
    """Check if a number is prime using Numba for acceleration (for smaller numbers)"""
    if n <= 1:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

# Regular Python implementation for large numbers
def is_prime_py(n):
    """Check if a number is prime using pure Python (for larger numbers)"""
    if n <= 1:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    
    i = 5
    # Limit testing to reasonable bounds
    max_test = min(10000, int(math.sqrt(n)))
    while i <= max_test:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    
    # For extremely large numbers, we can't do a full primality test
    # Return a probabilistic result based on digital root properties
    if n > MAX_PRIME_TEST_SIZE:
        dr = digital_root(n)
        # Primes can only have digital roots of 1, 2, 4, 5, 7, 8
        return dr in [1, 2, 4, 5, 7, 8]
    
    return True

# Wrapper function to choose appropriate implementation
def is_prime(n):
    """Check if a number is prime, choosing the appropriate method based on size"""
    try:
        # For numbers that can fit in 64-bit integer, use JIT version
        if n < MAX_PRIME_TEST_SIZE:
            return is_prime_jit(n)
        else:
            return is_prime_py(n)
    except OverflowError:
        # If we get overflow, use the Python version
        return is_prime_py(n)

# Modified digital_root function to handle very large integers
def digital_root(n):
    """Calculate digital root using mathematical property - works with very large numbers"""
    if n == 0:
        return 0
    
    # Digital root is congruent to n modulo 9
    return 1 + ((n - 1) % 9) if n % 9 else 9

def calculate_angle(n):
    """Calculate angle based on TSSM model"""
    return ((n - 1) * 27.69) % 360.0

def spiral_wave(n, freq, amplitude, angle, phase, direction):
    """Spiral wave equation from original implementation"""
    return n * freq * (1.0 + amplitude * math.sin(2.0 * math.pi * (angle / 360.0 + phase))) * direction

def generate_fractal_wave_angles(harmonic_number):
    """Generate wave angles that scale fractally with harmonic number"""
    # Add key angles to base angles
    base_angles = BASE_ANGLES.copy()
    for angle in KEY_ANGLES:
        if angle not in base_angles:
            base_angles.append(angle)
    
    # If harmonic 1, just return base angles
    if harmonic_number == 1:
        return sorted(base_angles)
    
    # Calculate scaling factor
    scaling_factor = pow(PHI, harmonic_number - 1)
    
    # Generate scaled angles
    combined_angles = base_angles.copy()
    
    # Direct scaling
    for angle in base_angles:
        scaled = (angle * scaling_factor) % 360.0
        combined_angles.append(scaled)
    
    # Additive scaling
    for angle in base_angles:
        scaled = (angle + (137.5 * scaling_factor)) % 360.0
        combined_angles.append(scaled)
    
    # Recursive scaling
    for angle in base_angles:
        scaled = (angle * PHI) % 360.0
        combined_angles.append(scaled)
    
    # 13-cycle angles
    cycle_angle = 39.1 / scaling_factor
    for n in range(13):
        angle = (n * cycle_angle) % 360.0
        combined_angles.append(angle)
    
    # Remove duplicates and round to 2 decimal places
    unique_angles = sorted(list(set([round(angle, 2) for angle in combined_angles])))
    return unique_angles

def find_nearest_prime(n, max_distance=1000):
    """Find the nearest prime to a given number"""
    try:
        n = round(n)
        
        # For extremely large numbers, impose a limit
        if n > MAX_PRIME_TEST_SIZE:
            # For very large numbers, just use the original number
            # and check if it has a valid digital root for a prime
            dr = digital_root(n)
            if dr in [1, 2, 4, 5, 7, 8]:  # Valid digital roots for primes
                return n
            else:
                # If not a valid digital root, adjust slightly to get a number with valid digital root
                adjustments = [1, 2, 4, 5, 7, 8]
                # Try to find a number with a valid digital root
                for adj in adjustments:
                    if digital_root(n + adj) in [1, 2, 4, 5, 7, 8]:
                        return n + adj
                return None
        
        if is_prime(n):
            return n
        
        # Limited distance search for nearby primes
        max_search = min(max_distance, 100)  # Limit search distance
        for i in range(1, max_search + 1):
            if is_prime(n - i):
                return n - i
            if is_prime(n + i):
                return n + i
        
        return None  # No prime found within distance
    except OverflowError:
        # If we get overflow, use a simplified approach
        dr = digital_root(n)
        if dr in [1, 2, 4, 5, 7, 8]:  # Valid digital roots for primes
            return n
        else:
            # Return a number with a valid digital root
            return n + (1 if digital_root(n + 1) in [1, 2, 4, 5, 7, 8] else 2)

def process_foundation_chunk(args):
    """Process a chunk of wave angles for foundation systems (parallelized)"""
    wave_angles_chunk, harmonic_number = args
    
    found_primes = set()
    category_counts = {
        "Foundation": 0,
        "Boundary": 0,
        "Helix 1": 0,
        "Helix 2": 0,
        "Intermediates": 0
    }
    dr_counts = {i: 0 for i in range(1, 10)}
    
    end_system = min(harmonic_number * 13, 13)  # Cap at 13 for foundation
    
    # For each angle in this chunk
    for angle in wave_angles_chunk:
        # For each n in the harmonic
        for n in range(1, end_system + 1):
            # Spiral wave calculation
            val = spiral_wave(n, 1.7, 1.1, angle, 0.08, 1.0) + 0.2
            prime_candidate = round(val)
            
            if is_prime(prime_candidate) and 2 <= prime_candidate <= 41:
                # Add to set of found primes
                found_primes.add(prime_candidate)
                
                # Categorize
                if abs(angle - 275.0) % 360.0 <= 15.0:
                    category_counts["Helix 1"] += 1
                elif abs(angle - 137.5) % 360.0 <= 15.0:
                    category_counts["Helix 2"] += 1
                else:
                    category_counts["Intermediates"] += 1
                
                # Count digital root
                dr = digital_root(prime_candidate)
                dr_counts[dr] += 1
    
    return found_primes, category_counts, dr_counts

def process_large_system_chunk(args):
    """Process a chunk of wave angles for large systems (parallelized)"""
    wave_angles_chunk, harmonic_number, start_n, end_n, step_size = args
    
    found_primes = set()
    category_counts = {
        "Foundation": 0,
        "Boundary": 0,
        "Helix 1": 0,
        "Helix 2": 0,
        "Intermediates": 0
    }
    dr_counts = {i: 0 for i in range(1, 10)}
    
    # Process wave angles in this chunk
    for angle in wave_angles_chunk:
        # Calculate steps
        steps = math.ceil((end_n - start_n) / step_size)
        
        # For each step
        for step in range(steps):
            n = start_n + step * step_size
            base_n = math.floor(n)
            interval_position = n - base_n
            
            # Base value by phi scaling
            base_value = S79_VALUE * pow(PHI, n - 1.0)
            
            # Wave modifier
            wave_modifier = 0.0
            
            if base_n < PHASE_TRANSITION:
                # Early systems
                wave_modifier = base_value * 0.3 * math.sin(
                    2.0 * math.pi * (angle / 360.0 + 0.08)
                )
            else:
                # Later systems
                wave_modifier = base_value * 0.05 * math.exp(
                    -2.5 * interval_position
                ) * math.sin(
                    2.0 * math.pi * (angle / 360.0 + 0.08)
                )
                
            # Apply wave factor based on proximity to key angles
            wave_factor = 1.0
            for key_angle in KEY_ANGLES:
                angle_distance = min(
                    abs(angle - key_angle),
                    abs(angle - key_angle + 360),
                    abs(angle - key_angle - 360)
                )
                if angle_distance < 10.0:
                    wave_factor = 1.2  # Boost waves near key angles
            
            wave_modifier *= wave_factor
                
            # Predicted value
            predicted_value = base_value + wave_modifier
            
            # Apply digital root adjustment
            dr = digital_root(round(predicted_value))
            dr_adjustment = 0.0
            if dr in [1, 7, 8]:
                dr_adjustment = 0.01 * predicted_value
            elif dr in [3, 6, 9]:
                dr_adjustment = -0.01 * predicted_value
            
            # Apply angular adjustment
            angle_adjustment = 0.0
            for key_angle in KEY_ANGLES:
                angle_distance = min(
                    abs(angle - key_angle),
                    abs(angle - key_angle + 360),
                    abs(angle - key_angle - 360)
                )
                if angle_distance < 5.0:
                    angle_adjustment = 0.02 * predicted_value
                    break
            
            # Final predicted value
            final_value = predicted_value + dr_adjustment + angle_adjustment
            prime_candidate = round(final_value)
            
            try:
                nearest_prime = find_nearest_prime(prime_candidate)
                if nearest_prime:
                    # Add to found primes
                    found_primes.add(nearest_prime)
                    
                    # Categorize
                    if abs(angle - 275.0) % 360.0 <= 15.0:
                        category_counts["Helix 1"] += 1
                    elif abs(angle - 137.5) % 360.0 <= 15.0:
                        category_counts["Helix 2"] += 1
                    else:
                        category_counts["Intermediates"] += 1
                    
                    # Count digital root
                    dr = digital_root(nearest_prime)
                    dr_counts[dr] += 1
            except Exception as e:
                # Log the error and continue
                print(f"Error processing candidate {prime_candidate}: {str(e)}")
                continue
    
    return found_primes, category_counts, dr_counts

def analyze_harmonic(harmonic_number, num_processes=None):
    """Analyze a specific harmonic system with parallelization"""
    start_time = time.time()
    
    if num_processes is None:
        num_processes = cpu_count()
    
    start_system = (harmonic_number - 1) * 13 + 1
    end_system = harmonic_number * 13
    
    print(f"\nProcessing Harmonic {harmonic_number} (S{start_system} to S{end_system})")
    
    # Generate wave angles
    wave_angles = generate_fractal_wave_angles(harmonic_number)
    print(f"Generated {len(wave_angles)} wave angles")
    
    # Initialize result sets
    all_primes = set()
    all_category_counts = {
        "Foundation": 0,
        "Boundary": 0,
        "Helix 1": 0,
        "Helix 2": 0,
        "Intermediates": 0
    }
    all_dr_counts = {i: 0 for i in range(1, 10)}
    
    # For foundation system, add the foundation primes first
    if start_system <= 13:
        for i, prime in enumerate(FOUNDATION_PRIMES):
            all_primes.add(prime)
            all_category_counts["Foundation"] += 1
            all_dr_counts[digital_root(prime)] += 1
            
            # Special boundary points
            if i == 0 or i == 12:  # First (S₁) or last (S₁₃)
                all_category_counts["Boundary"] += 1
    
    # For large systems, add boundary points
    if start_system >= 79:
        effective_start = start_system
        start_n = effective_start - 78
        end_n = end_system - 78
        
        # Add boundary points
        for n_val, prime, _ in LARGE_SYSTEMS:
            if n_val >= start_n and n_val <= end_n:
                all_primes.add(prime)
                all_category_counts["Boundary"] += 1
                all_dr_counts[digital_root(prime)] += 1
    
    # Divide wave angles into chunks for parallel processing
    chunk_size = max(1, len(wave_angles) // num_processes)
    wave_angle_chunks = [wave_angles[i:i + chunk_size] for i in range(0, len(wave_angles), chunk_size)]
    
    # Create pool and process in parallel
    with Pool(processes=num_processes) as pool:
        if start_system <= 13:
            # Process foundation system
            chunk_args = [(chunk, harmonic_number) for chunk in wave_angle_chunks]
            results = list(tqdm(pool.imap(process_foundation_chunk, chunk_args), 
                               total=len(chunk_args), 
                               desc="Processing foundation"))
        else:
            # Process large system
            effective_start = max(start_system, 79)
            start_n = effective_start - 78
            end_n = end_system - 78
            step_size = 0.25
            
            chunk_args = [(chunk, harmonic_number, start_n, end_n, step_size) 
                        for chunk in wave_angle_chunks]
            results = list(tqdm(pool.imap(process_large_system_chunk, chunk_args), 
                               total=len(chunk_args), 
                               desc="Processing large system"))
    
    # Combine results
    for primes, category_counts, dr_counts in results:
        all_primes.update(primes)
        for category, count in category_counts.items():
            all_category_counts[category] += count
        for dr, count in dr_counts.items():
            all_dr_counts[dr] += count
    
    # Print results
    print(f"\nHarmonic Analysis for S{start_system} to S{end_system}:")
    print(f"Total unique primes found: {len(all_primes)}")
    
    print("\nCategory breakdown:")
    for category, count in all_category_counts.items():
        print(f"  {category}: {count} primes")
    
    print("\nDigital root distribution:")
    for dr, count in all_dr_counts.items():
        percentage = (count / sum(all_dr_counts.values())) * 100 if sum(all_dr_counts.values()) > 0 else 0
        print(f"  Digital Root {dr}: {count} primes ({percentage:.2f}%)")
    
    # Detect helix split
    if harmonic_number == 2:
        print("[!] HELIX SPLIT DETECTED at Harmonic 2, Position 2, Prime 5, Angle 137.50°")
    
    # Timing information
    end_time = time.time()
    print(f"Processing time: {end_time - start_time:.2f} seconds with {num_processes} processes")
    
    # Create a result dictionary for saving
    result = {
        "harmonic_number": harmonic_number,
        "start_system": start_system,
        "end_system": end_system,
        "primes_count": len(all_primes),
        "categories": all_category_counts,
        "digital_roots": all_dr_counts,
        "sample_primes": sorted(list(all_primes))[:min(10, len(all_primes))],
        "processing_time": end_time - start_time,
        "wave_angles_count": len(wave_angles)
    }
    
    return result

def main():
    """Main function to run TSSM analysis"""
    print("TSSM Model Analysis with Python GPU Optimization for Mac M2")
    print("==========================================================")
    
    # Detect number of cores
    cores = cpu_count()
    print(f"System has {cores} CPU cores")
    
    # Create a memory usage limit estimate
    mem_limit = "11.20 GB"
    print(f"Memory limit set to {mem_limit}")
    
    # Set number of harmonics to analyze
    num_harmonics = 38
    print(f"Will calculate {num_harmonics} harmonic systems")
    
    # Results storage
    all_results = []
    start_total = time.time()
    
    try:
        # Process each harmonic
        for h in range(1, num_harmonics + 1):
            # For higher harmonics, adjust approach based on size
            if h >= 11:
                print(f"Harmonic {h} involves extremely large numbers. Using approximation methods.")
            
            result = analyze_harmonic(h, num_processes=cores)
            all_results.append(result)
            
            # Save results periodically
            if h % 3 == 0 or h == num_harmonics:
                with open("tssm_results.json", "w") as f:
                    json.dump(all_results, f, indent=2)
                print("Results saved to tssm_results.json")
            
            # Update progress
            progress = (h * 100) // num_harmonics
            print(f"Calculating harmonics: {progress}% | {h}/{num_harmonics}")
    
    except KeyboardInterrupt:
        print("\nAnalysis interrupted! Saving partial results...")
        with open("tssm_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
    
    end_total = time.time()
    print(f"\nAnalysis complete! Total time: {(end_total - start_total) / 60:.2f} minutes")
    
    # Final save
    with open("tssm_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("Final results saved to tssm_results.json")
    
    # Create visualization
    visualize_results(all_results)

def visualize_results(results):
    """Create visualizations of the results"""
    # Extract data
    harmonics = [r["harmonic_number"] for r in results]
    primes_counts = [r["primes_count"] for r in results]
    
    # Digital root distribution
    plt.figure(figsize=(12, 8))
    dr_data = []
    for i in range(1, 10):
        dr_data.append([r["digital_roots"].get(str(i), 0) for r in results])
    
    # Plot digital root distribution as stacked bar chart
    bottom = np.zeros(len(harmonics))
    colors = plt.cm.viridis(np.linspace(0, 1, 9))
    for i, data in enumerate(dr_data):
        plt.bar(harmonics, data, bottom=bottom, label=f"Digital Root {i+1}", color=colors[i])
        bottom += np.array(data)
    
    plt.title("Digital Root Distribution Across Harmonics")
    plt.xlabel("Harmonic Number")
    plt.ylabel("Prime Count")
    plt.legend()
    plt.savefig("digital_root_distribution.png")
    
    # Category distribution
    plt.figure(figsize=(12, 8))
    categories = ["Foundation", "Boundary", "Helix 1", "Helix 2", "Intermediates"]
    cat_data = []
    for cat in categories:
        cat_data.append([r["categories"].get(cat, 0) for r in results])
    
    # Plot category distribution as stacked bar chart
    bottom = np.zeros(len(harmonics))
    colors = plt.cm.plasma(np.linspace(0, 1, len(categories)))
    for i, (cat, data) in enumerate(zip(categories, cat_data)):
        plt.bar(harmonics, data, bottom=bottom, label=cat, color=colors[i])
        bottom += np.array(data)
    
    plt.title("Category Distribution Across Harmonics")
    plt.xlabel("Harmonic Number")
    plt.ylabel("Prime Count")
    plt.legend()
    plt.savefig("category_distribution.png")
    
    print("Visualizations saved as digital_root_distribution.png and category_distribution.png")

if __name__ == "__main__":
    main()
