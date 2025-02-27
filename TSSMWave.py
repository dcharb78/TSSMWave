import math
import numpy as np
import matplotlib.pyplot as plt
from sympy import isprime

# Constants
phi = 1.618033988749895
S79_value = 24157817

# System points for reference
systems = [
    {"name": "S₇₉", "n": 1, "value": 24157817},
    {"name": "S₈₀", "n": 2, "value": 39088169},
    {"name": "S₈₁", "n": 3, "value": 63245987},
    {"name": "S₈₂", "n": 4, "value": 102334157},
    {"name": "S₈₃", "n": 5, "value": 165580141},
    {"name": "S₈₄", "n": 6, "value": 267914293},
    {"name": "S₈₅", "n": 7, "value": 433494437}
]

# Interpolation points (known intermediate values)
interpolation_points = [
    {"n": 1.5, "value": 50234843},
    {"n": 2.5, "value": 81169957}
]

# TSSM key angles
key_angles = {
    "thirteen_cycle_boundary": 39.1,
    "digital_root_2": 56.3,
    "digital_root_1": 63.4,
    "helix_2": 137.5,
    "helix_1": 275.0
}

# Optimized parameters
params = {
    "early_peak_position": 0.6,
    "early_peak_height": 0.3,
    "late_decay_factor": 2.5,
    "late_initial_boost": 0.05,
    "dr_positive_adj": 0.01,
    "dr_negative_adj": -0.01,
    "angle_threshold": 5,
    "angle_boost": 0.02,
    "phase_transition": 3
}

def calculate_angle(n):
    """Calculate angle based on TSSM model"""
    return ((n - 1) * 27.69) % 360

def digital_root(n):
    """Calculate digital root"""
    if n == 0:
        return 0
    return 1 + ((n - 1) % 9)

def find_nearest_prime(n):
    """Find the nearest prime to a given number"""
    n = round(n)
    
    if isprime(n):
        return n
    
    i = 1
    while True:
        if isprime(n - i):
            return n - i
        if isprime(n + i):
            return n + i
        i += 1
        
        # Safety check to prevent infinite loops
        if i > 1000:
            return None

def tssm_wave_formula(n):
    """
    The optimized TSSM Wave Formula that predicts prime values
    based on the position parameter n.
    """
    # Determine which system interval we're in
    base_n = math.floor(n)
    
    # Calculate the position within the interval (0 to 1)
    interval_position = n - base_n
    
    # Calculate the base system value using phi scaling from S₇₉
    base_value = S79_value * (phi ** (base_n - 1))
    
    # Calculate the angle
    angle = calculate_angle(n)
    
    # Apply wave modification based on system
    wave_modifier = 0
    
    if base_n < params["phase_transition"]:
        # For early systems (before phase transition)
        # Use a quadratic function with peak at specified position
        peak_height = params["early_peak_height"]  # Peak height as % of base value
        peak_position = params["early_peak_position"]  # Peak position in interval
        
        # Quadratic function: a(x-p)² + h
        # where p is peak position, h is peak height
        a = -peak_height / (peak_position ** 2)
        wave_modifier = base_value * (a * ((interval_position - peak_position) ** 2) + peak_height)
    else:
        # For later systems (after phase transition)
        # Use an exponential decay function with maximum at beginning
        decay_factor = params["late_decay_factor"]
        initial_boost = params["late_initial_boost"]  # Initial boost as % of base value
        wave_modifier = base_value * initial_boost * math.exp(-decay_factor * interval_position)
    
    # Calculate predicted value before adjustments
    predicted_value = base_value + wave_modifier
    
    # Digital root adjustment
    dr = digital_root(round(predicted_value))
    
    # Apply adjustment based on digital root
    dr_adjustment = 0
    if dr in [1, 7, 8]:
        dr_adjustment = params["dr_positive_adj"] * predicted_value
    elif dr in [3, 6, 9]:
        dr_adjustment = params["dr_negative_adj"] * predicted_value
    
    # Angular adjustment
    # Apply adjustment based on proximity to key angles
    angle_adjustment = 0
    
    # Calculate distances to key angles
    distances = []
    for key_angle in key_angles.values():
        distance = min(
            abs(angle - key_angle),
            abs(angle - key_angle + 360),
            abs(angle - key_angle - 360)
        )
        distances.append(distance)
    
    # Find minimum distance
    min_distance = min(distances)
    
    # If close to a key angle, apply a boost
    if min_distance < params["angle_threshold"]:
        angle_adjustment = params["angle_boost"] * predicted_value
    
    # Combine all components
    result = predicted_value + dr_adjustment + angle_adjustment
    
    return {
        "predicted_value": result,
        "base_value": base_value,
        "wave_modifier": wave_modifier,
        "dr_adjustment": dr_adjustment,
        "angle_adjustment": angle_adjustment,
        "angle": angle,
        "digital_root": dr
    }

def fill_prime_gaps(start_n, end_n, step_count):
    """
    Generate prime predictions for a specific interval
    """
    step_size = (end_n - start_n) / step_count
    results = []
    
    print(f"\nFilling gaps between n={start_n} and n={end_n}:")
    
    success_count = 0
    
    for i in range(step_count + 1):
        n = start_n + i * step_size
        result = tssm_wave_formula(n)
        predicted_value = round(result["predicted_value"])
        nearest_prime = find_nearest_prime(predicted_value)
        
        # Skip if no prime was found
        if nearest_prime is None:
            print(f"  n = {n:.4f}: No prime found within ±1000")
            continue
            
        is_prime_result = isprime(nearest_prime)
        distance = abs(predicted_value - nearest_prime)
        distance_percent = (distance / predicted_value) * 100
        
        # Determine success criterion - distance less than 0.1%
        is_success = distance_percent < 0.1
        if is_success:
            success_count += 1
        
        results.append({
            "n": n,
            "predicted_value": predicted_value,
            "nearest_prime": nearest_prime,
            "is_prime": is_prime_result,
            "distance": distance,
            "distance_percent": distance_percent,
            "angle": result["angle"],
            "digital_root": result["digital_root"],
            "success": is_success
        })
        
        print(f"  n = {n:.4f}: Predicted = {predicted_value}, " +
              f"Nearest Prime = {nearest_prime}, " +
              f"Distance = {distance_percent:.6f}%, " +
              f"Angle = {result['angle']:.2f}°, " +
              f"DR = {result['digital_root']}")
    
    success_rate = (success_count / (step_count + 1)) * 100
    print(f"Generated {step_count + 1} predictions with success rate: {success_rate:.2f}%")
    
    return results

def create_optimized_prime_set(start_n, end_n, density=10):
    """
    Create an optimized set of primes with specified density
    """
    # Generate a dense set of candidate points
    dense_factor = 4  # Generate 4x more candidates than needed
    results = fill_prime_gaps(start_n, end_n, density * dense_factor)
    
    # Filter out results where no prime was found
    valid_results = [r for r in results if r["nearest_prime"] is not None]
    
    # If we have fewer valid results than requested density, return all we have
    if len(valid_results) <= density:
        valid_results.sort(key=lambda x: x["n"])
        return valid_results
    
    # Sort by distance percentage (ascending)
    valid_results.sort(key=lambda x: x["distance_percent"])
    
    # Take the top 'density' results
    top_results = valid_results[:density]
    
    # Sort back by n value
    top_results.sort(key=lambda x: x["n"])
    
    print("\nOptimized Prime Set:")
    for result in top_results:
        print(f"n = {result['n']:.4f}: " +
              f"Prime = {result['nearest_prime']}, " +
              f"Distance = {result['distance_percent']:.6f}%, " +
              f"Angle = {result['angle']:.2f}°, " +
              f"DR = {result['digital_root']}")
    
    return top_results

def analyze_digital_roots(primes):
    """
    Analyze digital root distribution
    """
    dr_counts = [0] * 10  # 0-9
    
    for prime in primes:
        dr_counts[prime["digital_root"]] += 1
    
    return dr_counts[1:]  # Exclude 0

def analyze_angles(primes):
    """
    Analyze angular distribution
    """
    # Define angle ranges for analysis
    angle_ranges = [
        {"name": "0° - 60°", "min": 0, "max": 60},
        {"name": "60° - 120°", "min": 60, "max": 120},
        {"name": "120° - 180°", "min": 120, "max": 180},
        {"name": "180° - 240°", "min": 180, "max": 240},
        {"name": "240° - 300°", "min": 240, "max": 300},
        {"name": "300° - 360°", "min": 300, "max": 360}
    ]
    
    angle_counts = [{"range": range_info["name"], "count": 0} for range_info in angle_ranges]
    
    for prime in primes:
        angle = prime["angle"]
        for i, range_info in enumerate(angle_ranges):
            if range_info["min"] <= angle < range_info["max"]:
                angle_counts[i]["count"] += 1
                break
    
    return angle_counts

def analyze_key_angle_proximity(primes):
    """
    Check proximity to key angles
    """
    key_counts = []
    for name, angle in key_angles.items():
        key_counts.append({
            "name": name,
            "angle": angle,
            "count": 0
        })
    
    for prime in primes:
        angle = prime["angle"]
        
        for key_count in key_counts:
            key_angle = key_count["angle"]
            distance = min(
                abs(angle - key_angle),
                abs(angle - key_angle + 360),
                abs(angle - key_angle - 360)
            )
            
            if distance < 10:  # Within 10 degrees
                key_count["count"] += 1
    
    return key_counts

def visualize_prime_distribution(prime_set):
    """
    Create visualizations of the prime distribution
    """
    n_values = [p["n"] for p in prime_set]
    prime_values = [p["nearest_prime"] if p["nearest_prime"] is not None else None for p in prime_set]
    angles = [p["angle"] for p in prime_set]
    digital_roots = [p["digital_root"] for p in prime_set]
    
    # Filter out None values
    valid_indices = [i for i, p in enumerate(prime_values) if p is not None]
    valid_n = [n_values[i] for i in valid_indices]
    valid_primes = [prime_values[i] for i in valid_indices]
    valid_angles = [angles[i] for i in valid_indices]
    valid_drs = [digital_roots[i] for i in valid_indices]
    
    # 1. Prime Distribution by n value
    plt.figure(figsize=(12, 8))
    plt.plot(valid_n, valid_primes, 'bo-', alpha=0.6)
    
    # Add system points
    sys_n = [s["n"] for s in systems]
    sys_values = [s["value"] for s in systems]
    plt.plot(sys_n, sys_values, 'ro', markersize=10, label='System Points')
    
    # Add labels
    for i, sys in enumerate(systems):
        plt.text(sys["n"], sys["value"]*1.05, sys["name"], fontsize=12)
    
    plt.title('TSSM Prime Distribution')
    plt.xlabel('n value')
    plt.ylabel('Prime Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('tssm_prime_distribution.png')
    
    # 2. Angular Distribution
    plt.figure(figsize=(10, 8))
    plt.scatter(valid_angles, valid_primes, c=valid_drs, cmap='viridis', alpha=0.7)
    
    # Add key angles
    for name, angle in key_angles.items():
        plt.axvline(x=angle, color='r', linestyle='--', alpha=0.5, label=f'{name} ({angle}°)')
    
    plt.title('TSSM Angular Distribution of Primes')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Prime Value')
    plt.colorbar(label='Digital Root')
    plt.grid(True, alpha=0.3)
    plt.savefig('tssm_angular_distribution.png')
    
    # 3. Digital Root Distribution
    dr_counts = analyze_digital_roots(prime_set)
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, 10), dr_counts)
    plt.title('Digital Root Distribution')
    plt.xlabel('Digital Root')
    plt.ylabel('Count')
    plt.xticks(range(1, 10))
    plt.grid(True, alpha=0.3)
    plt.savefig('tssm_digital_root_distribution.png')

def main():
    print("TSSM Prime Gap Analysis")
    print("======================")
    
    # Test the formula on system points
    print("\nTesting TSSM Wave Formula on System Points:")
    for system in systems:
        result = tssm_wave_formula(system["n"])
        predicted = result["predicted_value"]
        actual = system["value"]
        error = abs((predicted - actual) / actual * 100)
        
        print(f"{system['name']} (n={system['n']}): " +
              f"Predicted = {predicted:.2f}, " +
              f"Actual = {actual}, " +
              f"Error = {error:.2f}%")
    
    # Generate optimized prime sets for each system transition
    all_primes = []
    
    for i in range(len(systems) - 1):
        start_system = systems[i]
        end_system = systems[i + 1]
        transition_name = f"{start_system['name']}-{end_system['name']}"
        
        print(f"\n{transition_name} Gap Analysis:")
        prime_set = create_optimized_prime_set(start_system["n"], end_system["n"], density=10)
        all_primes.extend(prime_set)
    
    # Create a complete prime set including system points
    complete_prime_set = []
    
    # Add system points
    for system in systems:
        complete_prime_set.append({
            "n": system["n"],
            "nearest_prime": system["value"],
            "angle": calculate_angle(system["n"]),
            "digital_root": digital_root(system["value"]),
            "name": system["name"]
        })
    
    # Add all intermediate primes
    for prime in all_primes:
        complete_prime_set.append(prime)
    
    # Sort by n value
    complete_prime_set.sort(key=lambda x: x["n"])
    
    # Print the complete set
    print("\nComplete TSSM Prime Set (S₇₉ to S₈₅):")
    print("=======================================")
    for item in complete_prime_set:
        label = item.get("name", "")
        prime_value = item["nearest_prime"] if item["nearest_prime"] is not None else "No prime found"
        print(f"n = {item['n']:.4f}: {prime_value} {label}")
    
    print(f"\nTotal number of primes in the complete set: {len(complete_prime_set)}")
    
    # Analyze the prime set
    print("\nAnalysis of Optimized Prime Sets")
    print("===============================")
    
    # Analyze digital root distribution
    valid_primes = [p for p in all_primes if p["nearest_prime"] is not None]
    dr_counts = analyze_digital_roots(valid_primes)
    total_valid = sum(dr_counts)
    
    print("\nDigital Root Distribution:")
    for i, count in enumerate(dr_counts, 1):
        print(f"Digital Root {i}: {count} primes ({count/total_valid*100:.2f}%)")
    
    # Analyze angular distribution
    angle_counts = analyze_angles(valid_primes)
    
    print("\nAngular Distribution:")
    for range_info in angle_counts:
        print(f"{range_info['range']}: {range_info['count']} primes " +
              f"({range_info['count']/total_valid*100:.2f}%)")
    
    # Analyze key angle proximity
    key_angle_counts = analyze_key_angle_proximity(valid_primes)
    
    print("\nKey Angle Proximity (within 10°):")
    for key in key_angle_counts:
        print(f"{key['name']} ({key['angle']}°): {key['count']} primes " +
              f"({key['count']/total_valid*100:.2f}%)")
    
    # Visualize the results
    visualize_prime_distribution(complete_prime_set)
    
    print("\nAnalysis complete. Visualization images saved.")

if __name__ == "__main__":
    main()
