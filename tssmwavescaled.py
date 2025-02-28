import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sympy import isprime

# Constants
phi = 1.618033988749895
S79_value = 24157817

# System points for reference - these represent the full S₇₉-S₈₅ systems
LARGE_SYSTEMS = [
    {"name": "S₇₉", "n": 1, "value": 24157817},
    {"name": "S₈₀", "n": 2, "value": 39088169},
    {"name": "S₈₁", "n": 3, "value": 63245987},
    {"name": "S₈₂", "n": 4, "value": 102334157},
    {"name": "S₈₃", "n": 5, "value": 165580141},
    {"name": "S₈₄", "n": 6, "value": 267914293},
    {"name": "S₈₅", "n": 7, "value": 433494437}
]

# Small system points - first 13 primes for the foundation
SMALL_SYSTEMS = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]

# Key angles from the TSSM model
KEY_ANGLES = {
    "thirteen_cycle_boundary": 39.1,
    "digital_root_2": 56.3,
    "digital_root_1": 63.4,
    "helix_2": 137.5,
    "helix_1": 275.0
}

# Optimized parameters for the wave function
WAVE_PARAMS = {
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

def generate_fractal_wave_angles(base_harmonic, target_harmonic):
    """
    Generate wave angles that scale fractally from base harmonic to target harmonic
    """
    # Base angles for the foundation system
    base_angles = [0, 30, 60, 90, 120, 137.5, 180, 222.5, 275, 300, 330, 360]
    
    # Add key TSSM angles
    base_angles.extend([39.1, 56.3, 63.4])
    
    # If we're already at the base harmonic, return the base angles
    if target_harmonic == base_harmonic:
        return sorted(list(set([round(angle, 2) for angle in base_angles])))
    
    # Calculate scaling factor based on phi and harmonic difference
    harmonic_difference = target_harmonic - base_harmonic
    scaling_factor = phi ** harmonic_difference
    
    # Three possible scaling approaches:
    
    # 1. Direct scaling (multiply angles by scaling factor, modulo 360)
    scaled_angles_1 = [(angle * scaling_factor) % 360 for angle in base_angles]
    
    # 2. Additive scaling (add scaled offset to each angle)
    scaled_angles_2 = [(angle + (137.5 * scaling_factor)) % 360 for angle in base_angles]
    
    # 3. Recursive golden ratio rotation (phi * angle)
    scaled_angles_3 = [(angle * phi) % 360 for angle in base_angles]
    
    # Combine all approaches
    combined_angles = base_angles + scaled_angles_1 + scaled_angles_2 + scaled_angles_3
    
    # Also include the 13-cycle iterative angles for this harmonic
    cycle_angle = 39.1 / (scaling_factor)  # Shrinking the cycle angle as we go up in scale
    cycle_angles = [(n * cycle_angle) % 360 for n in range(13)]
    
    combined_angles.extend(cycle_angles)
    
    # Remove duplicates and sort
    return sorted(list(set([round(angle, 2) for angle in combined_angles])))

# Function to calculate digital root
def digital_root(n):
    """Calculate digital root of a number"""
    if n == 0:
        return 0
    return 1 + ((n - 1) % 9)

# Function to calculate angle based on TSSM model
def calculate_angle(n):
    """Calculate angle based on TSSM model"""
    return ((n - 1) * 27.69) % 360

# Function to check if a number is prime
def is_prime(n):
    """Check if a number is prime"""
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    
    # Use 6k±1 optimization
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

# Function to find the nearest prime to a given number
def find_nearest_prime(n, max_distance=1000):
    """Find the nearest prime to a given number"""
    n = round(n)
    
    if is_prime(n):
        return n
    
    i = 1
    while i <= max_distance:
        if is_prime(n - i):
            return n - i
        if is_prime(n + i):
            return n + i
        i += 1
        
    # If no prime found within the max distance, return None
    return None

# Original spiral wave function from the provided code
def spiral_wave(n, freq, amplitude, angle, phase, direction):
    """Spiral wave equation from original implementation"""
    return n * freq * (1 + amplitude * math.sin(2 * math.pi * (angle / 360 + phase))) * direction
# Advanced TSSM wave formula for larger systems
def tssm_wave_formula(n):
    """
    The optimized TSSM Wave Formula for larger systems
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
    
    if base_n < WAVE_PARAMS["phase_transition"]:
        # For early systems (before phase transition)
        # Use a quadratic function with peak at specified position
        peak_height = WAVE_PARAMS["early_peak_height"]  # Peak height as % of base value
        peak_position = WAVE_PARAMS["early_peak_position"]  # Peak position in interval
        
        # Quadratic function: a(x-p)² + h
        # where p is peak position, h is peak height
        a = -peak_height / (peak_position ** 2)
        wave_modifier = base_value * (a * ((interval_position - peak_position) ** 2) + peak_height)
    else:
        # For later systems (after phase transition)
        # Use an exponential decay function with maximum at beginning
        decay_factor = WAVE_PARAMS["late_decay_factor"]
        initial_boost = WAVE_PARAMS["late_initial_boost"]  # Initial boost as % of base value
        wave_modifier = base_value * initial_boost * math.exp(-decay_factor * interval_position)
    
    # Calculate predicted value before adjustments
    predicted_value = base_value + wave_modifier
    
    # Digital root adjustment
    dr = digital_root(round(predicted_value))
    
    # Apply adjustment based on digital root
    dr_adjustment = 0
    if dr in [1, 7, 8]:
        dr_adjustment = WAVE_PARAMS["dr_positive_adj"] * predicted_value
    elif dr in [3, 6, 9]:
        dr_adjustment = WAVE_PARAMS["dr_negative_adj"] * predicted_value
    
    # Angular adjustment
    # Apply adjustment based on proximity to key angles
    angle_adjustment = 0
    
    # Calculate distances to key angles
    distances = []
    for key_angle in KEY_ANGLES.values():
        distance = min(
            abs(angle - key_angle),
            abs(angle - key_angle + 360),
            abs(angle - key_angle - 360)
        )
        distances.append(distance)
    
    # Find minimum distance
    min_distance = min(distances)
    
    # If close to a key angle, apply a boost
    if min_distance < WAVE_PARAMS["angle_threshold"]:
        angle_adjustment = WAVE_PARAMS["angle_boost"] * predicted_value
    
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

# Basic TSSM prediction function (without fractal angles)
def unified_tssm_predict(system_range, step_size=0.1, foundation_layers=13):
    """
    Standard TSSM prediction function with fixed wave angles
    """
    predictions = {
        'Foundation': [],  # First 13 primes (spiral wave)
        'Boundary': [],    # System boundary points
        'Helix 1': [],     # Primary helix (275°)
        'Helix 2': [],     # Secondary helix (137.5°)
        'Intermediates': []  # All other prime predictions
    }
    
    start_system, end_system = system_range
    
    # 1. First, handle foundation system (S₁-S₁₃) using spiral wave approach
    if start_system <= foundation_layers:
        # Fixed wave angles (original)
        wave_angles = [0, 30, 60, 90, 120, 137.5, 180, 222.5, 275, 300, 330, 360]
        print(f"Original Wave Angles: {len(wave_angles)} angles")
        
        for n in range(1, min(foundation_layers + 1, end_system + 1)):
            # Add boundary points (system points)
            if n <= len(SMALL_SYSTEMS):
                predictions['Foundation'].append({
                    'n': n, 
                    'prime': SMALL_SYSTEMS[n-1],
                    'angle': calculate_angle(n),
                    'digital_root': digital_root(SMALL_SYSTEMS[n-1]),
                    'category': 'Foundation'
                })
                
                if n == 1 or n == 13:  # S₁ and S₁₃ are special boundary points
                    predictions['Boundary'].append({
                        'n': n, 
                        'prime': SMALL_SYSTEMS[n-1],
                        'angle': calculate_angle(n),
                        'digital_root': digital_root(SMALL_SYSTEMS[n-1]),
                        'category': 'Boundary'
                    })
            
            # Generate predictions using fixed wave angles
            for angle in wave_angles:
                val = spiral_wave(n, freq=1.7, amplitude=1.1, angle=angle, phase=0.08, direction=1)
                val += 0.2  # Positive adjustment as in original
                prime_candidate = round(val)
                
                if is_prime(prime_candidate) and 2 <= prime_candidate <= 41:
                    # Determine the category based on angle
                    if abs(angle - 275) % 360 <= 15:  # Near Helix 1
                        category = 'Helix 1'
                    elif abs(angle - 137.5) % 360 <= 15:  # Near Helix 2
                        category = 'Helix 2'
                    else:
                        category = 'Intermediates'
                    
                    predictions[category].append({
                        'n': n,
                        'prime': prime_candidate,
                        'angle': angle,
                        'digital_root': digital_root(prime_candidate),
                        'category': category,
                        'wave_distance': abs(val - prime_candidate)
                    })
    
    # 2. Handle larger systems (S₇₉ and beyond) using the advanced wave formula
    if end_system >= 79:
        # Adjust range to start from S₇₉ at minimum
        effective_start = max(start_system, 79)
        
        # Convert to n-value for TSSM formula (S₇₉ = n:1, S₈₀ = n:2, etc.)
        start_n = effective_start - 78
        end_n = end_system - 78
        
        # Generate n-values to test
        n_values = np.arange(start_n, end_n + step_size, step_size)
        
        for n in n_values:
            # Get the system index for boundary points
            sys_idx = round(n) - 1  # Convert n-value to index (0-based)
            
            # Add boundary points (system points)
            if abs(n - round(n)) < 1e-6 and 0 <= sys_idx < len(LARGE_SYSTEMS):
                system = LARGE_SYSTEMS[sys_idx]
                predictions['Boundary'].append({
                    'n': n + 78,  # Convert back to system number
                    'prime': system['value'],
                    'angle': calculate_angle(n),
                    'digital_root': digital_root(system['value']),
                    'category': 'Boundary',
                    'system_name': system['name']
                })
            
            # Generate prediction using the advanced wave formula
            result = tssm_wave_formula(n)
            predicted_value = round(result["predicted_value"])
            nearest_prime = find_nearest_prime(predicted_value)
            
            if nearest_prime is not None:
                # Determine the category based on angle
                angle = result["angle"]
                if abs(angle - 275) % 360 <= 15:  # Near Helix 1
                    category = 'Helix 1'
                elif abs(angle - 137.5) % 360 <= 15:  # Near Helix 2
                    category = 'Helix 2'
                else:
                    category = 'Intermediates'
                
                predictions[category].append({
                    'n': n + 78,  # Convert back to system number
                    'prime': nearest_prime,
                    'angle': angle,
                    'digital_root': digital_root(nearest_prime),
                    'category': category,
                    'wave_distance': abs(predicted_value - nearest_prime)
                })
    
    # Filter out duplicates within each category
    for category in predictions:
        unique_primes = {}
        for item in predictions[category]:
            prime = item['prime']
            n_val = item['n']
            
            # Keep the entry with the closest n to an integer value
            if prime not in unique_primes or abs(n_val - round(n_val)) < abs(unique_primes[prime]['n'] - round(unique_primes[prime]['n'])):
                unique_primes[prime] = item
        
        predictions[category] = list(unique_primes.values())
    
    return predictions
# Enhanced TSSM prediction function with fractal wave angles
def unified_tssm_predict_fractal(system_range, step_size=0.1, foundation_layers=13):
    """
    Unified TSSM prediction function that works across all scale ranges.
    Uses fractal wave angles that scale with harmonic level.
    
    Args:
        system_range: tuple of (start_system, end_system) to analyze
        step_size: granularity of n-values to test
        foundation_layers: number of layers for the foundation system
        
    Returns:
        Dictionary of predictions organized by category
    """
    predictions = {
        'Foundation': [],  # First 13 primes (spiral wave)
        'Boundary': [],    # System boundary points
        'Helix 1': [],     # Primary helix (275°)
        'Helix 2': [],     # Secondary helix (137.5°)
        'Intermediates': []  # All other prime predictions
    }
    
    start_system, end_system = system_range
    
    # 1. First, handle foundation system (S₁-S₁₃) using spiral wave approach
    if start_system <= foundation_layers:
        # Generate fractal wave angles for harmonic 1
        wave_angles = generate_fractal_wave_angles(1, 1)
        print(f"Harmonic 1 Wave Angles: {len(wave_angles)} angles")
        
        for n in range(1, min(foundation_layers + 1, end_system + 1)):
            # Add boundary points (system points)
            if n <= len(SMALL_SYSTEMS):
                predictions['Foundation'].append({
                    'n': n, 
                    'prime': SMALL_SYSTEMS[n-1],
                    'angle': calculate_angle(n),
                    'digital_root': digital_root(SMALL_SYSTEMS[n-1]),
                    'category': 'Foundation'
                })
                
                if n == 1 or n == 13:  # S₁ and S₁₃ are special boundary points
                    predictions['Boundary'].append({
                        'n': n, 
                        'prime': SMALL_SYSTEMS[n-1],
                        'angle': calculate_angle(n),
                        'digital_root': digital_root(SMALL_SYSTEMS[n-1]),
                        'category': 'Boundary'
                    })
            
            # Generate predictions using fractal wave angles
            for angle in wave_angles:
                val = spiral_wave(n, freq=1.7, amplitude=1.1, angle=angle, phase=0.08, direction=1)
                val += 0.2  # Positive adjustment as in original
                prime_candidate = round(val)
                
                if is_prime(prime_candidate) and 2 <= prime_candidate <= 41:
                    # Determine the category based on angle
                    if abs(angle - 275) % 360 <= 15:  # Near Helix 1
                        category = 'Helix 1'
                    elif abs(angle - 137.5) % 360 <= 15:  # Near Helix 2
                        category = 'Helix 2'
                    else:
                        category = 'Intermediates'
                    
                    predictions[category].append({
                        'n': n,
                        'prime': prime_candidate,
                        'angle': angle,
                        'digital_root': digital_root(prime_candidate),
                        'category': category,
                        'wave_distance': abs(val - prime_candidate)
                    })
    
    # 2. Handle larger systems (S₇₉ and beyond) using the advanced wave formula with fractal angles
    if end_system >= 79:
        # Adjust range to start from S₇₉ at minimum
        effective_start = max(start_system, 79)
        
        # Convert to n-value for TSSM formula (S₇₉ = n:1, S₈₀ = n:2, etc.)
        start_n = effective_start - 78
        end_n = end_system - 78
        
        # Calculate harmonic level
        harmonic_level = ((effective_start - 1) // 13) + 1
        
        # Generate fractal wave angles for this harmonic level
        wave_angles = generate_fractal_wave_angles(1, harmonic_level)
        print(f"Harmonic {harmonic_level} Wave Angles: {len(wave_angles)} angles")
        print(f"Sample angles: {wave_angles[:10]}...")
        
        # Generate n-values to test
        n_values = np.arange(start_n, end_n + step_size, step_size)
        
        for n in n_values:
            # Get the system index for boundary points
            sys_idx = round(n) - 1  # Convert n-value to index (0-based)
            
            # Add boundary points (system points)
            if abs(n - round(n)) < 1e-6 and 0 <= sys_idx < len(LARGE_SYSTEMS):
                system = LARGE_SYSTEMS[sys_idx]
                predictions['Boundary'].append({
                    'n': n + 78,  # Convert back to system number
                    'prime': system['value'],
                    'angle': calculate_angle(n),
                    'digital_root': digital_root(system['value']),
                    'category': 'Boundary',
                    'system_name': system['name']
                })
            
            # Generate predictions for each fractal wave angle
            for angle in wave_angles:
                # Use a modified wave formula that incorporates the specific angle
                base_n = math.floor(n)
                interval_position = n - base_n
                base_value = S79_value * (phi ** (base_n - 1))
                
                # Adjust wave based on the specific angle
                wave_factor = 1.0
                for key_angle, key_angle_value in KEY_ANGLES.items():
                    angle_distance = min(
                        abs(angle - key_angle_value),
                        abs(angle - key_angle_value + 360),
                        abs(angle - key_angle_value - 360)
                    )
                    if angle_distance < 10:  # Near a key angle
                        wave_factor = 1.2  # Boost waves near key angles
                
                # Apply the wave modifier based on the system phase
                if base_n < WAVE_PARAMS["phase_transition"]:
                    # Quadratic wave for early systems
                    wave_modifier = base_value * 0.3 * math.sin(2 * math.pi * (angle / 360 + 0.08))
                else:
                    # Exponential wave for later systems
                    wave_modifier = base_value * 0.05 * math.exp(-2.5 * interval_position) * math.sin(2 * math.pi * (angle / 360 + 0.08))
                
                # Apply wave factor
                wave_modifier *= wave_factor
                
                # Calculate predicted value
                predicted_value = base_value + wave_modifier
                
                # Find nearest prime
                nearest_prime = find_nearest_prime(predicted_value)
                
                if nearest_prime is not None:
                    # Determine the category based on angle
                    if abs(angle - 275) % 360 <= 15:  # Near Helix 1
                        category = 'Helix 1'
                    elif abs(angle - 137.5) % 360 <= 15:  # Near Helix 2
                        category = 'Helix 2'
                    else:
                        category = 'Intermediates'
                    
                    predictions[category].append({
                        'n': n + 78,  # Convert back to system number
                        'prime': nearest_prime,
                        'angle': angle,
                        'digital_root': digital_root(nearest_prime),
                        'category': category,
                        'wave_distance': abs(predicted_value - nearest_prime)
                    })
    
    # Filter out duplicates within each category
    for category in predictions:
        unique_primes = {}
        for item in predictions[category]:
            prime = item['prime']
            n_val = item['n']
            
            # Keep the entry with the closest n to an integer value
            if prime not in unique_primes or abs(n_val - round(n_val)) < abs(unique_primes[prime]['n'] - round(unique_primes[prime]['n'])):
                unique_primes[prime] = item
        
        predictions[category] = list(unique_primes.values())
    
    return predictions
# Function to analyze the harmonic patterns across system transitions
def analyze_harmonic_patterns(predictions):
    """
    Analyze harmonic patterns in the predicted primes across system transitions
    """
    # Extract all primes with their n-values and categories
    all_primes = []
    for category, items in predictions.items():
        for item in items:
            all_primes.append(item)
    
    # Sort by n-value
    all_primes.sort(key=lambda x: x['n'])
    
    # Define harmonic systems (groups of 13)
    harmonic_systems = {}
    
    for item in all_primes:
        n = item['n']
        harmonic = ((n - 1) // 13) + 1  # S₁-S₁₃ = H1, S₁₄-S₂₆ = H2, etc.
        
        if harmonic not in harmonic_systems:
            harmonic_systems[harmonic] = []
        
        harmonic_systems[harmonic].append(item)
    
    # Analyze each harmonic system
    harmonic_analysis = {}
    
    for harmonic, items in harmonic_systems.items():
        # Sort items by prime value
        items.sort(key=lambda x: x['prime'])
        
        # Count by category
        category_counts = {}
        for item in items:
            cat = item['category']
            if cat not in category_counts:
                category_counts[cat] = 0
            category_counts[cat] += 1
        
        # Count by digital root
        dr_counts = {}
        for item in items:
            dr = item['digital_root']
            if dr not in dr_counts:
                dr_counts[dr] = 0
            dr_counts[dr] += 1
        
        # Count by angle range
        angle_ranges = {
            "0°-60°": 0,
            "60°-120°": 0,
            "120°-180°": 0,
            "180°-240°": 0,
            "240°-300°": 0,
            "300°-360°": 0
        }
        
        for item in items:
            angle = item['angle']
            if 0 <= angle < 60:
                angle_ranges["0°-60°"] += 1
            elif 60 <= angle < 120:
                angle_ranges["60°-120°"] += 1
            elif 120 <= angle < 180:
                angle_ranges["120°-180°"] += 1
            elif 180 <= angle < 240:
                angle_ranges["180°-240°"] += 1
            elif 240 <= angle < 300:
                angle_ranges["240°-300°"] += 1
            else:
                angle_ranges["300°-360°"] += 1
        
        # Check for fibonacci primes
        fib_primes = []
        # Define a simple Fibonacci sequence check
        fibs = [1, 1]
        while fibs[-1] < max(item['prime'] for item in items):
            fibs.append(fibs[-1] + fibs[-2])
        
        for item in items:
            if item['prime'] in fibs and is_prime(item['prime']):
                fib_primes.append(item)
        
        harmonic_analysis[harmonic] = {
            'prime_count': len(items),
            'category_distribution': category_counts,
            'digital_root_distribution': dr_counts,
            'angle_distribution': angle_ranges,
            'fibonacci_primes': fib_primes,
            'system_range': (f"S{(harmonic-1)*13+1}", f"S{harmonic*13}")
        }
    
    return harmonic_analysis

# Function to visualize the TSSM predictions and harmonic patterns
def visualize_tssm(predictions, harmonic_analysis=None, save_prefix='tssm'):
    """
    Create visualizations of the TSSM predictions
    """
    # Extract all primes for visualization
    all_points = []
    categories = []
    for category, items in predictions.items():
        for item in items:
            all_points.append(item)
            categories.append(category)
    
    # Sort by n-value for consistent coloring
    sorted_indices = np.argsort([p['n'] for p in all_points])
    all_points = [all_points[i] for i in sorted_indices]
    categories = [categories[i] for i in sorted_indices]
    
    # 1. 3D Toroidal Visualization
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    category_colors = {
        'Foundation': 'black',
        'Boundary': 'red',
        'Helix 1': 'blue',
        'Helix 2': 'green',
        'Intermediates': 'purple'
    }
    
    for i, point in enumerate(all_points):
        # Convert angle to radians for plotting
        theta = np.radians(point['angle'])
        
        # Calculate toroidal coordinates
        R = 10  # Major radius
        r = 3   # Minor radius
        
        x = (R + r * np.cos(theta)) * np.cos(2 * np.pi * point['n'] / 13)
        y = (R + r * np.cos(theta)) * np.sin(2 * np.pi * point['n'] / 13)
        z = r * np.sin(theta)
        
        ax.scatter(x, y, z, c=category_colors[categories[i]], label=categories[i])
    
    # Remove duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    plt.title('TSSM Prime Distribution on Torus')
    plt.savefig(f'{save_prefix}_3d_torus.png')
    plt.close()
    
    # 2. 2D Angle vs N Plot
    plt.figure(figsize=(12, 8))
    
    for category in category_colors:
        points = [p for i, p in enumerate(all_points) if categories[i] == category]
        if points:
            plt.scatter([p['n'] for p in points], 
                       [p['angle'] for p in points],
                       c=category_colors[category],
                       label=category)
    
    plt.xlabel('System Number (n)')
    plt.ylabel('Angle (degrees)')
    plt.title('TSSM Prime Distribution by Angle')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_prefix}_angle_distribution.png')
    plt.close()

# Function to test the TSSM model on a harmonic system with fractal waves
def test_harmonic_system_fractal(harmonic_number, step_size=0.25):
    """
    Test the TSSM model on a whole harmonic system using fractal wave angles
    
    Args:
        harmonic_number: The harmonic to test (H1=1, H2=2, etc.)
        step_size: Granularity of n-values to test
    """
    print(f"TSSM Analysis of Harmonic System {harmonic_number} with Fractal Wave Angles")
    print("=" * 75)
    
    # Calculate system range for this harmonic
    start_system = (harmonic_number - 1) * 13 + 1
    end_system = harmonic_number * 13
    
    print(f"System Range: S{start_system} to S{end_system}")
    
    # Run the unified TSSM prediction function with fractal wave angles
    predictions = unified_tssm_predict_fractal((start_system, end_system), step_size)
    
    # Count the total unique primes
    total_primes = set()
    for category, items in predictions.items():
        for item in items:
            total_primes.add(item['prime'])
    
    print(f"\nTotal unique primes found: {len(total_primes)}")
    
    # Print category breakdown
    print("\nCategory breakdown:")
    for category, items in predictions.items():
        print(f"  {category}: {len(items)} primes")
    
    # Analyze harmonic patterns
    harmonic_analysis = analyze_harmonic_patterns(predictions)
    
    # Print harmonic analysis
    print("\nHarmonic Analysis:")
    for h, analysis in harmonic_analysis.items():
        print(f"\nHarmonic {h} ({analysis['system_range'][0]}-{analysis['system_range'][1]}):")
        print(f"  Total primes: {analysis['prime_count']}")
        print(f"  Category distribution: {analysis['category_distribution']}")
        print(f"  Digital root distribution: {analysis['digital_root_distribution']}")
        print(f"  Fibonacci primes: {[p['prime'] for p in analysis['fibonacci_primes']]}")
    
    return predictions, harmonic_analysis

# Run the test when the script is executed
if __name__ == "__main__":
    # Test the foundation harmonic (H1)
    test_harmonic_system_fractal(1, step_size=0.1)
    
    # Test a large harmonic (H7 corresponds to S₇₉-S₉₁)
    test_harmonic_system_fractal(7, step_size=0.25)
