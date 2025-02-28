#!/usr/bin/env python3
# TSSM Prime Calculator for MacBook Pro M2
# This script calculates prime numbers based on the Toroidal Structure Source Split Model
# optimized for memory and processing constraints of a MacBook Pro M2

import math
import multiprocessing
import numpy as np
import time
import psutil
import os
import json
from tqdm import tqdm

# TSSM Constants
PHI = 1.618034  # Golden ratio
GOLDEN_ANGLE = 137.5  # Golden angle in degrees
BOUNDARY_ANGLE = 39.1  # 13-cycle boundary angle
STAR_13_ANGLE = 360 / 13  # 13-point star angle ≈ 27.69°

# Initial Fibonacci primes for harmonic starts
FIBONACCI_PRIMES = [2, 3, 5, 13, 89, 233, 1597, 28657, 514229, 24157817, 39088169, 433494437]
FIBONACCI_POSITIONS = [3, 4, 5, 7, 11, 13, 17, 23, 29, 37, 39, 43]

# Key angles for prime clustering
KEY_ANGLES = [0, 137.5, 275]  # Primary angles

class TSMMCalculator:
    def __init__(self, max_harmonics=None, max_memory_pct=70, output_file="tssm_results.json"):
        """
        Initialize the TSSM Calculator
        
        Args:
            max_harmonics: Maximum number of harmonic systems to calculate
            max_memory_pct: Maximum percentage of system memory to use
            output_file: File to save results
        """
        self.max_harmonics = max_harmonics
        self.max_memory_pct = max_memory_pct
        self.output_file = output_file
        self.harmonics = {}
        self.total_primes = 0
        self.available_cores = multiprocessing.cpu_count()
        self.helix_split_detected = False
        self.helix_split_info = None
        
        # Determine system capabilities
        self.memory_limit = self._get_memory_limit()
        print(f"System has {self.available_cores} CPU cores")
        print(f"Memory limit set to {self.memory_limit / (1024**3):.2f} GB")
        
        # Initialize result data structure
        self.results = {
            "harmonics": {},
            "fibonacci_primes": [],
            "helix_split": None,
            "key_angle_primes": {
                "0": [],
                "137.5": [],
                "275": []
            },
            "stats": {
                "total_harmonics": 0,
                "total_primes": 0,
                "largest_prime": 0,
                "computation_time": 0
            }
        }
    
    def _get_memory_limit(self):
        """Calculate memory limit based on system memory and max percentage"""
        system_memory = psutil.virtual_memory().total
        return system_memory * (self.max_memory_pct / 100)
    
    def is_prime(self, n):
        """Check if a number is prime using an optimized algorithm"""
        if n <= 1:
            return False
        if n <= 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        
        # Only check divisors up to sqrt(n)
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    def find_next_prime(self, current_prime):
        """Find the next prime that approximately follows the golden ratio scaling"""
        target_value = current_prime * PHI
        candidate = round(target_value)
        
        # Ensure the candidate is odd (except for 2)
        if candidate > 2 and candidate % 2 == 0:
            candidate += 1
        
        # Find the next prime
        while not self.is_prime(candidate):
            candidate += 2
        
        return candidate
    
    def digital_root(self, n):
        """Calculate the digital root of a number"""
        if n == 0:
            return 0
        return 1 + ((n - 1) % 9)
    
    def calculate_harmonic_system(self, start_prime, harmonic_number):
        """Calculate a harmonic system starting from a given prime"""
        system = []
        current_number = start_prime
        
        for step in range(1, 14):  # 13 steps
            # Calculate angle using the golden angle
            angle = ((step - 1) * GOLDEN_ANGLE) % 360
            
            # Calculate radius using φ scaling
            radius = start_prime * (PHI ** (step - 1))
            
            # Determine helix strand (1 or 2)
            helix = 1 if step % 2 == 1 else 2
            
            # Calculate z-coordinate (depth) in the crystalline lattice
            h = PHI * start_prime / 13
            z = step * h
            
            # Calculate 3D position
            theta_rad = math.radians(angle)
            x = radius * math.cos(theta_rad)
            y = radius * math.sin(theta_rad)
            
            # Determine if this prime is at a key angle
            is_harmonic_start = step == 1
            is_harmonic_peak = abs(angle - 275) < 30  # Near 275°
            is_transition_point = abs(angle - 137.5) < 30  # Near 137.5°
            is_fibonacci_prime = current_number in FIBONACCI_PRIMES
            
            # Check for helix split
            if not self.helix_split_detected:
                # If a Fibonacci prime is detected at 137.5° but NOT at 275°, this may indicate a helix split
                if is_fibonacci_prime and is_transition_point and harmonic_number > 1:
                    self.helix_split_detected = True
                    self.helix_split_info = {
                        "harmonic": harmonic_number,
                        "position": step,
                        "prime": current_number,
                        "angle": angle
                    }
                    print(f"\n[!] HELIX SPLIT DETECTED at Harmonic {harmonic_number}, " 
                          f"Position {step}, Prime {current_number}, Angle {angle:.2f}°")
            
            # Add the prime to the system
            prime_data = {
                "value": current_number,
                "position": (harmonic_number - 1) * 13 + step,
                "system_position": step,
                "angle": angle,
                "radius": radius,
                "helix": helix,
                "z_depth": z,
                "coords": [x, y, z],
                "is_harmonic_start": is_harmonic_start,
                "is_harmonic_peak": is_harmonic_peak,
                "is_transition_point": is_transition_point,
                "is_fibonacci_prime": is_fibonacci_prime,
                "digital_root": self.digital_root(current_number)
            }
            
            system.append(prime_data)
            
            # Update results for key angles
            for key_angle in KEY_ANGLES:
                if abs(angle - key_angle) < 15:  # Within 15 degrees
                    key_str = str(key_angle)
                    self.results["key_angle_primes"][key_str].append({
                        "prime": current_number,
                        "harmonic": harmonic_number,
                        "position": step,
                        "exact_angle": angle,
                        "helix": helix
                    })
            
            # Check if we should track this Fibonacci prime
            if is_fibonacci_prime:
                self.results["fibonacci_primes"].append({
                    "prime": current_number,
                    "harmonic": harmonic_number,
                    "position": step,
                    "angle": angle,
                    "helix": helix
                })
            
            # Find the next prime for the next step
            if step < 13:
                current_number = self.find_next_prime(current_number)
        
        # Create the harmonic system object
        harmonic_system = {
            "harmonic_number": harmonic_number,
            "system_range": f"S{(harmonic_number - 1) * 13 + 1}-S{harmonic_number * 13}",
            "start_prime": start_prime,
            "end_prime": system[-1]["value"],
            "sum": sum(p["value"] for p in system),
            "sum_digital_root": self.digital_root(sum(p["value"] for p in system)),
            "z_depth": system[0]["z_depth"],
            "primes": system,
            "fibonacci_prime_count": sum(1 for p in system if p["is_fibonacci_prime"])
        }
        
        return harmonic_system
    
    def find_next_fibonacci_prime(self, current_fibonacci_prime):
        """Find the next Fibonacci prime after the current one"""
        try:
            idx = FIBONACCI_PRIMES.index(current_fibonacci_prime)
            if idx < len(FIBONACCI_PRIMES) - 1:
                return FIBONACCI_PRIMES[idx + 1]
        except ValueError:
            pass
        
        # If we don't have the next Fibonacci prime pre-calculated,
        # use an approximation by scaling by φ²
        # This is an approximation; true Fibonacci primes are sparser
        p1 = self.find_next_prime(current_fibonacci_prime)
        p2 = self.find_next_prime(p1)
        return p2
    
    def estimate_memory_usage(self, harmonic_number, largest_prime):
        """Estimate memory usage for the next harmonic system"""
        # Base memory per prime data structure (conservative estimate)
        bytes_per_prime = 500  # Each prime data structure with all properties
        
        # Estimate the next system's largest prime
        estimated_next_largest = largest_prime * (PHI ** 13)
        
        # Memory for the new harmonic system
        estimated_memory = 13 * bytes_per_prime  # 13 primes per harmonic
        
        # Add overhead for processing
        estimated_memory *= 2  # 2x for processing overhead
        
        return estimated_memory
    
    def calculate_max_harmonics(self):
        """Calculate the maximum number of harmonic systems we can compute given memory constraints"""
        # Start with the basic first harmonic system
        current_harmonic = 1
        largest_prime = 41  # End of first harmonic
        
        # Initial memory estimate
        memory_used = 0
        
        while True:
            # Estimate memory for next harmonic
            next_memory = self.estimate_memory_usage(current_harmonic + 1, largest_prime)
            
            # Check if adding the next harmonic would exceed our memory limit
            if memory_used + next_memory > self.memory_limit:
                break
            
            # Update tracking variables
            memory_used += next_memory
            current_harmonic += 1
            largest_prime *= (PHI ** 13)  # Rough estimate of largest prime after 13 steps
            
            # Stop if we hit user-specified max
            if self.max_harmonics is not None and current_harmonic >= self.max_harmonics:
                break
            
            # Safety check for unreasonably large numbers
            if largest_prime > 10**100:  # Arbitrary limit to prevent overflow
                break
        
        return current_harmonic
    
    def calculate_harmonics(self):
        """Calculate as many harmonic systems as memory allows"""
        start_time = time.time()
        
        # Determine how many harmonics we can calculate
        if self.max_harmonics is None:
            self.max_harmonics = self.calculate_max_harmonics()
        
        print(f"Will calculate {self.max_harmonics} harmonic systems")
        
        # First harmonic system S₁-S₁₃ starting with 2
        current_harmonic = 1
        start_prime = 2
        
        # Loop through and calculate each harmonic system
        progress_bar = tqdm(total=self.max_harmonics, desc="Calculating harmonics")
        
        while current_harmonic <= self.max_harmonics:
            # Calculate this harmonic system
            harmonic_system = self.calculate_harmonic_system(start_prime, current_harmonic)
            
            # Add to results
            self.harmonics[current_harmonic] = harmonic_system
            
            # Store in the results data structure
            self.results["harmonics"][current_harmonic] = {
                "system_range": harmonic_system["system_range"],
                "start_prime": harmonic_system["start_prime"],
                "end_prime": harmonic_system["end_prime"],
                "sum": harmonic_system["sum"],
                "sum_digital_root": harmonic_system["sum_digital_root"],
                "z_depth": harmonic_system["z_depth"],
                "fibonacci_prime_count": harmonic_system["fibonacci_prime_count"]
            }
            
            # Update statistics
            self.total_primes += 13
            if harmonic_system["end_prime"] > self.results["stats"]["largest_prime"]:
                self.results["stats"]["largest_prime"] = harmonic_system["end_prime"]
            
            # Determine the next starting prime
            if current_harmonic < len(FIBONACCI_PRIMES):
                # Use the next Fibonacci prime from our pre-calculated list
                start_prime = FIBONACCI_PRIMES[current_harmonic]
            else:
                # Find the next appropriate starting prime
                start_prime = self.find_next_fibonacci_prime(start_prime)
            
            # Save progress periodically
            if current_harmonic % 5 == 0:
                self.save_results(include_primes=False)
            
            current_harmonic += 1
            progress_bar.update(1)
        
        progress_bar.close()
        
        # Record helix split if detected
        if self.helix_split_detected:
            self.results["helix_split"] = self.helix_split_info
        
        # Update final statistics
        self.results["stats"]["total_harmonics"] = self.max_harmonics
        self.results["stats"]["total_primes"] = self.total_primes
        self.results["stats"]["computation_time"] = time.time() - start_time
        
        print(f"\nCalculation completed in {self.results['stats']['computation_time']:.2f} seconds")
        print(f"Calculated {self.max_harmonics} harmonic systems with {self.total_primes} primes")
        print(f"Largest prime found: {self.results['stats']['largest_prime']}")
        
        return self.harmonics
    
    def analyze_patterns(self):
        """Analyze patterns in the calculated primes"""
        # 1. Digital root analysis
        digital_root_counts = {i: 0 for i in range(1, 10)}
        
        for h_num, harmonic in self.harmonics.items():
            for prime_data in harmonic["primes"]:
                digital_root_counts[prime_data["digital_root"]] += 1
        
        print("\nDigital Root Distribution:")
        for root, count in digital_root_counts.items():
            percentage = (count / self.total_primes) * 100
            print(f"Digital Root {root}: {count} primes ({percentage:.2f}%)")
        
        # 2. Helix analysis
        helix_counts = {1: 0, 2: 0}
        for h_num, harmonic in self.harmonics.items():
            for prime_data in harmonic["primes"]:
                helix_counts[prime_data["helix"]] += 1
        
        print("\nHelix Distribution:")
        for helix, count in helix_counts.items():
            percentage = (count / self.total_primes) * 100
            print(f"Helix {helix}: {count} primes ({percentage:.2f}%)")
        
        # 3. Angular distribution around key angles
        angle_counts = {angle: 0 for angle in KEY_ANGLES}
        for h_num, harmonic in self.harmonics.items():
            for prime_data in harmonic["primes"]:
                for key_angle in KEY_ANGLES:
                    if abs(prime_data["angle"] - key_angle) < 15:
                        angle_counts[key_angle] += 1
        
        print("\nKey Angle Distribution:")
        for angle, count in angle_counts.items():
            percentage = (count / self.total_primes) * 100
            print(f"Angle {angle}°: {count} primes ({percentage:.2f}%)")
        
        # 4. Fibonacci prime analysis
        fibonacci_count = sum(1 for h_num, harmonic in self.harmonics.items() 
                            for prime_data in harmonic["primes"] 
                            if prime_data["is_fibonacci_prime"])
        
        print(f"\nFibonacci Primes: {fibonacci_count} out of {self.total_primes} " 
              f"({(fibonacci_count/self.total_primes)*100:.2f}%)")
        
        # Add analyses to results
        self.results["analysis"] = {
            "digital_root_distribution": digital_root_counts,
            "helix_distribution": helix_counts,
            "key_angle_distribution": angle_counts,
            "fibonacci_prime_percentage": (fibonacci_count/self.total_primes)*100
        }
        
        return self.results["analysis"]
    
    def save_results(self, include_primes=True):
        """Save results to a JSON file"""
        # Create a copy of results for saving
        save_data = self.results.copy()
        
        # Optionally exclude full prime details to save space
        if not include_primes:
            for h_num in save_data["harmonics"]:
                if "primes" in save_data["harmonics"][h_num]:
                    del save_data["harmonics"][h_num]["primes"]
        
        # Save to file
        with open(self.output_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"Results saved to {self.output_file}")
    
    def find_prime_at_position(self, position):
        """Find the prime at a specific position in the sequence"""
        harmonic_number = ((position - 1) // 13) + 1
        position_in_harmonic = ((position - 1) % 13) + 1
        
        if harmonic_number in self.harmonics:
            harmonic = self.harmonics[harmonic_number]
            for prime_data in harmonic["primes"]:
                if prime_data["system_position"] == position_in_harmonic:
                    return prime_data
        
        return None
    
    def search_prime(self, prime_value):
        """Search for a specific prime value in the calculated harmonics"""
        for h_num, harmonic in self.harmonics.items():
            for prime_data in harmonic["primes"]:
                if prime_data["value"] == prime_value:
                    return prime_data
        
        return None


def main():
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description='TSSM Prime Calculator for MacBook Pro M2')
    parser.add_argument('--max-harmonics', type=int, default=None, 
                        help='Maximum number of harmonic systems to calculate')
    parser.add_argument('--max-memory', type=int, default=70,
                        help='Maximum percentage of system memory to use (default: 70)')
    parser.add_argument('--output', type=str, default='tssm_results.json',
                        help='Output file for results (default: tssm_results.json)')
    parser.add_argument('--analyze-only', action='store_true',
                        help='Only analyze existing results without calculating new ones')
    
    args = parser.parse_args()
    
    # Create calculator instance
    calculator = TSMMCalculator(
        max_harmonics=args.max_harmonics,
        max_memory_pct=args.max_memory,
        output_file=args.output
    )
    
    # Perform calculations or analysis
    if not args.analyze_only:
        calculator.calculate_harmonics()
    
    # Analyze patterns
    calculator.analyze_patterns()
    
    # Save results
    calculator.save_results(include_primes=True)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
