import os
import json
import csv

def parse_estimates(criterion_dir, variant):
    results = []
    
    for group in os.listdir(criterion_dir):
        group_path = os.path.join(criterion_dir, group)
        
        if not os.path.isdir(group_path):
            continue
        
        variant_path = os.path.join(group_path, variant, "new")
        if not os.path.isdir(variant_path):
            continue

        try:
            with open(os.path.join(variant_path, 'benchmark.json')) as f:
                benchmark_data = json.load(f)
            with open(os.path.join(variant_path, 'estimates.json')) as f:
                estimates_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"Error reading files in {variant_path}")
            continue

        # Extract group dimensions (M, K, N) from the group_id (e.g., "4x256x256")
        group_id = benchmark_data['group_id']
        m, k, n = map(int, group_id.split('x'))

        # Get the number of elements and mean duration from JSON files
        num_elements = benchmark_data['throughput']['Elements']
        mean_duration_ns = estimates_data['mean']['point_estimate']
        throughput = (num_elements / mean_duration_ns)  # Convert ns to seconds & elements to GElements

        results.append({
            'variant': variant,
            'M': m,
            'K': k,
            'N': n,
            'throughput': throughput
        })
    
    return results

def write_to_csv(variant, results):
    filename = f"{variant}_throughput.csv"
    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['variant', 'M', 'K', 'N', 'throughput'])
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {filename}")

def main(criterion_dir):
    variants = ['float16', 'int8', 'int4']
    for variant in variants:
        results = parse_estimates(criterion_dir, variant)
        write_to_csv(variant, results)

if __name__ == "__main__":
    criterion_directory = 'target/criterion'
    main(criterion_directory)
