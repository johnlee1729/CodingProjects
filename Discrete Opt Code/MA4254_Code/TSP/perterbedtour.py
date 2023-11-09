import random

def generate_perturbed_tour(tour):
    n = len(tour)
    i = random.randint(0, n - 2)  # Randomly select i
    j = random.randint(i + 1, n - 1)  # Randomly select j such that i < j

    perturbed_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]

    return perturbed_tour

def generate_perturbed_tour_2(input_tour):
    n = len(input_tour)
    i = random.randint(0, n-2)  # Randomly select i such that 0 <= i <= n-2
    j = random.randint(i+1, n-1)  # Randomly select j such that i+1 <= j <= n-1

    perturbed_tour = input_tour[:i] + input_tour[j:j+1][::-1] + input_tour[i+1:j] + input_tour[i:i+1] + input_tour[j+1:]
    
    return perturbed_tour

input_tour = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
perturbed_tour = generate_perturbed_tour(input_tour)
perturbed_tour_2 = generate_perturbed_tour_2(input_tour)

print(f"Input Tour: {input_tour}")
print(f"Perturbed Tour (Method 1): {perturbed_tour}")
print(f"Perturbed Tour (Method 2): {perturbed_tour_2}")
