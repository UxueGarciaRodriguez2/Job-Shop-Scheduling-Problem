import random

##########################
#### DATA LOAD ###########
##########################

def extract_instances_to_dict(file_path, instances_to_extract):
    """
    Extracts and organizes specific instances of a file.

    param file_path: Path to the input file.
    param instances_to_extract_instances: List of names of instances to extract.
    return: Dictionary with the extracted instances.
    """
    instances_dict = {}
    current_instance = None
    instance_data = []

    with open(file_path, "r") as file:
        data = file.readlines()

    for line in data:
        line = line.strip()  # Eliminate blank spaces

        ## Skip lines with +++++++++++++++++++++
        if line.startswith("+++++++++++++++++++++++++++++") or not line:
            continue
        
        # Start of an instance
        if line.startswith("instance"):
            if current_instance:
                if current_instance in instances_to_extract:
                    instances_dict[current_instance] = parse_instance(instance_data)
            current_instance = line.split()[1].strip()  # name of the instance
            instance_data = [] 
        else:
            if current_instance:
                instance_data.append(line)  

    # Save the last instance processed if matches
    if current_instance and current_instance in instances_to_extract:
        instances_dict[current_instance] = parse_instance(instance_data)

    return instances_dict


def parse_instance(instance_data):
    """
    Converts the instance data into a structured format.

    param instance_data: List of lines with the instance data.
    return: Dictionary with the dimensions and the time matrix.
    """
    # Extract dimensions
    dimensions = instance_data[1].split()
    num_jobs = int(dimensions[0])  # N of jobs
    num_machines = int(dimensions[1])  # N of machines

    # Times
    processing_times = []
    for line in instance_data[2:]:
        row = list(map(int, line.split()))
        processing_times.append(row)

    return {
        "num_jobs": num_jobs,
        "num_machines": num_machines,
        "processing_times": processing_times
    }



####################################
####### CHROMOSOME ENCODING ########
####################################


def generate_chromosome(num_jobs, num_machines):
    """
    Generate a random chromosome for JSSP.
    Each job appears exactly num_machines times.
    """
    chromosome = []
    for job in range(num_jobs):
        chromosome.extend([job] * num_machines)
    random.shuffle(chromosome)  

    return chromosome


##################################
######### VALIDATION ############
#################################

def validate_chromosome(chromosome, num_jobs, num_machines, processing_times):
    """
    Validates a chromosome for JSSP constraints.
    Ensures:
    1. No task for a job can be started until the previous task for that job is completed.
    2. A machine can only work on one task at a time.
    3. A task, once started, must run to completion.
    """
    # Completion time tracking per job and machine
    job_completion_time = [0] * num_jobs  # Completion time for each job
    machine_completion_time = [0] * num_machines  # Completion time for each machine

    # Current operation index for each job
    job_operation_index = [0] * num_jobs

    # Iterate over the chromosome
    for gene in chromosome:
        job = gene  # Current job
        operation_idx = job_operation_index[job]  # Current operation index for this job

        # If the operation index exceeds the number of operations, it's invalid
        if operation_idx >= len(processing_times[job]) // 2:
            return False

        # Get machine and processing time for the current operation
        machine = processing_times[job][operation_idx * 2]  # Machine assigned
        processing_time = processing_times[job][operation_idx * 2 + 1]  # Processing time

        # Calculate the start and finish time for this task
        start_time = max(job_completion_time[job], machine_completion_time[machine])
        finish_time = start_time + processing_time

        # Check if the machine is available during this period
        if start_time < machine_completion_time[machine]:
            return False  # Machine is busy

        # Update completion times
        job_completion_time[job] = finish_time
        machine_completion_time[machine] = finish_time

        # Move to the next operation for this job
        job_operation_index[job] += 1

    # Ensure all operations for all jobs have been scheduled
    for job_idx in range(num_jobs):
        if job_operation_index[job_idx] != len(processing_times[job_idx]) // 2:
            return False

    return True


#################################
##### CALCULATE THE FITNESS #####
#################################

def compute_fitness_with_validation(chromosome, num_jobs, num_machines, processing_times):
    """
    Validates the chromosome and calculates the makespan if valid.
    
    Args:
        chromosome (list): Sequence of operations that defines a solution.
        num_jobs (int): Number of jobs.
        num_machines (int): Number of machines.
        processing_times (list): Array defining the machines and times for each operation.
    
    returns:
        int: Fitness of the solution if valid, or a high value if invalid.
    """
    # validate the chromosoma
    if not validate_chromosome(chromosome, num_jobs, num_machines, processing_times):
        return float('inf')  # High penalty if the chromosome is invalid

    # Initialise completion times for jobs and machines
    job_completion_time = [0] * num_jobs
    machine_completion_time = [0] * num_machines
    job_operation_index = [0] * num_jobs 

    for gene in chromosome:
        job = gene  # Trabajo actual
        operation_idx = job_operation_index[job] 

        machine = processing_times[job][operation_idx * 2]
        processing_time = processing_times[job][operation_idx * 2 + 1]

        start_time = max(job_completion_time[job], machine_completion_time[machine])
        finish_time = start_time + processing_time

        job_completion_time[job] = finish_time
        machine_completion_time[machine] = finish_time

        job_operation_index[job] += 1

    return max(machine_completion_time)



############################
#### PARENT SELECTION ####
############################

### Turnament Selection

def tournament_selection(population, fitness_values, tournament_size=3):
    """
    Selection by tournament.
    
    Args:
        population (list): List of chromosomes in the population.
        fitness_values (list): List of corresponding fitness values.
        tournament_size (int): Number of participants in the tournament.
    
    Returns:
        list: Chromosome selected.
    """
    # Randomly select individuals for the tournament
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament = [(fitness_values[i], population[i]) for i in tournament_indices]
    
    # Selecting the individual with the best fitness
    winner = min(tournament, key=lambda x: x[0])  
    return winner[1]


### Roulette Selection

def roulette_selection(population, fitness_values):
    """
    Roulette Wheel Selection (Roulette Wheel Selection).
    
    Args:
        population (list): List of chromosomes in the population.
        fitness_values (list): List of corresponding fitness values.
    
    Returns:
        list: Chromosome selected.
    """

    total_fitness = sum(fitness_values)
    
    # Select a random value
    selection_value = random.uniform(0, total_fitness)
    
    cumulative_sum = 0
    for i, fitness in enumerate(fitness_values):
        cumulative_sum += fitness
        if cumulative_sum >= selection_value:
            return population[i]

################################
######### CROSSOVER ##########
##############################

### Cycle Crossover
      
def cycle_crossover(parent1, parent2):

    # Validate that parents have the same size
    if len(parent1) != len(parent2):
        raise ValueError("Chromosomes must have the same size")

    child = [None] * len(parent1)

    copied = set()

    i = 0
    while len(copied) < len(parent1):

        cycle_start = i
        while child[cycle_start] is None:

            # Copy value from parent1 to child
            child[cycle_start] = parent1[cycle_start]
            copied.add(parent1[cycle_start])

            # Move to the next index in the cycle
            cycle_start = parent2.index(parent1[cycle_start])

        # Search for the following index that is not already copied
        i = next((index for index, value in enumerate(child) if value is None), None)
        if i is None:
            break  

    # Assign remaining parent2 values that have not been copied
    for i in range(len(parent2)):
        if child[i] is None:
            child[i] = parent2[i]

    return child

### One point crossover

def one_point_crossover(parent1, parent2):

    # Validate that parents have the same size
    if len(parent1) != len(parent2):
        raise ValueError("Chromosomes must have the same size")
    
    # Select a random cut-off point
    crossover_point = random.randint(1, len(parent1) - 1)  

    child = parent1[:crossover_point] + parent2[crossover_point:]
    
    return child


##########################
##### MUTATION #########
########################


def mutation(cromosoma):

    # Choosing two random indices on the chromosome
    idx1, idx2 = random.sample(range(len(cromosoma)), 2)
    
    # Swapping the genes in these indices
    cromosoma[idx1], cromosoma[idx2] = cromosoma[idx2], cromosoma[idx1]
    
    return cromosoma

def mutate_rotation(cromosoma):

    # Choosing two random indices on the chromosome
    punto_mutacion = random.randint(0, len(cromosoma) - 1)
    
    # Rotate: the genes after the mutation point go to the beginning.
    cromosoma_mutado = cromosoma[punto_mutacion:] + cromosoma[:punto_mutacion]
    
    return cromosoma_mutado


