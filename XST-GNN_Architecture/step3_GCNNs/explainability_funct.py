import math
import numpy as np
import pandas as pd

def nodeNames():
    """Generate a list of node names with time steps."""
    all_keys = ['AMG', 'ATF', 'ATI', 'ATP', 'CAR', 'CF1', 'CF2', 'CF3', 'CF4', 'Others',
        'GCC', 'GLI', 'LIN', 'LIP', 'MAC', 'MON', 'NTI', 'OTR', 'OXA', 'PAP',
        'PEN', 'POL', 'QUI', 'SUL', 'TTC',
        'MV hours', '# pat$_{atb}$', '# pat$_{MDR}$',
        'CAR$_{n}$', 'PAP$_{n}$', 'Others$_{n}$',
        'QUI$_{n}$', 'ATF$_{n}$', 'OXA$_{n}$', 'PEN$_{n}$',
        'CF3$_{n}$', 'GLI$_{n}$', 'CF4$_{n}$', 'SUL$_{n}$',
        'NTI$_{n}$', 'LIN$_{n}$', 'AMG$_{n}$', 'MAC$_{n}$',
        'CF1$_{n}$', 'GCC$_{n}$', 'POL$_{n}$', 'ATI$_{n}$',
        'MON$_{n}$', 'LIP$_{n}$', 'TTC$_{n}$', 'OTR$_{n}$',
        'CF2$_{n}$', 'ATP$_{n}$', 
        '# pat$_{tot}$',
        'Post change',
        'Insulin', 'Art nutrition', 'Sedation', 'Relax', 'Hepatic$_{fail}$',
        'Renal$_{fail}$', 'Coagulation$_{fail}$', 'Hemodynamic$_{fail}$',
        'Respiratory$_{fail}$', 'Multiorganic$_{fail}$',  '# transfusions',
        'Vasoactive drug', 'Dosis nems', 'Tracheo$_{hours}$', 'Ulcer$_{hours}$',
        'Hemo$_{hours}$', 'C01 PICC 1',
        'C01 PICC 2', 'C02 CVC - RJ',
        'C02 CVC - RS', 'C02 CVC - LS', 'C02 CVC - RF',
        'C02 CVC - LJ', 'C02 CVC - LF', '# catheters']

    node_names = []
    time_steps = 14
    for t in range(time_steps):
        for i in range(len(all_keys)):
            node_names.append(all_keys[i] + "_ts" + str(t))
            
    return node_names

def indicesByThreshold(lst, threshold):
    """Get indices of list elements above a certain threshold."""
    indices = [i for i, value in enumerate(lst) if value > threshold]
    return indices

def getValues(lst, indices):
    """Retrieve values from a list based on specified indices."""
    values = [lst[i] for i in indices]
    return values

def getMostImportantNodes(importance, percentage=0.75):
    """Identify the most important nodes based on importance scores."""
    importance = np.abs(importance)
    min_val = np.min(importance)
    max_val = np.max(importance)
    print("min:", min_val, "- max:", max_val)

    threshold = percentage * max_val
    print("Threshold selected:", threshold)

    node_names = nodeNames()

    idxs = indicesByThreshold(importance, threshold)
    important_nodes = getValues(node_names, idxs)
    
    return node_names, important_nodes

def commonFeatures(nodes1, nodes2):
    """Find common features between two lists of nodes."""
    set1 = set(nodes1)
    set2 = set(nodes2)

    # Find the intersection of the sets
    common_variables = set1.intersection(set2)
    
    return common_variables


def getIndexByRange(lst, lower_bound, upper_bound):
    """Get indices of elements within a specified range."""
    return [index for index, value in enumerate(lst) if lower_bound < value < upper_bound]


def splitListElements(lst):
    """Split list elements by time step identifiers."""
    return [(element.split("_t")[0], element.split("_ts")[1]) for element in lst]



def createTableWithSymbols(results):
    """Create a table with symbols from a list of (variable, time) tuples."""
    # Extract all unique variables and times
    variables = sorted(set(var for var, _ in results))
    times = sorted(set(time for _, time in results), key=int)
    min_time, max_time = int(times[0]), int(times[-1])

    # Create an empty DataFrame with labeled rows and columns
    table = pd.DataFrame(index=variables, columns=range(min_time, max_time + 1), dtype=str)

    # Mark the cells that we have in the results
    for variable, time in results:
        table.at[variable, int(time)] = "X"

    # Fill empty cells with "-"
    table.fillna("-", inplace=True)

    return table

def uniqueVariablesInSecondList(list1, list2):
    """Find variables that are unique to the second list."""
    set_list1 = set(list1)
    set_list2 = set(list2)
    unique_variables = set_list2 - set_list1
    return list(unique_variables)

def filterKeysByThreshold(occurrences, threshold):
    """Filter dictionary keys where values exceed a threshold."""
    filtered_keys = [key for key, value in occurrences.items() if value > threshold]
    return filtered_keys

def countIndexOccurrences(index_list):
    """Count occurrences of indices in a list."""
    counter = {}
    for index in index_list:
        if index in counter:
            counter[index] += 1
        else:
            counter[index] = 1
    return counter


def uncommonVariables(all_variables, list1, list2):
    """
    Function that returns the variables from all_variables that are not found in either of the two provided lists.
    
    Args:
        all_variables (list): List of all variables.
        list1 (list): First list of variables.
        list2 (list): Second list of variables.
        
    Returns:
        list: List of variables that are not found in either of the two provided lists.
    """
    # Concatenate the two lists into one
    combined_list = list1 + list2
    # Convert the combined list into a set to eliminate duplicates
    combined_set = set(combined_list)
    # Convert the list of all variables into a set
    all_variables_set = set(all_variables)
    # Get the variables that are in all variables but not in the combined list
    uncommon_variables = all_variables_set - combined_set
    # Convert the set of uncommon variables back into a list and return it
    return list(uncommon_variables)
