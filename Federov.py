""" 
Federov's Exchange Algorithm to do D-optimal design
"""

import numpy as np

### Federov Algorithm ###

# define the function to determine federov's delta
def federov_delta(row_in, row_out, info_mat_inv):
    A = row_in.T @ info_mat_inv @ row_in
    B = -(row_out.T @ info_mat_inv @ row_out)
    C = (row_in.T @ info_mat_inv @ row_in)
    D = row_in.T @ info_mat_inv @ row_out
    E = -(row_out.T @ info_mat_inv @ row_out)
    return A + (B * C) + (D ** 2) + E

# define the function to select random rows as the starting design
def random_starting_design(candidate_set_mat, num_rows):
    # select the row indices to use
    row_indices = np.random.choice(candidate_set_mat.shape[0], num_rows, replace=False)
    # form the design matrix
    design_mat = candidate_set_mat[row_indices, :]
    return design_mat, row_indices

# define the function to form the information matrix and its inverse
def form_info_mat(design_mat):
    info_mat = design_mat.T @ design_mat
    info_mat_inv = np.linalg.inv(info_mat)
    return info_mat, info_mat_inv

# define the function to determine the federov delta for each row pair
def determine_deltas(candidate_set_mat, design_mat, row_indices):
    info_mat, info_mat_inv = form_info_mat(design_mat)
    deltas = []
    switched_rows = []
    for row_in_index, row_in in enumerate(candidate_set_mat):
        for row_out_index, row_out in enumerate(design_mat):
            switched_rows.append((row_in_index, row_out_index))
            # if the row is already in the design, skip it
            if row_in_index in row_indices:
                deltas.append(0)
            else:
                delta = federov_delta(row_in, row_out, info_mat_inv)
                deltas.append(delta)

    return deltas, switched_rows

# define the function to update the design matrix
def update_design_mat(design_mat, row_indices, candidate_set_mat, deltas, switched_rows, epsilon):
    # find the row pair with the largest delta
    max_delta = max(deltas)

    gt_epsilon = False

    if max_delta > epsilon:
        # find the index of the row pair with the largest delta
        max_delta_index = deltas.index(max_delta)
        row_in_index, row_out_index = switched_rows[max_delta_index]

        # update the design matrix
        design_mat[row_out_index, :] = candidate_set_mat[row_in_index, :]
        row_indices[row_out_index] = row_in_index

        gt_epsilon = True

    return gt_epsilon

# function to determine the determinant of the information matrix
def info_mat_det(design_mat):
    info_mat = design_mat.T @ design_mat
    return np.linalg.det(info_mat)

# define the function to perform the federov algorithm
def federov(candidate_set_mat, num_rows, epsilon, return_evaluation_lists=False):
    # a variable to determine if the algorithm should continue
    gt_epsilon = True
    # a list to store the row indices of the design matrix
    row_indices_list = []
    # a list to store the determinant of the information matrix
    det_list = []
    # form the starting design
    design_mat, row_indices = random_starting_design(candidate_set_mat, num_rows)

    while gt_epsilon:
        # determine the deltas for the starting design
        deltas, switched_rows = determine_deltas(candidate_set_mat, design_mat, row_indices)
        # update the design matrix
        gt_epsilon = update_design_mat(design_mat, row_indices, candidate_set_mat, deltas, switched_rows, epsilon)
        
        if return_evaluation_lists:
            # determine the determinant of the information matrix
            det = info_mat_det(design_mat)

            det_list.append(det)
            row_indices_list.append(np.copy(row_indices))

    if return_evaluation_lists:
        return row_indices, design_mat, det, row_indices_list, det_list
    else:
        # determine the determinant of the information matrix
        det = info_mat_det(design_mat)
        return row_indices, design_mat, det    


### Adaption to make it not choose points that are too close together ###
def federov_adapted(candidate_set_mat, num_rows, epsilon, min_projection=0.9999):
    """
    This is an adaption of the federov algorithm that does not choose points that are too close.

    -> The algorithm first runs the normal federov algorithm.
    -> It then normalizes the vectors of the design matrix.
    -> It then determines the projection of each vector onto the other vectors.
    -> If some of the projections are above the minimum projection, it removes one of the points.
    -> Then the canditate set matrix is normalized.
    -> It then determines the projection of each vector in the candidate set matrix onto a vector in the design matrix.
    -> If a vector in the candidate set matrix has a projection above the minimum projection on a vector in the design matrix, it is removed.
    -> The algorithm then runs the federov algorithm on the new candidate set matrix.
    """

    # run the federov algorithm
    row_indices, design_mat, det = federov(candidate_set_mat, num_rows, epsilon, return_evaluation_lists=False)

    # normalize the vectors of the design matrix
    norm_design_mat = design_mat / np.linalg.norm(design_mat, axis=1)[:, None]

    # if some of the projections are above the minimum projection, remove one of the points
    deleted_rows = []
    indices_to_delete = []
    
    for row_index in range(norm_design_mat.shape[0]):
        if row_index not in indices_to_delete:
            # determine the projection of the vector onto the other vectors
            projections = np.abs(norm_design_mat @ norm_design_mat[row_index, :])
            # set the projection of the vector onto itself to 0
            projections[row_index] = 0
            # find the indices of the vectors with a projection above the minimum projection
            high_projections = np.where(projections > min_projection)[0]
            # if there are vectors with a projection above the minimum projection add the
            # indeces of the vectors with the high projections to the list of indeces to delete
            indices_to_delete += high_projections.tolist()

    # remove the rows with the high projections
    norm_design_mat = np.delete(norm_design_mat, indices_to_delete, axis=0)

    # figure out what row indices of the candidate set matrix were removed
    high_projection_row_indices = row_indices[indices_to_delete]

    # remove the rows with the high projections from the row indices
    row_indices = np.delete(row_indices, indices_to_delete)

    # normalize the vectors of the candidate set matrix
    norm_candidate_set_mat = candidate_set_mat / np.linalg.norm(candidate_set_mat, axis=1)[:, None]

    # determine the projection of each vector in the design matrix onto the vectors in the candidate set matrix
    high_projections = []
    for row_index in row_indices:
        # determine the vector
        row = norm_candidate_set_mat[row_index, :]
        # determine the projection of the vector onto the other vectors
        projections = np.abs(norm_candidate_set_mat @ row)
        # set the projection onto the vectors in the design matrix to 0
        projections[row_indices] = 0
        # find the indices of the vectors with a projection above the minimum projection
        high_projections += np.where(projections > min_projection)[0].tolist()

    # if there are duplicates, remove them
    high_projections = list(set(high_projections))

    # remove the rows with the high projections
    new_candidate_set_mat = np.copy(candidate_set_mat)
    new_candidate_set_mat = np.delete(new_candidate_set_mat, high_projections, axis=0)

    # determine what indices of the original candidate set matrix are in the new candidate set matrix
    og_row_indices = np.delete(np.arange(candidate_set_mat.shape[0]), high_projections)

    # run the federov algorithm on the new candidate set matrix
    row_indices, design_mat, new_det = federov(new_candidate_set_mat, num_rows, epsilon, return_evaluation_lists=False)

    # determine the row indices of the original candidate set matrix
    new_row_indices = og_row_indices[row_indices]

    return new_row_indices, design_mat, new_det
