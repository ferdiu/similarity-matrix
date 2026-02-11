import numpy as np
from scipy.optimize import fsolve


def _overlapping_windows_slices(
        input_text: str,
        window_size: int | None,
        delta: float):
    """
    Generate overlapping slices of a text with a given slice length in terms of words.
    It always produce windows of the same size thus if the last do not perfectly match the dimension
    the overlap will be larger automatically

    Parameters:
        input_text (str): The input text to be sliced.
        window_size (int): The length of each slice in terms of words.
        delta (float): The overlap factor, ranging from 0 (no overlap) to 0.8 (80% overlap) respect the window size.

    Returns:
        list of str: A list containing overlapping slices of the input text.

    Example:
        _overlapping_windows_slices("This is an example sentence for testing.", 3, 0.2)
        ['This is an', 'an example sentence', 'sentence for testing.'] # This is perfectly divied
        _overlapping_windows_slices("This is an example sentence for testing extra.", 3, 0.2)
        ['This is an', 'an example sentence', 'sentence for testing', 'for testing extra.']
        This is not perfectly divided and the last window is appended
    """
    if delta > 0.8:
        raise ValueError("The overlap factor (delta) cannot exceed 0.8.")
    if window_size is None:
        slices = [input_text]
    else:
        words = input_text.split()
        overlap_size = round(window_size * delta)
        step_size = window_size - overlap_size

        if len(words) <= window_size:
            return [input_text.strip()] if len(words) > 0 else []

        slices = []
        n_slices = 0
        for i in range(0, len(words) - window_size + 1, step_size):
            slices.append(" ".join(words[i:i + window_size]))
            n_slices += 1

        # Handle the ramining part if the overlapping windows do not cover perfectly all the words of the sentence then replate the last part
        #   As the first sentence has lenght window_size while the other have lenght step_size
        # thus calculate all the words taken from the previeus process and if
        # there is something left tale the right most window and append it
        if (window_size + ((n_slices - 1) * step_size)) != len(words):
            slices.append(" ".join(words[-window_size:]))

    return slices


def extract_window_and_pointers_from_text_sections(
        texts: np.ndarray, window_size: int, delta: float = 0.2) -> tuple[np.ndarray, list[list[int]]]:
    """
    Extract overlapping windows from a various texts each splitted in sections.
    The sections that are empty are discarted but before that are used for copy the positions of the windows in the supported array.
    Use a support matrix that has a row for each text and for every row an array of pointers that reference each window with its section.

    Parameters:
        texts (numpy.ndarray): Matrix of sections of a big text. If the passed array is of one dimension it will become a column matrix (# columns = 1).
        window_size (int): The length of each window in terms of words.
        delta (float): The overlap factor, ranging from 0 (no overlap) to 0.8 (80% overlap) respect the window size.

    Returns:
        numpy.ndarray: An array containing overlapping windows extracted from text sections.
        list of list of int: A matrix containing pointers to the sections corresponding to each window, each section is a row of the matrix.
            given that it is an inhomogeneous must be a list of list
    """
    pointers = []
    windows = []

    # For inhomogeneous arrays the second dimension is represented
    # as objects so we should check this condition before reshaping
    if texts.ndim == 1 and texts.dtype != object:
        # If the passed array is one dimensional it will become a column
        # matrix (# columns = 1).
        texts = texts.reshape(-1, 1)

    for text_sections in texts:
        pointers_array = []
        for i, section in enumerate(text_sections):
            # Skip empty sections
            if len(section) == 0:
                continue

            # Extract windows from non-empty sections
            section_windows = _overlapping_windows_slices(
                section, window_size, delta)
            section_pointers = [i + 1] * len(section_windows)

            # Extend the array with new elements: this way we create
            # an array of windows with no separation between sections
            windows += section_windows
            pointers_array += section_pointers

        pointers.append(pointers_array)

    return np.array(windows), pointers


def _superellipse_aggregation_function(
        x: np.ndarray,
        alpha: float,
        beta: float,
        gamma: float) -> np.ndarray:
    """
    Compute the y-values of a superellipse aggregation function for given x-values, alpha, beta, and gamma parameters.

    Parameters:
        x (numpy.ndarray): Input x-values.
        alpha (float): Distortion of the x-axis.
        beta (float): Maximum value of y for normalization purposes.
        gamma (float): Curve parameter.

    Returns:
        numpy.ndarray: Corresponding y-values.
    """
    # Equation of superllipse of the top right quadrant:
    #   (x/a)**(2/n) + (y/b)**(2/n) = 1
    #   solving for y we get the following equation
    #   alpha is the distorsion of the x axes thus we want to match it with the
    #   number of points we want, in this way we calculate the y on only x natural numbers
    #   beta is the maximum value of y and we keep 1 for normalization purpose
    #   gamma is the curve parameter
    y = beta * (1 - (np.abs(x) / alpha)**(2 / gamma))**(gamma / 2)
    return y


def _constraint_function(
        gamma: float,
        n_points: int,
        beta: float,
        omega: float):
    """
    Constraint function to solve for gamma using fsolve.

    Parameters:
        gamma (float): Curve parameter.
        n_points (int): Number of points.
        beta (float): Maximum value of y for normalization purposes.
        omega (float): Weight constraint.

    Returns:
        float: Zero value, difference between computed omega and desired omega.
    """
    alpha = n_points
    # Output x_values natural nubmers
    x_values = np.linspace(0, alpha, n_points + 1)
    y_values = _superellipse_aggregation_function(x_values, alpha, beta, gamma)

    # Objective function: omega correspond to the amount of the sum off all the y_values thus
    #   the sum of all the weights discarded the first that is always 1
    #   e.g. with a 0.5 omega it means that all the tail weight (all without the first)
    #   correspond to the 50% of the first weight that it is always 1
    zero = omega - np.sum(y_values[1:])

    return zero


def _set_omega(n_points: int, tail_weight: float, tolerance: float):
    """
    Set the weight constraint (omega) based on the number of points and tail weight.

    Parameters:
        n_points (int): Number of points.
        tail_weight (float): Percentage of tail weight.
        tolerance (float): Tolerance level for rounding.

    Returns:
        float: Weight constraint (omega).
    """
    # Check if tail_weight is within the valid range
    if tail_weight < 0 or tail_weight > 1:
        raise ValueError("Tail weight must be between 0 and 1")

    # Round weight to second decimal point
    tail_weight = round(tail_weight, 2)

    # Make omega
    # Average: array of weights like [0.2. 0.2. 0.2. 0.2. 0.2.]
    if tail_weight == 1:
        epsilon = tail_weight / n_points
        # -epsilon because the x starts from zero and the last x has weight y = 0
        ceil_value = 1 - epsilon
        omega = ceil_value - tolerance
    elif tail_weight == 0:  # Maximum: [1. 0. 0. 0. 0.]
        floor_value = 0
        omega = floor_value + tolerance
    else:  # Percentage of the tail (array without the first value) relative to 1
        omega = tail_weight

    return omega


def _find_gamma(n_points, beta, omega, tolerance):
    """
    Find the optimal gamma parameter satisfying the weight constraint using fsolve.

    Parameters:
        n_points (int): Number of points.
        beta (float): Maximum value of y for normalization purposes.
        omega (float): Weight constraint.
        tolerance (float): Tolerance level for rounding.

    Returns:
        float: Optimal gamma value.
    """
    gamma_initial_guess = 2  # Gamma 2 is 45 degree line
    xtol = tolerance

    gamma_solution = fsolve(
        _constraint_function,
        gamma_initial_guess,
        xtol=xtol,
        args=(n_points, beta, omega))

    return gamma_solution[0]


def _set_beta(n_points: int, tail_weight: float):
    """
    Set the beta parameter based on the tail weight.

    Parameters:
        tail_weight (float): Percentage of tail weight.

    Returns:
        float: Beta parameter.
    """
    if tail_weight == 1:
        beta = tail_weight / n_points
    elif tail_weight == 0:
        beta = 1
    else:
        beta = 1 - tail_weight

    return beta


def _edge_case_cover(n_points: int, tail_weight: float):
    """
    Generates a NumPy array with a specified number of points,
    where the first element is set to `beta` (1 - tail_weight) and
    all other elements in the array are set to `weight` (1 / (n_points - 1)).

    Parameters:
    - n_points (int): The number of points in the array.
    - tail_weight (float): The tail weight to determine the value of beta.

    Returns:
    - tuple: A tuple containing:
        - np.ndarray: A NumPy array of length `n_points`, with the first element set to `beta`
                      and all other elements set to `weight`.
        - np.ndarray: A NumPy array of length `n_points` containing a range from 0 to `n_points - 1`.
        - float: A zero float value.
    """
    beta = 1 - tail_weight
    weight = tail_weight / (n_points - 1)

    array = np.full(n_points, weight)
    array[0] = beta

    return array, np.arange(n_points), 0.0


def _get_weights(n_points, tail_weight, decimal_precision=7):
    """
    Compute weights based on the number of points and tail weight.
    The objective is to find the optimal gamma thus the curvature of the superellipse
    in function with the number of points and the tail weight
    The tail weight is the amount of weight we want to give to the tail thus all the
    array value minus the head that is only the first one
    E.g. with a tail_weight of 0.2 we give 0.8 weight to the head and 0.2 to the tail
    The tail take an ellipse configuration given that the elements of the array are ordinated
    and we want to give more importantce to the values nearer the head

    NOTE: the sum of weights is always 1 (normalized)

    Parameters:
        n_points (int): Number of points.
        tail_weight (float): Percentage of tail weight.
        decimal_precision (int): Decimal precision for rounding.

    Returns:
        numpy.ndarray, numpy.ndarray, float: Indexes, weights, and optimal gamma value.
    """
    # Float64 take around 15-17 decimal digits
    # Given a n decimal digits precision then set the 1e-(n+1) precision
    tolerance = 10 ** (-(decimal_precision + 1))

    # Cover the case if there is only one window
    if n_points == 1:
        return np.array([1.0]), np.array([1]), 0.0

    # Find the beta parameter that is the y value of the x_0
    beta = _set_beta(n_points, tail_weight)

    # If the the number of points are too and the tail weight is too high like
    #   n_points = 5 and tail_weight 0.9 is not possible to find a gamma as
    #   the beta is 0.1 and the maximum sum of weights is n_points - 1 = 5 - 1 = 4 * 0.1 = 0.4 < 0.9
    #   while n_points 5 and tail_weight = 0.8: 4 * beta (0.2) = 0.8
    if tail_weight != 0 and tail_weight != 1 and (
            n_points - 1) * beta < tail_weight - tolerance:
        return _edge_case_cover(n_points, tail_weight)

    # Given the wanted tail weight get che correct rounded omega value
    #   we do this because tail weight is limited between 0 and 1 for simplicity
    #   while omega is more complex
    omega = _set_omega(n_points, tail_weight, tolerance)

    # Given an omega value find the best gamma that satify the omega or in other words
    #   the tail weight condition
    gamma = _find_gamma(n_points, beta, omega, tolerance)

    # Set the alpha to be equal to the number of point plus 1 as there is the 0
    #   in this way the x values will be natural numbers (int) and not float
    alpha = n_points
    x_values = np.linspace(0, alpha, n_points + 1)
    y_values = _superellipse_aggregation_function(x_values, alpha, beta, gamma)

    # Convert indexes to integers
    indexes = x_values.astype(int)
    # Round output y values with set decimal precision
    # Remove the last value that it is always 0
    weights = np.round(y_values, decimal_precision)[:-1]

    return weights, indexes[:-1], gamma


def compute_single_aggregation_tail_weight(cos_sim_windows: np.ndarray,
                                           pointers: list[list[int]],
                                           tail_weight: float,
                                           axis: int = 0,
                                           precalculated_weights=None,
                                           max_precalculated=1000):
    """
    Computes the aggregation of cosine similarity values using a weighted sum approach.
    Can aggregate along rows (axis=0) or columns (axis=1).

    Please note that this function assumes that `pointers` is provided and valid.
    When a row or column group is empty, the corresponding aggregated value will be set
    to zero.

    Parameters:
        cos_sim_windows (numpy.ndarray): Cosine similarity matrix for multiple windows.
        pointers (list): List of lists containing pointers indicating the sections of windows
                        corresponding to each row (for axis=0) or each column (for axis=1).
        tail_weight (float): Weight assigned to the tail elements for computing the weighted sum.
        axis (int): Aggregation axis. 0 for row aggregation, 1 for column aggregation.
        precalculated_weights (json, optional): Pre calculated aggregation function weights.
        max_precalculated (int, optional): The maximum window length where there are precalculated values.

    Returns:
        numpy.ndarray: Matrix containing aggregated cosine similarity values.
    """
    if axis == 0:
        return _aggregate_along_rows(cos_sim_windows, pointers, tail_weight,
                                     precalculated_weights, max_precalculated)
    elif axis == 1:
        return _aggregate_along_columns(
            cos_sim_windows,
            pointers,
            tail_weight,
            precalculated_weights,
            max_precalculated)
    else:
        raise ValueError("axis must be 0 (rows) or 1 (columns)")


def _aggregate_along_rows(cos_sim_windows: np.ndarray,
                          pointers: list[list[int]],
                          tail_weight: float,
                          precalculated_weights=None,
                          max_precalculated=1000) -> np.ndarray:
    """
    Aggregate along rows.
    Each element in pointers represents windows for a row text.
    """
    # Initialize cos_sim_rows matrix
    cos_sim_rows = np.zeros((len(pointers), cos_sim_windows.shape[1]))

    windows_counter = 0
    # Iterate over each row
    for row_index, row_pointers in enumerate(pointers):
        n_windows = len(row_pointers)

        # Cover the case of empty rows
        # The row will be all zeros
        if n_windows == 0:
            cos_sim_rows[row_index, :] = 0
            continue

        # Get the weights
        if precalculated_weights is not None and n_windows <= max_precalculated:
            weights = np.array(
                precalculated_weights[str(tail_weight)][str(n_windows)])
        else:
            weights, _, _ = _get_weights(n_windows, tail_weight)

        # Left and right boundaries for selecting the appropriate windows of a
        # given row
        left_counter = windows_counter
        right_counter = windows_counter + n_windows

        window_slice = cos_sim_windows[left_counter:right_counter, :]

        # Iterate over each column
        for column_index in range(cos_sim_windows.shape[1]):
            # Take the windows of the i-row and sort
            cos_sim_array_ordered = np.sort(
                window_slice[:, column_index])[::-1]
            # Calculate the weighted sum
            weighted_sum = np.dot(cos_sim_array_ordered, weights)
            cos_sim_rows[row_index, column_index] = weighted_sum

        # Update windows_counter
        windows_counter += n_windows

    return cos_sim_rows


def _aggregate_along_columns(
        cos_sim_windows: np.ndarray,
        *args,
        **kwargs) -> np.ndarray:
    """
    Aggregate along columns.
    Each element in pointers represents windows for a column text.
    """
    return _aggregate_along_rows(
        cos_sim_windows.transpose(), *args, **kwargs).transpose()


def compute_dual_aggregation_tail_weight(cos_sim_windows: np.ndarray,
                                         row_pointers: list[list[int]],
                                         column_pointers: list[list[int]],
                                         tail_weight: float):
    """
    Computes the aggregation of cosine similarity values using a weighted sum approach
    along both rows and columns simultaneously by flattening the submatrix and applying
    weights directly to the linearized values.

    Please note that this function assumes that both `row_pointers` and `column_pointers`
    are provided and valid. When a row or column group is empty, the corresponding
    aggregated value will be set to zero.

    Parameters:
        cos_sim_windows (numpy.ndarray): Cosine similarity matrix for multiple windows.
        row_pointers (list): List of lists containing pointers for row groupings.
        column_pointers (list): List of lists containing pointers for column groupings.
        tail_weight (float): Weight assigned to the tail elements for computing the weighted sum.

    Returns:
        numpy.ndarray: Matrix containing aggregated cosine similarity values with shape
                      (len(row_pointers), len(column_pointers)).
    """
    # Initialize the final aggregated matrix
    cos_sim_aggregated = np.zeros((len(row_pointers), len(column_pointers)))

    row_windows_counter = 0
    # Iterate over each row group
    for row_index, row_group_pointers in enumerate(row_pointers):
        n_row_windows = len(row_group_pointers)

        # Cover the case of empty rows
        # The row will be all zeros
        if n_row_windows == 0:
            cos_sim_aggregated[row_index, :] = 0
            continue

        # Row boundaries for selecting the appropriate windows
        row_left = row_windows_counter
        row_right = row_windows_counter + n_row_windows

        col_windows_counter = 0
        # Iterate over each column group
        for col_index, col_group_pointers in enumerate(column_pointers):
            n_col_windows = len(col_group_pointers)

            # Cover the case of empty columns
            # The column will be all zeros
            # NOTE: if the code reached here it means that the row is not empty
            #   thus the row is not all zeros but all the values correpsonding to
            #   this column will be zeros after every iteration
            if n_col_windows == 0:
                cos_sim_aggregated[row_index, col_index] = 0
                continue

            # Column boundaries for selecting the appropriate windows
            col_left = col_windows_counter
            col_right = col_windows_counter + n_col_windows

            # Extract the submatrix for this row-column group combination
            submatrix = cos_sim_windows[row_left:row_right, col_left:col_right]

            # Flatten the submatrix and sort in descending order
            flattened_values = submatrix.flatten()
            sorted_values = np.sort(flattened_values)[::-1]

            # Calculate weights for the total number of elements in the
            # submatrix
            n_total_elements = len(flattened_values)
            weights, _, _ = _get_weights(n_total_elements, tail_weight)

            # Calculate the weighted sum directly on the linearized submatrix
            final_value = np.dot(sorted_values, weights)
            cos_sim_aggregated[row_index, col_index] = final_value

            col_windows_counter += n_col_windows

        row_windows_counter += n_row_windows

    return cos_sim_aggregated


def compute_aggregation_tail_weight(cos_sim_windows: np.ndarray,
                                    row_pointers: list[list[int]] | None,
                                    column_pointers: list[list[int]] | None,
                                    tail_weight: float):
    """
    Aggregates a cosine similarity matrix along specified dimensions using tail weighting.

    Depending on which pointers are provided, the function performs aggregation along rows, columns, both, or returns the original matrix:
    - If both `row_pointers` and `column_pointers` are provided, aggregation is performed along both dimensions.
    - If only `row_pointers` is provided, aggregation is performed along rows.
    - If only `column_pointers` is provided, aggregation is performed along columns.
    - If neither is provided, the original matrix is returned.

    Please note that this function assumes that both `row_pointers` and `column_pointers`,
    when provided, are valid. When a row or column group is empty, the corresponding
    aggregated value will be set to zero.

    Args:
        cos_sim_windows (np.ndarray): The input cosine similarity matrix to aggregate.
        row_pointers (list[list[int]] | None): Indices specifying how to aggregate rows, or None.
        column_pointers (list[list[int]] | None): Indices specifying how to aggregate columns, or None.
        tail_weight (float): The weight to apply to the tail elements during aggregation.

    Returns:
        np.ndarray: The aggregated matrix according to the specified pointers and tail weighting.
    """
    if row_pointers and column_pointers:
        # Aggregation needed on both matrix dimensions
        return compute_dual_aggregation_tail_weight(
            cos_sim_windows, row_pointers, column_pointers, tail_weight)
    elif row_pointers:
        # Aggregation needed only on rows
        return compute_single_aggregation_tail_weight(
            cos_sim_windows, row_pointers, tail_weight, axis=0)
    elif column_pointers:
        # Aggregation needed only on columns
        return compute_single_aggregation_tail_weight(
            cos_sim_windows, column_pointers, tail_weight, axis=1)
    else:
        # None of the pointers were passed: return the matrix as it is
        return cos_sim_windows
