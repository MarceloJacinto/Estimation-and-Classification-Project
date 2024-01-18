function [A] = compute_transition_matrix(grid)
% compute_transition_matrix Compute the transition matrix from the prior
% map
% grid - A NxN grid that represents the binary map
% A - A N^2*N^2 where the rows represent the previous state and the columns
% represent the next state

% Get the size of the grid
N = size(grid,1);

% Get the possible number of states
num_states = N*N;

% Compute the transition matrix A from each cell to every other cell
A = zeros(num_states, num_states);

for i=1:N
    for j=1:N

        % Get the hypotetical robot position
        robot_pos = [i j];
        robot_pos_index = sub2ind([N N], i, j);

        % Check if the cell in the grid has obstacle. If so, we 
        % say in the transition matrix that the probability of leave such
        % cell is 0 (just a hack so that A is well behaved and we can pre-alocate memory).
        if grid(i,j) == 1
            A(robot_pos_index, robot_pos_index) = 1;
        % Otherwise, check which other adjacent cells are available and
        % set their transition probabilities accordingly
        else
            
            % Get the top-left corner of the hypotetical robot position
            top_corner = robot_pos - 2;
            free_cells = {};
            
            % Check for free cells adjacent to the hypotetical robot pos
            for k=1:3
                for m=1:3
                    % Check if we are outside the grid
                    if top_corner(1) + k < 1 || top_corner(2) + m < 1 || top_corner(1) + k > N || top_corner(2) + m > N
                        continue
                    end
                    
                    % Check if the adjacent cell is free
                    if grid(top_corner(1) + k, top_corner(2) + m) == 0
                        free_cells = [free_cells, [top_corner(1) + k, top_corner(2) + m]];
                    end
                end
            end

            % Get the probability of transition based on the number of free
            % adjacent cells
            prob = 1 / length(free_cells);
            
            % Go through the list of free cells
            for k=1:length(free_cells)

                indices = cell2mat(free_cells(k));
                
                % Get the column of the adjacent cell in the transition
                % matrix A
                adjcent_cell_index = sub2ind([N N], indices(1), indices(2));
                
                % Set the probability in the transition matrix
                A(robot_pos_index, adjcent_cell_index) = prob;
            end
        end
    end
end
end