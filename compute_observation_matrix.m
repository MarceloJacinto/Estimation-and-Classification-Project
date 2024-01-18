function [B] = compute_observation_matrix(grid)

% Get the size of the grid
N = size(grid,1);

% Get the possible number of states
num_states = N*N;

% Get the number of possible diferent observations
max_number_obs = 2^8;   % Number of cells that can be filled with sensor

% Compute the transition matrix A from each cell to every other cell
B = zeros(num_states, max_number_obs);

% Fill the matrix assuming the 0 probability of bit-flip 
for i=1:N
    for j=1:N
        
        % Consider an hypotetical position for the robot
        robot_pos = [i j];
        robot_pos_index = sub2ind([N N], i, j);
        
        % Get the top_corner of a 3x3 square where the robot is positioned
        top_corner = robot_pos - 2;

        % Consider an empty measurement set
        sensor_obs = zeros(3,3);
        
        % Get the measurement for that given state
        for k=1:3
            for m=1:3
                % Check if we are outside the grid. If so, place an obstacle
                if top_corner(1) + k < 1 || top_corner(2) + m < 1 || top_corner(1) + k > N || top_corner(2) + m > N
                    sensor_obs(k,m) = 1;
                else
                    % If we are inside the grid, just check if there is an obstacle or not
                    sensor_obs(k,m) = grid(top_corner(1) + k, top_corner(2) + m);
                end
            end
        end
        
        % Encode the measurement into a number between 1-256
        obs_index = encode_measurement(sensor_obs);
        
        % Update the measurement matrix B
        B(robot_pos_index, obs_index) = 1;
    end
end

end