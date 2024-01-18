function [B] = compute_observation2_matrix(grid)

% Probabilities of correct measurements and bit flips
prob_correct_measurement = 0.5;
prob_bit_flip = 1 - prob_correct_measurement;

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
        B(robot_pos_index, obs_index) = prob_correct_measurement;

        % Compute variations of the sensor measurement considering 1-bit
        % flip possibility in the noise
        noisy_obs_indexes = zeros(8,1);
        counter = 1;
        for k=1:3
            for m=1:3
                if k ~= 2 || m ~=2

                    % Get a copy of the actual observation in that cell
                    noisy_observation = sensor_obs;
                    
                    % Flip one bit to add noise
                    if noisy_observation(k,m) == 0
                        noisy_observation(k,m) = 1;
                    else
                        noisy_observation(k,m) = 0;
                    end

                    % Encode the noisy measurement into a number between
                    % 1-256
                    noisy_obs_indexes(counter) = encode_measurement(noisy_observation);
                    counter = counter + 1;
                end
            end
        end

        % Update the measurement matrix with 8 possible noisy measurements
        for k=1:8
            B(robot_pos_index, noisy_obs_indexes(k)) = prob_bit_flip / 8;
        end
    end
end

end