clear all; close all; clc; clear uifig
% Authors: Marcelo Fialho Jacinto (marcelo.jacinto@tecnico.ulisboa.pt)
%          Jose Pedro Gomes (josepgomes@tecnico.ulisboa.pt)
% Ph.D. course on Estimation and Classification (Project)

% Initialize the seed and random number generator
rng(10);

% Create the grid with obstacles
N = 20;
n_obs = 25;
simulation_steps = 0;
sensor_direction = 4;
bitflip_prob = 0;       % Internal model of the probability of bitflip
bitflip_sensor = 0;     % The actual sensor bitflip probability
                        % This is done, because in real-life, our model
                        % is never models perfectly the sensor
robot_pos = [-1 -1];    % Unitialized robot position
timer_period = 1;
first_step = 1;         % Marks the first simulation step
outside_walls = 1;

% Create a UI to ask for the size of the grid, 4 or 8 movement directions 
fig = uifigure('Name', 'Simulation Configuration', 'HandleVisibility', 'on');
grid_layout = uigridlayout(fig, [7,2]);
grid_layout.RowHeight = {22,22,22,22,22,22,22,'1x'};
grid_layout.ColumnWidth = {150,'1x'};

p = uilabel(grid_layout, 'Text', 'Grid layout size (NxN):');
p.Layout.Row = 1;
p.Layout.Column = 1;

grid_N = uieditfield(grid_layout, 'numeric');
grid_N.Value = 20;
grid_N.Limits = [20 50];
grid_N.Layout.Row = 1;
grid_N.Layout.Column = 2;

p2 = uilabel(grid_layout, 'Text', 'Obstacle density:');
p2.Layout.Row = 2;
p2.Layout.Column = 1;

obstacle_N = uieditfield(grid_layout, 'numeric');
obstacle_N.Value = n_obs;
obstacle_N.Limits = [0 100];
obstacle_N.Layout.Row = 2;
obstacle_N.Layout.Column = 2;

p3 = uilabel(grid_layout, 'Text', 'Robot sensor directions:');
p3.Layout.Row = 3;
p3.Layout.Column = 1;

directions_N = uidropdown(grid_layout, 'Items', ["4","8"]);
directions_N.Layout.Row = 3;
directions_N.Layout.Column = 2;

p4 = uilabel(grid_layout, 'Text', 'Model 1 Bit-flip probability:');
p4.Layout.Row = 4;
p4.Layout.Column = 1;

bit_flip_prob_N = uieditfield(grid_layout, 'numeric');
bit_flip_prob_N.Value = bitflip_prob;
bit_flip_prob_N.Limits = [0 1];
bit_flip_prob_N.Layout.Row = 4;
bit_flip_prob_N.Layout.Column = 2;

p5 = uilabel(grid_layout, 'Text', 'Sensor 1 Bit-flip probability:');
p5.Layout.Row = 5;
p5.Layout.Column = 1;

bit_flip_sensor_N = uieditfield(grid_layout, 'numeric');
bit_flip_sensor_N.Value = bitflip_sensor;
bit_flip_sensor_N.Limits = [0 1];
bit_flip_sensor_N.Layout.Row = 5;
bit_flip_sensor_N.Layout.Column = 2;

p6 = uilabel(grid_layout, 'Text', 'Timer period (s):');
p6.Layout.Row = 6;
p6.Layout.Column = 1;

timer_period_N = uieditfield(grid_layout, 'numeric');
timer_period_N.Value = timer_period;
timer_period_N.Limits = [0.1 5];
timer_period_N.Layout.Row = 6;
timer_period_N.Layout.Column = 2;

p7 = uibutton(grid_layout, 'Text', "Generate Map", 'ButtonPushedFcn', @generate_map_btn_click);
p7.Layout.Row = 7;
p7.Layout.Column = [1 2];

%% Function definitions 

function generate_map_btn_click(src, event)

    % Get the values from the workspace
    grid_N = evalin('base', 'grid_N');
    obstacle_N = evalin('base', 'obstacle_N');
    directions_N = evalin('base', 'directions_N');
    bit_flip_prob_N = evalin('base', 'bit_flip_prob_N');
    bit_flip_sensor_N = evalin('base', 'bit_flip_sensor_N');
    timer_period_N = evalin('base', 'timer_period_N');
    fig = evalin('base', 'fig');
    
    % Get the parameters from the UI
    N = grid_N.Value;
    n_obs = obstacle_N.Value;
    sensor_direction = str2double(directions_N.Value);
    bitflip_prob = bit_flip_prob_N.Value;
    bitflip_sensor = bit_flip_sensor_N.Value;
    timer_period = timer_period_N.Value;
    
    % Save the parameters to the workspace
    % Update the value of grid size N in the workspace
    assignin('base', 'N', N);
    assignin('base', 'n_obs', n_obs);
    assignin('base', 'sensor_direction', sensor_direction);
    assignin('base', 'bitflip_prob', bitflip_prob);
    assignin('base', 'bitflip_sensor', bitflip_sensor);
    assignin('base', 'timer_period', timer_period);
        
    % Generate a new grid according to the defined parameters
    grid = create_grid(N, n_obs);
    assignin('base', 'grid', grid);
    
    % Close the figure and create a new one where the generated map is shown
    % and editable
    close(fig);
    
    % Create a new figure where the generated map is shown and can be edited
    fig = figure('Name', 'Map Configuration', 'HandleVisibility', 'on');
    grid_ax = gca;
    display_grid(grid_ax, grid, [-1 -1], 'Click to edit the map', 1);
    grid_ax = gca;
    set(fig, 'CloseRequestFcn', @close_map_callback);
    set(grid_ax, 'ButtonDownFcn', @edit_map_callback);
    assignin('base', 'fig', fig);
    assignin('base', 'grid_ax', grid_ax);

end

function edit_map_callback(src, event)
    % Get the current point where the click occurred
    currentPoint = get(gca, 'CurrentPoint');

    % Get the current grid from the workspace 
    grid = evalin('base', 'grid');
    
    % Extract x and y coordinates
    xCoord = round(currentPoint(1, 1));
    yCoord = round(currentPoint(1, 2));

    % Change the value if the cell is occupied or not
    if grid(xCoord, yCoord) == 0
        grid(xCoord, yCoord) = 1;
    else
        grid(xCoord, yCoord) = 0;
    end
    
    % Display the clicked position
    disp(['Clicked position: (' num2str(xCoord) ', ' num2str(yCoord) ')']);

    % Update the grid in the workspace
    assignin('base', 'grid', grid);

    % Display the new grid
    grid_ax = gca;
    display_grid(grid_ax, grid, [-1 -1], 'Click to edit the map', 1);
end

function close_map_callback(src, event)

    % Delete the actual figure and close it
    delete(gcf);

    % Get the size of the grid
    N = evalin('base', 'N');
    grid = evalin('base', 'grid');

    % Initialize the robot position randomly
    robot_pos = init_robot_position(grid);

    % Create a UI to ask for the size of the grid, 4 or 8 movement directions 
    fig = uifigure('Name', 'Robot Initial Position', 'HandleVisibility', 'on');
    grid_layout = uigridlayout(fig, [3,3]);
    grid_layout.RowHeight = {22,22,'1x'};
    grid_layout.ColumnWidth = {150,'1x'};
    
    p = uilabel(grid_layout, 'Text', 'Robot Position (Row, Col):');
    p.Layout.Row = 1;
    p.Layout.Column = 1;
    
    grid_row = uieditfield(grid_layout, 'numeric');
    grid_row.Value = robot_pos(1);
    grid_row.Limits = [1 N];
    grid_row.Layout.Row = 1;
    grid_row.Layout.Column = 2;

    grid_col = uieditfield(grid_layout, 'numeric');
    grid_col.Value = robot_pos(2);
    grid_col.Limits = [1 N];
    grid_col.Layout.Row = 1;
    grid_col.Layout.Column = 3;

    p5 = uibutton(grid_layout, 'Text', "Set Robot Position", 'ButtonPushedFcn', @set_robot_pos_btn_click);
    p5.Layout.Row = 2;
    p5.Layout.Column = [1 3];
    
    grid_ax = uiaxes(grid_layout);
    grid_ax.Layout.Row = 3;
    grid_ax.Layout.Column = [1 3];
    display_grid(grid_ax, grid, robot_pos, '', 0);

    assignin('base', 'grid_row', grid_row);
    assignin('base', 'grid_col', grid_col);
    assignin('base', 'p5', p5);

end

function set_robot_pos_btn_click(src, event)
    
    % Get the text fields from the UI (variables in the workspace)
    grid_row = evalin('base', 'grid_row');
    grid_col = evalin('base', 'grid_col');
    grid = evalin('base', 'grid');
    
    row = grid_row.Value;
    col = grid_col.Value;
    
    % If the selected position is an empty cell, just delete the window
    % and proceed to the simulation
    if grid(row, col) == 0

        % Update the robot position
        robot_pos = [row col];
        assignin('base', 'robot_pos', robot_pos);

        % Close the current window
        delete(gcf);
        
        % Setup the simulation window
        setup_simulation();
    else

        % Ignore the current selection, as we should not be able to place
        % robot in an occupied cell
        disp(["Occupied cell selected. Choose another initial position for the robot!"]);
    end
end

function setup_simulation()

    % Delete the actual figure and close it
    delete(gcf);
    
    % ---------------------------------------------------------------
    % Start the actual simulation here
    % ---------------------------------------------------------------
    grid = evalin('base', 'grid');
    simulation_steps = evalin('base', 'simulation_steps');
    sensor_direction = evalin('base', 'sensor_direction');
    bitflip_prob = evalin('base', 'bitflip_prob');
    timer_period = evalin('base', 'timer_period');
    robot_pos = evalin('base', 'robot_pos');

    % Compute the transition matrix A from the prior grid
    A = compute_transition_matrix(grid);
    B = compute_observation_matrix(grid, sensor_direction, bitflip_prob);

    % Create the probability matrix that will hold the probability of each cell
    prob_matrix = init_cell_prob(grid);
    
    % Create a new figure where the UI will be displayed
    fig = uifigure('Name', 'Robot Motion and Localization', 'Position', [200, 200, 900, 600]);
    grid_layout = uigridlayout(fig,[3 2]);
    
    p = uipanel(grid_layout,'Title','Configuration');
    p.Layout.Row = 3;
    p.Layout.Column = [1 2];
    
    grid_ax = uiaxes(grid_layout);
    grid_ax.Layout.Row = [1 2];
    grid_ax.Layout.Column = 1;
    
    heat_map_ax = uipanel(grid_layout);
    heat_map_ax.Layout.Row = [1 2];
    heat_map_ax.Layout.Column = 2;
    
    % Setup the robot grid for the first time
    display_grid(grid_ax, grid, robot_pos, '', 0);
    display_heatmap(heat_map_ax, prob_matrix);

    % Configuration information in the UI
    N = evalin('base', 'N');
    n_obs = evalin('base', 'n_obs');
    sensor_direction = evalin('base', 'sensor_direction');
    bitflip_prob = evalin('base', 'bitflip_prob');
    bitflip_sensor = evalin('base', 'bitflip_sensor');
    timer_period = evalin('base', 'timer_period');
    
    % Create the button interface to control the simulation
    start_button = uibutton(p, 'Text', 'Start', 'Position', [20, 20, 100, 30], 'ButtonPushedFcn', @start_btn_click);
    pause_button = uibutton(p,'Text', 'Pause', 'Position', [130, 20, 100, 30], 'ButtonPushedFcn', @pause_btn_click);
    next_step_button = uibutton(p,'Text', 'Next Step', 'Position', [240, 20, 100, 30], 'ButtonPushedFcn', @next_step_btn_click);
    
    uilabel(p, 'Text', sprintf('Grid Size: %d x %d', N,N), 'Position', [20, 150, 200, 30]);
    uilabel(p, 'Text', sprintf('Sensor measurement directions: %d ', sensor_direction), 'Position', [20, 130, 200, 30]);
    uilabel(p, 'Text', sprintf('Bit-flip probability (HMM): %d ', bitflip_prob), 'Position', [20, 110, 200, 30]);
    uilabel(p, 'Text', sprintf('Bit-flip probability (Sensor): %d ', bitflip_sensor), 'Position', [20, 90, 200, 30]);
    iterations_text = uilabel(p, 'Text', sprintf('Number of Iterations: %d', simulation_steps), 'Position', [20, 70, 200, 30]);

    % Save the variables to the workspace
    assignin('base', 'fig', fig);
    assignin('base', 'A', A);
    assignin('base', 'B', B);
    assignin('base', 'robot_pos', robot_pos);
    assignin('base', 'prob_matrix', prob_matrix);
    assignin('base', 'grid_ax', grid_ax);
    assignin('base', 'heat_map_ax', heat_map_ax);
    assignin('base', 'iterations_text', iterations_text);
    % assignin('base', 'robot_pos_x_field', robot_pos_x_field);

    % Create a timer to execute the motion of the robot automatically
    t = timer('TimerFcn',@next_step_btn_click, 'ExecutionMode', 'fixedRate', 'Period', timer_period, 'BusyMode','drop');
    assignin('base', 't', t);
end

function start_btn_click(~,~)

    % Get the timer from the workspace and stop it
    t = evalin('base', 't');
    
    % Start the timer
    if isequal(t.Running, 'off')
        start(t);
    end

end

function pause_btn_click(~,~)

    % Get the timer from the workspace and stop it
    t = evalin('base', 't');
    
    % Stop the timer
    if isequal(t.Running, 'on')
        stop(t);
    end

end

function next_step_btn_click(~,~)

    % Get robot_pos and grid from the workspace
    robot_pos = evalin('base', 'robot_pos');
    grid = evalin('base', 'grid');
    prob_matrix = evalin('base', 'prob_matrix');
    grid_ax = evalin('base', 'grid_ax');
    heat_map_ax = evalin('base', 'heat_map_ax');
    simulation_steps = evalin('base', 'simulation_steps');
    iterations_text = evalin('base', 'iterations_text');
    A = evalin('base', 'A');
    B = evalin('base', 'B');
    sensor_direction = evalin('base', 'sensor_direction');
    bitflip_sensor = evalin('base', 'bitflip_sensor');
    first_step = evalin('base', 'first_step');
    
    % Increment the simulation step
    simulation_steps = simulation_steps + 1;
    assignin('base', 'simulation_steps', simulation_steps);
    
    % Perform one simulation step update
    [robot_pos, prob_matrix] = simulate_one_step(robot_pos, prob_matrix, grid, A, B, sensor_direction, bitflip_sensor, first_step);
    
    % Mark that we are no longer in the first step
    first_step = 0;

    % Update the grid visualization
    display_grid(grid_ax, grid, robot_pos, '', 0);
    display_heatmap(heat_map_ax, prob_matrix);
    iterations_text.Text = sprintf('Number of Iterations: %d', simulation_steps);
    
    % Update robot_pos, prob_matrix, grid and num_iterations in the workspace
    assignin('base', 'robot_pos', robot_pos);
    assignin('base', 'prob_matrix', prob_matrix);
    assignin('base', 'iterations_text', iterations_text);
    assignin('base', 'first_step', first_step);

end

function [robot_pos, prob_matrix] = simulate_one_step(robot_pos, prob_matrix, grid, A, B, sensor_direction, bitflip_sensor, first_step)
    
    simulation_steps = evalin('base', 'simulation_steps');

    % Move the robot
    if simulation_steps > 1
        robot_pos = robot_motion(robot_pos, grid);
    end

    % Perform a measurement 
    measurement = sensor_model(robot_pos, grid, bitflip_sensor, sensor_direction);
    
    % Compute the probabilities of each cell
    prob_matrix = compute_cell_probabilities(prob_matrix, measurement, A, B, sensor_direction, first_step, grid);
end

function [prob_matrix] = compute_cell_probabilities(prob_matrix, measurement, A, B, sensor_direction, first_step, grid)
% compute_cell_probabilities - Computes the probability of each cell
% given the robot measurements of the environment, the probability
% transition matrix, the measurement matrix and the previous probability
% matrix from the last iteration

    % Get the number of possible states
    N = size(A,1);
    
    % Encode the measurements matrix into a measurement index
    obs_index = encode_measurement(measurement, sensor_direction);
    
    % Compute the cell probability using the forward algorithm
    D = diag(B(:, obs_index));
    
    % Flatten the probability matrix of each cell to multiply by the matrix eqn
    alpha_prev = reshape(prob_matrix, N, []);
    
    if first_step == 1
        alpha = D * alpha_prev;
    else
        alpha = D * A' * alpha_prev;
    end
    
    % Get the new state vector to a matrix
    prob_matrix = reshape(alpha, sqrt(N), sqrt(N));
    
    % Renormalize the probabilities at each step
    total_sum = sum(prob_matrix, 'all');
    
    % Check if the total sum if really really small (almost zero). If so, 
    % all the boxes have almost zero probability therefore we should
    % renormalize everything with equal proability
    if (total_sum ~= 0)
        prob_matrix = prob_matrix / total_sum;
    else
        prob_matrix = init_cell_prob(grid);
    end

end

function [sensor_obs] = sensor_model(robot_pos, grid, bitflip_sensor, sensor_direction)
% sensor_model Implements the robot sensor model
% robot_pos - [row,col] of the actual robot position
% grid - the grid of the world with obstacles

    outside_walls = evalin('base', 'outside_walls');

    % Get the size of the grid
    N = size(grid,1);
    
    % Create an empty measurement of the sensor
    sensor_obs = zeros(3,3);
    
    % Get the top_corner of a 3x3 square where the robot is positioned
    top_corner = robot_pos - 2;
    
    % Check which cells surrounding the robot are obstacles
    for i=1:3
        for j=1:3
            % Check if we are outside the grid. If so, measure like an empty
            % state
            if top_corner(1) + i < 1 || top_corner(2) + j < 1 || top_corner(1) + i > N || top_corner(2) + j > N
                sensor_obs(i,j) = outside_walls;
            else
                % If we are inside the grid, just check if there is an obstacle or
                % not
                sensor_obs(i,j) = grid(top_corner(1) + i, top_corner(2) + j);
            end
        end
    end
    
    % Clean the position in the middle, because the sensor never measures the
    % middle position
    sensor_obs(2,2) = 0;
    
    % Add sensor noise (possibility of 1 bit-flip)
    if bitflip_sensor ~= 0
         
        % Generate a random sample from a binomial distribution to check
        % whether to flip or not a bit
        flip = binornd(1, bitflip_sensor);
        
        % If we are going to flip a bit, choose from a uniform distribution 
        % which bit to flip
        if flip == 1
            
            % Get the coordinate randomly where the bit will flip
            flip_map_4 = [[1 2]; [2 1]; [2 3]; [3 2]];
            flip_map_8 = [[1 1]; [1 2]; [1 3]; [2 1]; [2 3]; [3 1]; [3 2]; [3 3]];
            
            index = round(unifrnd(1,sensor_direction, 1));
    
            if sensor_direction == 4
                flip_coord = flip_map_4(index,:);
            else
                flip_coord = flip_map_8(index,:);
            end
    
            % Flip the bit 
            disp(['Sensor bit flip at coordinate: (' num2str(flip_coord(1)) ',' num2str(flip_coord(2)) ')']);
            if sensor_obs(flip_coord) == 0
                sensor_obs(flip_coord) = 1;
            else
                sensor_obs(flip_coord) = 0;
            end
    
            
        end
    end

end

function prob_matrix = init_cell_prob(grid)

    % Get the size of the grid
    N = size(grid,1);
    
    % Initialize the prob_matrix
    prob_matrix = zeros(N,N);
    
    % Count the number of empty cells
    num_empty_cells = sum(grid == 0, 'all');
    
    % Initial probability of all empty cells
    prob = 1 / num_empty_cells;
    prob_matrix(grid == 0) = prob;
    
    end
    
    function [init_pos] = init_robot_position(grid)
    
    % Get the size of the grid
    N = size(grid,1);
    
    % Initialize the free cells list
    free_cells = {};
    
    % Get which cells are free to spawn the robot into position
    for i=1:N
        for j=1:N
            if grid(i, j) == 0
                free_cells = [free_cells, [i, j]];
            end
        end
    end
    
    % Select one of the cells according to a uniform probability distribution
    index = round(unifrnd(1,length(free_cells), 1));
    
    % Set the next robot position
    init_pos = cell2mat(free_cells(index));

end

function [next_pos] = robot_motion(robot_pos, grid)
% robot_motion Implements the uniform probabilistic motion model
% robot_pos - [row, col] of the current robot position
% grid - the grid of the world with obstacles

    % Get the size of the grid
    N = size(grid,1);
    
    % Initialize the free cells list with the current robot position
    free_cells = {};
    
    % Get the top_corner of a 3x3 square where the robot is positioned
    top_corner = robot_pos - 2;
    
    % Check which cells have obstacles
    for i=1:3
        for j=1:3
            % Check if we are outside the grid. If so, those positions are not
            % possible
            if top_corner(1) + i < 1 || top_corner(2) + j < 1 || top_corner(1) + i > N || top_corner(2) + j > N
                continue;
            % If we are inside the grid, check if the cell if free. If so, add
            % it to the free-cells list
            else
                % If we are inside the grid, just check if there is an obstacle or
                % not
                if grid(top_corner(1) + i, top_corner(2) + j) == 0
                    free_cells = [free_cells, [top_corner(1) + i, top_corner(2) + j]];
                end
            end
        end
    end

    % Make sure the current robot position is also set to a possible next state
    index = round(unifrnd(1,length(free_cells), 1));
    
    % Set the next robot position
    next_pos = cell2mat(free_cells(index));
end
    
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


function [B] = compute_observation_matrix(grid, num_observations, prob_1_bit_flip)

    assert(num_observations == 4 || num_observations == 8, 'Sensor must measure in 4 or 8 directions.');
    
    outside_walls = evalin('base', 'outside_walls');

    % Get the probability of the correct measurement
    prob_correct_measurement = 1 - prob_1_bit_flip;
    
    % Get the size of the grid
    N = size(grid,1);
    
    % Get the possible number of states
    num_states = N*N;
    
    % Get the number of possible diferent observations
    max_number_obs = 2^num_observations;   % Number of cells that can be filled with sensor
    
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
                        sensor_obs(k,m) = outside_walls;
                    else
                        % If we are inside the grid, just check if there is an obstacle or not
                        sensor_obs(k,m) = grid(top_corner(1) + k, top_corner(2) + m);
                    end
                end
            end
            
            % Encode the measurement into a number between 1-256
            obs_index = encode_measurement(sensor_obs, num_observations);
            
            % Update the measurement matrix B
            B(robot_pos_index, obs_index) = prob_correct_measurement;
    
            % -----------------------------------------------------
            % Handle the probabilities for 1 bitflip
            % -----------------------------------------------------
            
            % Compute variations of the sensor measurement considering 1-bit
            % flip possibility in the noise
            noisy_obs_indexes = zeros(num_observations, 1);
            counter = 1;

            for k=1:3
                for m=1:3
                    % Ignore the middle cell
                    if k ~= 2 || m ~=2
                           
                        % Ignore the diagonal cells if we consider only a
                        % sensor with 4 directions for the observations
                        if num_observations == 4 && ((k==1 && m==1) || (k==1 && m==3) || (k==3 && m==1) || (k==3 && m==3))
                            continue;
    
                        % Actually apply a bit flip a get the measurement
                        % encoding
                        else
                            % Get a copy of the actual observation in that cell
                            noisy_observation = sensor_obs;
    
                            % Flip one bit to add noise
                            if noisy_observation(k,m) == 0
                                noisy_observation(k,m) = 1;
                            else
                                noisy_observation(k,m) = 0;
                            end
    
                            % Encode the noisy measurement into a number between
                            % 1-2^num_observations
                            noisy_obs_indexes(counter) = encode_measurement(noisy_observation, num_observations);
                            counter = counter + 1;
                        end
                    end
                end
            end
    
            % Update the measurement matrix with the possible noise
            % measurements
            for k=1:num_observations
                B(robot_pos_index, noisy_obs_indexes(k)) = prob_1_bit_flip / num_observations;
            end
        
        end
    end

end

function [observation_index] = encode_measurement(measurement, num_observations)
%ENCODE_MEASUREMENT Takes a 3x3 matrix of measurements and converts it into
% an encoded measurement from 1-2^num_observations
% It is assumed that num_observations is 4 or 8

    % Encode the individual obstacle measurements into one unique
    % measurement
    observation_index = 0;
    counter = 0;
    
    % Iterate over each cell
    for k=1:3
        for m=1:3
            if k==2 && m==2
                % If we are in the midle square, skip it
                continue;
            
            % Case where the sensor measures in 4 directions (top, right, left
            % down)
            elseif num_observations == 4
    
                if (k==1 && m==1) || (k==1 && m==3) || (k==3 && m==1) || (k==3 && m==3)
                    continue
                else
                    if measurement(k,m) == 1
                        observation_index = observation_index + (2^counter);
                    end
                    counter = counter + 1;
                end
    
            % Case where the sensor measures in 8 directions (includes the
            % diagonals)
            else
                if measurement(k,m) == 1
                    observation_index = observation_index + (2^counter);
                end
                counter = counter + 1;
            end
        end
    end
    
    % Shift everything by 1 such that we start in 1 and stop at 16 or 256
    observation_index = observation_index + 1;
end


function display_grid(ax, grid, robot_pos, grid_title, hit_test)

    % Clear previous visualization
    cla(ax);
    
    % Get the size of the grid
    N = size(grid,1);
    
    % Draw grid lines
    for i=1:N+1
        % Draw horizontal lines
        line(ax, [1 - 0.5, N+1 - 0.5], [i-0.5, i-0.5], 'Color', 'k');
        % Draw vertical lines
        line(ax, [i-0.5, i-0.5], [1-0.5, N+1-0.5], 'Color', 'k');
    end
    
    % Draw the obstacles in the grid
    for i=1:N
        for j=1:N
            if grid(i,j) == 1
                % Check if we want to detect clicks on the rectangles or not
                if hit_test == 0
                    rectangle(ax, 'Position', [i-0.5, j-0.5, 1, 1], 'FaceColor', 'k');
                else
                    rectangle(ax, 'Position', [i-0.5, j-0.5, 1, 1], 'FaceColor', 'k', 'HitTest', 'on', 'ButtonDownFcn', @edit_map_callback);
                end
            end
        end
    end
    
    % Draw the robot position in the grid as a circle
    if robot_pos ~= [-1 -1]
        r = 0.5;
        d = 1;
        px = robot_pos(1)-r;
        py = robot_pos(2)-r;
        rectangle(ax, 'Position',[px py d d],'Curvature',[1,1], 'FaceColor', 'r');
    end

    % Set axis limits to show the board
    axis(ax, [0.5, N+0.5, 0.5, N+0.5]); 
    
    % --------------------------------------------------------
    % Make the grid look exactly like the data in the matrices 
    % --------------------------------------------------------
    % Flip the x and y axis
    view(ax, [90 -90]);
    % Reverse the order of the vertical direction
    set(ax, 'XDir','reverse')
    set(ax,'XTick',1:N);
    set(ax,'YTick',1:N);
    ax.DataAspectRatio = [1 1 1];
    title(ax,grid_title);
end

function display_heatmap(ax, prob_matrix)

    % Get the size of the probability matrix
    N = size(prob_matrix, 1); 
    
    % Create heatmap using heatmap function
    heatmap(ax, 1:N, 1:N, prob_matrix*100, 'Colormap', summer);
end

function [grid] = create_grid(N, n_obs)
% create_grid Function that generates a grid for the robot
% N - The size of the grid
% n_obs - The number of L shaped obstacles to place in the map

    % Create a template obstacle (an L)
    L = [1 0 0; 1 0 0; 1 1 1];
    
    % Create an empty environment
    grid = zeros(N,N);
    
    % Generate the position for the set of obstacles
    obs_pos = round(unifrnd(1,N,[n_obs, 2]));
    
    % Generate a random rotation for the obstacles
    rot = round(unifrnd(1,4,[n_obs, 1]));
    
    % Place the obstacles in the grid
    for i=1:n_obs
    
        % Make a copy of the obtacle that will be rotated 
        obs = L;
    
        % Rotate the obstacle by the times specified in rot
        for j=1:rot(i)
            obs = obs';
        end
    
        % Get the center of the obstacle position and subtract 2 to get the top
        % corner
        pos = obs_pos(i,:) - 2;
    
        % Copy the template obstacle to the grid
        for j=1:3
            for k=1:3
                if pos(1)+j >= 1 && pos(1)+j <= N && pos(2)+k >= 1 && pos(2)+k <= N 
                    grid(pos(1)+j, pos(2)+k) = obs(j,k);
                end
            end
        end
    end

    % Check for individual holes in the diagonals and fill them
    changes = 1;
    
    while changes == 1
        changes = 0;
        % Check for the pattern: x 0
        %                        0 x
        for i=1:N
            for j=1:N
                if i < N && j < N
                    if grid(i,j) == 1 && grid(i+1, j+1) == 1 && grid(i+1,j) == 0 && grid(i,j+1) == 0
                        % Set the right top value to 1
                        grid(i, j+1) = 1;
                        changes = 1;
                    end
                end
            end
        end
        % Check for the pattern: 0 x
        %                        X 0
        for i=1:N
            for j=1:N
                if i < N && j < N
                    if grid(i,j) == 0 && grid(i+1, j+1) == 0 && grid(i+1,j) == 1 && grid(i,j+1) == 1
                        % Set the left top value to 1
                        grid(i,j) = 1;
                        changes = 1;
                    end
                end
            end
        end
    end
end