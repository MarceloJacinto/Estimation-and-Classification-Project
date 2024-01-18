clear all; close all; clc;

% Initialize the seed and random number generator
rng("default");

% Create the grid with obstacles
N = 20;
n_obs = 30;
simulation_steps = 0;

% Create the grid where the robot will navigate
grid = create_grid(N, n_obs);

% Compute the transition matrix A from the prior grid
A = compute_transition_matrix(grid);
B = compute_observation2_matrix(grid);

% Place the robot in some random initial position
robot_pos = [-1,-1];

% Create the probability matrix that will hold the probability of each cell
prob_matrix = init_cell_prob(grid);

% Create a new figure where the UI will be displayed
fig = uifigure('Name', 'Robot Motion and Localization', 'Position', [200, 200, 900, 500]);
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
display_grid(grid_ax, grid, robot_pos);
display_heatmap(heat_map_ax, prob_matrix);

% Create the button interface to control the simulation
start_button = uicontrol(p,'Style', 'pushbutton', 'String', 'Start', 'Position', [20, 20, 100, 30], 'Callback', @start_btn_click);
pause_button = uicontrol(p,'Style', 'pushbutton', 'String', 'Pause', 'Position', [130, 20, 100, 30], 'Callback', @pause_btn_click);
next_step_button = uicontrol(p,'Style', 'pushbutton', 'String', 'Next Step', 'Position', [240, 20, 100, 30], 'Callback', @next_step_btn_click);
iterations_text = uicontrol(p,'Style', 'text', 'String', sprintf('Number of Iterations: %d', simulation_steps), 'Position', [5, 50, 200, 30]);

% Create a timer to execute the motion of the robot automatically
t = timer('TimerFcn',@next_step_btn_click, 'ExecutionMode', 'fixedRate', 'Period', 1, 'BusyMode','drop');

%% Function definitions 
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

% Increment the simulation step
simulation_steps = simulation_steps + 1;

% Perform one simulation step update
[robot_pos, prob_matrix] = simulate_one_step(robot_pos, prob_matrix, grid, A, B);

% Update the grid visualization
display_grid(grid_ax, grid, robot_pos);
display_heatmap(heat_map_ax, prob_matrix);
iterations_text.String = sprintf('Number of Iterations: %d', simulation_steps);

% Update robot_pos, prob_matrix, grid and num_iterations in the workspace
assignin('base', 'robot_pos', robot_pos);
assignin('base', 'prob_matrix', prob_matrix);
assignin('base', 'simulation_steps', simulation_steps);
assignin('base', 'iterations_text', iterations_text);

end

function [robot_pos, prob_matrix] = simulate_one_step(robot_pos, prob_matrix, grid, A, B)

% If the robot position was not initialized, yet - initialize it
if robot_pos == [-1 -1]
    robot_pos = init_robot_position(grid);
    first_step = 1;
else
    % Move the robot to the next position
    first_step = 0;
    robot_pos = robot_motion(robot_pos, grid);
end

% Perform a measurement 
measurement = sensor_model(robot_pos, grid);

% Compute the probabilities of each cell
prob_matrix = compute_cell_probabilities(prob_matrix, measurement, A, B, first_step);

end

function [prob_matrix] = compute_cell_probabilities(prob_matrix, measurement, A, B, first_step)
% compute_cell_probabilities - Computes the probability of each cell
% given the robot measurements of the environment, the probability
% transition matrix, the measurement matrix and the previous probability
% matrix from the last iteration

% Get the number of possible states
N = size(A,1);

% Encode the measurements matrix into a measurement index
obs_index = encode_measurement(measurement);

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
prob_matrix = prob_matrix / total_sum;

end

function [sensor_obs] = sensor_model(robot_pos, grid)
% sensor_model Implements the robot sensor model
% robot_pos - [row,col] of the actual robot position
% grid - the grid of the world with obstacles

% Get the size of the grid
N = size(grid,1);

% Create an empty measurement of the sensor
sensor_obs = zeros(3,3);

% Get the top_corner of a 3x3 square where the robot is positioned
top_corner = robot_pos - 2;

% Check which cells surrounding the robot are obstacles
for i=1:3
    for j=1:3
        % Check if we are outside the grid. If so, place an obstacle
        if top_corner(1) + i < 1 || top_corner(2) + j < 1 || top_corner(1) + i > N || top_corner(2) + j > N
            sensor_obs(i,j) = 1;
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


function display_grid(ax, grid, robot_pos)

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
            rectangle(ax, 'Position', [i-0.5, j-0.5, 1, 1], 'FaceColor', 'k');
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
title(ax,'Robot Motion');
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
% TODO
end