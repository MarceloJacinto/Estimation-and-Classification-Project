function [observation_index] = encode_measurement(measurement)
%ENCODE_MEASUREMENT Takes a 3x3 matrix of measurements and converts it into
% an encoded measurement from 1-256

% Encode the individual obstacle measurements into one unique
% measurement
observation_index = 0;
counter = 0;

% Iterate over each cell
for k=1:3
    for m=1:3
        if k==2 && m==2
            % If we are in the midle square, skip it
            continue
        % Check whether to include the cell or not
        else
            if measurement(k,m) == 1
                observation_index = observation_index + (2^counter);
            end
            counter = counter + 1;
        end
    end
end

% Shift everything by 1 such that we start in 1 and stop at 256
observation_index = observation_index + 1;

end

