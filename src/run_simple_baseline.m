function result = run_simple_baseline(calcium, fps, params)
% run_simple_baseline
%
% Goal:
%   Run a simple calcium-event detection baseline on one fluorescence trace.
%
% Model idea:
%   The model detects calcium events using a combined evidence score:
%
%       event_score =
%           derivative_weight * positive_derivative
%         + amplitude_weight  * smoothed_calcium
%
% Inputs:
%   calcium - calcium fluorescence trace
%   fps     - sampling rate of the calcium trace
%   params  - struct with model parameters:
%       params.smoothing_window_sec
%       params.derivative_weight
%       params.amplitude_weight
%       params.event_score_threshold
%       params.min_event_distance_sec
%
% Output:
%   result - struct containing:
%       predicted_events
%       event_score
%       calcium_z
%       calcium_smooth
%       d_calcium_z
%       score_peaks
%       peak_locs
%       params

    %% Basic input checks

    if nargin ~= 3
        error('run_simple_baseline requires exactly 3 inputs: calcium, fps, params.');
    end

    if isempty(calcium)
        error('Input calcium trace is empty.');
    end

    if isempty(fps) || ~isnumeric(fps) || ~isscalar(fps)
        error('fps must be a single numeric value.');
    end

    if fps <= 0
        error('fps must be positive. Received: %.4f', fps);
    end

    if ~isstruct(params)
        error('params must be a struct.');
    end

    required_fields = { ...
        'smoothing_window_sec', ...
        'derivative_weight', ...
        'amplitude_weight', ...
        'event_score_threshold', ...
        'min_event_distance_sec'};

    for i = 1:length(required_fields)
        field_name = required_fields{i};

        if ~isfield(params, field_name)
            error('params is missing required field: %s', field_name);
        end
    end

    %% Convert calcium to column vector

    calcium = double(calcium(:));

    if any(isnan(calcium))
        error('Calcium trace contains NaN values.');
    end

    if any(isinf(calcium))
        error('Calcium trace contains Inf values.');
    end

    if std(calcium) == 0
        error('Calcium trace has zero standard deviation. Cannot z-score.');
    end

    %% Convert time parameters to samples

    smoothing_window_samples = max(1, round(params.smoothing_window_sec * fps));
    min_event_distance_samples = max(1, round(params.min_event_distance_sec * fps));

    %% Normalize calcium trace

    % Z-score normalization makes thresholds easier to interpret.
    calcium_z = (calcium - mean(calcium)) / std(calcium);

    %% Smooth calcium trace

    % Smoothing reduces small fluctuations before derivative calculation.
    calcium_smooth = movmean(calcium_z, smoothing_window_samples);

    %% Calculate derivative

    % The derivative highlights rising phases of the calcium trace.
    d_calcium = [0; diff(calcium_smooth)];

    if std(d_calcium) == 0
        error('Calcium derivative has zero standard deviation. Cannot z-score derivative.');
    end

    % Normalize derivative to z-score units.
    d_calcium_z = (d_calcium - mean(d_calcium)) / std(d_calcium);

    %% Combined evidence score

    % Keep only positive derivative values.
    % Falling calcium should not support detection of a new event.
    positive_derivative = max(d_calcium_z, 0);

    event_score = ...
        params.derivative_weight * positive_derivative + ...
        params.amplitude_weight  * calcium_smooth;

    %% Detect events

    [score_peaks, peak_locs] = findpeaks(event_score, ...
        'MinPeakHeight', params.event_score_threshold, ...
        'MinPeakDistance', min_event_distance_samples);

    predicted_events = false(size(calcium));
    predicted_events(peak_locs) = true;

    %% Package output

    result = struct();

    result.predicted_events = predicted_events;
    result.event_score = event_score;

    result.calcium_z = calcium_z;
    result.calcium_smooth = calcium_smooth;
    result.d_calcium_z = d_calcium_z;

    result.score_peaks = score_peaks;
    result.peak_locs = peak_locs;

    result.params = params;
    result.smoothing_window_samples = smoothing_window_samples;
    result.min_event_distance_samples = min_event_distance_samples;

    result.n_predicted_events = sum(predicted_events);
end