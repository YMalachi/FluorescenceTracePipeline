function result = run_template_matching_model(calcium, fps, params)
% run_template_matching_model
%
% Goal:
%   Detect calcium events using template matching / matched filtering.
%
% Model idea:
%   A neural event creates a calcium transient with a fast rise and slower decay.
%   We create a simple calcium-event template, slide it across the calcium trace,
%   and detect peaks in the template-matching score.
%
% Inputs:
%   calcium - calcium fluorescence trace
%   fps     - sampling rate in Hz
%   params  - struct with model parameters:
%       params.smoothing_window_sec
%       params.template_rise_sec
%       params.template_decay_sec
%       params.template_duration_sec
%       params.score_threshold_z
%       params.min_event_distance_sec
%
% Output:
%   result - struct containing:
%       predicted_events
%       event_score
%       event_score_z
%       calcium_z
%       calcium_smooth
%       template
%       template_time
%       peak_locs
%       score_peaks
%       params
%       n_predicted_events

    %% Basic input checks

    if nargin ~= 3
        error('run_template_matching_model requires exactly 3 inputs: calcium, fps, params.');
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
        'template_rise_sec', ...
        'template_decay_sec', ...
        'template_duration_sec', ...
        'score_threshold_z', ...
        'min_event_distance_sec'};

    for i = 1:length(required_fields)
        field_name = required_fields{i};

        if ~isfield(params, field_name)
            error('params is missing required field: %s', field_name);
        end
    end

    %% Validate parameters

    if params.smoothing_window_sec < 0
        error('smoothing_window_sec must be non-negative.');
    end

    if params.template_rise_sec <= 0
        error('template_rise_sec must be positive.');
    end

    if params.template_decay_sec <= 0
        error('template_decay_sec must be positive.');
    end

    if params.template_duration_sec <= 0
        error('template_duration_sec must be positive.');
    end

    if params.template_duration_sec <= params.template_rise_sec
        error('template_duration_sec should be larger than template_rise_sec.');
    end

    if params.min_event_distance_sec < 0
        error('min_event_distance_sec must be non-negative.');
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

    template_n_samples = max(3, round(params.template_duration_sec * fps));
    template_time = (0:template_n_samples-1)' / fps;

    %% Normalize and smooth calcium

    % Z-score calcium so template matching is less sensitive to absolute scale.
    calcium_z = (calcium - mean(calcium)) / std(calcium);

    % Optional light smoothing before template matching.
    calcium_smooth = movmean(calcium_z, smoothing_window_samples);

    %% Build calcium-event template

    % Template form:
    %   fast rise term:       1 - exp(-t / rise_tau)
    %   slow decay term:      exp(-t / decay_tau)
    %
    % Together this creates a transient that rises and then decays.
    rise_component = 1 - exp(-template_time / params.template_rise_sec);
    decay_component = exp(-template_time / params.template_decay_sec);

    template = rise_component .* decay_component;

    % Remove DC component so the filter responds to shape, not baseline level.
    template = template - mean(template);

    % Normalize template energy.
    template_norm = norm(template);

    if template_norm == 0
        error('Template norm is zero. Check template parameters.');
    end

    template = template / template_norm;

    %% Matched filtering / template correlation

    % We use convolution with the reversed template.
    % This gives a score that is high where the calcium trace locally
    % resembles the template.
    event_score = conv(calcium_smooth, flipud(template), 'same');

    if std(event_score) == 0
        error('Template-matching score has zero standard deviation. Cannot z-score.');
    end

    % Z-score score for thresholding.
    event_score_z = (event_score - mean(event_score)) / std(event_score);

    %% Detect peaks in template-matching score

    [score_peaks, peak_locs] = findpeaks(event_score_z, ...
        'MinPeakHeight', params.score_threshold_z, ...
        'MinPeakDistance', min_event_distance_samples);

    predicted_events = false(size(calcium));
    predicted_events(peak_locs) = true;

    %% Package output

    result = struct();

    result.predicted_events = predicted_events;

    result.event_score = event_score;
    result.event_score_z = event_score_z;

    result.calcium_z = calcium_z;
    result.calcium_smooth = calcium_smooth;

    result.template = template;
    result.template_time = template_time;

    result.score_peaks = score_peaks;
    result.peak_locs = peak_locs;

    result.params = params;
    result.smoothing_window_samples = smoothing_window_samples;
    result.min_event_distance_samples = min_event_distance_samples;
    result.template_n_samples = template_n_samples;

    result.n_predicted_events = sum(predicted_events);
end