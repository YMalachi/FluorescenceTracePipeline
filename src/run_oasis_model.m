function result = run_oasis_model(calcium, fps, params)
% run_oasis_model
%
% Goal:
%   Run OASIS calcium deconvolution on one fluorescence trace and convert
%   the inferred activity signal into predicted event times.
%
% Inputs:
%   calcium - calcium fluorescence trace
%   fps     - sampling rate of the calcium trace
%   params  - struct with OASIS/event-conversion parameters:
%       params.ar_model
%       params.method
%       params.optimize_b
%       params.optimize_pars
%       params.oasis_threshold_z
%       params.min_event_distance_sec
%
% Output:
%   result - struct containing:
%       predicted_events
%       c_oasis
%       s_oasis
%       s_oasis_z
%       score_peaks
%       peak_locs
%       params
%       n_predicted_events

    %% Basic input checks

    if nargin ~= 3
        error('run_oasis_model requires exactly 3 inputs: calcium, fps, params.');
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
        'ar_model', ...
        'method', ...
        'optimize_b', ...
        'optimize_pars', ...
        'oasis_threshold_z', ...
        'min_event_distance_sec'};

    for i = 1:length(required_fields)
        field_name = required_fields{i};

        if ~isfield(params, field_name)
            error('params is missing required field: %s', field_name);
        end
    end

    %% Convert calcium to row vector for OASIS

    calcium = double(calcium(:));

    if any(isnan(calcium))
        error('Calcium trace contains NaN values.');
    end

    if any(isinf(calcium))
        error('Calcium trace contains Inf values.');
    end

    if std(calcium) == 0
        error('Calcium trace has zero standard deviation.');
    end

    % deconvolveCa commonly expects a row vector.
    y = calcium(:)';

    %% Run OASIS

    try
        [c_oasis, s_oasis, options_oasis] = deconvolveCa( ...
            y, ...
            params.ar_model, ...
            'method', params.method, ...
            'optimize_b', params.optimize_b, ...
            'optimize_pars', params.optimize_pars);

    catch ME
        error(['OASIS deconvolution failed.\n', ...
               'ar_model = %s\n', ...
               'method = %s\n', ...
               'Original error:\n%s'], ...
               params.ar_model, params.method, ME.message);
    end

    %% Convert outputs to column vectors

    c_oasis = double(c_oasis(:));
    s_oasis = double(s_oasis(:));

    %% OASIS output checks

    if length(c_oasis) ~= length(calcium)
        error(['OASIS calcium estimate length does not match input calcium length. ', ...
               'c_oasis length = %d, calcium length = %d.'], ...
               length(c_oasis), length(calcium));
    end

    if length(s_oasis) ~= length(calcium)
        error(['OASIS activity estimate length does not match input calcium length. ', ...
               's_oasis length = %d, calcium length = %d.'], ...
               length(s_oasis), length(calcium));
    end

    if any(isnan(c_oasis)) || any(isinf(c_oasis))
        error('OASIS calcium estimate contains NaN or Inf values.');
    end

    if any(isnan(s_oasis)) || any(isinf(s_oasis))
        error('OASIS activity estimate contains NaN or Inf values.');
    end

    %% Normalize OASIS activity signal

    % s_oasis is the inferred activity/spike-like signal.
    % We z-score it so the threshold is easier to reuse across neurons.
    if std(s_oasis) == 0
        error('OASIS activity estimate has zero standard deviation. Cannot z-score.');
    end

    s_oasis_z = (s_oasis - mean(s_oasis)) / std(s_oasis);

    %% Convert OASIS activity into predicted events

    min_event_distance_samples = max(1, round(params.min_event_distance_sec * fps));

    [score_peaks, peak_locs] = findpeaks(s_oasis_z, ...
        'MinPeakHeight', params.oasis_threshold_z, ...
        'MinPeakDistance', min_event_distance_samples);

    predicted_events = false(size(s_oasis_z));
    predicted_events(peak_locs) = true;

    %% Package output

    result = struct();

    result.predicted_events = predicted_events;

    result.c_oasis = c_oasis;
    result.s_oasis = s_oasis;
    result.s_oasis_z = s_oasis_z;

    result.score_peaks = score_peaks;
    result.peak_locs = peak_locs;

    result.params = params;
    result.options_oasis = options_oasis;

    result.min_event_distance_samples = min_event_distance_samples;
    result.n_predicted_events = sum(predicted_events);
end