% Callback function to update outlier lines
function update_limits()
    % Read input from the text boxes
    lower_limit = str2double(lower_input.String);
    upper_limit = str2double(upper_input.String);
    
    % Validate input
    if isnan(lower_limit) || isnan(upper_limit) || lower_limit < 0 || upper_limit > 100 || lower_limit >= upper_limit
        disp('Invalid limits. Please enter values between 0 and 100, with lower < upper.');
        return;
    end
    
    % Update outliers
    out = isoutlier(sss, 'percentiles', [lower_limit, upper_limit]);
    
    % Update black lines
    delete(line_handles); % Remove old outlier lines
    line_handles = plot(ax, freq, pow_median(:, out)', '-k', 'LineWidth', 1);
    
    % Update legend
    legend(ax, rawData.label(out), 'Interpreter', 'none');
    
    % Display updated limits and outlier sensors
    disp(['Updated Percentiles: [', num2str(lower_limit), ', ', num2str(upper_limit), ']']);
    disp('Outliers:');
    disp(rawData.label(out));
end