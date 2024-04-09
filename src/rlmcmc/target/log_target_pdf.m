function [res] = log_target_pdf(x)

[res, ~] = wrapped_log_target_pdf(x', 'earnings_log10earn_height');

end
