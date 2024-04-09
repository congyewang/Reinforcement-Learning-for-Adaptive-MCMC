function [res] = grad_log_target_pdf(x)

[~, res] = wrapped_log_target_pdf(x', 'earnings-log10earn_height');

end
