function [val, grad] = log_target_pdf_val_and_grad(x)

[val, grad] = wrapped_log_target_pdf(x, 'earnings_log10earn_height');

end
