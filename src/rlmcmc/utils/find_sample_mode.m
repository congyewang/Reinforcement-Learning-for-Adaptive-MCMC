function sample_mode = find_sample_mode(x0)

neg_log_target_pdf = @(x) -log_target_pdf(x);

options = optimoptions('fminunc', 'Algorithm', 'quasi-newton');

[sample_mode, ~] = fminunc(neg_log_target_pdf, x0, options);

end
