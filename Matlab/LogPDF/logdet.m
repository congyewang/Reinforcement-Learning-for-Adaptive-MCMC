function y = logdet(A)
U = chol(A);
y = 2*sum(log(diag(U)));
end