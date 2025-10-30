ErrMu1_ins <- as.matrix(read.csv("./res/ErrMu1_ins.csv", header = F))
ErrMu1_bag <- as.matrix(read.csv("./res/ErrMu1_bag.csv", header = F))
ErrMu0_ins <- as.matrix(read.csv("./res/ErrMu0_ins.csv", header = F))
ErrMu0_bag <- as.matrix(read.csv("./res/ErrMu0_bag.csv", header = F))
ErrSigma_ins <- as.matrix(read.csv("./res/ErrSigma_ins.csv", header = F))
ErrSigma_bag <- as.matrix(read.csv("./res/ErrSigma_bag.csv", header = F))
ErrPi_ins <- as.matrix(read.csv("./res/ErrPi_ins.csv", header = F))
ErrPi_bag <- as.matrix(read.csv("./res/ErrPi_bag.csv", header = F))

p = 50; replication = 500
mu1.true <- rep(1,p); mu0.true <- rep(-1,p)
sigma2List <- c(0.5, 1, 4, 9)
SNR <- c()
for(sigma2 in sigma2List){
  Sigma = diag(sigma2, p)
  SNR <- c(SNR, t(mu1.true - mu0.true) %*% solve(Sigma) %*% (mu1.true - mu0.true))
}
SNR <- round(SNR, 1)

colnames(ErrMu1_ins) <- sigma2List; colnames(ErrMu1_bag) <- sigma2List
ins.mu1 <- apply(ErrMu1_ins, 2, mean); bag.mu1 <- apply(ErrMu1_bag, 2, mean)
opar <- par(mai=c(0.8,1,0.4,0.2), cex.main = 1.8)
plot(sigma2List, log(bag.mu1), type = "b", lty = 2,
     xlab=expression(sigma), ylab="log(MSE)", main = expression(mu[1]),
     cex.axis=1.2, cex.lab = 1.5, lwd = 2.5)
lines(sigma2List, log(ins.mu1), type = "b", pch = 2, lwd = 2.5)
par(opar)

colnames(ErrMu0_ins) <- sigma2List; colnames(ErrMu0_bag) <- sigma2List
ins.mu0 <- apply(ErrMu0_ins, 2, mean); bag.mu0 <- apply(ErrMu0_bag, 2, mean)
opar <- par(mai=c(0.8,1,0.4,0.2), cex.main = 1.8)
plot(sigma2List, log(bag.mu0), type = "b", lty = 2,
     xlab=expression(sigma), ylab="log(MSE)", main = expression(mu[0]),
     cex.axis=1.2, cex.lab = 1.5, lwd = 2.5)
lines(sigma2List, log(ins.mu0), type = "b", pch = 2, lwd = 2.5)
par(opar)

colnames(ErrSigma_ins) <- sigma2List; colnames(ErrSigma_bag) <- sigma2List
ins.Sigma <- apply(ErrSigma_ins, 2, mean); bag.Sigma <- apply(ErrSigma_bag, 2, mean)
opar <- par(mai=c(0.8,1,0.4,0.2), cex.main = 1.8)
plot(sigma2List, log(bag.Sigma), type = "b", lty = 2,
     xlab=expression(sigma), ylab="log(MSE)", main = expression(Omega),
     cex.axis=1.2, cex.lab = 1.5, lwd = 2.5)
lines(sigma2List, log(ins.Sigma), type = "b", pch = 2, lwd = 2.5)
par(opar)

colnames(ErrPi_ins) <- sigma2List; colnames(ErrPi_bag) <- sigma2List
ins.pi <- apply(ErrPi_ins, 2, mean); bag.pi <- apply(ErrPi_bag, 2, mean)
opar <- par(mai=c(0.8,1,0.4,0.2), cex.main = 1.8)
plot(sigma2List, log(bag.pi), type = "b", lty = 2,
     xlab=expression(sigma), ylab="log(MSE)", main = expression(pi),
     cex.axis=1.2, cex.lab = 1.5, lwd = 2.5,  ylim = c(-13.5, -9.4))
lines(sigma2List, log(ins.pi), type = "b", pch = 2, lwd = 2.5)
legend(6,-11, legend = c("IMLE", "BMLE"),
       pch = c(2,1), lty = c(1,2), lwd = c(2.5,2.5), cex = 1.2)
par(opar)

