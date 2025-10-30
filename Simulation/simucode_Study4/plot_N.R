ErrMu1_sub <- as.matrix(read.csv("./res/mu1_sub.csv", header = F))
ErrMu0_sub <- as.matrix(read.csv("./res/mu0_sub.csv", header = F))
ErrSigma_sub <- as.matrix(read.csv("./res/Sigma_sub.csv", header = F))
ErrPi_sub <- as.matrix(read.csv("./res/pi_sub.csv", header = F))

Err <- ErrMu1_sub + ErrMu0_sub + ErrSigma_sub + ErrPi_sub

pilotList <- c(0,0.1, 0.25, 0.5, 1)
pilotList <- as.character(pilotList)
pilotList <- c(pilotList, "Inf")

colnames(Err) <- pilotList
opar <- par(mai=c(0.9,1,0.1,0.1), cex.main = 1.8)
boxplot(log(Err), ylim = c(-10,-8.8),
        xlab="Pilot Sample Fraction", ylab="log(MSE)",
        cex.axis=1.2, cex.lab = 1.5, lwd = 2)
par(opar)
