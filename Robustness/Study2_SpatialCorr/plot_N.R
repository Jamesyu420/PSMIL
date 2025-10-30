ErrMu1_ins <- as.matrix(read.csv("mu1_ins.csv", header = F))
ErrMu1_bag <- as.matrix(read.csv("mu1_bag.csv", header = F))
ErrMu1_com <- as.matrix(read.csv("mu1_com.csv", header = F))

ErrMu0_ins <- as.matrix(read.csv("mu0_ins.csv", header = F))
ErrMu0_bag <- as.matrix(read.csv("mu0_bag.csv", header = F))
ErrMu0_com <- as.matrix(read.csv("mu0_com.csv", header = F))

ErrSigma_ins <- as.matrix(read.csv("Sigma_ins.csv", header = F))
ErrSigma_bag <- as.matrix(read.csv("Sigma_bag.csv", header = F))
ErrSigma_com <- as.matrix(read.csv("Sigma_com.csv", header = F))

ErrPi_ins <- as.matrix(read.csv("pi_ins.csv", header = F))
ErrPi_bag <- as.matrix(read.csv("pi_bag.csv", header = F))
ErrPi_com <- as.matrix(read.csv("pi_com.csv", header = F))


Nlist <- c(50, 100, 200, 500, 1000)

library(reshape2)
library(ggplot2)
library(ggpattern)

Err.ins <- ErrMu1_ins + ErrMu0_ins + ErrSigma_ins
Err.bag <- ErrMu1_bag + ErrMu0_bag + ErrSigma_bag
Err.com <- ErrMu1_com + ErrMu0_com + ErrSigma_com

colnames(Err.ins) <- Nlist
msedata <- rbind(log(Err.ins[,2:5]),
                 log(Err.bag[,2:5]),
                 log(Err.com[,2:5]))
msedata <- data.frame(melt(msedata))
colnames(msedata) <- c("Estimator", "N", "value")
msedata$Estimator <- factor(as.integer(msedata$Estimator / 501) +1,
                            levels = c(1,2,3), labels = c("IMLE","BMLE", "SMLE"))
msedata$N <- factor(msedata$N)
msedata$level <- factor(as.numeric(msedata$Estimator), 
                        levels = c(1,2,3), labels = c(1,2,3))
ggplot(data=msedata, aes(x=N, y=value, fill=Estimator)) + 
  geom_boxplot_pattern(
    aes(pattern = Estimator),
    fill = 'gray'
  ) + 
  labs(x="N", y="log(MSE)") +
  theme(
    plot.title = element_text(hjust = 0.5, size = 20),  # Adjust title size
    axis.title = element_text(size = 16),  # Adjust axis title size
    axis.text = element_text(size = 14),  # Adjust axis text size
    panel.grid.major = element_blank(), 
    panel.grid.minor = element_blank(),
    panel.background = element_blank(),
    panel.border = element_rect(fill = NA),
    legend.position = "none"  # Remove the legend
  )

