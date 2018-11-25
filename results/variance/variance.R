



library(tidyr)
library(dplyr)
library(ggplot2)




dataS = read.csv("CS_SCR/deps/plane-fixed.tsv", sep="\t")

dataS = dataS %>% filter(Model %in% c(3516168, 2138590))

plot = ggplot(dataS, aes(x=Surp, group=Type, color=Type)) + geom_histogram()

ggsave(plot, file="variance.pdf")


