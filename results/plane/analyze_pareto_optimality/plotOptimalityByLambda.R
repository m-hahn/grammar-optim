data = read.csv("optimality-by-lambda.tsv", sep="\t")

library(tidyr)
library(dplyr)
library(ggplot2)

plot = ggplot(data, aes(x=Lambda, y=Quantile)) + geom_line() + facet_wrap(~Language)




