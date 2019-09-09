data = read.csv("optimality-by-lambda.tsv", sep="\t")

library(tidyr)
library(dplyr)
library(ggplot2)

plot = ggplot(data, aes(x=Lambda, y=Quantile)) + geom_line() + facet_wrap(~Language)

cBound = c()
for(i in (1:nrow(data))) {
   quantile = data$Quantile[[i]]
   samples = data$Samples[[i]]
   test = binom.test(round(quantile*samples), round(samples), alternative="greater", conf.level=0.99)
   ci = test$conf.int[[1]]
   cBound = c(cBound, ci)
}
data$LowerBound = cBound


plot = ggplot(data, aes(x=Lambda, y=Quantile)) + geom_line() + geom_line(aes(x=Lambda,y=LowerBound)) + facet_wrap(~Language)


