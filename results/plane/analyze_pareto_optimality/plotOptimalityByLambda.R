data = read.csv("optimality-by-lambda.tsv", sep="\t")

library(tidyr)
library(dplyr)
library(ggplot2)

plot = ggplot(data, aes(x=Lambda, y=Quantile)) + geom_line() + facet_wrap(~Language)

cBound = c()
pValue = c()
for(i in (1:nrow(data))) {
   quantile = data$Quantile[[i]]
   samples = data$Samples[[i]]
   test = binom.test(round(quantile*samples), round(samples), alternative="greater", conf.level=1-0.05/3)
   ci = test$conf.int[[1]]
   cBound = c(cBound, ci)
   pValue = c(pValue, test$p.value*3)
}
data$LowerBound = cBound
data$pValue = pValue


plot = ggplot(data, aes(x=Lambda, y=Quantile)) + geom_line() + geom_line(aes(x=Lambda,y=LowerBound)) + facet_wrap(~Language)


lambdas = c(0.0, 0.5, 0.98)
plot = ggplot(data %>% filter(Lambda %in% lambdas), aes(x=Lambda, y=Quantile)) + geom_line() + geom_line(aes(x=Lambda,y=LowerBound)) + facet_wrap(~Language)


data_ = data %>% filter(Lambda %in% lambdas)


