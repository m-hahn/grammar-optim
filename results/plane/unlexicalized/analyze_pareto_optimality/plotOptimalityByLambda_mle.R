data = read.csv("optimality-by-lambda-mle.tsv", sep="\t")

library(tidyr)
library(dplyr)
library(ggplot2)

plot = ggplot(data, aes(x=Lambda, y=Quantile)) + geom_line() + facet_wrap(~Language)

cBoundUpper = c()
cBoundLower = c()
pValue = c()
for(i in (1:nrow(data))) {
   quantile = data$Quantile[[i]]
   samples = data$Samples[[i]]
   if(is.na(quantile)) {
      cBoundUpper = c(cBoundUpper, 1)
      cBoundLower = c(cBoundLower, 0)
      pValue = c(pValue, 1)

   } else{
      test = binom.test(round(quantile*samples), round(samples), alternative="two.sided", conf.level=1-0.05)
      cBoundUpper = c(cBoundUpper, test$conf.int[[2]])
      cBoundLower = c(cBoundLower, test$conf.int[[1]])
      pValue = c(pValue, test$p.value*3)
   }
}
data$UpperBound = cBoundUpper
data$LowerBound = cBoundLower
data$pValue = pValue


plot = ggplot(data, aes(x=Lambda, y=Quantile)) + geom_line() + geom_line(aes(x=Lambda,y=LowerBound)) + facet_wrap(~Language)


#lambdas = c(0.0, 0.5, 0.98)
#plot = ggplot(data %>% filter(Lambda %in% lambdas), aes(x=Lambda, y=Quantile)) + geom_line() + geom_line(aes(x=Lambda,y=LowerBound)) + facet_wrap(~Language)

plot = ggplot(data, aes(x=Lambda, y=Quantile)) + geom_line() + geom_line(aes(x=Lambda,y=UpperBound), linetype="dotted") + geom_line(aes(x=Lambda,y=LowerBound), linetype="dotted") + facet_wrap(~Language)


ggsave(plot, file="figures/quantileByLambda-mle.pdf")


