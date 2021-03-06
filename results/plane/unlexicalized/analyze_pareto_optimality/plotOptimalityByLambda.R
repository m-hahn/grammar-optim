data = read.csv("optimality-by-lambda.tsv", sep="\t")

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
   test = binom.test(round(quantile*samples), round(samples), alternative="two.sided", conf.level=1-0.05)
   cBoundUpper = c(cBoundUpper, test$conf.int[[2]])
   cBoundLower = c(cBoundLower, test$conf.int[[1]])
   pValue = c(pValue, test$p.value*3)
}
data$UpperBound = cBoundUpper
data$LowerBound = cBoundLower
data$pValue = pValue


plot = ggplot(data, aes(x=Lambda, y=Quantile)) + geom_line() + geom_line(aes(x=Lambda,y=LowerBound)) + facet_wrap(~Language)



corpusSize = read.csv("../../../corpus-size/corpus-sizes.tsv", sep="\t")
languagesOrdered = corpusSize$language[order(-corpusSize$sents_train)]

data$Language = factor(data$Language, levels=languagesOrdered)



plot = ggplot(data, aes(x=Lambda, y=Quantile)) + geom_line() + geom_line(aes(x=Lambda,y=UpperBound), linetype="dotted") + geom_line(aes(x=Lambda,y=LowerBound), linetype="dotted") + facet_wrap(~Language)
plot = plot + theme(axis.text = element_text(size=10), axis.text.x = element_text(angle=90, hjust=1)) 

ggsave(plot, file="figures/quantileByLambda.pdf")


