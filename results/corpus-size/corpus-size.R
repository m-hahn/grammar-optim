data = read.csv("corpus-sizes.tsv", sep="\t")

sum(data$sents_train) + sum(data$sents_dev)
sum(data$words_train) + sum(data$words_dev)

median(data$sents_train + data$sents_dev)
median(data$words_train + data$words_dev)

