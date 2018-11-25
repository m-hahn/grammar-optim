dataP = read.csv("adversarial-parser.tsv", sep="\t")
dataS = read.csv("adversarial-lm.tsv", sep="\t")

library(ggplot2)
plot = ggplot(data=dataP, aes(x=UAS, y=1, group=Model, color=Model, fill = Model)) + 
      geom_bar(stat="identity") + facet_wrap(~Type)

ggsave(plot=plot, filename="figures/adversarial-parse.pdf")



plot = ggplot(data=dataS, aes(x=LM_Loss, y=1, group=Model, color=Model, fill = Model)) + 
      geom_bar(stat="identity") + facet_wrap(~Type)

ggsave(plot=plot, filename="figures/adversarial-lm.pdf")
