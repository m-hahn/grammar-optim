data = read.csv("czech-learning-parser.tsv", sep="\t")

library(ggplot2)

plot = ggplot(data=data, aes(x=Fraction, y=UAS, group=Model+RNN_Dim, color=Model, fill = Model)) + 
      geom_line() + theme_bw() + theme(legend.position="none")

ggsave(plot=plot, filename="figures/learning-parser-czech.pdf") #, height=5, width=5)

plot = ggplot(data=data, aes(x=log(Fraction), y=UAS, group=Model, color=Model, fill = Model)) + 
      geom_line() + theme_bw()

ggsave(plot=plot, filename="figures/learning-parser-czech-log.pdf") #, height=5, width=5)



