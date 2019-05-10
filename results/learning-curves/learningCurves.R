data = read.csv("czech-learning-parser.tsv", sep="\t")

library(ggplot2)

plot = ggplot(data=data, aes(x=Fraction, y=UAS, group=Model+RNN_Dim, color=Model, fill = Model)) + 
      geom_line() + theme_bw() + theme(legend.position="none")
plot = plot + theme(axis.title.x = element_text(size=17))
plot = plot + theme(axis.title.y = element_text(size=17))
plot = plot + xlab("Fraction of Data")
plot = plot + ylab("Parsing Accuracy")

ggsave(plot=plot, filename="figures/learning-parser-czech.pdf") #, height=5, width=5)

plot = ggplot(data=data, aes(x=log(Fraction), y=UAS, group=Model, color=Model, fill = Model)) + 
      geom_line() + theme_bw()

ggsave(plot=plot, filename="figures/learning-parser-czech-log.pdf") #, height=5, width=5)

plot = ggplot(data=data, aes(x=Fraction, y=Loss, group=Model+RNN_Dim, color=Model, fill = Model)) + 
      geom_line() + theme_bw() + theme(legend.position="none")
plot = plot + theme(axis.title.x = element_text(size=17))
plot = plot + theme(axis.title.y = element_text(size=17))
plot = plot + xlab("Fraction of Data")
plot = plot + ylab("Parsing Loss")
plot = plot + ylim(NA, 2.6)
ggsave(plot=plot, filename="figures/learning-parser-czech-logloss.pdf") #, height=5, width=5)



