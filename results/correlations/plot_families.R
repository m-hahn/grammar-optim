

plot_orders_real = ggplot(D %>% filter(Type == "Real Languages"), aes(x = 1, y = Language_Ordered, group=CoarseDependency)) + 
  geom_point(aes(fill=DirB, colour = DirB, size =1), position = position_dodge(width=2.0)) +
#  scale_color_gradient() + #values=c("blue", "green")) +
  theme_bw() + theme(axis.text.x=element_blank(), #element_text(size=9, angle=0, vjust=0.3),
                     axis.text.y=element_blank(),axis.ticks=element_blank(),
                     plot.title=element_text(size=11)) +
  theme(axis.title=element_blank()) + 
  theme(legend.position="none") + labs(x=NULL)
plot_orders_eff = ggplot(D %>% filter(Type == "Efficiency"), aes(x = 1, y = Language_Ordered, group=CoarseDependency)) + 
  geom_point(aes(fill=DirB, colour = DirB, size =1), position = position_dodge(width=2.0)) +
#  scale_color_gradient() + #values=c("blue", "green")) +
  theme_bw() + theme(axis.text.x=element_blank(), #element_text(size=9, angle=0, vjust=0.3),
                     axis.text.y=element_blank(),axis.ticks=element_blank(),
                     plot.title=element_text(size=11)) +
  theme(axis.title=element_blank()) + 
  theme(legend.position="none") + labs(x=NULL)
D$LanguageNumeric = as.numeric(D$Language_Ordered)
DLang = unique(D %>% select(Language_Ordered, iso_Ordered))
DFam = D %>% group_by(Family) %>% summarise(Start = min(Language_Ordered), End = max(Language_Ordered), Mean = mean(LanguageNumeric))
DFam = DFam %>% mutate(Family = ifelse(Family == "Malayo-Sumbawan", "Mal.-Sum.", as.character(Family)))
DFam = DFam %>% mutate(Family = ifelse(Family == "Sino-Tibetan", "Sin.-Tib.", as.character(Family)))
plot_langs = ggplot(DLang) 
plot_langs = plot_langs +  theme_bw() + theme(axis.text.x=element_blank(), #element_text(size=9, angle=0, vjust=0.3),
                     axis.text.y=element_blank(),
                     plot.title=element_text(size=11)) 
plot_langs = plot_langs + geom_text(aes(x=1.7 + 0.05, y=Language_Ordered, label=Language), hjust=1, size=3, colour="grey30")
plot_langs = plot_langs +      	theme(axis.title=element_blank()) 
plot_langs = plot_langs + xlim(-2.0, 1.9)
#plot_langs = plot_langs + geom_segment(aes(x=0, y=Language_Ordered, xend=1, yend=Language_Ordered)) 
plot_langs = plot_langs + geom_segment(data=DFam, aes(x=0, y=Start, xend=0.5, yend=Start)) 
plot_langs = plot_langs + geom_segment(data=DFam, aes(x=0, y=End, xend=0.5, yend=End)) 
plot_langs = plot_langs + geom_segment(data=DFam, aes(x=0, y=Start, xend=0, yend=End))
plot_langs = plot_langs + geom_text(data=DFam, aes(x=-0.1, y=Mean , label=Family), hjust=1, size=3, colour="grey30")
plot_langs = plot_langs + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
		    panel.background = element_blank(), axis.line = element_blank(),
                    plot.margin=unit(c(0,0,0,0), "mm"),
		    axis.ticks = element_blank()) + labs(x=NULL)
library("gridExtra")
plot_orders_real = plot_orders_real + theme(                    plot.margin=unit(c(0,0,0,0), "mm"))
plot_orders_eff = plot_orders_eff + theme(                    plot.margin=unit(c(0,0,0,0), "mm"))

grid.arrange(plot_langs, plot_orders_real, plot_orders_eff, nrow=1, widths=c(1, 1.2, 1.2))


