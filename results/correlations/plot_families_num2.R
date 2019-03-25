# tiles-real_eff_large-byObj-restricted-viz-pred-large.R



D$LanguageNumeric = as.numeric(D$Language_Ordered)

D$FamilyPrint = as.character(D$Family)
D = D %>% mutate(FamilyPrint = ifelse(FamilyPrint == "Malayo-Sumbawan", "Mal.-Sum.", as.character(FamilyPrint)))
D = D %>% mutate(FamilyPrint = ifelse(FamilyPrint == "Sino-Tibetan", "Sin.-Tib.", as.character(FamilyPrint)))
D = D %>% mutate(FamilyPrint = ifelse(FamilyPrint == "Viet-Muong", "Viet-M.", as.character(FamilyPrint)))


DFam = D %>% group_by(FamilyPrint) %>% summarise(Start = min(LanguageNumeric), End = max(LanguageNumeric), Mean = mean(LanguageNumeric))

DFam$yOffset = 0.2*(1:(nrow(DFam))) 
D$yOffset=NULL
D = merge(D, DFam %>% select(FamilyPrint, yOffset), by=c("FamilyPrint"))


DLang = unique(D %>% select(Language_Ordered, iso_Ordered, LanguageNumeric, yOffset))


plot_orders_real = ggplot(D %>% filter(Type == "Real Languages"), aes(x = 1, y = LanguageNumeric+yOffset, group=CoarseDependency)) + 
  geom_point(aes(fill=DirB, colour = DirB, size =1), position = position_dodge(width=2.0)) +
#  scale_color_gradient() + #values=c("blue", "green")) +
  theme_bw() + theme(axis.text.x=element_blank(), #element_text(size=9, angle=0, vjust=0.3),
                     axis.text.y=element_blank(),axis.ticks=element_blank(),
                     plot.title=element_text(size=11)) +
  theme(axis.title=element_blank()) + 
  theme(legend.position="none") + labs(x=NULL)
plot_orders_eff = ggplot(D %>% filter(Type == "Efficiency"), aes(x = 1, y = LanguageNumeric+yOffset, group=CoarseDependency)) + 
  geom_point(aes(fill=DirB, colour = DirB, size =1), position = position_dodge(width=2.0)) +
#  scale_color_gradient() + #values=c("blue", "green")) +
  theme_bw() + theme(axis.text.x=element_blank(), #element_text(size=9, angle=0, vjust=0.3),
                     axis.text.y=element_blank(),axis.ticks=element_blank(),
                     plot.title=element_text(size=11)) +
  theme(axis.title=element_blank()) + 
  theme(legend.position="none") + labs(x=NULL)
plot_langs = ggplot(DLang) 
plot_langs = plot_langs +  theme_bw() 
plot_langs = plot_langs + theme(axis.text.x=element_blank(), #element_text(size=9, angle=0, vjust=0.3),
                     axis.text.y=element_blank(),
                     plot.title=element_text(size=11)) 
plot_langs = plot_langs + geom_text(aes(x=1.2 + 0.07, y=LanguageNumeric+yOffset, label=iso_Ordered), hjust=1, size=3, colour="grey30")
plot_langs = plot_langs +      	theme(axis.title=element_blank()) 
plot_langs = plot_langs + xlim(-2.0, 1.35)
plot_langs = plot_langs + geom_segment(data=DFam, aes(x=0, y=Start+yOffset, xend=0.5, yend=Start+yOffset)) 
plot_langs = plot_langs + geom_segment(data=DFam, aes(x=0, y=End+yOffset, xend=0.5, yend=End+yOffset)) 
plot_langs = plot_langs + geom_segment(data=DFam, aes(x=0, y=Start+yOffset, xend=0, yend=End+yOffset))
plot_langs = plot_langs + geom_text(data=DFam, aes(x=-0.1, y=Mean+yOffset , label=FamilyPrint), hjust=1, size=3, colour="grey30")
plot_langs = plot_langs + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
		    panel.background = element_blank(), axis.line = element_blank(),
                    plot.margin=unit(c(0,0,0,0), "mm"),
		    axis.ticks = element_blank()) + labs(x=NULL)
library("gridExtra")
plot_orders_real = plot_orders_real + theme(                    plot.margin=unit(c(0,0,0,0), "mm"))
plot_orders_eff = plot_orders_eff + theme(                    plot.margin=unit(c(0,0,0,0), "mm"))

plot = grid.arrange(plot_langs, plot_orders_real, plot_orders_eff, nrow=1, widths=c(1, 1.2, 1.2))
plot


ggsave(plot=plot, "figures/pred-eff-families.pdf")

