source("../tiles-real_eff_large-byObj-restricted-viz-pred-large-dlm.R")



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


D = D %>% mutate(CoarseDependency = recode(CoarseDependency, lifted_case=1, lifted_cop=2, aux=3, nmod=4, acl=5, lifted_mark=6, obl=7, xcomp=8))

plot_orders_real = ggplot(D %>% filter(Type == "Real Languages"), aes(x = 1, y = LanguageNumeric+yOffset, group=CoarseDependency)) + 
  geom_point(aes(fill=DirB, colour = DirB, size =1), position = position_dodge(width=2.0)) +
#  scale_color_gradient() + #values=c("blue", "green")) +
  theme_classic() +
  #theme_bw() + 
  theme(axis.text.x=element_blank(), #element_text(size=9, angle=0, vjust=0.3),
                     axis.text.y=element_blank(),axis.ticks=element_blank(),
                     plot.title=element_text(size=11)) +
  theme(axis.title=element_blank()) + 
  theme(legend.position="none") + labs(x=NULL) +
  scale_x_continuous(breaks = NULL) +
  scale_y_continuous(breaks = NULL)

plot_orders_eff = ggplot(D %>% filter(Type == "Efficiency"), aes(x = 1, y = LanguageNumeric+yOffset, group=CoarseDependency)) + 
  geom_point(aes(fill=DirB, colour = DirB, size =1), position = position_dodge(width=2.0)) +
#  scale_color_gradient() + #values=c("blue", "green")) +
  theme_classic() +
 theme(axis.text.x=element_blank(), #element_text(size=9, angle=0, vjust=0.3),
                     axis.text.y=element_blank(),axis.ticks=element_blank(),
                     plot.title=element_text(size=11)) +
  theme(axis.title=element_blank()) + 
  theme(legend.position="none") + labs(x=NULL) +
  scale_x_continuous(breaks = NULL) +
  scale_y_continuous(breaks = NULL)

plot_langs = ggplot(DLang) 
plot_langs = plot_langs +  theme_classic() 
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
ggsave(plot=plot, "../figures/pred-eff-families-dlm.pdf", width=4, height=8)





plot_langs2 = plot_langs + annotate("text", label="", x=1, y=58.5, size=6)





plot_orders_real2 = plot_orders_real + annotate("text", label="Real", x=1, y=58.5, size=6)
plot_orders_real2 = plot_orders_real2 + geom_point(data=data.frame(num=c(1,2,3,4,5,6,7,8)), aes(x=0.25 * num - 0.12, group=NA, y=56.7, colour=NA, fill=NA), color="black", fill=NA, size=4.5, shape=21)
plot_orders_real2 = plot_orders_real2 + geom_text(data=data.frame(CoarseDependency=unique(D$CoarseDependency), num=c(1,2,3,4,5,6,7,8)), aes(x=0.25 * num - 0.12, group=CoarseDependency, y=56.55, label=as.character(num)))
plot_orders_real2



plot_orders_eff2 = plot_orders_eff + annotate("text", label="Optimized", x=1, y=58.5, size=6)
plot_orders_eff2 = plot_orders_eff2 + geom_point(data=data.frame(num=c(1,2,3,4,5,6,7,8)), aes(x=0.25 * num - 0.12, group=NA, y=56.7, colour=NA, fill=NA), color="black", fill=NA, size=4.5, shape=21)
plot_orders_eff2 = plot_orders_eff2 + geom_text(data=data.frame(CoarseDependency=unique(D$CoarseDependency), num=c(1,2,3,4,5,6,7,8)), aes(x=0.25 * num - 0.12, group=CoarseDependency, y=56.55, label=as.character(num)))
plot_orders_eff2




plot = grid.arrange(plot_langs2, plot_orders_real2, plot_orders_eff2, nrow=1, widths=c(1, 1.2, 1.2))
plot

ggsave(plot=plot, "../figures/pred-eff-families-2-dlm.pdf", width=4, height=8)

D2 = (D %>% select(Family, Language, CoarseDependency, DirB, Type) %>% spread(Type, DirB) %>% rename(Real = 'Real Languages') %>% rename(Predicted = Efficiency))

D2$Agree = (D2$Real == D2$Predicted)
summary(glmer(Agree ~ (1|CoarseDependency) + (1|Family), data=D2, family="binomial"))

mean(D2$Agree)



D %>% filter(Type == "Efficiency") %>% filter(Language == "English")




