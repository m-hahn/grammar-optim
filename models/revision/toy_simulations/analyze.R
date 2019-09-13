data = read.csv("results.tsv", sep="\t")

library(dplyr)
library(tidyr)
library(ggplot2)

data2 = data %>% filter(probNPBranching == 0.0)


data3 = data2 %>% mutate(probVPBranching_ = round(probVPBranching*10)/10, probObj_ = round(probObj*10)/10) %>% filter(probVPBranching_ > 0.0,probVPBranching < 0.5,  probObj_ < 1.0, probObj_ > 0.1)
data3 = data3 %>% group_by(probVPBranching_, probObj_) %>% summarise(Diff = mean(Diff))

#plot = ggplot(data=data2, aes(x=probVPBranching, y=probObj, color=Diff)) + geom_point()
#plot = ggplot(data=data3, aes(x=probVPBranching_, y=probObj_, color=-Diff)) + geom_point()

plot = ggplot(data=data3, aes(x=probVPBranching_, y=probObj_)) + geom_tile(aes(fill=-Diff))
plot = plot +     theme(text = element_text(size=20),    axis.text = element_text(size=15)) 
plot = plot + xlab("p(branching)") + ylab("p(obj)")
plot = plot + theme(legend.title=element_blank())
plot = plot + labs(title="p(acl) = 0.0")
plot = plot + theme(plot.title = element_text(hjust = 0.5, size=25))

plot = plot +  scale_fill_gradient2()

ggsave(plot, file="result_NPBranching0_0.pdf")


data4 = data %>% filter(probNPBranching == 0.3)
data5 = data4 %>% mutate(probVPBranching_ = round(probVPBranching*10)/10, probObj_ = round(probObj*10)/10) %>% filter(probVPBranching_ > 0.0, probVPBranching < 0.5, probObj_ < 1.0, probObj_ > 0.1)
data5 = data5 %>% group_by(probVPBranching_, probObj_) %>% summarise(Diff = mean(Diff))


plot = ggplot(data=data5, aes(x=probVPBranching_, y=probObj_)) + geom_tile(aes(fill=-Diff))
plot = plot +     theme(text = element_text(size=20),    axis.text = element_text(size=15)) 
plot = plot + xlab("p(branching)") + ylab("p(obj)")
plot = plot + theme(legend.title=element_blank())
plot = plot + labs(title="p(acl) = 0.3")
plot = plot + theme(plot.title = element_text(hjust = 0.5, size=25))

plot = plot +  scale_fill_gradient2()

ggsave(plot, file="result_NPBranching0_3.pdf")



