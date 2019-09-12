data = read.csv("results.tsv", sep="\t")

library(dplyr)
library(tidyr)
library(ggplot2)

data2 = data %>% filter(probNPBranching == 0.0)


data3 = data2 %>% mutate(probVPBranching_ = round(probVPBranching*10)/10, probObj_ = round(probObj*10)/10)
data3 = data3 %>% group_by(probVPBranching_, probObj_) %>% summarise(Diff = mean(Diff))

plot = ggplot(data=data2, aes(x=probVPBranching, y=probObj, color=Diff)) + geom_point()


plot = ggplot(data=data3, aes(x=probVPBranching_, y=probObj_, color=Diff)) + geom_point()


