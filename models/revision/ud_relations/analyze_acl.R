data = read.csv("relations.tsv", header=FALSE, sep="\t")
names(data) <- c("Coarse", "Dep", "Language", "Count", "DH_Proportion")


library(tidyr)
library(dplyr)

data_acl = data %>% filter(Dep == "acl") %>% rename(Count_acl = Count, DH_Proportion_acl = DH_Proportion) %>% mutate(Dep = NULL)
data_relcl = data %>% filter(Dep == "acl:relcl") %>% rename(Count_relcl = Count, DH_Proportion_relcl = DH_Proportion) %>% mutate(Dep = NULL)

data2 = merge(data_acl, data_relcl, by=c("Language", "Coarse"))


