data = read.csv("pareto-data.tsv")


library(dplyr)
library(tidyr)

total = data.frame()
for(language in unique(data$Language)) {

   data_ = read.csv(paste("~/CS_SCR/posteriors/pareto-smooth/pareto-", language, sep="")) %>% mutate(Language=language)
   total = rbind(total, data_)
}	

write.csv(total, "~/CS_SCR/posteriors/pareto-smooth/pareto-total.csv")
