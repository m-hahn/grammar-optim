# Control using frequentist regression model

data = read.csv("../../../../grammars/manual_output_funchead_two_coarse_lambda09_best_large/auto-summary-lstm.tsv", sep="\t")
best = read.csv("../../../../grammars/manual_output_funchead_two_coarse_lambda09_best_large/successful-seeds.tsv")

library(forcats)
library(plyr)
library(dplyr)
library(tidyr)
library(ggplot2)
data = data %>% mutate(Language = fct_recode(Language, "Ancient_Greek" = "Ancient", "Old_Church_Slavonic" = "Old"))

data = merge(data %>% mutate(FileName = as.character(FileName)), best %>% rename(FileName = Model), by=c("Language", "FileName"))

languages = read.csv("../../../languages/languages-iso_codes.tsv")
data  = merge(data, languages, by=c("Language"), all.x=TRUE)
dependency = "nmod"
dependencies = c("acl", "aux", "lifted_case", "lifted_cop", "lifted_mark", "nmod", "obl", "xcomp", "obj")
objs = data %>% filter(CoarseDependency == "obj") %>% mutate(obj = pmax(0, sign(DH_Weight))) %>% select(obj, Language, FileName)

data = data %>% filter((CoarseDependency %in% dependencies))
data = merge(data, objs, by=c("Language", "FileName"))
data = data %>% mutate(dir = pmax(0, sign(DH_Weight)))
data = data %>% mutate(dir = ifelse(DH_Weight == 0, NA, dir))
data = unique(data %>% select(Family, Language, FileName, CoarseDependency, dir)) %>% spread(CoarseDependency, dir)

library(lme4)

data_ = data %>% select(obj, acl, aux, lifted_case, lifted_cop, lifted_mark, nmod, obl, xcomp) %>% group_by(obj, acl, aux, lifted_case, lifted_cop, lifted_mark, nmod, obl, xcomp) %>% summarise(count = NROW(acl))%>% group_by() %>% mutate(acl.C = acl-0.5) 
data_ = data_ %>% mutate(aux.C = aux-0.5)
data_ = data_ %>% mutate(lifted_case.C = lifted_case-0.5)
data_ = data_ %>% mutate(lifted_cop.C = lifted_cop-0.5)
data_ = data_ %>% mutate(lifted_mark.C = lifted_mark-0.5)
data_ = data_ %>% mutate(nmod.C = nmod-0.5)
data_ = data_ %>% mutate(obl.C = obl-0.5)
data_ = data_ %>% mutate(xcomp.C = xcomp-0.5)
data_ = data_ %>% mutate(obj.C = obj-0.5)



size = nrow(data)


formula = "count/size ~ obj.C + acl.C + aux.C + lifted_case.C + lifted_cop.C + lifted_mark.C + nmod.C + obl.C + xcomp.C"

while(TRUE) {
  model0 = glm(formula, data=data_, family="binomial", weights=size+0*data_$count)
  bestImprovement = 0
  bestImprovementTerm = NULL
  for(d1 in dependencies) {
	  for(d2 in dependencies) {
		  if(d1 < d2) {
			  cat(d1, " ", d2, "\n")
			  formula2 = paste(formula, "+", paste(d1, ".C", sep=""), "*", paste(d2, ".C", sep=""), sep=" ")
			  model1 = glm(formula2, data=data_, family="binomial", weights=size+0*data_$count)
			  AICDiff = AIC(model0) - AIC(model1)
			  if(AICDiff > bestImprovement) {
				  bestImprovement = AICDiff
				  bestImprovementTerm = formula2
			  }

		  }
	  }
  }
  if(bestImprovement == 0) {
	  break
  }
  formula = bestImprovementTerm
}



edges = data.frame(from=c(), to=c())
vertices = data.frame(from=c("obj", dependencies), type=c("v", "n", "v", "p", "v", "p", "n", "v", "v", "v"))

for(d in vertices$from) {
     edges = rbind(edges, data.frame(from=c(d), to=c(d), thickness=c(0.1)) )   

}
chosen = variable.names(model0)
for(i in (1:length(chosen))) {
  name = chosen[i]
  if(grepl(":", name)) {
     names = strsplit(name, ":")
     name1 = gsub(".C", "", names[[1]][1])
     name2 = gsub(".C", "", names[[1]][2])
     coefficient = abs(coef(model0)[[i]])

     edges = rbind(edges, data.frame(from=c(name1), to=c(name2), thickness=c(coefficient)) )   
  }
}


edges = merge(edges, vertices, by=c("from"))

vertices$from = revalue(vertices$from, c("obj"="Verb-Object"))
vertices$from = revalue(vertices$from, c("obl"="Verb-Adp.Phr."))
vertices$from = revalue(vertices$from, c("acl"="Noun-RelCl"))
vertices$from = revalue(vertices$from, c("nmod"="Noun-Genitive"))
vertices$from = revalue(vertices$from, c("lifted_cop"="Copula-Noun"))
vertices$from = revalue(vertices$from, c("lifted_mark"="Comp.-Sent."))
vertices$from = revalue(vertices$from, c("lifted_case"="Adp.-Noun"))
vertices$from = revalue(vertices$from, c("xcomp"="Verb-Verb Phr."))
vertices$from = revalue(vertices$from, c("aux"="Auxiliary-Verb"))



edges$from = revalue(edges$from, c("obj"="Verb-Object"))
edges$from = revalue(edges$from, c("obl"="Verb-Adp.Phr."))
edges$from = revalue(edges$from, c("acl"="Noun-RelCl"))
edges$from = revalue(edges$from, c("nmod"="Noun-Genitive"))
edges$from = revalue(edges$from, c("lifted_cop"="Copula-Noun"))
edges$from = revalue(edges$from, c("lifted_mark"="Comp.-Sent."))
edges$from = revalue(edges$from, c("lifted_case"="Adp.-Noun"))
edges$from = revalue(edges$from, c("xcomp"="Verb-Verb Phr."))
edges$from = revalue(edges$from, c("aux"="Auxiliary-Verb"))


edges$to = revalue(edges$to, c("obj"="Verb-Object"))
edges$to = revalue(edges$to, c("obl"="Verb-Adp.Phr."))
edges$to = revalue(edges$to, c("acl"="Noun-RelCl"))
edges$to = revalue(edges$to, c("nmod"="Noun-Genitive"))
edges$to = revalue(edges$to, c("lifted_cop"="Copula-Noun"))
edges$to = revalue(edges$to, c("lifted_mark"="Comp.-Sent."))
edges$to = revalue(edges$to, c("lifted_case"="Adp.-Noun"))
edges$to = revalue(edges$to, c("xcomp"="Verb-Verb Phr."))
edges$to = revalue(edges$to, c("aux"="Auxiliary-Verb"))



library(geomnet)

#  , aes(linewidth=thickness)       ) + 

plot = ggplot(data = edges, aes(from_id = from, to_id = to)) +
  geom_net(labelon = TRUE, labelcolour="black", ecolour="grey70", aes(colour=type)) + 
  theme_net()  + 
    scale_x_continuous(expand = c(0.2, 0.2)) +
     theme(legend.position = "none")

ggsave(plot, file="loglinear-pairwise-correlations.pdf")


#
#
#data_ = data %>% select(obj, acl, aux, lifted_case, lifted_cop, lifted_mark, nmod, obl, xcomp, Family) %>% group_by(Family, obj, acl, aux, lifted_case, lifted_cop, lifted_mark, nmod, obl, xcomp) %>% summarise(count = NROW(acl))%>% group_by() %>% mutate(acl.C = acl-0.5) 
#data_ = data_ %>% mutate(aux.C = aux-0.5)
#data_ = data_ %>% mutate(lifted_case.C = lifted_case-0.5)
#data_ = data_ %>% mutate(lifted_cop.C = lifted_cop-0.5)
#data_ = data_ %>% mutate(lifted_mark.C = lifted_mark-0.5)
#data_ = data_ %>% mutate(nmod.C = nmod-0.5)
#data_ = data_ %>% mutate(obl.C = obl-0.5)
#data_ = data_ %>% mutate(xcomp.C = xcomp-0.5)
#data_ = data_ %>% mutate(obj.C = obj-0.5)
#
#
#
#size = nrow(data)
#
#
#
#summary(glmer(count/size ~ obj.C * acl.C + obj.C*aux.C + obj.C*lifted_case.C + obj.C*lifted_cop.C + obj.C*lifted_mark.C + obj.C*nmod.C + obj.C*obl.C + obj.C*xcomp.C + nmod.C * acl.C + lifted_case.C * obl.C + aux.C * xcomp.C + obl.C * nmod.C + lifted_case.C * acl.C + obl.C * acl.C + aux.C * lifted_cop.C + lifted_mark.C * aux.C + lifted_mark.C * aux.C + lifted_mark.C * xcomp.C + (1+obj.C|Family), data=data_, family="binomial", weights=size+0*data_$count))
#
#













#
#
#sink("../analysis_frequentist/efficiency-results-lme4.tsv")
#for(dependency in dependencies) {
#     model = summary(glmer(paste(dependency, " ~ (1|Family) + (1|Language)"), family="binomial", data=data))
#     p = (coef(model)[4])
#     beta = coef(model)[1]
#     se = coef(model)[2]
#     z = coef(model)[3]
#     cat(dependency,  beta, se, z, p, "\n", sep="\t")
#}
#sink()
#
#
#
###################################
#
#data = merge(data, objs, by=c("Language", "FileName"))
#summary(glmer(paste(dependency, " ~ obj + (1|Family) + (1|Language)"), family="binomial", data=data))
#
